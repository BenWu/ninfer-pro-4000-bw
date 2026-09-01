# E8 lattice KV cache port — working note / handoff (2026-08-31)

Status: **Fixed, with op-test coverage.** The port compiles, runs under CUDA Graphs, and
produces correct text in both `rk4v4-e8` and `rk2v4-e8`. Five defects were found and repaired,
all in the E8 attention path (section 2). Debug instrumentation has been removed and the port now
has op tests (section 4), which is how the last three defects surfaced.

---

## 1. Goal & design contract (short)

Port E8 lattice KV-cache storage from the `ninfer-4090` fork into `ninfer`, on top of the
existing INT8 G64 codec instead of a parallel storage family.

- `RK4V4E8` : K = E8 lattice-projected 4-bit codes packed 2-per-byte (head extent `D/2`),
  V = packed 4-bit (head extent `D/2`).
- `RK2V4E8` : K = 2-byte E8 root cylinder code per 8 rotated dims (head extent `D/4`),
  V = packed 4-bit (head extent `D/2`).
- Both modes H64-rotate K and V per 64-dim group (`kv_cache_hadamard64`) before quantization.
  Per-group FP16 scale uses the `/7` boundary (`/127` for INT8).
- Q is **H64-rotated per 64-dim group only** (matches the H64-rotated K cache), NOT the D256
  prepass the INT8 path uses.
- The PV result lives in the H64-rotated value space and is un-rotated by
  `detail::kv_cache_inverse_rotate_output_kernel` after attention. Both launchers
  (`prompt.cu`, `small_t.cu`) apply it under `if (cache.v_packed_i4)`.
- E8 key byte layout: RK2 stores **consecutive `(root, rad)` byte pairs** per 8-dim block
  (`raw[2b], raw[2b+1]`); RK4 stores packed-4 nibbles. Both match `kv_cache_append_e8_group`.

Constants: `kKVCacheInt8HeadDim=256`, `kKVCacheInt8Group=64`; E8 key raw bytes/row =
`D/2` (RK4) or `D/4` (RK2); V raw = `D/2`.

## 2. Defects found and fixed

Five defects, all in the E8 attention/codec path, none of them in the E8 encoder arithmetic.
The first two produced the all-zero tokens; the last three were found later by the new op tests
(section 4) and would each have taken the engine down on their own.

### 2.1 Prompt kernel decoded only a quarter of each key tile

`prompt_e8.cuh`, per-key-block loop. The staged E8 key codes were decoded into the INT8 QK arena
by the four producer warps with

```cpp
if (warp < ProducerWarps) {
    for (int chunk = lane; chunk < 16 * Groups; chunk += 32) {   // 16, not Bc
        const int key_l = chunk / Groups;                        // 0..15 only
```

`Bc` is 64, so `key_l` never exceeded 15: rows 16..63 of `k_i8` were never written, and all four
producer warps redundantly decoded the same rows 0..15. Every QK `ldmatrix` covers all `Bc` rows,
so three quarters of every key tile was stale shared memory left by whatever CTA previously
occupied the SM. The INT8 kernel does not have this bug because its key codes land in `k_i8`
directly via `cp.async` over the full `Bc * (D/16)` chunk range.

Fix: decode the whole tile cooperatively across the CTA and publish it with a `__syncthreads()`
before the producer/consumer split. Distributing the decode per producer warp is not an option:
a warp reads rows it did not write, and the QK branch cannot host a `__syncthreads()` because the
V-worker warps are in the sibling branch.

### 2.2 Small-T kernel fed FP16 bit patterns to a BF16 MMA

`small_t_e8.cuh`, `causal_small_t_e8_dequant_f16x8`. The helper built its values with
`__floats2half2_rn` / `__hmul2`, i.e. **FP16** pairs, and stored them into `v_bf16` - a
`__nv_bfloat16` tile consumed by `mma_bf16`. Every decoded V element was therefore an FP16 bit
pattern reinterpreted as BF16: `1.0f` decodes as `0.0078`, `100.0f` as `~5e13`. The INT8 sibling
`kv_cache_int8_dequant_i8x8_from` correctly uses `pack_bf16x2`; the E8 helper had been copied from
the prompt kernel, where the V tile genuinely is `__half` and the PV MMA is `mma_f16`.

This is the source of the NaN chain the earlier investigation chased. Scrambled V of magnitude
~1e13 flows into the residual stream, the SwiGLU MLP squares it, and within a layer or two the
BF16 residual saturates to inf; RMSNorm then produces `inf * 0 = NaN`. That is why the very first
NaN appeared in `input.v` at the layer-1 append rather than in the attention output being printed
- the corruption was upstream of the append, in the previous layer's attention result.

Fix: `causal_small_t_e8_dequant_bf16x8`, packing with `pack_bf16x2` and taking the scale as
`float`, mirroring the INT8 path.

### 2.3 Prompt V dequant stored 4-byte values at 2-byte strides

Same file, the packed-value decode loop:

```cpp
for (int i = 0; i < 8; ++i) { *reinterpret_cast<unsigned*>(&dst[i]) = ...; }  // dst is __half*
```

`&dst[i]` advances one `__half` (2 bytes) per step while each store writes 4 bytes, so the writes
overlap, spill past the row, and `&dst[1]` is only 2-byte aligned. Confirmed by A/B: rebuilding
with the pre-fix header aborts on `cudaDeviceSynchronize: misaligned address` at T=17. The 16
codes also span two swizzle blocks, which a single linear run cannot address.

Fix: compute two swizzled bases (`d`, `d + 8`) and store four `unsigned` at each.

### 2.4 Divergent full-mask `__shfl_sync` in the prompt V decode (kernel hang)

The same loop walks `Bc * (D/16)` chunks, i.e. **sixteen** chunks per key row against 32 lanes,
so one warp covers **two** key rows. The causal test inside it,

```cpp
if (key <= max_query_abs) { ... __shfl_sync(FullMask, vs, ...); ... } else { ...zeros... }
```

is therefore not warp-uniform: whenever a tile's last visible key lands on an even row, lanes
0-15 take the decode branch and lanes 16-31 take the zero branch, and the full-mask shuffle
inside the taken branch hangs the warp. cuda-gdb on the stalled kernel shows exactly that - warp
12 with active mask `0xffff0000` and divergent mask `0x0000ffff` while every other warp sits at
the next barrier.

The condition is a parity: a CTA is safe only when the absolute position of the last query it
covers is odd. For a fresh prefill the full 64-row blocks always end on an odd position, but the
final partial block ends at position `N - 1`, so **every odd-length prompt hangs** and every even
one survives. The engine runs that appeared to validate the port (68 tokens, 32K, the 32K needle)
were all even-length. Continuation chunks starting at an odd base position hang on full blocks
too. The INT8 kernel is immune because its V loop uses `D/8` chunks per row, so one warp covers
exactly one key row and the branch is warp-uniform.

Fix: hoist the scale load and its broadcast above the causal branch, so the shuffle is executed
by the whole warp unconditionally. An out-of-range key row still indexes inside `v_scale_s`.

### 2.5 Duplicate kernel registration for the inverse rotation

`kv_cache_inverse_rotate_output_kernel` was moved into `namespace detail` with a comment claiming
internal linkage. A namespace does not change linkage: the template still had external linkage,
every TU including the header emitted the same mangled entry, and the RDC device linker
registered each one. compute-sanitizer reports `Duplicate entry kernels named
"_ZN6ninfer3ops6detail37kv_cache_inverse_rotate_output_kernelILi24EEE..." detected. The Runtime
will use the one that was registered first`. It is latent in the product binary and immediate in
any test binary, which links the ops archive whole and so pulls in both `prompt.cu` and
`small_t.cu`. This is the same hazard the earlier session saw as a graph-capture MMU fault.

Fix: `static __global__`, which does give the template internal linkage, so each TU gets its own
entry and there is nothing to collide.

### 2.6 What was NOT the problem

- The E8 encoder arithmetic, pack/unpack, H64 rotation, E8 root byte order, scales, and index
  math are all correct.
- The inverse output rotation was already enabled in both launchers; the earlier note that it was
  stubbed out with `if (false && ...)` was stale.
- There is no workspace/page aliasing between the shrunk E8 code planes and the BF16 workspace.

## 3. Uncommitted real fixes (KEEP)

| File | Change | Why |
|---|---|---|
| `.../causal_cache/prompt_e8.cuh` | cooperative full-`Bc` key decode + `__syncthreads()` before the producer split | §2.1 |
| `.../causal_cache/prompt_e8.cuh` | V decode stores each 8-half swizzle block at its own base | §2.3 |
| `.../causal_cache/prompt_e8.cuh` | scale broadcast hoisted above the causal branch, 4-lane quadrant source | §2.4 |
| `.../causal_cache/small_t_e8.cuh` | `causal_small_t_e8_dequant_bf16x8` replaces the FP16 helper | §2.2 |
| `.../causal_cache/small_t_e8.cuh` | `__syncthreads()` after the `physical_pages_s` fill, before the writes_cache append | page-table smem race; was the graph-capture MMU fault |
| `src/ops/kv_cache/int8_g64_codec.cuh` | `static __global__` on `kv_cache_inverse_rotate_output_kernel` | §2.5 |
| `src/ops/kv_cache/d256_profile.h` | `D256KVCacheProfile` gains `k_leading_extent`/`v_leading_extent`; `d256_kv_cache_profile(dtype, v_packed, k_packed, k_e8_root)` returns `U8` code dtype + halved/quartered extents | op validators rejected the U8/partial planes |
| `.../causal_cache/causal_softmax_attention.cpp`, `kv_cache/append/kv_cache_append.cpp` | validators pass view flags into the profile and use the per-plane extents | same |
| `.../causal_cache/small_t.cu` | `single_row_batch_view` copies `.v_packed_i4/.k_packed_i4/.k_e8_root` | cached small-T route would otherwise run the INT8 kernel on U8 pointers |
| `tests/CMakeLists.txt` | link `ninfer_*` static archives whole; register the two new E8 tests | RDC kernel entries live in `.cudaubsdata`, which the host linker cannot see |
| `tests/ops/test_e8_root_codec.cu` (new) | ported from the fork | §4 |
| `tests/ops/softmax_attention/causal_cache_e8.cu` (new) | E8 attention correctness | §4 |
| `tests/ops/softmax_attention/main.cpp` | runs the E8 entry | §4 |

`src/ops/softmax_attention/dense/causal_cache/e8_debug_scan.cuh` and `tests/cuda_rdc_stub.cu` are
untracked leftovers; the debug scanner is no longer referenced from any source file and can be
deleted.

## 4. Op-test coverage (new)

The fork's only E8 op test is `tests/ops/test_e8_root_codec.cu`, which covers the root-cylinder
*encoder* and nothing else - no test at any level touches the E8 cache planes or the attention
kernels that read them. That is how §2.1-§2.5 reached the engine. Both gaps are now closed.

- **`tests/ops/test_e8_root_codec.cu`** - ported from the fork verbatim apart from the codec's
  path (`src/ops/kernel/` -> `src/ops/kv_cache/`). Warp-encoder/scalar-encoder parity plus the
  zero-subgroup contamination guard. `op_tester.h` and `e8_root_codec.cuh` are byte-identical
  between the trees, so nothing else needed adapting. Passes on sm_120a.
- **`tests/ops/softmax_attention/causal_cache_e8.cu`** - new, no fork counterpart. Drives both
  entries (`causal_softmax_attention`, `causal_softmax_attention_cached`) for both modes, both
  head geometries, all three page mappings, and token counts that straddle the small-T/prompt
  boundary and a 64-key tile. The reference reads the code and scale planes back after the call
  and decodes them on the host (unpack -> per-group scale -> inverse H64), so it reproduces no
  tile, swizzle, MMA, split, or reduction tree from the kernels; cache *encoding* is out of
  scope by design. It is a `.cu` only so the RK2 root decode can be tabulated once from the
  production header over all 65536 code pairs.

The E8 criterion is set from measurement, not guesswork: over all 72 comparisons, max
relative-L2 3.65e-3 and max absolute error 5.62e-3 against a max reference of 1.16. The gross
term is dominated by the inverse H64 rotation, which reads BF16, mixes 64 components and writes
BF16, and is widest on a single-key tile where nothing averages the error down.

## 4b. Verification (PRO 4000, device 1, `models/qwen3_8_27b.ninfer`)

- `17 * 23` with `--no-thinking --greedy`: `int8`, `rk4v4-e8`, `rk2v4-e8` all answer `391`.
- 32K prefill (`models/long_prompt_32tok.json`): all three reply `OK`.
- Needle at 32K (passphrase planted mid-document): all three return `HORIZON-4471-QUARTZ`.
- `ninfer_softmax_attention_test` (incl. the new E8 entries), `ninfer_e8_root_codec_test`,
  `ninfer_kv_cache_append_test`, `ninfer_kv_cache_test` pass.

## 5. Remaining work for Workstream B

- **Done 2026-09-01.** The B.3 measurement sweep ran; results and verdict are in
  `docs/plans/rtx-pro-4000-sm120-port-plan.md` under "B.3 results". Capacity at 262144 and all
  retrieval gates pass; prefill misses the ~2% gate at every length (4.9% to 21.9%, widening
  with context) and rk2v4-e8 exceeds the 6% decode budget at 128K (7.7%) while rk4v4-e8 stays
  flat. The 5-needle @118K fixture does not exist in either tree and was not run.
- Open question left by that sweep: whether the prefill cost is acceptable, and whether
  rk2v4-e8 ships given its long-context decode tax. Product call, not a measurement one.
- **Done 2026-09-01.** Append-side E8 coverage now lives in `tests/ops/test_kv_cache_append_e8.cu`
  (target `ninfer_kv_cache_append_e8_test`), which compares whole planes byte for byte against an
  oracle rather than round-tripping through the read path. RK4V4E8 key codes are recomputed on the
  host (H64 rotation, group scale, E8 projection, 4-bit pack); RK2V4E8 key codes come from running
  the warp cylinder encoder over the host-rotated key. Both modes also check V codes, group
  scales, guard bytes, the zeroed-group path, and a page-crossing 129-token case.
- Odd-length prompt shapes: done. Beyond the 31 and 23 token engine runs, the B.3 sweep
  prefilled 62,905 and 123,575 token prompts (both odd) in both E8 modes without a hang, which
  exercises the §2.4 parity fix at scale.
- **Done 2026-09-01.** Decode CUDA Graph profiles re-qualified for both E8 modes against the
  ordinary profile family: 15 of 15 graph vs `--no-cuda-graph` comparisons identical across five
  frontier ranges. Details and the caveats on shape weighting are in the port plan under "Decode
  CUDA Graph qualification". The MTP and dflash profile families are still unqualified for E8.
- **Done 2026-09-01.** NVFP4-weight coverage recorded in the port plan under D.3. NVFP4 is the
  larger artifact here (18.98 GiB versus 15.92), which halves every context ceiling, while prefill
  runs about 2.4x faster; the E8 prefill tax roughly doubles under it. With NVFP4 weights
  `rk2v4-e8` is the only mode reaching 262144.
- **Done 2026-09-01.** Workstream C (MTP3 fits at 262144 on both E8 modes, rk4v4-e8 at full
  acceptance parity with int8) and D.5 (prefill preset written, d2d transfer fit rejected) are
  recorded in the port plan. New retrieval fixtures that withhold their answers now exist; both E8
  modes retrieve correctly at 111K and on the 5-needle 118K case.

## 6. Environment / toolchain notes

- RTX PRO 4000 Blackwell (GPU1), CC 12.0 / sm_120a, 70 SMs, ECC on, 24 GiB. CUDA 13.3
  (`/opt/cuda`), nvcc V13.3.73, driver 610.57.04.
- Tests: `CUDA_VISIBLE_DEVICES=1 ./build/tests/<test>`. `NINFER_OP_REPORT_STATS=1` dumps the
  measured error for every comparison, which is how the E8 criterion above was set.
- `ptrace_scope` blocks attaching to a running process, but cuda-gdb can launch one:
  `cuda-gdb -q -batch -x cmds --args ./build/tests/<test>`, then SIGINT the debugger from another
  shell and let the script run `info cuda warps`. The active/divergent lane masks in that table
  are what identified §2.4.
- **PTX JIT: measured, and it does not happen on this target.** `CMAKE_CUDA_ARCHITECTURES` is a
  bare `120a` rather than `120a-real`, so every relocatable object ships PTX beside its cubin
  (`build/apps/ninfer` carries a 123 MB `__nv_relfatbin`), and on the sm_89 fork the driver
  JIT-links that on first launch: 380.9 s of silent startup and 1.6 GB written to the JIT cache.
  Measured on this tree by the ninfer-4090 session (2026-09-01, device 1, int8, 8192 context,
  `CUDA_CACHE_PATH` pointed at an empty scratch dir): weights 2.553 s, engine construction
  4.086 s, prefill 2.071 s, **0 bytes written to the JIT cache and 0 stack samples in
  libnvidia-ptxjitcompiler**. So the mechanism is present in the binary but the driver never
  invokes it on 120a, and the ~3.7 s engine construction seen all session is the real cost, not
  a warm-cache artifact. No timing in this document is polluted by JIT.
  If you ever need to re-check on another arch, the signature is distinctive - silent, one core
  at 100% user, GPU at 0% SM - and `--no-cuda-graph` skips the first launch to isolate it.
  `~/.nv/ComputeCache` defaults to a 1 GiB cap and each device-code rebuild invalidates entries.
  Why sm_89 triggers a runtime device link and 120a does not is unexplained and unpursued.
- Ninja does not always recompile `small_t.cu`/`prompt.cu` when an included `.cuh` changes.
  Force with
  `rm -f build/src/CMakeFiles/ninfer_ops.dir/ops/softmax_attention/dense/causal_cache/{small_t,prompt}.cu.o build/src/CMakeFiles/ninfer_ops.dir/ops/kv_cache/append/launch.cu.o`
  before rebuilding. The `small_t.cu` TU takes ~5-10 min.

### Reproduction
```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer --device 1 \
  --prompt 'What is 17 * 23? Answer with only the number.' \
  --max-context 16384 --kv-capacity auto --kv-dtype rk2v4-e8 \
  --max-new 64 --greedy --no-thinking
```
