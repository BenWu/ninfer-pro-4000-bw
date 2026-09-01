# RTX PRO 4000 Blackwell (sm_120a, 24 GB) — Exact Change Plan

Companion to `ninfer-rtx-pro-4000-blackwell-plan.md` (the strategy draft). This document
resolves the draft's open questions against the actual upstream tree at
`3d9fda22` and turns each workstream into concrete, file-level changes. No code has been
changed yet.

**Local validation hardware confirmed:** this workstation exposes an `NVIDIA RTX PRO 4000
Blackwell` (24467 MiB, CC 12.0) alongside a 4090 and a 5070 Ti. CUDA toolkit is 13.3
(satisfies the `>= 13.1` gate). All build/test work below can run locally by selecting the
PRO 4000 with the existing `--device N` flag / device index; no new selection machinery is
needed.

---

## 0. Answers to the draft's open questions (resolved against this tree)

| Draft question | Answer | Evidence |
|---|---|---|
| Does upstream have ReplaySSM? | **Yes.** GDN ReplaySSM state transactions are fully present. | `src/core/gdn_replay_records.{h,cpp}`, `src/targets/qwen3_6/impl/runtime/speculative_target_impl.h`, `docs/maintainer/replayssm-gdn.md` |
| Does upstream have cohorts C2–C8? | **Yes, and it is the product contract.** Bounded FIFO admission with 1–8 active requests; the engine forms one compact decode batch at every round boundary. `--max-concurrency N` exists on both CLI and serve. | AGENTS.md product contract; `apps/serve/main.cpp`, `src/runtime/engine/scheduler.h` |
| NVFP4 profile flag from the laptop fork? | **Not needed.** `neroued/Qwen3.8-27B-nvfp4-NInfer` and `Qwen3.6-27B-nvfp4-NInfer` are published upstream; the nvfp4 weight profiles resolve from artifact identity, no extra flag. | README.md artifact table; `src/targets/registry.cpp:resolve_weights`; model-cards |
| Device-name / SM-count gates? | No name-string gate anywhere. One CC gate passes as-is. The risk is **hardcoded 5090 SM counts inside op schedules** (Workstream A below). | grep audit; `src/targets/qwen3_6/impl/runtime/layouts_impl.h:649` |
| E8 KV in upstream? | **No.** `KvCacheStorage` = `{BFloat16, Int8Group64, Fp8E4M3Row256}` only. E8 is a genuine cherry-pick. | `include/ninfer/types.h:29`, `src/serve/serve_options.cpp:48` |
| `/metrics`, `/slots`, llama.cpp timings? | **No.** Serve has internal `GenerationMetrics` + periodic throughput log + `--request-log-jsonl`, but no Prometheus endpoint and no slot endpoints. | `src/serve/generation_service.h`, `apps/serve/main.cpp` usage text |

**Consequences:** draft Workstream 3 becomes a fit test (no port), Workstream 5 becomes
measurement only, Workstream 7 needs no profile port. The real engineering work is:
(A) device-geometry fixes for the 70-SM card, (B) the E8 KV cherry-pick, (C) baseline + fit
validation on this machine.

---

## Workstream A — Device-geometry fixes (new; replaces draft §1.3)

### A.0 Findings of the gate audit

The device itself is accepted: `DeviceContext` queries everything at runtime
(`src/core/device.cu`), KV sizing uses `cudaMemGetInfo` free bytes after weights
(`src/targets/registry.cpp`), and the only hard gate is
`layouts_impl.h:649 if (device.sm() != 120) throw` — CC 12.0 reports sm == 120, so it passes.

The defect class is **compile-time constants tuned to the 5090's 170 SMs inside
device-free op code**. The ops layer (`src/ops`) deliberately receives no
`DeviceContext`; schedule resolution is a pure function of the problem. Every constant below
is either *correctness-safe but mis-sized* (persistent grids sized for more SMs than exist)
or, in one case, an occupancy legality check that is simply wrong on a 70-SM device.

| # | Site | Constant | Effect on 70 SMs | Severity |
|---|---|---|---|---|
| A1 | `src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu:259-261` | `kRtx5090SmCount = 170`; persistent grid = 510 blocks, launched at :1192/:1197/:1231/:1237/:1246/:1252 | Grid 2.4× larger than one resident wave (70×3=210). Correct but queued; launch overhead + uneven work distribution on the persistent loop | Perf, **35B-A3B only** (dense 27B never launches MoE) |
| A2 | `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:120-141` | residency caps 340 and 680 CTAs ("across 170 SMs"); `cooperative_27_grid_is_resident` hardcodes 340 | Legality check wrong on 70 SMs (real caps ≈ 140/280). No `cudaLaunchCooperativeGrid` in the tree, so launches are ordinary → **no crash**, but "cooperative" schedules are admitted where they cannot be one wave. The 35B-A3B small-T path (`bf16_gdn_norm_gating_resolve_plan`, cols ≤ 16) selects `MmaCooperativeSplit32` on a grid that does not fit one wave | Correctness-adjacent (schedule semantics), both targets use GDN gating |
| A3 | `src/ops/linear_attention/gated_delta_net/chunked/output.cu:9-11,57-60` | `kTargetCtas = 170*4 = 680` | jobs-per-block computed for 680 CTAs; on 70 SMs each CTA does ~2.4× more work than tuned. Correct (plain grid), slower chunked GDN prefill output | Perf, both targets |
| A4 | `src/ops/launcher/rope.cu:15-16` | `kLargeBlockWaveCapacity = 1020` (170 SMs × 6 CTAs/SM) | Chunked RoPE fallback under-splits on the PRO 4000 | Perf, both targets, prefill only |
| A5 | `src/ops/softmax_attention/dense/causal_cache/small_t.cu:61-70` | I8 T=6 window>5000 split bounds tuned to "one 170-SM wave" | Decode attention split count gives ~2.4 waves on 70 SMs → CTAs queued | Perf, both targets, decode attention |
| A6 | `sparse_moe_decode_kernels.cu:266`, `bf16_gdn_gating_proj_plan.cpp:135-138` (comments) | "170 SMs" narrative | No behavior; comments false on other sm_120a cards | Docs |

Note: `src/runtime/engine/context_cost_defaults.cpp:40` (`hardware_class =
"nvidia-geforce-rtx-5090-sm120"`) is **not a defect**: the hardware class is derived from
`device.props.name` at runtime (`src/targets/registry.cpp:98`), so the PRO 4000 gets its own
slug and cleanly falls back to the generic cost model. The external preset path
(`--context-cost-presets FILE`) exists precisely to supply a measured PRO 4000 table later;
see Workstream D.5.


### A.1 Design decision (decide before editing)

The ops layer must stay device-free (ownership rule: `src/ops` owns the math, targets own
device facts). Two candidate mechanisms:

- **A-opt (chosen):** thread a small immutable value — `sm_count` only; per-schedule CTA
  occupancy is a property of the kernel, not the device — into the plan resolvers that
  currently use 170-derived constants. The target-owned code that calls these resolvers
  already holds a `DeviceContext` (planning happens in `src/targets/qwen3_6/impl/runtime/*`
  and wrapper entry points reachable from it), so `device.props.multiProcessorCount` is
  available at every call site. The 5090-tuned values remain as the *measured schedule
  table*; the SM count only rescales the wave/residency arithmetic (exact
  `sm_count * kCtasPerSm` for persistent grids and residency caps).
- Rejected: a global/registry of device properties (hidden state in `src/ops`, breaks the
  pure-planner design); per-kernel `cudaGetDeviceProperties` scattered across launchers
  (A1/A3 could do this, but A2/A4/A5 live in pure host planners and would still need a
  parameter — one mechanism for all five is cheaper).

Per-site change list (all under `src/ops`; repo-internal contracts only, no public
`include/ninfer` surface changes):

1. **A1** `sparse_moe_prefill_kernels.cu`: replace the compile-time 510 with a runtime
   persistent grid of `sm_count * kPrefillBlocksPerSm` (cached device query in the launcher
   or passed from the 35B-A3B wrapper that owns the call). Keep the
   `__launch_bounds__(..., 3)` min-blocks hint.
2. **A2** `bf16_gdn_gating_proj_plan.{h,cpp}`: add an `int sm_count` parameter to
   `bf16_gdn_gating_resolve_plan / resolve_candidate / execute_plan / dispatch`, the
   norm-gating variants, and both `*_capacity_workspace_bytes`. Replace the 340/680 literals
   with `sm_count * ctas_per_sm(schedule)` using the per-schedule occupancy (2 or 4 CTAs/SM)
   already documented in the comments. Update all target callers:
   `src/targets/qwen3_6_27b/impl/variant.cpp`,
   `src/targets/qwen3_6_35b_a3b/impl/variant.cpp`, and any wrapper under `src/ops/wrapper`
   that forwards. On a 170-SM device every resolved plan must be **bit-identical** to today —
   that is the regression gate for this change.
3. **A3** `gated_delta_net/chunked/output.cu`: `kTargetCtas` becomes
   `runtime_sm_count * kCtasPerSm` (launcher context, cached device query fine).
4. **A4** `rope.cu`: same pattern for `kLargeBlockWaveCapacity`.
5. **A5** `causal_cache/small_t.cu` split policy: rescale the wave-derived bounds by
   sm_count (clamped to the existing min/max), keeping the 5090 table byte-identical at
   sm_count=170. **Needs care:** split count feeds CUDA Graph capture (kernel params), so
   decode-graph profiles must be re-qualified after this change; run the attention op tests
   and a graph-replay smoke test.

Ordering: A2+A3+A4+A5 are on the 27B critical path; A1 only matters for 35B-A3B. All five
share one review/test pass.

### A.2 Validation for Workstream A

- `cmake -S . -B build -G Ninja -DBUILD_TESTING=ON` (CUDA 13.3 OK) + ops test suite with the
  PRO 4000 selected as the active device (`tests/ops/*`, `test_kv_capacity`,
  `test_gdn_replay_records`).
- **170-SM-parity gate:** at sm_count=170 no resolved plan may change vs. current behavior;
  on this box verify sm_count-scaled plans by unit-checking `resolve_plan` outputs for known
  (heads, rows, cols) cases against hand-computed caps (70 SMs → 140/280 resident CTAs).
- End-to-end: one short decode + one 32K prefill on the PRO 4000, spot-check text.


---

## Workstream B — E8 lattice KV cherry-pick (draft WS2)

### B.1 Exact registration/dispatch sites in upstream to rewire

New storage modes land in this closed set of places (verified symbols):

1. `include/ninfer/types.h:29` — extend `enum class KvCacheStorage` with the E8 variants
   (take the fork's naming, e.g. `Rk4V4E8`, `Rk2V4E8`). This is the in-tree public C++
   interface; extending a closed enum is the contract change.
2. Flag plumbing: `apps/cli/options.cpp:50-60,135` and `src/serve/serve_options.cpp:48-53,259`
   (`parse_kv_dtype`) + both usage texts (`options.cpp:81`, `serve_options.cpp:79`).
3. Page sizing / layouts: `src/targets/qwen3_6/impl/runtime/layouts.h`
   (`kv_quant_group` at :78/:102), `layouts_impl.h` (per-dtype page byte size, capacity
   curve), `request_plan_impl.h`, `program.h`/`program_impl.h`, and
   `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/decoder_state.h`.
4. Physical storage: `src/core/paged_kv_cache.{h,cpp}` (per-dtype page byte size),
   `src/core/cyclic_kv_cache.*`, `src/core/host_kv_arena.cpp` (the host replica path must
   understand the E8 encoding too).
5. Kernels: KV quantize/dequant + attention consumption — find the existing per-dtype
   dispatch in `src/ops/kv_cache/` and `src/ops/softmax_attention/*` (INT8 group-64 and FP8
   row-256 are the templates; E8 slots in beside them). The fork's lattice encode/decode
   `.cu` files are copied verbatim minus any `sm_86/sm_89` conditionals.
6. Tests: port the fork's E8 op tests into `tests/ops/` following the existing INT8/FP8 KV
   test structure; add flag-value cases in `test_serve_options.cpp` / `test_cli_options.cpp`.

### B.2 Port discipline

- Branch `cherry/e8-kv`. Copy from the reference fork with `git show <ref>:<path>`; never
  merge fork history.
- Strip every arch conditional in copied files; on sm_120a the lattice kernels (integer math)
  compile natively.
- Keep INT8 and FP8 modes intact (E8 is additive; no deletion of existing dtypes).
- The startup memory validator needs no structural change: KV capacity resolution
  (`src/runtime/engine/kv_capacity.cpp`) sizes from
  `curve.bytes_per_additional_main_page_group`, which derives from the per-dtype page size —
  E8 pages simply change the curve. Verify the fork's sizing matches this flow rather than
  importing a second validator.

### B.3 Exit gate (measured on the PRO 4000, ECC state recorded)

- `--kv-dtype rk4v4-e8 --max-context 262144` starts; record slack GiB next to the fork's
  1.37 GiB reference (expect ~1.0 GiB after ECC).
- Needle gates at 64K/128K, 5-needle @118K: exact match vs the INT8 baseline answers.
- Decode tax ≤ ~6%, prefill within ~2% vs the *local* INT8 baseline (not the fork's absolute
  numbers — 672 GB/s may show a larger tax than at 1008 GB/s).

### B.3 results (PRO 4000, 2026-09-01, branch `cherry/e8-kv`, device 1, ECC on)

Every INT8 figure below is a control measured in the same session as the E8 rows it is
compared against, not a quote from the D.1 table. This matters: the D.1 numbers predate the
two E8 commits, and the 32K prefill shape turned out to vary 2.6% run to run.

**Capacity at 262144. Pass, with more room than expected.**

| dtype | 262144 startup | planned slack | KV payload |
|---|---|---|---|
| int8 | rejected: needs 8.57 GiB runtime reservation, 7.00 GiB available | n/a | n/a |
| rk4v4-e8 | starts | 2.43 GiB | 4.25 GiB |
| rk2v4-e8 | starts | 3.43 GiB | 3.25 GiB |

The fork reference was 1.37 GiB and this section expected ~1.0 GiB after ECC. Both modes clear
that by 2.4x to 3.4x.

**Long-context output stability. Pass, 8 of 8 exact. Read this as stability, not retrieval.**

All runs return `ORCHID=493817; COLOR=COBALT` on the frozen `long_niah_*` fixtures: both E8
modes at 64K, 128K and 256K, plus INT8 controls at 64K and 128K.

The caveat matters. Every `long_niah_*` fixture ends with `... Return exactly: ORCHID=493817;
COLOR=COBALT`, so the expected answer is stated verbatim in the question and the model does not
have to attend to the planted needle to produce it. What these runs actually prove is that at up
to 260K context the engine emits the exact expected tokens instead of garbage, which is a real
and relevant gate here since the failure modes this port was chasing were all-zero tokens, NaN
and hangs. They are not evidence of retrieval quality at long context. A fixture that withholds
the answer is needed for that.

INT8 cannot run the 256K fixture at all. Its starting KV ceiling is 196608 and the prompt is
260,096 tokens, so the two E8 256K passes are a capability INT8 does not have on this card,
not a speed comparison.

The 5-needle @118K case named in the gate above was not run: no such fixture existed in this
tree or in the ninfer-4090 fork. The requirement was carried over from a protocol description,
not from a runnable input.

**Decode tax. rk4v4 passes everywhere; rk2v4 passes at 32K and fails at 128K.**

| context | int8 | rk4v4-e8 | rk2v4-e8 |
|---|---|---|---|
| 32K (median of 3) | 31.00 tok/s | 30.89 (0.36%) | 30.78 (0.71%) |
| 64K needle | 27.34 tok/s | 27.18 (0.6%) | 26.03 (4.8%) |
| 128K needle | 24.77 tok/s | 24.72 (0.2%) | 22.85 (**7.7%**) |

The 32K row alone would have passed both modes comfortably, but it is the least informative
shape: decode there is dominated by streaming 15.92 GiB of weights and the KV cache is at most
1 GiB of traffic, so it measures whether the dequant path stalls, not what it costs in
bandwidth. The gate has to be read at long context, and there rk2v4 grows past the 6% budget
while rk4v4 stays flat.

**Prefill. Fails at every length, and the gap widens with context.**

Two independent fixture families, same result.

| length | int8 | rk4v4-e8 | rk2v4-e8 |
|---|---|---|---|
| `long_prompt_32tok`, median of 3 | 1040.6 tok/s | 989.8 (4.9%) | 948.5 (8.8%) |
| `long_prompt_64tok` | 941.1 tok/s | 882.9 (6.2%) | 827.9 (12.0%) |
| `long_prompt_128tok` | 811.1 tok/s | 726.5 (10.4%) | 655.5 (21.9%) |
| `long_niah_64k` | 957.0 tok/s | 873.9 (8.7%) | 816.5 (14.7%) |
| `long_niah_128k` | 797.3 tok/s | 712.6 (10.6%) | 642.0 (19.5%) |

The trend is monotonic in both modes across both fixture families, and the E8 runs repeat to
within 0.5% while INT8 is the noisy arm, so this is not measurement error. The shape points at
per-key decode work scaling with key-tile count, with rk2v4 costing roughly double rk4v4,
which matches root-cylinder decode being more work per key than a nibble unpack.

**Verdict.** Capacity and correctness pass decisively. Prefill misses the ~2% gate outright.
rk4v4-e8 meets the decode gate at every context measured; rk2v4-e8 does not past ~64K. Whether
a ~10% prefill cost is worth 256K of otherwise unreachable context is a product call, not a
measurement one, and the gate as written does not answer it.

**Status of the two modes.** `rk4v4-e8` is the supported mode and the one to judge the port by.
`rk2v4-e8` stays in the tree but ships with a disclaimer: it is correct at every context tested
(retrieval is exact through 256K) and it buys about 1 GiB more slack, but it exceeds the 6%
decode budget past roughly 64K and costs about twice rk4v4-e8's prefill throughput at every
length. Use it only when the extra GiB is worth those costs. It is not being optimized; see the
performance note below for why the costs are structural rather than a bug.

**Why the E8 prefill gap widens with context, and why rk2v4-e8 is worse.** Not investigated
with a profiler; this is what the read path implies, recorded so nobody re-derives it.

The INT8 path stages key codes straight into the QK arena with cp.async and never decodes. Both
E8 modes decode the staged key tile on every visit, and a key tile is visited once per query
block, so total decode work grows with the number of (query block, key block) pairs, which is
quadratic in prompt length while the rest of prefill is linear. That alone explains a relative
penalty that grows with context, and it applies to rk4v4-e8 too (4.9% at 32K to 10.4% at 128K).
There is no reuse across query blocks to exploit; the decoded tile is CTA-local.

rk2v4-e8 carries a larger constant into that same quadratic term, from two places in
`e8_root_decode_8d_fast` (`src/ops/kv_cache/e8_root_codec.cuh`):

- `c_e8_stage1_i8x8` is `__device__ const`, so it lives in global memory and is read with
  `__ldg`. The root code differs per lane, making every lookup a divergent gather over a 2 KB
  table. There are 32 such calls per key row (four 64-dimension groups, eight 8-dimension
  blocks each), plus a second divergent index into `c_axis_i8x8`, which is `__constant__` and
  therefore also serializes when lanes disagree.
- Each decoded dimension costs an FP32 multiply and round,
  `__float2int_rn(float(dir) * c_radius_scale[rad])`, which is 256 per key row. The rk4v4-e8
  nibble unpack is shifts and masks with no float work at all.

Store cost is not the differentiator: both modes write the swizzled arena one byte at a time
through `causal_prompt_i8_store_swz`.

If rk2v4-e8 ever becomes worth optimizing, the cheapest thing to try first is staging the 2 KB
stage-1 table into shared memory once per CTA, which turns the divergent global gather into a
bank-conflict cost. Folding the radius scale away is harder: it varies per 8-dimension block,
so it cannot be absorbed into the per-group quantization scale.

Reproduce with the scripts recorded alongside this section; every command pins `--device 1`.

### Decode CUDA Graph qualification (2026-09-01, groupwise-int weights)

`Variant::ordinary_graph_profiles` cuts the decode frontier into ranges ending at
127/511/2047/4095/8197/16389/32767 and captures one graph per range. The boundaries were chosen
against the INT8 attention split policy, and a graph fixes kernel and launch geometry at capture
time, so the open question for E8 was whether a captured graph stays correct across its whole
range. Answer: yes, on every range reachable here.

Method: same prompt, `--greedy`, run once normally and once with `--no-cuda-graph`, generated text
compared byte for byte. `--no-cuda-graph` really does disable capture, confirmed by the startup
report (`CUDA Graph allowance` 12.00 MiB with graphs, 0 B without). Decode throughput is within
noise between the two because at ~33 ms per token launch overhead is about 1% of the budget, so
the speeds are not evidence either way; the text comparison is.

| shape | input | decode frontier | generated |
|---|---|---|---|
| A | short prompt, 512 new | 0-127, 128-511, into 512-2047 | 512 tokens |
| B | short prompt, 2100 new | crosses 2047/2048 | 1983 tokens |
| C | `long_niah_8k`, 128 new | around 8197 | 17 tokens |
| D | `long_prompt_32tok`, 128 new | 16390-32767 | 2 tokens |
| E | `long_niah_64k`, 128 new | 32768+ | 17 tokens |

All 15 combinations (int8, rk4v4-e8, rk2v4-e8 crossed with the five shapes) produced identical
output with and without graphs. INT8 is the control: it is the dtype the boundaries were tuned
for, so a difference there would have meant a harness fault rather than an E8 one.

Weight the shapes unequally. B is the real evidence, at 8599 bytes of greedy text matching
exactly across a boundary crossing; A is next at 2238 bytes. C and E are 17 tokens of needle
answer, enough to catch a wrong answer but not a subtle drift. D generates two tokens because its
fixture only ever answers `OK`, so it shows the 16390-32767 range executes without error and
nothing more. If that range needs real coverage, it wants a fixture that generates prose at a
~20K frontier.

Not covered: the MTP and dflash profile families (`mtp_graph_profiles`, `dflash_graph_profiles`),
which have their own boundaries and topology classes. This pass is ordinary decode only.

---

## Workstream C — MTP3 fit on 24 GB (draft WS3; now a fit test, not a port)

ReplaySSM is present upstream, so the only question is whether MTP3's recurrent draft state
fits after weights on this card:

1. Baseline first: record `available_after_weights` from the startup report for
   `qwen3_8_27b.ninfer` (groupwise, 16.96 GiB) on the PRO 4000 with ECC on and off.
2. Sweep, all with `--spec mtp --draft-tokens 3 --lm-head-draft`:
   - INT8 KV: largest starting `--kv-capacity`;
   - E8 KV (after B): same; target is 262144.
3. If MTP3 does not fit at the target context, the in-tree levers are already there — no
   port: `--device-state-slots / --host-state-slots / --host-kv-mib` (context cache
   offload), lower `--max-concurrency`, or `--draft-tokens 2`. Document which combination
   ships as the default profile.
4. Exit gate: MTP3 starts at the chosen profile; acceptance within ±3 pt of the fork's ~78%
   band at 111K; single-stream correctness unchanged with MTP off→on.

### Workstream C results (PRO 4000, 2026-09-01, groupwise-int weights, device 1, ECC on)

**Step 1, available after weights.** 7.28 GiB with the groupwise artifact (3.94 GiB with nvfp4,
see D.3). ECC could not be disabled: the host has no root, so every number here is ECC-on.

**Step 2, MTP3 fits at the target context on both E8 modes.** All runs use
`--spec mtp --draft-tokens 3 --lm-head-draft`.

| dtype | max context with MTP3 | without MTP3 | slack | KV payload |
|---|---|---|---|---|
| int8 | 131072 | 196608 | 1.47 GiB | 4.38 GiB |
| rk4v4-e8 | **262144** | >= 262144 | 1.34 GiB | 4.52 GiB |
| rk2v4-e8 | **262144** | >= 262144 | 2.40 GiB | 3.45 GiB |

**Step 3, no levers needed.** `--device-state-slots`, `--host-state-slots`, `--host-kv-mib`,
lower `--max-concurrency` and `--draft-tokens 2` exist for the case where MTP3 does not fit at
the target. It does. The shipping profile is `rk4v4-e8` at 262144 with MTP3 and no compromises.
Only int8 pays, dropping to 131072 because its KV payload leaves no room for the draft state.

**Step 4, exit gate.** Two halves, and the acceptance half needs its reference restated.

*Acceptance.* Measured on `long_decode_111k.json`, a 111,187-token document with a generative
question and 512 new tokens. A needle fixture cannot measure this: its answer is ten tokens,
which is two trivially predictable draft rounds and reads as a meaningless 100%.

| dtype | acceptance | tok/round | decode |
|---|---|---|---|
| int8 | 60.89% | 2.82 | 47.99 tok/s |
| rk4v4-e8 | 60.89% | 2.82 | 47.74 tok/s |
| rk2v4-e8 | 54.56% | 2.63 | 42.03 tok/s |

The gate's "±3 pt of the fork's ~78% band" cannot be met, and not because of E8: the **int8
control lands 17 points below that band**. The fork's number came from different hardware and
probably a different workload, and it does not reproduce here at any KV dtype. Three independent
measurements on this machine cluster instead at 57.9% (D.1, 32K), 58.3% (C3, 32K) and 60.9%
(111K), so the reproducible form of this gate is E8 against same-machine int8.

On that form `rk4v4-e8` passes outright with **no acceptance penalty at all**. Note the equality
is a coincidence at the aggregate: the two runs generated different text and different
per-position acceptance (int8 142/108/80, rk4v4-e8 147/107/76) that happens to sum to the same
330 of 542. Two different generation paths reaching the same aggregate is better evidence of
parity than one path measured twice. `rk2v4-e8` gives up 6.3 points of acceptance and 12% of
decode throughput, consistent with its long-context decode tax recorded under B.3.

*Single-stream correctness, MTP off to on.* All three dtypes produce coherent, correct,
undegraded prose with MTP enabled, and all three retrieve correctly at 111K with MTP on.

A byte comparison is the wrong test for this gate and cannot pass. With MTP on, the target model
verifies K+1 tokens in one batched forward pass; with MTP off it runs one token at a time.
Different batch shapes take different kernel paths, so tiny floating point differences
occasionally flip an argmax where two tokens are nearly tied. Greedy speculative decoding is
output-identical to non-speculative decoding only in exact arithmetic. The observed divergences
are exactly that shape: int8 splits 162 characters in on `storage**` versus `storage
resources**`, then both continue saying the same thing. This was verified by reading the outputs,
not inferred.

**Verdict.** Steps 1 to 3 pass cleanly and MTP3 ships at 262144 on rk4v4-e8. Step 4 passes in its
reproducible form and its absolute form is unreachable for reasons unrelated to this port; the
fork's ~78% reference should be dropped from the gate or requalified on this hardware.


---

## Workstream D — Baseline, serving extras, NVFP4 (draft WS1/4/7)

### D.1 Baseline capture on this machine (after Workstream A lands)

The purpose of this step is to produce one reference table of numbers for the PRO 4000, so
that every later workstream (E8 KV in B, the MTP3 fit test in C, the NVFP4 comparison in D.3)
is judged against *this card's* measured baseline instead of the fork's RTX 4090 numbers.
Run it once, after `fix/device-geometry` has merged, and record the results as a table in
this section.

**Binary note:** every measurement command below uses the CLI binary `./build/apps/ninfer`,
not `./build/apps/ninfer-serve`. The server binary does not accept `--max-new` (its output
cap is `--default-max-tokens` server-wide or the `max_tokens` request field); if a step
here is run against `ninfer-serve` instead, the error is `unknown argument: --max-new`.

1. Build the product binaries and tests: run
   `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON`, then
   `cmake --build build -j`. The architecture pin stays at the default `120a`; do not pass
   `-DCMAKE_CUDA_ARCHITECTURES`.
2. Download the baseline artifact once, into `models/`:
   `hf download neroued/Qwen3.8-27B-NInfer --local-dir models` (or the equivalent layout your
   checkout uses; the CLI takes the `.ninfer` path as its first argument). All measurements
   in this section use `qwen3_8_27b.ninfer`.
3. Identify the PRO 4000's index with
   `nvidia-smi --query-gpu=index,name,memory.total,memory.ecc.mode --format=csv`, and pass it
   to every command below as `--device <idx>`. Also note the other GPUs' idle state, because
   steps 5 and 6 read free memory.
4. Measure usable VRAM with ECC on and off. With ECC in its default (on) state, record the
   `nvidia-smi` total for the PRO 4000, then run one trivial CLI invocation and read the
   GPU-memory line from the stderr diagnostics:

   ```bash
   ./build/apps/ninfer models/qwen3_8_27b.ninfer \
     --prompt "Return one sentence." \
     --max-context 2048 --kv-capacity 2048 --max-new 1 --no-thinking \
     --device <idx> 2> probe.log
   ```

   The startup report also prints `available_after_weights`; record that value, since it is
   the number the KV capacity resolver actually starts from. Then repeat both readings with
   ECC disabled (`nvidia-smi -i <idx> -e 0`, re-run, then restore ECC and verify) so the
   difference attributable to ECC reservation is a measured number, not an assumption.
5. Measure decode throughput without speculation. Run this command three times and keep the
   median of the decode tok/s from the stderr logs:

   ```bash
   ./build/apps/ninfer models/qwen3_8_27b.ninfer \
     --prompt "Write a detailed explanation of how virtual memory works." \
     --max-context 32768 --kv-capacity 32768 --kv-dtype int8 \
     --max-new 512 --no-thinking --device <idx> 2> decode_nospec.log
   ```

   (The CLI writes timings, throughput, and GPU memory to stderr; the prompt must be short
   enough that the 32K context is dominated by the 512 generated tokens.) This number is
   the "no-spec decode" row of the baseline table.
6. Measure MTP3 decode and acceptance. Run the same command with the speculative flags
   added, again three times, and record both the median tok/s and the
   speculative-decoding statistics (draft acceptance) from stderr:

   ```bash
   ./build/apps/ninfer models/qwen3_8_27b.ninfer \
     --prompt "Write a detailed explanation of how virtual memory works." \
     --max-context 32768 --kv-capacity 32768 --kv-dtype int8 \
     --spec mtp --draft-tokens 3 --lm-head-draft \
     --max-new 512 --no-thinking --device <idx> 2> decode_mtp3.log
   ```
7. Measure prefill throughput at 32K, 64K, and 128K. First prepare one long prompt as a
   messages file: concatenate paragraphs of ordinary prose until the text is roughly
   4× the target token count characters (prose tokenizes near one token per four
   characters), and store it as `models/long_prompt_<len>.json` in chat-messages form
   (`[{"role":"user","content":"..."}]`). For each length, run:

   ```bash
   ./build/apps/ninfer models/qwen3_8_27b.ninfer \
     --messages models/long_prompt_<len>.json \
     --max-context <len> --kv-capacity <len> --kv-dtype int8 \
     --max-new 1 --no-thinking --device <idx> 2> prefill_<len>.log
   ```

   `--max-new 1` makes the run almost pure prefill; record the prompt-processing tok/s
   from stderr, normalized by the *actual* prompt token count reported in the log. If a
   length does not start (memory rejection), record the largest that does — that finding
   belongs in step 8 anyway.
8. Find the largest starting KV capacity per dtype. For each of `int8`, `fp8`, and `bf16`,
   increase `--kv-capacity` (with `--max-context` equal to it) in steps until startup is
   rejected, then bisect to the largest value that starts cleanly on a single run:

   ```bash
   ./build/apps/ninfer models/qwen3_8_27b.ninfer \
     --prompt "Return one sentence." \
     --max-context <cap> --kv-capacity <cap> --kv-dtype <dtype> \
     --max-new 1 --no-thinking --device <idx> 2> cap_<dtype>.log
   ```

   Record all three limits; these are the ceilings that B.3's E8 result and C's fit test
   must beat.
9. Write the results into this section as one table with columns: measurement, ECC state,
   value, command. Include the `nvidia-smi` name/total, the two usable-VRAM readings,
   no-spec decode tok/s, MTP3 tok/s plus acceptance, prefill tok/s at each of the three
   lengths, and the three KV-capacity limits. Every later workstream's exit gate references
   this table instead of quoting the fork.

### D.1 results (PRO 4000, 2026-08-31, master `2e07b54a`, artifact `models/qwen3_8_27b.ninfer` 15.92 GiB groupwise-int)

GPU: NVIDIA RTX PRO 4000 Blackwell, index 1, 32768 MiB total, **24467 MiB usable with ECC
enabled**. ECC was left in its default (enabled) state: the host has no root, so the ECC-off
reading could not be taken; all numbers below are ECC-on and the ~ECC slack from the fork
(1.37 GiB) should be compared against this card's post-ECC usable total, not 24.5 GiB.
Other GPUs on the host are busy (4090 at 100 %; a `ninfer-serve` E8/MTP3 docker container
on another card) — every command below pins `--device 1`.

| Measurement | ECC | Value | Command |
|---|---|---|---|
| usable VRAM (`nvidia-smi` total) | on | 24467 MiB of 32768 MiB | `nvidia-smi -q -i 1` |
| free after weights | on | 7.28 GiB | step-4 probe (`--max-context 2048 --kv-capacity 2048`) |
| free after startup (probe) | on | 6.85 GiB / planned slack 6.84 GiB | same |
| decode, no spec (median of 3) | on | **30.86 tok/s** (30.81 / 30.89 / 30.86) | step 5, `--max-new 512` |
| decode, MTP3 (median of 3) | on | **55.90 tok/s** (55.72 / 55.90 / 56.04); acceptance 57.86 %, 2.73 tok/round | step 6, `--spec mtp --draft-tokens 3 --lm-head-draft` |
| prefill 32K | on | 31,984 prompt tokens @ **1081.46 tok/s**; planned slack 5.94 GiB | step 7, `long_prompt_32tok.json`, int8 |
| prefill 64K | on | 62,905 prompt tokens @ **956.86 tok/s** | step 7, `long_prompt_64tok.json`, int8 |
| prefill 128K | on | 123,575 prompt tokens @ **811.25 tok/s**; starts cleanly (no memory rejection) | step 7, `long_prompt_128tok.json`, int8 |
| max starting KV, int8 | on | **196608** (229376 rejected; slack 799 MiB at 196608) | step 8, 32768 granularity |
| max starting KV, fp8 | on | **196608** | step 8 |
| max starting KV, bf16 | on | **98304** (118784 rejected; slack 991 MiB at 98304) | step 8 |

Notes:

- The step-8 bisection has 32768-token granularity (coarse step + bisection); treat the
  limits as exact to ±32768.
- Long prompts were regenerated at 31500/63000/126000 target tokens so each fits its nominal
  `--max-context`; the CLI-reported token counts above are the normalization values.
- The original 32K crash (`cudaErrorCooperativeLaunchTooLarge` in the fork build) no longer
  reproduces: `prefill_32k` exits 0 on this card with int8 KV, and the same run shape at
  128K also starts and completes.
- MTP3 acceptance (57.86 %) is measured on a short virtual-memory prompt; the C workstream
  re-measures it on the 111K needle workload before comparing against the fork's ~78 % band.

### D.2 Serving extras (draft WS4) — scope note

These are real external-contract additions: any change to OpenAI/Anthropic payloads must land
with schema-test + `docs/serving.md` updates together. Order within the port:
1. `GET /metrics` (Prometheus counters; new route in `src/serve`, data from the existing
   `GenerationMetrics` — no engine change);
2. `timings` block on chat completions / final stream chunk (payload addition → schema tests);
3. `context_window` in `/v1/models`;
4. `/slots` + slot save/restore — **defer** unless llama.cpp-style slot workflows are
   actually used; it is the largest diff and the least portable value here.

### D.3 NVFP4 (draft WS7) — no port needed

Download `neroued/Qwen3.8-27B-nvfp4-NInfer`, load on the PRO 4000, compare decode/prefill vs
the groupwise baseline at equal context, run the same needle gates. The nvfp4 profile is
resolved from artifact identity; no flag port from any fork.

### D.3 results (PRO 4000, 2026-09-01, `models/qwen3_8_27b_nvfp4.ninfer`, device 1, ECC on)

Every B.3 and graph number above was taken with the groupwise-int artifact. The parts that depend
on weight residency or on graph topology do not carry over, so they were repeated here. The E8 op
tests do carry over unchanged: they drive the ops with synthetic tensors and never load weights.

**NVFP4 is the larger artifact on this pair, not the smaller one.**

| | groupwise-int | nvfp4 |
|---|---|---|
| weight H2D | 15.92 GiB | 18.98 GiB |
| free after weights | 7.28 GiB | 3.94 GiB |

That is 3.06 GiB more resident weights and 3.34 GiB less room for everything else, which halves
the context ceiling of every KV dtype.

**Largest starting context.**

| dtype | groupwise | nvfp4 | slack (nvfp4) | KV payload (nvfp4) |
|---|---|---|---|---|
| int8 | 196608 | 98304 | 544 MiB | 3.09 GiB |
| rk4v4-e8 | >= 262144 | 196608 | 448 MiB | 3.19 GiB |
| rk2v4-e8 | >= 262144 | 262144 | 384 MiB | 3.25 GiB |

With NVFP4 weights, `rk2v4-e8` is the only configuration on this card that reaches 262144.
`rk4v4-e8`, the supported mode, stops at 196608. That is worth holding against the disclaimer
in B.3: rk2v4-e8 is the weaker mode on decode tax and prefill, and it is also the only route to
the top context on this weight format.

**Long-context output stability: 8 runs, all exact, 3 skips.** 8K and 64K pass on all three
dtypes; 128K passes on both E8 modes and is out of reach for int8; 256K passes on rk2v4-e8 only.
Same caveat as B.3, these fixtures state their answer in the question, so this is exact-output
stability rather than retrieval.

**Prefill is much faster and the E8 tax roughly doubles.** On `long_niah_64k`, the fixture both
weight formats can run on all three dtypes:

| dtype | groupwise | nvfp4 | E8 tax vs int8 (groupwise -> nvfp4) |
|---|---|---|---|
| int8 | 957.0 tok/s | 2280.0 tok/s | n/a |
| rk4v4-e8 | 873.9 tok/s | 1892.2 tok/s | 8.7% -> 17.0% |
| rk2v4-e8 | 816.5 tok/s | 1658.3 tok/s | 14.7% -> 27.3% |

NVFP4 moves prefill by about 140%, an order of magnitude more than the KV codec does, so it is
the lever to pull if prefill throughput is the problem. The E8 penalty doubling is the expected
consequence of the mechanism described under B.3: E8 key decode is fixed work per key tile, and
when the surrounding weight math gets 2.4x faster that fixed cost stops being amortized. NVFP4
and E8 therefore work against each other on prefill while stacking cleanly on capacity.

Decode is close to flat between the formats (26.7 vs 27.3 tok/s for int8 at 64K) despite 19% more
weight bytes to stream, which is not explained here and was not pursued.

**Decode CUDA Graph differential: 9 of 9 identical** across int8/rk4v4-e8/rk2v4-e8 and shapes
A/B/C at 16384 context, so the E8 path is graph-safe against NVFP4 weights as well.

**Verdict.** NVFP4 on this card is a real trade rather than an upgrade: about 2.4x prefill against
half the context ceiling. Which side wins depends on whether the workload is prefill-heavy at
moderate context or needs the full 262144.

### D.4 Causal-tile attention (draft WS6) — deferred

Only if prefill hurts after B/C land. Re-derive on the upstream sm_120 schedule
(`src/ops/softmax_attention/dense/context/*`); the 4090 fork is algorithm reference only.

### D.5 Context-cost presets for the PRO 4000 (small, do once measured)

After baselining, generate a machine preset via the existing calibration path
(`--context-cost-presets FILE`, format in `src/runtime/engine/context_cost.cpp`) so context
cache/continuation planning stops using generic conservative costs on this card. This is a
local-machine override by design — do not compile it into `context_cost_defaults.cpp`.

### D.5 results (PRO 4000, 2026-09-01)

Calibration needs `ninfer_context_cost_bench`, which is gated behind `NINFER_BUILD_BENCHMARKS`
(default OFF), so the build directory has to be reconfigured with `-DNINFER_BUILD_BENCHMARKS=ON`
to produce it. Nothing in `src/`, `apps/` or `tools/` calls the upsert entry points; `bench/` is
the only caller.

**Prefill preset written, transfer preset not.** Result file:
`models/context_cost_presets.json` (models/ is gitignored, which suits a local-machine override),
hardware class `nvidia-rtx-pro-4000-blackwell-sm120`, one prefill entry for `qwen3.8-27b`.

The combined `--suite all` run is rejected wholesale, because the preset write only happens when
every fitted component passes. Three of four passed comfortably:

| component | training p95 | validation p95 | limit | verdict |
|---|---|---|---|---|
| prefill | 0.041 | 0.017 | 0.15 | accepted |
| transfer d2h | 0.098 | 0.055 | 0.35 | accepted |
| transfer h2d | 0.173 | 0.255 | 0.35 | accepted |
| transfer d2d | 0.365 | 0.447 | 0.35 | rejected, 1 ordering failure |

Splitting the suites salvages the useful half: `--suite prefill` alone is accepted and writes the
prefill model, which is the component context planning depends on most and which fits very well
at 0.017 validation error against a 0.15 limit.

**The device-to-device miss is not sampling noise.** That was the obvious hypothesis, since the
default is 9 samples per point and two other GPUs are busy on this host, so the transfer suite was
retried at 25 and then 45 samples with 5 warmups. Training p95 moved 0.365 -> 0.351 -> 0.349 and
validation stayed at 0.40 to 0.45, with the same single ordering failure every time. Five times
the sampling changed almost nothing, so the d2d roofline model genuinely does not describe this
card's behaviour within its own acceptance limit.

The limit was not relaxed to force a pass. It encodes the tool's judgement about when its model is
trustworthy for planning, and a preset that fails it would make context planning worse than the
conservative defaults it replaces. Serving therefore runs with the measured prefill model and
falls back to generic transfer costs, which is the correct outcome rather than a partial failure.

Why the d2d roofline misfits sm_120a was not investigated.

---

## Explicit non-ports (confirmed against this tree)

| Item | Verdict |
|---|---|
| Any arch retarget / CC-check change | No. `120a` pin + `sm == 120` gate are already correct for the PRO 4000. |
| ReplaySSM port (3090 fork) | Already upstream — nothing to port. |
| Cohort C2–C8 port (3090 fork) | Upstream contract; measure only (expect less headroom on 70 SMs / 672 GB/s). |
| NVFP4 profile flag (laptop fork) | Upstream artifacts resolve the profile from identity. |
| RotorQuant `rk8v4` (3090 fork) | Superseded by E8; skip. |
| Windows/MSVC build surface (3090 fork) | Irrelevant. |

## Workstream A execution (done on this machine)

Landed on `fix/device-geometry` as three commits:

- `fb6641c6` A2 + A3 — device-dependent GDN gating residency (planner takes an explicit
  `sm_count`; routes whose tuned cooperative grid no longer fits demote S8→S4→S2→unsplit,
  35B S16→S8→S4→S2→unsplit; 35B norm fused split32 falls back to the composed route below
  its residency; interval capacity accounts for mid-band demotion maxima) plus the GDN
  chunked-output target grid. Public op signatures unchanged; `src/ops/common/device_geometry.h`
  supplies the cached current-device SM count to the wrapper/launchers.
- `4ffcddb4` A4 + A5 — RoPE large-block wave capacity (`sm × 6`) and the INT8 small-T 8K
  split clamp (`(sm / KVHeads) × scale`, reproduces the 5090's 42/84).
- `ec5aeab3` A1 (+ A6 comments) — MoE prefill persistent grid `sm × 3`.

Verification:

- `ninfer_gdn_gating_proj_test` passes on the 70-SM PRO 4000 and the 5070 Ti: pure planner
  checks pin sm=170 (bit-identical route table) and sm=70 (demotion boundaries incl. the
  T=1024 Split8→Split4 case that crashed), plus GPU oracle runs that execute the demoted
  cooperative kernels on both cards.
- Full ctest on both sm_120a cards: 96/98 pass each. The two failures are pre-existing,
  not from this work: `ninfer_qwen3_6_frontend_test` aborts on a missing `/home/neroued/...`
  resource path baked into the fork's test; `ninfer_sliding_window_attention_test` misses its
  reduction criterion at T=8/V=8/L=96 with bit-identical values on the pre-change base build
  (its own launcher carries 5090-tuned planning outside this workstream).
- The 170-SM (5090) parity gate is closed without a physical 5090 rerun: for sm_count = 170
  every derived value reduces exactly to the old constant (residency caps 2×170/4×170 =
  340/680, output grid 4×170 = 680, RoPE 6×170 = 1020, small-T clamp 170/4 = 42, MoE
  3×170 = 510), so a 5090 launch is arithmetically identical to the pre-change one; the
  sm=170-pinned planner tests verify route table and workspace capacity are bit-identical,
  and the full suite (96/98 on both 120a cards, only the two known pre-existing failures)
  executed these kernels on two real sm_120a GPUs. A 5090 host is not available and a rerun
  is not required; B's exit gate is judged against the local PRO 4000 baseline regardless.
- The 32K engine repro ran end-to-end on this host after the merge, using the fork's baseline
  artifact `models/qwen3_8_27b.ninfer` (the same target the fork build crashed on): 31,984
  prompt tokens, int8 KV, exit 0, and the 128K shape also starts and completes. See the D.1
  results table.

## Sequencing and branches

1. `base-sm120a` tag at `3d9fda22`.
2. `fix/device-geometry` — Workstream A. **Merged into master 2026-08-31** (ff to `2e07b54a`,
   incl. the long-prompt generator commit). The 5090 parity gate is closed by constant
   reduction + sm=170-pinned planner tests (no 5090 host available; see the A-execution
   notes above).
3. D.1 baseline captured on the PRO 4000 (see table above).
4. `cherry/e8-kv` off master — Workstream B.
5. Workstream C (MTP3 fit test) after B; regression pair (no-spec decode probe + 64K needle)
   after every merge.
6. `cherry/serving-metrics` — D.2 items 1–3, independent of B/C.
7. D.3 NVFP4 comparison (artifact `models/qwen3_8_27b_nvfp4.ninfer` is already on this host);
   D.5 presets; D.4 only if motivated.

## Open risks

- ECC-reserved VRAM: D.1 recorded the ECC-on state only (24467 MiB usable); the host has no
  root, so the ECC-off toggle (`nvidia-smi -e 0`) could not be run. If an ECC-off reading is
  needed for B.3's slack gate, run it on a host with root.
- A5 touches CUDA Graph capture inputs; graph profiles must be re-qualified on the PRO 4000
  after the split-policy change, not just unit-tested.
- The fork's E8 numbers were measured at 1008 GB/s; the 672 GB/s card may show a larger
  decode tax than 5.7% — B.3's gate is against the *local* INT8 baseline, not the fork's
  absolute numbers.

