# NInfer

> Selected checkpoints. Maximum single-GPU inference performance.

---

## Fork note: RTX PRO 4000 Blackwell port

Upstream targets the RTX 5090; this fork runs on an **RTX PRO 4000 Blackwell**
(sm_120a, 70 SM, 24467 MiB usable with ECC on, CUDA 13.3, driver 610.57.04) and ports the E8
lattice KV cache quantization from the ninfer-3090/4090 forks. Two new KV dtypes:
`rk4v4-e8` (supported) and `rk2v4-e8` (kept with a disclaimer, see below).

Full detail lives in `docs/plans/rtx-pro-4000-sm120-port-plan.md` and
`docs/plans/e8-kv-port-handoff-2026-08-31.md`. This section is the overview.

### Status

Working. The engine produces correct text in both E8 modes at every context up to 262144, under
CUDA Graphs, with and without MTP3 speculative decoding.

### What changed on this branch

Five defects were found and fixed in the E8 attention path. None were in the E8 encoder math.

| Defect | Effect |
|---|---|
| `prompt_e8.cuh` decoded 16 of 64 staged key rows | all-zero tokens |
| `small_t_e8.cuh` wrote FP16 bit patterns into a BF16 tile | NaN via inf residual into RMSNorm |
| V dequant stored 4-byte values at 2-byte strides | misaligned address fault |
| `kv_cache_inverse_rotate_output_kernel` lacked `static` | duplicate kernel registration, launch hang |
| full-mask `__shfl_sync` under a non warp-uniform causal branch | hang on roughly every odd-length prompt |

The last one is the interesting one: sixteen chunks per key row against 32 lanes puts two key rows
in one warp, so the causal test is not warp uniform and the shuffle deadlocks. It is guarded now
by an explicit parity pair in the attention test (`{17,0,17}` even against `{17,1,18}` odd).

Also on this branch: per-plane leading extents in `d256_profile.h`, new op tests, a fixture
generator (`tools/gen_niah_fixtures.py`), four new long-context fixtures, and the measurement
write-ups in `docs/plans/`.

### Serving on this card

`./serve.sh` starts `ninfer-serve` with the NVFP4 artifact, NVFP4 KV, vision and MTP3, using the
context measured below. Everything is env-overridable (`CTX`, `PORT`, `DEVICE`, `KV_DTYPE`,
`MAX_CONCURRENCY`, `VISION=0`, `MTP=0`) and extra flags pass through to the server.

Largest `--max-context` that starts, NVFP4 weights with NVFP4 KV at `--max-concurrency 1`:

| config | max context | slack at that context |
|---|---|---|
| plain | 196608 | 109 MiB |
| `--vision` | 131072 | 307 MiB |
| MTP3 | 131072 | 266 MiB |
| `--vision` + MTP3 | **90112** | 75 MiB |

Cost of each feature before KV is allocated: vision 282 MiB, MTP3 771 MiB. MTP3 is the expensive
one, at nearly three times vision, because of its recurrent draft state.

`serve.sh` defaults to 81920 rather than the 90112 ceiling. Starting with 75 MiB to spare works but
leaves nothing for a raised `--max-concurrency`, a larger media payload, or a different driver
state, and 81920 keeps 228 MiB. Note that `--max-concurrency` lowers every number in this table:
the context-cache defaults scale off it, with device-state slots equal to concurrency, private
continuations at twice that, and shared prefixes at one times.

A ladder that steps by 32768 reports 65536 for the vision plus MTP3 row, because 98304 misses by
only 82 MB. The real ceiling is 38% higher, so probe the gap rather than trusting a coarse sweep.

### Running the tests

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
CUDA_VISIBLE_DEVICES=1 ctest --test-dir build --output-on-failure
```

98 of 100 pass. The two failures are pre-existing and unrelated to this port:
`ninfer_qwen3_6_frontend_test` aborts on a tokenizer path that does not exist on this machine, and
`ninfer_sliding_window_attention_test` misses its reduction criterion at T=8/V=8/L=96.

E8-specific targets, all passing:

```bash
CUDA_VISIBLE_DEVICES=1 ./build/tests/ninfer_e8_root_codec_test        # encoder, warp vs scalar
CUDA_VISIBLE_DEVICES=1 ./build/tests/ninfer_kv_cache_append_e8_test   # append planes, byte exact
CUDA_VISIBLE_DEVICES=1 ./build/tests/ninfer_softmax_attention_test    # includes causal_cache_e8
```

`NINFER_OP_REPORT_STATS=1` dumps measured error for every comparison, which is how the tolerance
in the attention test was set. Note `ninja` does not always rebuild `small_t.cu` / `prompt.cu` when
an included `.cuh` changes; delete those `.o` files to force it.

### Measurements

Every number below is **Qwen3.8-27B**, artifact `models/qwen3_8_27b.ninfer` (`groupwise-int`,
15.92 GiB resident), on device 1 with ECC on, unless the NVFP4 subsection says otherwise. The
NVFP4 rows use `models/qwen3_8_27b_nvfp4.ninfer` (same model, `nvfp4` weights, 18.98 GiB
resident). None of the other supported checkpoints were measured on this card, so do not carry
these numbers to Qwen3.6-27B or Qwen3.6-35B-A3B.

Baseline for comparison: int8 KV, 31.00 tok/s decode, 196608 max starting context.

These tables were taken before the rebase onto upstream's nvfp4/k8v4 work. They remain valid for
what they measured, but do not compare them across that boundary: int8 decode moved from 31.00 to
31.69 tok/s over the rebase, most likely from upstream's `fp16 V storage and PV compute` change.
The comparison section below is a single post-rebase session and is internally consistent.

**Capacity.** E8 buys context that int8 cannot reach.

| dtype | max context | slack | with MTP3 |
|---|---|---|---|
| int8 | 196608 | 799 MiB | 131072 |
| rk4v4-e8 | >= 262144 | 2.43 GiB | 262144 |
| rk2v4-e8 | >= 262144 | 3.43 GiB | 262144 |

**Decode throughput, tok/s.** Single stream, greedy, no speculation. Absolute numbers first,
tax against same-session int8 in parentheses. rk4v4-e8 is effectively free; rk2v4-e8 is not, at
long context.

| context | int8 | rk4v4-e8 | rk2v4-e8 |
|---|---|---|---|
| 32K | **31.00** | 30.89 (0.36%) | 30.78 (0.71%) |
| 64K | **27.34** | 27.18 (0.6%) | 26.03 (4.8%) |
| 128K | **24.77** | 24.72 (0.2%) | 22.85 (**7.7%**) |
| 256K | n/a | 20.86 | 18.27 |

The 32K row is the median of three runs at 32768 capacity. The 64K and 128K rows come from needle
runs with the context matched to the prompt. The 256K row was measured under a 262144 reservation
and has no int8 entry because int8 cannot hold a 260096 token prompt on this card. Reservation
size affects decode, so compare rows within a row, not across.

**Decode with MTP3, tok/s.** `--spec mtp --draft-tokens 3 --lm-head-draft`. Roughly 1.7x to 1.8x
over unspeculated decode.

| context | int8 | rk4v4-e8 | rk2v4-e8 |
|---|---|---|---|
| 32K | **55.83** | 51.79 | 53.98 |
| 111K | **47.99** | 47.74 | 42.03 |

The 32K row is a single run per dtype and its spread tracks per-run acceptance variation
(58.3 / 51.4 / 55.4%), so do not read the ordering there as a property of the codec. The 111K row
is the reliable one: 512 generated tokens, acceptance 60.9 / 60.9 / 54.6%.

**Prefill tax vs int8.** Both modes miss the plan's 2% gate, and the gap grows with context.

| context | int8 | rk4v4-e8 | rk2v4-e8 |
|---|---|---|---|
| 32K | **1040.6** | 989.8 (4.9%) | 948.5 (8.8%) |
| 64K | **941.1** | 882.9 (6.2%) | 827.9 (12.0%) |
| 128K | **811.1** | 726.5 (10.4%) | 655.5 (21.9%) |

The cause is structural. int8 stages key codes with cp.async and never decodes; E8 decodes the
staged tile on every visit, and a tile is visited once per query block, so decode work grows with
the number of (query block, key block) pairs while the rest of prefill grows linearly.

**Retrieval.** Exact on both E8 modes at 111K single needle and 118K five needle, matching int8.
These fixtures withhold the answer from the question. The committed `long_niah_*` fixtures do not:
they end with `Return exactly: ORCHID=...`, so passing them is output stability, not retrieval.

**CUDA Graphs.** 15 of 15 graph vs `--no-cuda-graph` comparisons identical across five frontier
ranges, plus 9 of 9 on NVFP4 weights and 3 of 3 at a 20K frontier. Ordinary decode profiles only;
the MTP and dflash profile families are not qualified for E8.

**MTP3.** Fits at 262144 on both E8 modes with no fallback levers. Acceptance at 111K over 512
generated tokens: int8 60.89%, rk4v4-e8 60.89%, rk2v4-e8 54.56%. rk4v4-e8 has no acceptance
penalty. The plan's reference band of ~78% from the 4090 fork does not reproduce here at any KV
dtype, including int8, so it should be requalified or dropped.

**NVFP4 weights** (`models/qwen3_8_27b_nvfp4.ninfer`). A real trade rather than an upgrade.
The NVFP4 artifact is *larger* here (18.98 GiB resident against 15.92), so every context ceiling
roughly halves, while prefill runs
about 2.4x faster. The E8 prefill tax doubles under it (rk4v4-e8 8.7% to 17.0% at 64K) because
the fixed per-key decode cost stops being amortized. With NVFP4 weights `rk2v4-e8` is the only
mode that still reaches 262144.

| | groupwise-int | nvfp4 |
|---|---|---|
| weights resident | 15.92 GiB | 18.98 GiB |
| max context, rk4v4-e8 | >= 262144 | 196608 |
| prefill @64K, int8 | 957 tok/s | 2280 tok/s |

NVFP4 decode, tok/s, from the needle runs at matched context. Close to the groupwise numbers
despite 19% more weight bytes to stream, which is not explained here. `n/a` means the dtype
cannot hold that prompt with these weights.

| context | int8 | rk4v4-e8 | rk2v4-e8 |
|---|---|---|---|
| 8K | 28.90 | 28.73 | 28.51 |
| 64K | 26.67 | 26.48 | 25.14 |
| 128K | n/a | 23.87 | 22.08 |
| 256K | n/a | n/a | 17.72 |

**Context-cost presets.** `models/context_cost_presets.json` holds the measured prefill model
for `qwen3.8-27b` / `groupwise-int` under hardware class
`nvidia-rtx-pro-4000-blackwell-sm120`
(0.017 validation error against a 0.15 limit). The device-to-device transfer fit is rejected by
the tool's own limit (0.35 against a 0.365 p95) and is not sampling noise: 5x the samples moved it
to 0.349 with the same ordering failure. Serving falls back to generic transfer costs. Building
the calibrator needs `-DNINFER_BUILD_BENCHMARKS=ON`.

### Comparison against upstream nvfp4 and k8v4

Upstream added two low-bit KV modes of its own, `nvfp4` (group-16 scales on both planes) and
`k8v4` (FP8 key plus NVFP4 value). This branch is rebased onto them, so all five modes were
measured on the same card in the same session against the same fixtures. Groupwise-int weights,
Qwen3.8-27B, device 1, ECC on.

**Storage.** Measured KV payload at 32768 capacity, which matches the per-token arithmetic from
the layout tables to within 0.001 in every case.

| mode | payload @32768 | vs int8 | K + V bytes per token per head | max context |
|---|---|---|---|---|
| int8 g64 | 1.03 GiB | 1.000 | 528 | 196608 |
| k8v4 | 804 MiB | 0.762 | 402 | 262144 |
| nvfp4 g16 | 576 MiB | 0.546 | 288 | 262144 |
| rk4v4-e8 | 544 MiB | 0.516 | 272 | 262144 |
| rk2v4-e8 | 416 MiB | 0.394 | 208 | 262144 |

`rk4v4-e8` is only 5.6% smaller than `nvfp4`. Only `rk2v4-e8` is meaningfully more compact, at 28%
below it.

**Decode, tok/s at 32K, median of three.** This shape cannot separate the modes: decode is
dominated by streaming 15.92 GiB of weights and everything lands within 2%. The ordering still
tracks decode cost per key.

| mode | median | vs int8 |
|---|---|---|
| nvfp4 | 31.70 | 0.0% |
| int8 | 31.69 | n/a |
| k8v4 | 31.47 | 0.7% |
| rk4v4-e8 | 31.26 | 1.4% |
| rk2v4-e8 | 31.07 | 2.0% |

**Prefill, tok/s. This is the decisive measurement.**

| mode | 32K | 64K | 128K | vs int8 @128K |
|---|---|---|---|---|
| k8v4 | 1063.24 | 979.70 | 858.31 | +5.1% |
| nvfp4 | 1058.34 | 974.30 | 847.46 | +3.8% |
| int8 | 1045.04 | 954.14 | 816.45 | n/a |
| rk4v4-e8 | 1004.04 | 887.18 | 727.85 | -10.9% |
| rk2v4-e8 | 969.02 | 832.61 | 658.38 | -19.4% |

Both upstream modes are faster than int8 at every length and pull further ahead as context grows.
Both E8 modes are slower and fall further behind. One mechanism explains both directions. Every
mode except int8 pays a decode per key tile per query block, and int8 pays none but reads 528 bytes
per token per head. The question is only whether that decode costs less than the bandwidth it
saves. For NVFP4's multiply by a group-16 scale it does, so the advantage grows as KV traffic
becomes a larger share of prefill. For the E8 lattice decode it does not, so the same rising share
turns into a widening penalty.

Against `nvfp4` specifically, `rk4v4-e8` trails by 5.1%, 8.9% and 14.1% at 32K, 64K and 128K. The
storage advantage that buys is a fixed 5.6%, so the trade is underwater from roughly 64K onward.

**Retrieval.** All ten runs exact, on both withheld-answer fixtures, for every mode. Accuracy does
not separate these codecs at this scale, so the decision rests on the prefill table above.

Decode at long context does separate them, unlike the 32K shape, because KV traffic is now a real
share of the work. The ordering matches prefill exactly.

| mode | 111K single needle | 118K five needles | answers |
|---|---|---|---|
| k8v4 | 26.85 | 26.37 | all exact |
| nvfp4 | 26.78 | 26.31 | all exact |
| int8 | 25.61 | 25.36 | all exact |
| rk4v4-e8 | 25.36 | 25.18 | all exact |
| rk2v4-e8 | 23.70 | 23.39 | all exact |

Both upstream modes decode faster than int8 at this context; both E8 modes decode slower.

**Same comparison on NVFP4 weights.** `nvfp4` KV against `rk4v4-e8` KV, artifact
`models/qwen3_8_27b_nvfp4.ninfer`. NVFP4 weights make prefill roughly 2 to 3x faster, and that
makes the E8 penalty worse, not better.

| context | nvfp4 KV | rk4v4-e8 | gap | gap on groupwise weights |
|---|---|---|---|---|
| 32K | 3005.48 | 2606.43 | 13.3% | 5.1% |
| 64K | 2451.70 | 1936.86 | 21.0% | 8.9% |
| 128K | 1763.00 | 1299.68 | 26.3% | 14.1% |

Two axes amplify the same fixed cost independently. Longer context raises KV's share of prefill,
and faster weights raise the codec's share of what is left, so the penalty roughly doubles at every
length. The E8 decode is per key tile per query block and does not shrink when anything else gets
faster.

Both modes cap at 196608 on these weights, so E8 buys no extra context here, though it does leave
about twice the slack at that ceiling (447.94 MiB against 256.00 MiB). Retrieval is again exact for
both on both fixtures, with decode 24.58 against 25.84 tok/s at 111K and 24.31 against 25.57 at
118K.

**Why E8 is still here.** On these numbers `nvfp4` dominates `rk4v4-e8`: 5.6% more memory for
strictly better prefill at every length, and it is upstream code that will keep getting attention.
`rk2v4-e8` keeps a real 28% storage edge over `nvfp4` but pays about 22% of prefill for it. This is
a personal fork and the E8 modes are kept deliberately for their pedagogical value: the port is a
worked example of E8 lattice quantization, warp-cooperative encoding, and the debugging that came
with it. They are not the modes to reach for on throughput.

### Known limits

- `rk2v4-e8` ships with a disclaimer. It is correct everywhere tested, and with NVFP4 weights it
  is the only route to 262144, but it exceeds the 6% decode budget past roughly 64K and costs
  about twice rk4v4-e8's prefill. It is not being optimized. The likely costs are a divergent
  gather over a 2 KB `__device__ const` table and 256 FP32 convert/round operations per key row,
  neither of which rk4v4-e8 pays.
- The prefill gate fails for both E8 modes and is structural, not a bug. Read it alongside the
  upstream comparison below: `nvfp4` and `k8v4` are *faster* than int8 on the same measurement, so
  the E8 prefill cost is not the price of low-bit KV in general, only of this codec.
- The `examples/cli/messages/long_niah_*.json` fixtures state their answers in the question.
- MTP and dflash CUDA Graph profile families are unqualified for E8.
- Unexplained and unpursued: NVFP4 decode is flat despite 19% more weight bytes to stream, and the
  d2d transfer roofline does not fit this card.
- Serving extras (plan D.2: `/metrics`, `timings`, `context_window`) are not started.

---

NInfer is a from-scratch C++/CUDA inference engine for explicitly registered Qwen checkpoints on a
single NVIDIA GeForce RTX 5090. It runs text, image, and video prompts through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs. The runtime is deliberately specialized: one GPU, one
resident model, and a startup-fixed capacity of one to eight active requests.

NInfer supports five artifact identities. The quick-start commands use Qwen3.8-27B NVFP4.

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

The artifact identity fixes the exact model and weight profile. Every artifact also embeds the
tokenizer, chat template, and media frontend resources required by its registered target.

## Quick start

NInfer requires 64-bit Linux, an NVIDIA GeForce RTX 5090, CUDA Toolkit 13.1 or newer, CMake 3.28 or
newer, a C++20 host compiler, Ninja, `pkg-config`, FFmpeg development libraries
(`libavformat >= 60`, `libavcodec >= 60`, `libavutil >= 58`, and `libswscale >= 7`), and
`libcurl >= 7.85`. The build rejects CUDA architectures other than `sm_120a`.

Build the product binaries:

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Tests, benchmarks, and maintainer tools are excluded from the default build. There is no install
target or packaged binary distribution; run NInfer from its source build tree.

Download the artifact used by this example with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Start a long-running text/agent server with two active-request lanes and explicit Device/Host
checkpoint capacity:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. A shared 240,000-token Device KV pool serves
admitted requests; two requests run concurrently when their combined reservations fit. The cache
tiers provide two Device checkpoint slots, eight pinned Host State slots, and 8 GiB of pinned Host
KV beyond the two active StateImages.

Send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

Run a one-shot CLI request with a 32,768-token allocation:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Answer content is written to stdout. Structured startup/runtime-error records and the CLI-owned
reasoning, timing, throughput, memory, and speculative-decoding report are written to stderr;
reasoning and the result report remain unprefixed product output. On a terminal, weight
materialization additionally uses one transient progress line. Redirected stderr receives only
persistent structured phase records, including rate-limited progress for long loads. Option and
local input errors remain direct command diagnostics. Use `--messages FILE` and `--vision` for
structured image/video input; see the [CLI guide](docs/cli.md) and
[committed examples](examples/cli/).

## Resource-aware long-context reuse

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Performance

Published measurements use an RTX 5090. [Performance](docs/performance.md) records the exact
benchmark profiles and methodology.

### Concurrent MTP3 decode

Saturated decode used INT8 group-64 KV, CUDA Graphs, MTP3, and one 8,192-token generation per active
request. Values are aggregate committed decode throughput and MTP acceptance from complete
intervals whose actual decode batch equaled the configured concurrency.

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept | C8 / C1 |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| Qwen3.6-27B `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| Qwen3.6-35B-A3B `groupwise-int` | 593.0 / 67.2% | 877.7 / 68.2% | 1,166.0 / 69.8% | 1,313.8 / 67.3% | 2.22× |
| Qwen3.8-27B `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

### Single-request serving

The serial serving corpus used INT8 group-64 KV, CUDA Graphs, a 1,024-token prefill chunk, and five
fixed seeds after warm-up. The table keeps one short-prefill, one extreme-prefill, and one
structured-output MTP3 point for each published profile; the full context and scenario matrices are
in the performance document.

| Model profile | 7,680-token prefill | 260,096-token prefill | Structured MTP3 decode |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B `groupwise-int` | 15,544.3 tok/s | 5,157.1 tok/s | 770.9 tok/s |
| Qwen3.6-27B `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| Qwen3.6-27B `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| Qwen3.8-27B `groupwise-int` | 3,274.7 tok/s | 1,609.7 tok/s | 224.4 tok/s |
| Qwen3.8-27B `nvfp4` | 8,340.4 tok/s | 2,203.1 tok/s | 219.8 tok/s |

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens to
fit the RTX 5090 after weights. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

### Perplexity

Run the fixed four-domain quick corpus through the artifact's tokenizer and Text model:

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_nvfp4.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick --kv-dtype fp8
```

The evaluator reports token-weighted fixed-window causal perplexity and writes a complete JSON
record under `profiles/perplexity/`. See [Perplexity evaluation](docs/perplexity.md) for the metric,
corpus, custom-text mode, and comparison rules.

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` selects Vision residency. DFlash is available for text-only Qwen3.6-35B-A3B execution.

## Docker

Build the runtime image on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Mount the downloaded model and run the same example server profile:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

## Capabilities and limits

All registered model IDs support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16, INT8 group-64, row-scaled FP8 E4M3, NVFP4 group-16, and asymmetric K8V4 KV storage;
- offline causal-perplexity scoring with the same Text model and selectable KV storage;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The exact low-bit KV selectors are `--kv-dtype nvfp4` (144-byte K and V vectors) and
`--kv-dtype k8v4` (258-byte FP8 K plus 144-byte NVFP4 V vectors), all for D256. These runtime
cache choices are independent of the registered artifact's weight format. NVFP4 attention does
not quantize Q: prompt and small-T use FP16 rotated Q and exactly expanded FP16 K with FP32 QK
accumulation; K8V4 retains its FP8 Q/K path.

The 35B-A3B target additionally supports text-only DFlash with draft windows from one to fifteen.

The product boundary remains intentionally small:

- one RTX 5090 and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress;
- no request preemption, priority/QoS, active-request swapping, weight offload, multi-GPU, or
  distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- no runtime model discovery or unregistered checkpoint fallback;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights while keeping 1 GiB of sizing headroom. Explicit
capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
