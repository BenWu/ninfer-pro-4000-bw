# Speculative pairing acceptance probe

Measures whether cross-GPU speculative decoding between the NVFP4 build and the
groupwise INT build is worth implementing. It answers one question: when the INT
weights draft greedily, how many of those tokens does the NVFP4 build accept?

The mean committed tokens per round is the speedup ceiling. Everything a real
implementation adds (transport, scheduling, KV rollback) only subtracts from it.

## Picking the device

This machine has more than one GPU, so each server needs `--device N` to land on
the right card. Get the indices from nvidia-smi:

    nvidia-smi --query-gpu=index,name,memory.total --format=csv

which currently reports:

    0, NVIDIA GeForce RTX 4090, 24564 MiB
    1, NVIDIA RTX PRO 4000 Blackwell, 24467 MiB
    2, NVIDIA GeForce RTX 5070 Ti, 16303 MiB

Do not hardcode these. nvidia-smi orders by PCI bus id while CUDA defaults to
enumerating fastest first, so `--device 0` is not always nvidia-smi's GPU 0.
Either export `CUDA_DEVICE_ORDER=PCI_BUS_ID` so the two agree, or confirm the
card name in the server startup log. The 4090 and the PRO 4000 Blackwell report
nearly the same total memory, so check the name rather than the capacity.

## Recommended flow: record then replay

Only one server runs at a time, so the two models never need to be resident
together, and both phases can use the same binary on the Blackwell card. That
leaves `weights_id` as the only variable, which is what the measurement is about.

Record the NVFP4 trajectories:

    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    ninfer-serve models/qwen3_8_27b_nvfp4.ninfer --device 1 --port 8080 --no-thinking

    python3 tools/spec_probe/acceptance_probe.py record \
        --server http://127.0.0.1:8080 --out trajectories.json

Stop that server, start the INT weights, and replay:

    ninfer-serve models/qwen3_8_27b.ninfer --device 1 --port 8080 --no-thinking

    python3 tools/spec_probe/acceptance_probe.py replay \
        --server http://127.0.0.1:8080 --trajectories trajectories.json -k 8

Neither server should be started with `--spec`.

### Why replay is equivalent to running both at once

In a greedy round every committed token is the target's own greedy token.
Accepted drafts match the target by definition, and the correction token is the
target's. So the agreed prefix is exactly the target's greedy self-continuation
and never depends on the drafter. Recording that trajectory once is enough to
replay every round boundary against it.

Within a round the drafter still conditions on its own drafts, because replay
asks for all K tokens in a single call. The semantics match the live version.

### Recording regime

`record --regime prefill` (default) generates one token per request, re-prefilling
the accumulated text each step, so the reference token at each position is the same
computation replay performs. `--regime decode` is one continuous generation: much
faster, but every token after the first comes from incremental decode while replay
probes with a prefill. Those paths use different kernels and disagree on near-ties.

Measured on qwen3.8-27b nvfp4, 5 prompts, 128 tokens each, K=8:

| regime | self-check mean accepted | position 1 | usable rounds |
|---|---|---|---|
| decode | 6.10 / 8 | 97.8% | 90 |
| prefill | 7.80 / 8 | 100.0% | 15 |

Prefill is exact but yields far fewer usable rounds, because its trajectories are
text-unstable: about 101 to 112 of 128 positions get excluded, against about 10 of
128 for a decode recording. Re-prefilling joined text lets the tokenization drift,
and every drifted position has to be dropped. Compensate with more prompts rather
than longer ones, and prefer prose prompts, since markdown emits many `**` and
`\n\n` tokens and those are where merges happen.

### Sanity check

Record and replay against the *same* weights. With `--regime prefill` this comes
out at exactly 100% per position. Anything lower means the harness, not the models,
is wrong.

Rounds are also clamped so they never span a position whose text fails to round
trip. Both regimes need this. A decode recording can retokenize differently when
its tokens are joined; a prefill recording bakes the merge into the reference
instead, giving a token sequence no continuous generation can reproduce. Skipping
only the round *start* is not enough, since a merge inside a round breaks it too.

## Live mode

    python3 tools/spec_probe/acceptance_probe.py live \
        --target http://127.0.0.1:8080 --drafter http://127.0.0.1:8081 -k 8

Both servers up at once, drafter and target queried per round. This needs a
drafter that supports assistant prefill continuation. The ninfer-4090 fork does
not: it has an older serve layer with no `ContinueFinalAssistant` handling, so a
trailing assistant message becomes a closed turn plus a fresh generation prompt
and the drafter answers a different prompt than the target. Use record/replay
unless both servers are known to support prefill.

## Requirements on the server

`--no-thinking` is required, not optional. Continuation refuses to start in
thinking mode (`chat_template.cpp`, continuation branch), the Anthropic endpoint
has no per request thinking switch, and the server default is thinking on, so a
server started without the flag rejects every prefill request with a 400.

This also depends on the continuation fix in `chat_template.cpp` that renders the
empty thinking block before the prefill text. Without it the prefill looks like a
finished turn to the model, which emits end of turn immediately and the probe
reports zero rounds. The preflight check calls this out.

Rounds are driven through assistant prefill on `/v1/messages`. That is the only
endpoint that continues a partial completion. The OpenAI chat and responses
endpoints never set `ContinueFinalAssistant`, so a trailing assistant message
there starts a new answer that can read like a continuation but is not one.

## Reading the output

`acceptance by draft position` is the most informative part. Acceptance decays
with position because one early mismatch invalidates everything after it. A curve
that stays high through position 4 or 5 means a larger K pays off. A curve that
collapses after position 1 means the two quantizations diverge quickly and only a
small K makes sense.

Rough reading of mean committed tokens per round:

- above 2.5, the pairing is worth building
- 1.8 to 2.5, marginal, measure transport cost first
- below 1.8, the quantizations disagree too often to bother

## Limitations

Tokens are compared as decoded text, not token ids, because the HTTP layer does
not expose ids or logprobs (`src/serve/openai_chat_request.cpp:115`). Two distinct
ids that decode to the same string would count as agreement. This is rare with
BPE and does not change the conclusion. Both artifacts embed a byte-identical
tokenizer, so the comparison is at least consistent across the two.

The probe measures greedy agreement only. That is the right measurement: the
accept kernel in `include/ninfer/ops/speculative_round.h` assumes one-hot draft
proposals, so a real implementation must draft greedily.
