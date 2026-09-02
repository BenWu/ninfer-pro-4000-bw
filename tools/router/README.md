# Two card router

Fronts two ninfer servers with one endpoint and decides which card each request
goes to. It exists because of three things measured on this machine, not because
of the prefill and decode asymmetry between the two cards.

## What actually decides placement

**A prefix hit is worth 108x.** A cold 32k prompt on the Blackwell takes 10.98
seconds to first token. The same prompt again takes 0.102 seconds, with
`cache_read_input_tokens` reporting 32025 of 32030 reused. No other decision the
router can make is within two orders of magnitude of that, so cache affinity
outranks every other rule.

**A server runs one request at a time.** `--max-concurrency 1` is what both
`serve.sh` and the 4090's `start_with_vision.sh` use. A request that takes 0.60
seconds on an idle card measured 11.64 seconds when it arrived 0.3 seconds
behind a cold 32k prefill. The queue therefore lives in the router, one in
flight per backend, and never in the server, where it is FIFO and invisible:
`/health` returns a static `{"status":"ok"}` with no depth or load.

**The cards do not hold the same context.** `serve.sh` gives the Blackwell 81920
tokens with vision and MTP3 enabled, against 262144 on the 4090. A prompt above
the smaller ceiling is a placement constraint before it is a preference.

Shape only breaks ties between backends that are equally warm and equally idle.

## Running it

    python3 tools/router/ninfer_router.py --port 8090 \
        --backend blackwell=http://127.0.0.1:8080,max_context=81920,prefill=3860,decode=57.4,attn=2.493e-9 \
        --backend rtx4090=http://127.0.0.1:8081,max_context=262144,prefill=2115,decode=108.0,attn=1.619e-9

Clients point at the router instead of a server. `/router/status` reports queue
depth, committed work, and placement counts.

### Which endpoints are routed

| endpoint | treatment |
|---|---|
| `/v1/chat/completions` | fully routed: affinity, queue, context fit, streamed |
| `/v1/messages` | fully routed, identically |
| `/v1/responses` | proxied only, **not** balanced, see below |
| everything else | proxied to the least loaded backend, streamed if streaming |

The two generating endpoints get the same treatment because they converge below
the frontend: both parse into the same `ChatTurn` list, so prefix reuse is
identical. Measured over a five turn conversation on one card, cached tokens ran
4635, 4654, 4673 on `/v1/messages` against 4636, 4655, 4674 on
`/v1/chat/completions`, with matching latencies. Adding an endpoint is an
adapter over body shape, not a second routing policy: OpenAI keeps the system
turn inside `messages` where Anthropic hoists it to a top level field, and the
output cap is `max_completion_tokens` or `max_tokens` rather than `max_tokens`.

`/v1/responses` is deliberately excluded from routing. Its store is process
local (`src/serve/openai_responses_store.h:3`), so `previous_response_id`
resolves through a `store.get()` that 404s when the request lands on the card
that did not create the response (`src/serve/openai_responses_state.cpp:154`),
as do the `/v1/responses/{id}` subpaths. Balancing it would need sticky routing
keyed by response id, which pins that traffic from creation and orphans ids on a
card restart. It is proxied unchanged instead.

### Backend options

| option | meaning |
|---|---|
| `max_context` | hard fit constraint, from the server's `--max-context` |
| `prefill` | linear prefill rate in tokens per second |
| `attn` | quadratic attention coefficient in seconds per token squared |
| `decode` | decode rate in tokens per second |
| `affinity_min_tokens` | below this a prefix is assumed not retained, default 2048 |
| `affinity_slots` | contexts the card holds, default 3, **use 1 for the 4090** |

### The two forks retain differently

They are different engine lineages, and it changes how many conversations each
card can keep warm. The Blackwell fork has a cost-model driven context cache
(`src/runtime/engine/context_cost.cpp`, about 9600 lines in `runtime/engine`).
The 4090 fork has none: reuse is a retained-sequence check
(`request_plan_impl.h:182`, about 2500 lines), and it holds exactly one context.

Alternating two 4000 word documents, time to first token:

| | cold | repeat | alternating |
|---|---|---|---|
| Blackwell, calibrated presets | 7.9 s | 0.09 s | 6 of 6 hits |
| RTX 4090 | 7.8 s | **0.028 s** | 0 of 6, every one pays 7.85 s |

The 4090 is faster on a hit and useless on an alternation. Setting
`affinity_slots=1` for it stops the router promising a hit that card cannot
give. This is architectural rather than a misconfiguration:
`--context-cost-presets` cannot be ported to that fork without backporting the
whole context-cache subsystem, since it has no `ContextCacheOptions`,
`max_shared_prefixes` or `max_private_continuations` at all.

## Measured coefficients

Both cards in the configuration their launch scripts use, meaning
`--spec mtp --draft-tokens 3 --lm-head-draft` on both.

| | Blackwell nvfp4 | RTX 4090 int |
|---|---|---|
| decode, mean of 4 prompts | 57.4 tok/s (50.9 to 65.2) | **108.0 tok/s** (94.5 to 124.2) |
| decode, speculation off | 29.7 tok/s (29.3 to 29.9) | not measured |
| prefill at 2442 tokens | **0.77 s** | 1.28 s |
| prefill at 32045 tokens | **10.84 s** | 16.79 s |
| prefill at 62966 tokens | **26.20 s** | 36.20 s |
| prefill at 123636 tokens | over context | 83.19 s |
| fitted `prefill` | 3860 tok/s | 2115 tok/s |
| fitted `attn` | 2.493e-9 | 1.619e-9 |
| `max_context` | 81920 | 262144 |

`prefill` alone is not enough at long lengths. Fitting `t/prefill + attn * t^2`
reproduces every point above 30k within 0.2%, while a flat rate tuned to the
short end underestimates a 63k prompt by 28%. The 4090 has the smaller attention
coefficient, so the Blackwell's prefill lead narrows with length: 1.66x at 2.4k,
1.55x at 32k, 1.38x at 63k.

### The resulting shape rule

For a cold request the Blackwell wins when the prompt is more than about 40
times the generation, near enough flat across the range (38 at 1k, 44 at 32k,
57 at the 81920 ceiling). Worked examples:

| shape | Blackwell | 4090 | winner |
|---|---|---|---|
| 32k document, 200 token answer | 14.35 s | 18.67 s | Blackwell |
| 63k document, 400 token answer | 33.17 s | 39.89 s | Blackwell |
| 16k code review, 800 tokens out | 18.85 s | 15.59 s | 4090 |
| 800 token chat turn, 300 out | 5.44 s | 3.16 s | 4090 |
| short prompt, 2000 token essay | 34.97 s | 18.76 s | 4090 |

This only applies to cold requests. A warm prefix on the other card overrides it.

## Calibrating a backend

    python3 tools/router/shape_bench.py --server http://127.0.0.1:8080 \
        --label blackwell --out bw.json --max-context 81920 --repeat 1 \
        --prompts models/retrieval_spot.json models/long_prompt_32tok.json \
                  models/long_prompt_64tok.json

Reports time to first token against prompt length and the decode rate
separately, which is what the two coefficients are fitted from. Run it against
each card in the configuration it will actually serve in. Decode measured under
`--spec` varies with content, 13% across four prompts against 1% with
speculation off, so use the mean and do not expect a single request to match it.

## Tests

`test_router.py` runs four request traces against three policies using
`mock_backend.py`, which reproduces one request in flight, a prefix cache of
about three entries, and separate prefill and decode rates. It is not a model,
so a green run is evidence about placement decisions, not inference speed.

    python3 tools/router/test_router.py

| trace | one card | round robin | router |
|---|---|---|---|
| 1 cold 30k prefill then 6 short calls | 0.227 | 0.124 | **0.047** |
| 2 long documents, 8 bunched turns | 0.320 | 0.387 | **0.252** |
| 3 long documents, 9 bunched turns | 0.466 | 0.330 | **0.255** |
| 3 long documents among 11 chat turns | 0.394 | 0.226 | **0.200** |

`live_bench.py` runs the same comparison against both real servers, which is the
claim that matters. Every policy gets its own nonce so all four face equally
cold caches; without that the policies run in order and each is warmed by the
last, which silently flatters whichever runs last.

    python3 tools/router/live_bench.py \
        --blackwell http://127.0.0.1:8080 --rtx4090 http://127.0.0.1:8081 \
        --endpoint /v1/chat/completions

Mean time to first token, seconds:

| trace | blackwell | rtx4090 | round robin | router |
|---|---|---|---|---|
| 1 cold 32k prefill then 6 short calls | 12.19 | 17.15 | 6.52 | **1.82** |
| 2 documents of 32k, 8 bunched turns | 21.60 | 55.59 | 23.94 | **14.58** |
| 2 documents of 2.4k, 8 bunched turns | 4.18 | 4.64 | **1.92** | 1.94 |

The first row is head of line blocking: the router placed the long prefill on
the Blackwell and all six short calls on the 4090, for a median of 0.29 seconds
against round robin's 10.94.

The second is affinity. Each card took one document and hit cache on all three
follow ups, and round robin came out **worse than the Blackwell alone**, because
it splits each conversation across both cards and destroys the affinity a single
card gets for free. That failure mode is why the router leads with affinity
rather than load.

The third row is the same trace with 2.4k documents, and the router only ties.
That is correct rather than disappointing: re-prefilling 2.4k costs 0.77
seconds, so the queue override declines to wait for a warm card, and there is
little left to win. Affinity pays in proportion to what a cold prefill costs.

Running the first trace on `/v1/messages` instead gives 1.85 seconds against
1.82, with identical placements, which is the endpoint parity claim above
measured end to end.

## Why not just raise --max-concurrency

The obvious cheaper alternative to a second card is to let one server take more
than one request at a time. It was measured, and it does not substitute.

Both servers run `--max-concurrency 1`. Raising it costs context, because the
context-cache capacities scale off it: the CUDA graph allowance alone doubles
from 82 to 164 MiB. Measured on the Blackwell with vision and MTP3, on an 8192
step grid:

| concurrency | ceiling | `serve.sh` default | 1 stream | loaded per-stream | aggregate decode | 8 short calls |
|---|---|---|---|---|---|---|
| 1 | 106496 | 98304 | 56.0 tok/s | 56.0 | 52.5 tok/s | 2.83 s |
| 2 | 81920 | 73728 | 56.0 | **57.1** | **104.7** tok/s | 2.11 s |
| 3 | 65536 | 57344 | | | | |
| 4 | 40960 | 32768 | 55.8 | 46.9 | 144.8 tok/s | 2.59 s |

Concurrency 2 costs 24576 tokens of context, a quarter of it, which is real on a
card where NVFP4 weights already hold the context down. What it buys is
aggregate decode: two concurrent generations run at 57.1 tokens per second each
against 56.0 alone, so throughput doubles for no per-stream cost, because decode
is bound on loading weights and a second sequence rides along. Concurrency 4
reaches 2.76x aggregate but gives up 16% per stream and most of the context.

Pool size does not affect prefill speed, so choosing a context is about what fits
and what slack is left rather than throughput:

| prompt | ctx 106496 | ctx 81920 | ctx 81920, concurrency 2 |
|---|---|---|---|
| 2.4k | 0.76 s | 0.77 s | 0.76 s |
| 32k | 10.79 s | 11.01 s | 11.12 s |
| 63k | 25.75 s | 26.37 s | 26.63 s |
| 100k | 50.44 s | rejected | rejected |

What it does not do is fix head of line blocking, which is the reason the router
exists. A short call arriving 0.3 seconds behind a cold 32k prefill:

| concurrency | idle | behind a 32k prefill | penalty |
|---|---|---|---|
| 1 | 173 ms | 10750 ms | 62x |
| 2 | 161 ms | 10696 ms | 67x |
| 4 | 160 ms | 10708 ms | 67x |

Unchanged at every level. A lane's prefill runs to completion before other lanes
are served, so extra lanes let a request be admitted but not scheduled. Lowering
`--prefill-chunk` does not interleave it either, it only makes the prefill
slower: 10.7 s at 1024, 12.8 s at 256, 15.3 s at 128. There is no scheduler or
admission flag that changes this.

So the two are complementary rather than alternatives. Concurrency 2 on each
card buys aggregate throughput; the router buys protection from long prefills.
`serve.sh` picks a measured context for concurrency 1 to 4.

## Limitations

Prompt length is estimated from serialized character count at 3.0 characters per
token. The corpora here run 3.45 to 3.87, so the estimate is deliberately high
and errs toward the larger context. `/v1/messages/count_tokens` is exact if a
round trip per request is acceptable.

Affinity is tracked by hashing the request's messages, so it follows exact
prefixes only. A client that rewrites earlier turns loses its affinity, which is
correct, because the server's cache would miss too.

`affinity_min_tokens` is a measured floor, not a policy, and it moves. With the
context cost presets calibrated, cycling two prompts on the Blackwell retained
nothing at 349 tokens, a third of the time at 949, half at 1949, and every time
at 3943 and above. Re-measure after changing `--max-concurrency` or the presets.
