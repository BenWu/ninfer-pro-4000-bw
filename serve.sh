#!/usr/bin/env bash
# Start ninfer-serve on the RTX PRO 4000 with NVFP4 weights and NVFP4 KV.
#
# With CTX unset the context is chosen from the feature combination, using values measured on this
# machine (ECC on, 24467 MiB usable, NVFP4 weights, NVFP4 KV). Each ceiling below is the largest
# 8192 step that started; the default is one step under it so there is room to spare:
#
#   features            ceiling             default picked      slack at the default
#   plain               212992 (104 MiB)    204800              248 MiB
#   --vision            163840 ( 12 MiB)    155648              156 MiB
#   MTP3                155648 (160 MiB)    147456              312 MiB
#   --vision + MTP3     106496 (122 MiB)     98304              274 MiB
#
# Running at a ceiling works but leaves nothing for a larger media payload or a different driver
# state, so the defaults trade a step of context for headroom. Override with CTX=... to run at the
# ceiling, or anywhere else.
#
# These ceilings were re-measured on 2026-09-02 and every one came out 8192 to 16384 tokens above
# the values previously recorded here, so the old defaults were leaving context unused. Re-measure
# after a driver change rather than trusting the table.
#
# Both features cost memory before KV is allocated. MTP3 is the expensive one because of its
# recurrent draft state, which is why enabling it costs more context than vision does.
#
# Raising MAX_CONCURRENCY lowers every entry, because the context-cache defaults scale off it
# (device-state slots = concurrency, private continuations = 2x, shared prefixes = 1x). Measured
# for vision + MTP3, the combination this script defaults to:
#
#   concurrency   ceiling             default picked      slack at the default
#   1             106496 (122 MiB)     98304              274 MiB
#   2              81920 (272 MiB)     73728              426 MiB
#   3              65536 (272 MiB)     57344              426 MiB
#   4              40960 (426 MiB)     32768              578 MiB
#
# Concurrency 2 costs 24576 tokens of context against concurrency 1, because the CUDA graph
# allowance doubles (82 MiB to 164 MiB). What it buys is aggregate decode: two concurrent
# generations run at 57.1 tok/s each against 56.0 alone, so throughput doubles to 104.7 tok/s for
# no per-stream cost, since decode is bound on loading weights and a second sequence rides along.
# Concurrency 4 reaches 144.8 tok/s aggregate but costs 16% per stream and most of the context.
#
# What concurrency does not buy is protection from a long prefill. A short call arriving behind a
# cold 32k prefill waited 10.7s at every concurrency level, because a lane's prefill runs to
# completion before other lanes are served. Lowering --prefill-chunk does not interleave it either,
# only slows the prefill down: 10.7s at 1024, 12.8s at 256, 15.3s at 128. Mixing long and short
# requests on one card needs a second card and a router, not a bigger concurrency number.
#
# Pool size does not affect prefill speed. A 32k prompt took 10.79s at a 106496 context, 11.01s at
# 81920 and 11.12s at concurrency 2, and a 63k prompt 25.7s, 26.4s and 26.6s. Choosing a context is
# about what fits and what slack is left, not throughput.
#
# For any other KV dtype, or above concurrency 1 with a different feature combination, the script
# asks for CTX explicitly.
#
# Usage:  ./serve.sh                     vision + MTP3, context chosen automatically
#         VISION=0 ./serve.sh            drop vision, context rises to 147456
#         CTX=106496 ./serve.sh          run at the measured ceiling instead
#         MAX_CONCURRENCY=2 ./serve.sh   double aggregate decode, context falls to 73728
#         ./serve.sh --greedy            extra flags are passed through to ninfer-serve
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL=${MODEL:-models/qwen3_8_27b_nvfp4.ninfer}
BIN=${BIN:-./build/apps/ninfer-serve}
DEVICE=${DEVICE:-1}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
CTX=${CTX:-}
KV_DTYPE=${KV_DTYPE:-nvfp4}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-1}
DEFAULT_MAX_TOKENS=${DEFAULT_MAX_TOKENS:-8192}
PREFILL_CHUNK=${PREFILL_CHUNK:-1024}
DRAFT_TOKENS=${DRAFT_TOKENS:-3}
VISION=${VISION:-1}
MTP=${MTP:-1}
PRESETS=${PRESETS:-models/context_cost_presets.json}

[ -x "$BIN" ] || { echo "no server binary at $BIN (cmake --build build -j)" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "no artifact at $MODEL" >&2; exit 1; }

# Context defaults measured per feature combination. See the table at the top of this file.
if [ -z "$CTX" ]; then
    if [ "$KV_DTYPE" != nvfp4 ]; then
        echo "serve.sh: the measured context table covers KV_DTYPE=nvfp4 only." >&2
        echo "          You have KV_DTYPE=$KV_DTYPE, so set CTX explicitly." >&2
        exit 1
    fi
    if [ "$MAX_CONCURRENCY" = 1 ]; then
        if [ "$VISION" = 1 ] && [ "$MTP" = 1 ]; then CTX=98304
        elif [ "$VISION" = 1 ];                then CTX=155648
        elif [ "$MTP" = 1 ];                   then CTX=147456
        else                                        CTX=204800
        fi
    elif [ "$VISION" = 1 ] && [ "$MTP" = 1 ]; then
        # Only this feature combination was measured above concurrency 1.
        case "$MAX_CONCURRENCY" in
            2) CTX=73728 ;;
            3) CTX=57344 ;;
            4) CTX=32768 ;;
            *) echo "serve.sh: no measured context for MAX_CONCURRENCY=$MAX_CONCURRENCY, set CTX explicitly." >&2
               exit 1 ;;
        esac
    else
        echo "serve.sh: above MAX_CONCURRENCY=1 only vision=1 mtp=1 was measured." >&2
        echo "          You have vision=$VISION mtp=$MTP, so set CTX explicitly." >&2
        exit 1
    fi
    echo "serve.sh: vision=$VISION mtp=$MTP concurrency=$MAX_CONCURRENCY -> --max-context $CTX (measured, see header)" >&2
fi

ARGS=("$MODEL"
      --host "$HOST" --port "$PORT" --device "$DEVICE"
      --max-context "$CTX" --kv-capacity "$CTX" --kv-dtype "$KV_DTYPE"
      --max-concurrency "$MAX_CONCURRENCY"
      --default-max-tokens "$DEFAULT_MAX_TOKENS"
      --prefill-chunk "$PREFILL_CHUNK")

[ "$VISION" = 1 ] && ARGS+=(--vision)
[ "$MTP" = 1 ] && ARGS+=(--spec mtp --draft-tokens "$DRAFT_TOKENS" --lm-head-draft)

# Measured cost model for this card, driving context-cache retention. Both the transfer and
# prefill components are calibrated and accepted; startup logs transfer_source and prefill_source
# as "external" when they load. The file is keyed by model_id and weights_id, so re-run the
# calibrator after switching artifacts or the entry misses and generic defaults are used silently.
# Skipped if absent.
[ -f "$PRESETS" ] && ARGS+=(--context-cost-presets "$PRESETS")

echo "+ $BIN ${ARGS[*]} $*" >&2
exec "$BIN" "${ARGS[@]}" "$@"
