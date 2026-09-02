#!/usr/bin/env bash
# Start ninfer-serve on the RTX PRO 4000 with NVFP4 weights and NVFP4 KV.
#
# With CTX unset the context is chosen from the feature combination, using values measured on this
# machine (ECC on, 24467 MiB usable, NVFP4 weights, NVFP4 KV, --max-concurrency 1). Each pair below
# is the largest context that started, and the default this script picks, which is one step below
# it so there is room to spare:
#
#   features            ceiling            default picked      slack at the default
#   plain               200704 ( 37 MiB)   196608              109 MiB
#   --vision            147456 ( 17 MiB)   139264              161 MiB
#   MTP3                139264 (113 MiB)   131072              266 MiB
#   --vision + MTP3      90112 ( 75 MiB)    81920              228 MiB
#
# Running at a ceiling works but leaves nothing for a larger media payload or a different driver
# state, so the defaults trade a step of context for headroom. Override with CTX=... to run at the
# ceiling, or anywhere else.
#
# Both features cost memory before KV is allocated: vision 282 MiB, MTP3 771 MiB. MTP3 is the
# expensive one because of its recurrent draft state, which is why enabling it costs more context
# than vision does.
#
# The table above is for --max-concurrency 1. Raising MAX_CONCURRENCY lowers every entry, because
# the context-cache defaults scale off it (device-state slots = concurrency, private continuations
# = 2x, shared prefixes = 1x). Measured for vision + MTP3, the combination this script defaults to:
#
#   concurrency   largest that started   next step up failed   default picked
#   1                          106496                 114688           81920
#   2                           86016                  90112           81920
#   3                           65536                  81920           57344
#   4                           40960                  49152           32768
#
# Concurrency 2 is the good trade. Two concurrent generations run at 57.1 tok/s each against 56.0
# alone, so aggregate decode doubles to 104.7 tok/s for no per-stream cost, and the context it gives
# up is the headroom step this script was already leaving unused. Concurrency 4 reaches 144.8 tok/s
# aggregate but costs 16% per stream and most of the context.
#
# What concurrency does not buy is protection from a long prefill. A short call arriving behind a
# cold 32k prefill waited 10.7s at every concurrency level, because a lane's prefill runs to
# completion before other lanes are served. Lowering --prefill-chunk does not interleave it either,
# only slows the prefill down: 10.7s at 1024, 12.8s at 256, 15.3s at 128. Mixing long and short
# requests on one card needs a second card and a router, not a bigger concurrency number.
#
# For any other KV dtype or feature combination the script still asks for CTX explicitly.
#
# Usage:  ./serve.sh                     vision + MTP3, context chosen automatically
#         VISION=0 ./serve.sh            drop vision, context rises to 131072
#         CTX=90112 ./serve.sh           run at the measured ceiling instead
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
        if [ "$VISION" = 1 ] && [ "$MTP" = 1 ]; then CTX=81920
        elif [ "$VISION" = 1 ];                then CTX=139264
        elif [ "$MTP" = 1 ];                   then CTX=131072
        else                                        CTX=196608
        fi
    elif [ "$VISION" = 1 ] && [ "$MTP" = 1 ]; then
        # Only this feature combination was measured above concurrency 1.
        case "$MAX_CONCURRENCY" in
            2) CTX=81920 ;;
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
