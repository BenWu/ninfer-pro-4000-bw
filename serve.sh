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
# The table is only valid for NVFP4 KV at --max-concurrency 1. Raising MAX_CONCURRENCY lowers every
# entry, because the context-cache defaults scale off it (device-state slots = concurrency, private
# continuations = 2x, shared prefixes = 1x). Change either and the script warns and asks for CTX.
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
    if [ "$KV_DTYPE" != nvfp4 ] || [ "$MAX_CONCURRENCY" != 1 ]; then
        echo "serve.sh: the measured context table covers KV_DTYPE=nvfp4 at MAX_CONCURRENCY=1." >&2
        echo "          You have KV_DTYPE=$KV_DTYPE MAX_CONCURRENCY=$MAX_CONCURRENCY, so set CTX explicitly." >&2
        exit 1
    fi
    if [ "$VISION" = 1 ] && [ "$MTP" = 1 ]; then CTX=81920
    elif [ "$VISION" = 1 ];                then CTX=139264
    elif [ "$MTP" = 1 ];                   then CTX=131072
    else                                        CTX=196608
    fi
    echo "serve.sh: vision=$VISION mtp=$MTP -> --max-context $CTX (measured, see header)" >&2
fi

ARGS=("$MODEL"
      --host "$HOST" --port "$PORT" --device "$DEVICE"
      --max-context "$CTX" --kv-capacity "$CTX" --kv-dtype "$KV_DTYPE"
      --max-concurrency "$MAX_CONCURRENCY"
      --default-max-tokens "$DEFAULT_MAX_TOKENS"
      --prefill-chunk "$PREFILL_CHUNK")

[ "$VISION" = 1 ] && ARGS+=(--vision)
[ "$MTP" = 1 ] && ARGS+=(--spec mtp --draft-tokens "$DRAFT_TOKENS" --lm-head-draft)

# Measured prefill cost model for this card. Only the prefill component was accepted by the
# calibrator; transfer falls back to the generic defaults. Skipped silently if absent.
[ -f "$PRESETS" ] && ARGS+=(--context-cost-presets "$PRESETS")

echo "+ $BIN ${ARGS[*]} $*" >&2
exec "$BIN" "${ARGS[@]}" "$@"
