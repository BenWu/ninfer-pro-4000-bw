#!/usr/bin/env bash
# Start ninfer-serve on the RTX PRO 4000 with NVFP4 weights and NVFP4 KV.
#
# Every default here was measured with ECC on, 24467 MiB usable.
# Largest --max-context that starts, NVFP4 weights + NVFP4 KV, at --max-concurrency 1:
#
#   plain                 196608   (109 MiB slack)
#   --vision              131072   (307 MiB slack)
#   MTP3                  131072   (266 MiB slack)
#   --vision + MTP3        90112   ( 75 MiB slack)   <- both features, the ceiling
#
# The default below is 81920 rather than the 90112 ceiling. At 90112 startup succeeds with only
# 75 MiB to spare, which a raised --max-concurrency, a larger media payload, or a slightly
# different driver state can erase. 81920 keeps 228 MiB and still gives an 80K context.
#
# Raising MAX_CONCURRENCY lowers all of these: the context-cache defaults scale off it
# (device-state slots = concurrency, private continuations = 2x, shared prefixes = 1x), so it
# costs memory beyond the KV pool itself.
#
# Usage:  ./serve.sh                     start with the defaults below
#         CTX=65536 ./serve.sh           override any single setting
#         ./serve.sh --greedy            extra flags are passed through to ninfer-serve
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL=${MODEL:-models/qwen3_8_27b_nvfp4.ninfer}
BIN=${BIN:-./build/apps/ninfer-serve}
DEVICE=${DEVICE:-1}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
CTX=${CTX:-81920}
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
