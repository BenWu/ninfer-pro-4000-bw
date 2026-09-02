#!/usr/bin/env bash
# Regenerate models/context_cost_presets.json for this machine.
#
# The engine uses these coefficients to decide whether keeping a context is
# cheaper than rebuilding it. Without them it falls back to generic defaults and
# declines to retain prefixes it should keep: cycling two 2.4k prompts went from
# 0 of 6 cache hits at 772ms mean to 6 of 6 at 87ms once calibrated.
#
# Entries are keyed by hardware class plus the artifact's model_id and
# weights_id, so a preset measured on one quantization does not apply to
# another. Re-run this after converting or replacing an artifact, and check the
# startup log for prefill_source=external rather than generic-default.
#
# Both cards must be idle: the prefill suite loads the artifact.
#
# Usage:  ./tools/calibrate_context_cost.sh                      nvfp4 on device 1
#         ARTIFACT=models/qwen3_8_27b.ninfer ./tools/...         the INT artifact
#         DEVICE=0 ./tools/calibrate_context_cost.sh             another card
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ARTIFACT=${ARTIFACT:-models/qwen3_8_27b_nvfp4.ninfer}
DEVICE=${DEVICE:-1}
PRESETS=${PRESETS:-models/context_cost_presets.json}
CORPUS=${CORPUS:-bench/fixtures/bench_corpus.ids}
REPORTS=${REPORTS:-/tmp}
BIN=build/bench/ninfer_context_cost_bench

[ -f "$ARTIFACT" ] || { echo "no artifact at $ARTIFACT" >&2; exit 1; }

# The benchmarks are off by default, so enable them, build, and put the flag back.
if [ ! -x "$BIN" ]; then
    echo "+ building $BIN" >&2
    cmake -S . -B build -DNINFER_BUILD_BENCHMARKS=ON > /dev/null
    cmake --build build -j --target ninfer_context_cost_bench
    cmake -S . -B build -DNINFER_BUILD_BENCHMARKS=OFF > /dev/null
fi

# transfer is machine level and loads no model; prefill is per artifact. Each
# --preset-out replaces only the component it measured, so entries for other
# artifacts survive.
echo "+ transfer suite" >&2
"$BIN" --suite transfer --device "$DEVICE" \
    --json "$REPORTS/context_cost_transfer.json" --preset-out "$PRESETS"

echo "+ prefill suite for $ARTIFACT" >&2
"$BIN" --suite prefill --device "$DEVICE" \
    --artifact "$ARTIFACT" --corpus "$CORPUS" \
    --max-context 8192 --prefill-chunk 1024 \
    --json "$REPORTS/context_cost_prefill.json" --preset-out "$PRESETS"

echo "wrote $PRESETS" >&2
