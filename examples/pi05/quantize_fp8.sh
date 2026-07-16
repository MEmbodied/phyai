#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 INPUT_CHECKPOINT OUTPUT_DIRECTORY" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# note(chenghua): Quantize only the language-model and action-expert Transformer MLP linears (gate/up/down).
# note(chenghua): Keep the vision MLP, time MLP, attention, embeddings, norms, and heads in their original precision.
# note(chenghua): Block-128 uses 128x128 weight scales and dynamic group-128 activation scales required by the SM110 FlashInfer path.
MLP_TARGET='re:^paligemma_with_expert\.(?:paligemma\.model\.language_model|gemma_expert\.model)\.layers\.[0-9]+\.mlp\.'

cd "${REPO_ROOT}"
uv run phyai-optimize quantize \
  --input "$1" \
  --output "$2" \
  --weight-dtype fp8_e4m3 \
  --activation-dtype fp8_e4m3 \
  --fp8-scheme block-128 \
  --pack-format compressed-tensors \
  --targets "${MLP_TARGET}"
