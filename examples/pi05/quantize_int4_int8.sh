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
# note(chenghua): INT4 weights use symmetric group-128 scales; activations use dynamic per-token INT8.
# note(chenghua): Runtime execution requires a GPU architecture supported by Humming; Humming 0.1.10 does not support SM110.
MLP_TARGET='re:^paligemma_with_expert\.(?:paligemma\.model\.language_model|gemma_expert\.model)\.layers\.[0-9]+\.mlp\.'

cd "${REPO_ROOT}"
uv run phyai-optimize quantize \
  --input "$1" \
  --output "$2" \
  --weight-dtype int4 \
  --activation-dtype int8 \
  --group-size 128 \
  --pack-format compressed-tensors \
  --targets "${MLP_TARGET}"
