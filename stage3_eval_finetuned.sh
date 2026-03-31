#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH="${1:-configs/crohme_2019_lora.yaml}"
EVAL_SPLIT="${2:-2019}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
FT_EVAL_BATCH_SIZE="${FT_EVAL_BATCH_SIZE:-1}"
DATASET_ID="${DATASET_ID:-Neeze/CROHME-full}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

OUTPUT_DIR="$(python - <<'PY' "$CONFIG_PATH"
import sys
from hme_vlm.config import load_yaml_config
config = load_yaml_config(sys.argv[1])
print(config["output_dir"])
PY
)"

TRAINED_CHECKPOINT="${OUTPUT_DIR}/checkpoint-final"
FT_EVAL_DIR="${TRAINED_CHECKPOINT}/eval_${EVAL_SPLIT}"

CMD=(
  python -m scripts.eval_generate
  --checkpoint "$TRAINED_CHECKPOINT"
  --output-dir "$FT_EVAL_DIR"
  --batch-size "$FT_EVAL_BATCH_SIZE"
  --split "$EVAL_SPLIT"
  --config "$CONFIG_PATH"
  --dataset-id "$DATASET_ID"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi

echo "[Stage 3] Evaluating fine-tuned checkpoint: ${TRAINED_CHECKPOINT}"
"${CMD[@]}"
echo "[Stage 3] Output: ${FT_EVAL_DIR}"
