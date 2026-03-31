#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH="${1:-configs/crohme_2019_lora.yaml}"
EVAL_SPLIT="${2:-2019}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BASE_EVAL_BATCH_SIZE="${BASE_EVAL_BATCH_SIZE:-1}"
DATASET_ID="${DATASET_ID:-Neeze/CROHME-full}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

BASE_MODEL_ID="$(python - <<'PY' "$CONFIG_PATH"
import sys
from hme_vlm.config import load_yaml_config
config = load_yaml_config(sys.argv[1])
print(config["model_id"])
PY
)"

OUTPUT_DIR="$(python - <<'PY' "$CONFIG_PATH"
import sys
from hme_vlm.config import load_yaml_config
config = load_yaml_config(sys.argv[1])
print(config["output_dir"])
PY
)"

BASE_EVAL_DIR="${OUTPUT_DIR}/base_model_eval_${EVAL_SPLIT}"

CMD=(
  python -m scripts.eval_generate
  --checkpoint "$BASE_MODEL_ID"
  --output-dir "$BASE_EVAL_DIR"
  --batch-size "$BASE_EVAL_BATCH_SIZE"
  --split "$EVAL_SPLIT"
  --config "$CONFIG_PATH"
  --dataset-id "$DATASET_ID"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi

echo "[Stage 1] Evaluating base model: ${BASE_MODEL_ID}"
"${CMD[@]}"
echo "[Stage 1] Output: ${BASE_EVAL_DIR}"
