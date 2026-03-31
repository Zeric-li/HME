#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH="${1:-configs/crohme_2019_lora.yaml}"
EVAL_SPLIT="${2:-2019}"

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

BASE_EVAL_DIR="${OUTPUT_DIR}/base_model_eval_${EVAL_SPLIT}"
TRAINED_CHECKPOINT="${OUTPUT_DIR}/checkpoint-final"
FT_EVAL_DIR="${TRAINED_CHECKPOINT}/eval_${EVAL_SPLIT}"

./stage1_eval_base.sh "$CONFIG_PATH" "$EVAL_SPLIT"
./stage2_train.sh "$CONFIG_PATH"
./stage3_eval_finetuned.sh "$CONFIG_PATH" "$EVAL_SPLIT"

echo "Pipeline completed."
echo "Base model eval: ${BASE_EVAL_DIR}"
echo "Fine-tuned model eval: ${FT_EVAL_DIR}"
