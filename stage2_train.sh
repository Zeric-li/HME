#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH="${1:-configs/crohme_2019_lora.yaml}"

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

echo "[Stage 2] Training with config: ${CONFIG_PATH}"
python -m scripts.train_lora --config "$CONFIG_PATH"
echo "[Stage 2] Final checkpoint: ${OUTPUT_DIR}/checkpoint-final"
