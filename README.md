# Qwen3-VL HME LoRA MVP

A minimal pure-Python project for supervised LoRA fine-tuning of `Qwen/Qwen3-VL-4B-Instruct` on handwritten mathematical expression (HME) transcription.

## Scope

This project is intentionally small and direct:

- one training script
- one evaluation script
- one simple dataset adapter
- one metrics module
- no framework-specific project scaffolding

The default dataset target is `Neeze/CROHME-full` from Hugging Face.

## Environment

Recommended hardware for the default config:

- 1x A100 80GB
- CUDA-enabled PyTorch
- Linux

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If `transformers` support for Qwen2.5-VL lags in your environment, install the latest version from source:

```bash
pip install -U "git+https://github.com/huggingface/transformers"
```

## Train

```bash
python -m scripts.train_lora \
  --config configs/crohme_2019_lora.yaml
```

## Evaluate

```bash
python -m scripts.eval_generate \
  --checkpoint ./outputs/qwen3vl-crohme-lora/checkpoint-final \
  --config configs/crohme_2019_lora.yaml \
  --split 2019
```

## Dataset assumptions

The code expects a Hugging Face dataset with these fields:

- `image`: PIL image or image-like object accepted by `datasets`
- `label`: ground-truth LaTeX string

This matches `Neeze/CROHME-full` for the benchmark splits `2014`, `2016`, and `2019`.

## Notes

- The training objective masks prompt tokens and only computes loss on the assistant answer tokens.
- The default LoRA target list focuses on the language model attention projections.
- The vision backbone stays frozen in the MVP.
- Math-Verify is used only for evaluation, not for the training loss.
