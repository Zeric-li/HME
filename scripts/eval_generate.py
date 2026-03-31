from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from hme_vlm.config import load_yaml_config
from hme_vlm.data import load_hf_hme_records, build_prompt_messages
from hme_vlm.metrics import clean_model_text, math_verify_match, normalize_latex_for_exact_match
from hme_vlm.modeling import load_model_for_inference
from qwen_vl_utils import process_vision_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="2019")
    parser.add_argument("--dataset-id", type=str, default="Neeze/CROHME-full")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def chunked(items, n):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def make_eval_output_dir(checkpoint: str, split: str, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)

    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        return checkpoint_path / f"eval_{split}"

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", checkpoint).strip("-")
    return Path("outputs") / "evals" / safe_name / f"eval_{split}"


def load_model_and_processor(args):
    if args.config:
        config = load_yaml_config(args.config)
    else:
        config = {
            "torch_dtype": "bfloat16",
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "system_prompt": (
                "You are a handwritten mathematical expression transcription engine. "
                "Your entire reply must be exactly one raw LaTeX expression. "
                "Output only the expression content itself, with no surrounding formatting or wrapper symbols."
            ),
            "user_prompt": (
                "Transcribe the handwritten mathematical expression in the image into one raw LaTeX expression. "
                "Strict output contract:\n"
                "- Output exactly one line.\n"
                "- Output only the raw LaTeX tokens for the expression.\n"
                "- Do not wrap the answer in $...$, $$...$$, \\(...\\), or \\[...\\].\n"
                '- Do not add "latex:", "answer:", markdown fences, explanations, spaces before or after, or any extra text.\n'
                "Correct format example: \\frac{a}{b}+c\n"
                "Wrong format examples: $\\frac{a}{b}+c$, $$\\frac{a}{b}+c$$, \\(\\frac{a}{b}+c\\), LaTeX: \\frac{a}{b}+c"
            ),
            "max_new_tokens": 128,
        }

    model, processor = load_model_for_inference(
        model_id_or_adapter_path=args.checkpoint,
        min_pixels=config["min_pixels"],
        max_pixels=config["max_pixels"],
        torch_dtype=config["torch_dtype"],
    )
    return model, processor, config


def main() -> None:
    args = parse_args()
    model, processor, config = load_model_and_processor(args)
    max_new_tokens = int(config.get("max_new_tokens", 128))

    records = load_hf_hme_records(
        dataset_id=args.dataset_id,
        split=args.split,
        max_samples=args.max_samples,
        shuffle=False,
        seed=42,
    )

    out_dir = make_eval_output_dir(args.checkpoint, args.split, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for batch_records in tqdm(list(chunked(records, args.batch_size)), desc=f"Evaluating {args.split}"):
        messages = [
            build_prompt_messages(
                image=r.image,
                system_prompt=config["system_prompt"],
                user_prompt=config["user_prompt"],
            )
            for r in batch_records
        ]

        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(model.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        latency_s = time.perf_counter() - t0

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        texts_out = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for record, pred_text in zip(batch_records, texts_out):
            pred = clean_model_text(pred_text)
            gold = record.latex
            exact = normalize_latex_for_exact_match(pred) == normalize_latex_for_exact_match(gold)
            mv = math_verify_match(gold, pred)
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "source": record.source,
                    "gold_latex": gold,
                    "pred_latex": pred,
                    "exact_match": exact,
                    "math_verify_match": mv,
                    "latency_s": latency_s / max(1, len(batch_records)),
                }
            )

    if not rows:
        raise ValueError("No evaluation samples were loaded. Check --split, --dataset-id, or --max-samples.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "predictions.csv", index=False)

    metrics = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "num_samples": int(len(df)),
        "exact_match_rate": float(df["exact_match"].mean()),
        "math_verify_rate": float(df["math_verify_match"].mean()),
        "avg_latency_s": float(df["latency_s"].mean()),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions to: {out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
