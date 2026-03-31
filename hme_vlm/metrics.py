from __future__ import annotations

import re
from math_verify import parse, verify


def normalize_latex_for_exact_match(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", "", text)
    return text


def clean_model_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = text.replace("```latex", "").replace("```", "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]
    text = re.sub(r"^(latex\s*:\s*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(answer\s*:\s*)", "", text, flags=re.IGNORECASE)
    return text.strip()


def wrap_for_math_verify(expr: str) -> str:
    expr = (expr or "").strip()
    if not expr:
        return expr
    if expr.startswith("$") and expr.endswith("$"):
        return expr
    return f"${expr}$"


def math_verify_match(gold_latex: str, pred_latex: str) -> bool:
    try:
        gold_parsed = parse(wrap_for_math_verify(gold_latex), raise_on_error=False)
        pred_parsed = parse(wrap_for_math_verify(pred_latex), raise_on_error=False)
        return bool(verify(gold_parsed, pred_parsed, raise_on_error=False))
    except Exception:
        return False
