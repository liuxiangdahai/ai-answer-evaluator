"""Deterministic answer evaluator for simple AI-assistant assessments."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class EvaluationError(Exception):
    """Raised when evaluation input cannot be read or validated."""


def normalize_answer(answer: str) -> str:
    """Trim surrounding whitespace and lowercase an answer."""
    return answer.strip().lower()


def tokenize_answer(answer: str) -> list[str]:
    """Extract lowercase alphanumeric tokens for partial scoring."""
    return TOKEN_PATTERN.findall(normalize_answer(answer))


def token_overlap_f1(reference: str, model: str) -> float:
    """Compute token-overlap F1 using alphanumeric tokens and token counts."""
    reference_tokens = Counter(tokenize_answer(reference))
    model_tokens = Counter(tokenize_answer(model))

    if not reference_tokens or not model_tokens:
        return 0.0

    overlap = sum((reference_tokens & model_tokens).values())
    if overlap == 0:
        return 0.0

    precision = overlap / sum(model_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return (2 * precision * recall) / (precision + recall)


def load_items(path: str | Path) -> list[Any]:
    """Load and validate the top-level JSON shape from a file."""
    input_path = Path(path)
    if not input_path.exists():
        raise EvaluationError(f"Missing file: {input_path}")
    if not input_path.is_file():
        raise EvaluationError(f"Input path is not a file: {input_path}")

    try:
        with input_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise EvaluationError(
            f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(data, list):
        raise EvaluationError("Input JSON must be a list of evaluation items.")

    return data


def evaluate_items(items: list[Any]) -> dict[str, Any]:
    """Evaluate all items and return per-item results plus a summary."""
    if not isinstance(items, list):
        raise EvaluationError("Evaluation input must be a list.")

    results = []
    exact_matches = 0

    for index, item in enumerate(items):
        result, is_exact = evaluate_item(item, index)
        results.append(result)
        if is_exact:
            exact_matches += 1

    scores = [result["score"] for result in results]
    number_of_items = len(results)
    average_score = sum(scores) / number_of_items if number_of_items else 0.0

    return {
        "results": results,
        "summary": {
            "average_score": round(average_score, 4),
            "number_of_items": number_of_items,
            "exact_matches": exact_matches,
            "failed_items": sum(1 for score in scores if score == 0.0),
        },
    }


def evaluate_item(item: Any, index: int) -> tuple[dict[str, Any], bool]:
    """Evaluate one JSON item and report whether it was an exact match."""
    if not isinstance(item, dict):
        raise EvaluationError(f"Item at index {index} must be an object.")

    _validate_answer_field(item, "reference_answer", index)
    _validate_answer_field(item, "model_answer", index)

    item_id = item.get("id", index)

    if "reference_answer" not in item or "model_answer" not in item:
        return _result(item_id, 0.0, "Missing reference_answer or model_answer."), False

    reference_answer = item.get("reference_answer")
    model_answer = item.get("model_answer")
    if reference_answer is None or model_answer is None:
        return _result(item_id, 0.0, "Missing reference_answer or model_answer."), False

    reference_normalized = normalize_answer(reference_answer)
    model_normalized = normalize_answer(model_answer)

    if not reference_normalized or not model_normalized:
        return _result(item_id, 0.0, "Empty reference_answer or model_answer."), False

    if reference_normalized == model_normalized:
        return _result(item_id, 1.0, "Exact match after normalization."), True

    score = token_overlap_f1(reference_normalized, model_normalized)
    if score == 0.0:
        reason = "No token overlap after normalization."
    else:
        reason = "Partial token-overlap F1 after normalization."

    return _result(item_id, score, reason), False


def _validate_answer_field(item: dict[str, Any], field_name: str, index: int) -> None:
    if field_name in item and item[field_name] is not None:
        if not isinstance(item[field_name], str):
            raise EvaluationError(
                f"Item at index {index} has invalid {field_name}; expected a string."
            )


def _result(item_id: Any, score: float, reason: str) -> dict[str, Any]:
    return {"id": item_id, "score": round(score, 4), "reason": reason}
