"""Recall@k utilities."""

from __future__ import annotations

from collections.abc import Callable

from flyaoc.evaluation.constants import RECALL_K_VALUES

DEFAULT_K_VALUES = RECALL_K_VALUES


def exact_recall_at_k(predictions: list[str], ground_truth: set[str], k: int) -> float:
    if not ground_truth:
        return 0.0
    return len(set(predictions[:k]) & ground_truth) / len(ground_truth)


def semantic_recall_at_k(
    predictions: list[str],
    ground_truth: set[str],
    k: int,
    similarity_fn: Callable[[str, str], float],
) -> float:
    if not ground_truth:
        return 0.0
    top_k = predictions[:k]
    if not top_k:
        return 0.0
    total = 0.0
    for gt_item in ground_truth:
        total += max((similarity_fn(pred, gt_item) for pred in top_k), default=0.0)
    return total / len(ground_truth)


def recall_series(
    predictions: list[str],
    ground_truth: set[str],
    *,
    similarity_fn: Callable[[str, str], float] | None = None,
    k_values: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    values = k_values or DEFAULT_K_VALUES
    result = {
        "exact_recall_at_k": {str(k): exact_recall_at_k(predictions, ground_truth, k) for k in values}
    }
    if similarity_fn is not None:
        result["semantic_recall_at_k"] = {
            str(k): semantic_recall_at_k(predictions, ground_truth, k, similarity_fn)
            for k in values
        }
    return result
