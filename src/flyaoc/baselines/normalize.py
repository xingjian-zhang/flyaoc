"""Normalize raw baseline outputs into evaluator-ready prediction rows."""

from __future__ import annotations

from typing import Any

from flyaoc.baselines.types import RawBaselineResult


def normalize_result(result: RawBaselineResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, RawBaselineResult):
        raw = result.to_dict()
    else:
        raw = result

    output = raw.get("output") or {}
    run_status = raw.get("run_status")
    if run_status is None:
        run_status = _infer_status(output, raw.get("error"))
    return {
        "gene_id": raw["gene_id"],
        "gene_symbol": raw["gene_symbol"],
        "baseline": raw["baseline"],
        "provider": raw["provider"],
        "model": raw["model"],
        "paper_budget": raw["paper_budget"],
        "run_status": run_status,
        "task1_function_predictions": _strip_evidence(
            output.get("task1_function_predictions", output.get("task1_function", []))
        ),
        "task2_expression_predictions": _strip_evidence(
            output.get("task2_expression_predictions", output.get("task2_expression", []))
        ),
        "task3_synonym_predictions": _normalize_synonyms(
            output.get("task3_synonym_predictions", output.get("task3_synonyms", {}))
        ),
    }


def _strip_evidence(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = {key: value for key, value in row.items() if key != "evidence"}
        if item:
            normalized.append(item)
    return normalized


def _normalize_synonyms(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {"fullname_synonyms": [], "symbol_synonyms": []}
    return {
        "fullname_synonyms": [str(item) for item in value.get("fullname_synonyms", [])],
        "symbol_synonyms": [str(item) for item in value.get("symbol_synonyms", [])],
    }


def _infer_status(output: dict[str, Any], error: str | None) -> str:
    if error:
        return "failed"
    row = {
        "task1_function_predictions": _strip_evidence(
            output.get("task1_function_predictions", output.get("task1_function", []))
        ),
        "task2_expression_predictions": _strip_evidence(
            output.get("task2_expression_predictions", output.get("task2_expression", []))
        ),
        "task3_synonym_predictions": _normalize_synonyms(
            output.get("task3_synonym_predictions", output.get("task3_synonyms", {}))
        ),
    }
    if (
        row["task1_function_predictions"]
        or row["task2_expression_predictions"]
        or row["task3_synonym_predictions"]["fullname_synonyms"]
        or row["task3_synonym_predictions"]["symbol_synonyms"]
    ):
        return "ok"
    return "empty_output"
