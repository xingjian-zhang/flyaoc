"""Generate table-ready summaries from normalized prediction artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from flyaoc.evaluation import evaluate_prediction_file


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return json.load(handle)


def evaluate_manifest(
    manifest_path: str | Path, *, include_optional: bool = False
) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    for artifact in manifest["prediction_files"]:
        if not include_optional and not artifact.get("required_for_paper", True):
            continue
        prediction_path = Path(manifest_path).parent / artifact["path"]
        if not prediction_path.exists():
            if artifact.get("required_for_paper", True):
                raise FileNotFoundError(f"Missing required prediction artifact: {prediction_path}")
            continue
        evaluation = evaluate_prediction_file(prediction_path)
        aggregate = evaluation["aggregate"]
        row = {
            "artifact": artifact["path"],
            "baseline": artifact.get("baseline", ""),
            "provider": artifact.get("provider", ""),
            "model": artifact.get("model", ""),
            "paper_budget": artifact.get("paper_budget", ""),
            "n_genes": evaluation["n_genes"],
            "task1_verified_fact_count": aggregate["task1_verified_fact_count"],
            "task2_verified_fact_count": aggregate["task2_verified_fact_count"],
            "task3_verified_fact_count": aggregate["task3_verified_fact_count"],
            "task1_semantic_recall_at_20_micro": aggregate[
                "task1_semantic_recall_at_20_micro"
            ],
            "task2_anatomy_semantic_recall_at_10_micro": aggregate[
                "task2_anatomy_semantic_recall_at_10_micro"
            ],
            "task3_combined_exact_recall_at_20_micro": aggregate[
                "task3_combined_exact_recall_at_20_micro"
            ],
            "task1_semantic_recall_at_20_macro": aggregate[
                "task1_semantic_recall_at_20_macro"
            ],
            "task2_anatomy_semantic_recall_at_10_macro": aggregate[
                "task2_anatomy_semantic_recall_at_10_macro"
            ],
            "task3_combined_exact_recall_at_20_macro": aggregate[
                "task3_combined_exact_recall_at_20_macro"
            ],
        }
        _add_k_columns(row, "task1_semantic", aggregate["task1_semantic_recall_at_k_micro"])
        _add_k_columns(
            row,
            "task2_anatomy_semantic",
            aggregate["task2_anatomy_semantic_recall_at_k_micro"],
        )
        _add_k_columns(
            row,
            "task3_combined_exact",
            aggregate["task3_combined_exact_recall_at_k_micro"],
        )
        _add_k_columns(
            row,
            "task1_semantic_macro",
            aggregate["task1_semantic_recall_at_k_macro"],
        )
        _add_k_columns(
            row,
            "task2_anatomy_semantic_macro",
            aggregate["task2_anatomy_semantic_recall_at_k_macro"],
        )
        _add_k_columns(
            row,
            "task3_combined_exact_macro",
            aggregate["task3_combined_exact_recall_at_k_macro"],
        )
        rows.append(row)
    return rows


def _add_k_columns(row: dict[str, Any], prefix: str, values: dict[str, float]) -> None:
    for k_value in sorted(values, key=lambda value: int(value)):
        row[f"{prefix}_recall_at_{k_value}"] = values[k_value]


def write_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("")
        return
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("No prediction artifacts were evaluated.\n")
        return
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row[key]) for key in headers) + " |")
    output.write_text("\n".join(lines) + "\n")


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
