"""Generate table-ready summaries from normalized prediction artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from flyaoc.evaluation import evaluate_prediction_file
from flyaoc.evaluation.constants import PRIMARY_TASK1_K, PRIMARY_TASK2_K, PRIMARY_TASK3_K

MODEL_COMPARISON_ARTIFACTS = [
    "multi_agent/openai_gpt-5-mini_budget16.jsonl",
    "multi_agent/openai_gpt-4o_budget16.jsonl",
    "multi_agent/openai_gpt-5_budget16.jsonl",
    "multi_agent/bedrock_claude-sonnet-4_budget16.jsonl",
    "multi_agent/bedrock_minimax-m2.5_budget16.jsonl",
    "multi_agent/bedrock_deepseek-v3.2_budget16.jsonl",
]

MAIN_ARCHITECTURE_ARTIFACTS = [
    "memorization/openai_gpt-5-mini_budget0.jsonl",
    "pipeline/openai_gpt-5-mini_budget1.jsonl",
    "pipeline/openai_gpt-5-mini_budget2.jsonl",
    "pipeline/openai_gpt-5-mini_budget4.jsonl",
    "pipeline/openai_gpt-5-mini_budget8.jsonl",
    "pipeline/openai_gpt-5-mini_budget16.jsonl",
    "single_agent/openai_gpt-5-mini_budget1.jsonl",
    "single_agent/openai_gpt-5-mini_budget2.jsonl",
    "single_agent/openai_gpt-5-mini_budget4.jsonl",
    "single_agent/openai_gpt-5-mini_budget8.jsonl",
    "single_agent/openai_gpt-5-mini_budget16.jsonl",
    "multi_agent/openai_gpt-5-mini_budget1.jsonl",
    "multi_agent/openai_gpt-5-mini_budget2.jsonl",
    "multi_agent/openai_gpt-5-mini_budget4.jsonl",
    "multi_agent/openai_gpt-5-mini_budget8.jsonl",
    "multi_agent/openai_gpt-5-mini_budget16.jsonl",
]

CROSS_FAMILY_ARTIFACTS = [
    "memorization/bedrock_claude-sonnet-4_budget0.jsonl",
    "memorization/openai_gpt-5-mini_budget0.jsonl",
    "memorization/bedrock_minimax-m2.5_budget0.jsonl",
    "memorization/bedrock_deepseek-v3.2_budget0.jsonl",
    "single_agent/bedrock_claude-sonnet-4_budget16.jsonl",
    "single_agent/openai_gpt-5-mini_budget16.jsonl",
    "single_agent/bedrock_minimax-m2.5_budget16.jsonl",
    "single_agent/bedrock_deepseek-v3.2_budget16.jsonl",
    "multi_agent/bedrock_claude-sonnet-4_budget16.jsonl",
    "multi_agent/openai_gpt-5-mini_budget16.jsonl",
    "multi_agent/bedrock_minimax-m2.5_budget16.jsonl",
    "multi_agent/bedrock_deepseek-v3.2_budget16.jsonl",
]

MODEL_DISPLAY_NAMES = {
    "gpt-5-mini": "GPT-5-mini",
    "gpt-4o": "GPT-4o",
    "gpt-5": "GPT-5",
    "claude-sonnet-4": "Claude Sonnet 4.6",
    "minimax-m2.5": "MiniMax M2.5",
    "deepseek-v3.2": "DeepSeek V3.2",
}


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return json.load(handle)


def evaluate_manifest(
    manifest_path: str | Path, *, include_optional: bool = False
) -> list[dict[str, Any]]:
    evaluations = evaluate_prediction_artifacts(manifest_path, include_optional=include_optional)
    return manifest_rows_from_evaluations(evaluations)


def manifest_rows_from_evaluations(
    evaluations: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        _manifest_row(artifact, evaluation)
        for artifact, evaluation in evaluations.values()
    ]


def evaluate_prediction_artifacts(
    manifest_path: str | Path, *, include_optional: bool = False
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    manifest = load_manifest(manifest_path)
    evaluations: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for artifact in manifest["prediction_files"]:
        if not include_optional and not artifact.get("required_for_paper", True):
            continue
        prediction_path = Path(manifest_path).parent / artifact["path"]
        if not prediction_path.exists():
            if artifact.get("required_for_paper", True):
                raise FileNotFoundError(f"Missing required prediction artifact: {prediction_path}")
            continue
        evaluation = evaluate_prediction_file(prediction_path)
        evaluations[artifact["path"]] = (artifact, evaluation)
    return evaluations


def _manifest_row(artifact: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    aggregate = evaluation["aggregate"]
    task1_k = aggregate.get("task1_primary_k", PRIMARY_TASK1_K)
    task2_k = aggregate.get("task2_primary_k", PRIMARY_TASK2_K)
    task3_k = aggregate.get("task3_primary_k", PRIMARY_TASK3_K)
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
        f"task1_semantic_recall_at_{task1_k}_micro": aggregate[
            f"task1_semantic_recall_at_{task1_k}_micro"
        ],
        f"task2_anatomy_semantic_recall_at_{task2_k}_micro": aggregate[
            f"task2_anatomy_semantic_recall_at_{task2_k}_micro"
        ],
        f"task3_combined_exact_recall_at_{task3_k}_micro": aggregate[
            f"task3_combined_exact_recall_at_{task3_k}_micro"
        ],
        f"task1_semantic_recall_at_{task1_k}_macro": aggregate[
            f"task1_semantic_recall_at_{task1_k}_macro"
        ],
        f"task2_anatomy_semantic_recall_at_{task2_k}_macro": aggregate[
            f"task2_anatomy_semantic_recall_at_{task2_k}_macro"
        ],
        f"task3_combined_exact_recall_at_{task3_k}_macro": aggregate[
            f"task3_combined_exact_recall_at_{task3_k}_macro"
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
    return row


def evaluate_main_architecture_bootstrap(
    manifest_path: str | Path,
    *,
    n_bootstrap: int = 20_000,
    seed: int = 20_260_503,
    evaluations_by_path: dict[str, tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Compute gene-bootstrap intervals for the main harness-scaling figure."""
    return evaluate_artifact_bootstrap(
        manifest_path,
        MAIN_ARCHITECTURE_ARTIFACTS,
        n_bootstrap=n_bootstrap,
        seed=seed,
        evaluations_by_path=evaluations_by_path,
    )


def evaluate_model_comparison_bootstrap(
    manifest_path: str | Path,
    *,
    n_bootstrap: int = 20_000,
    seed: int = 20_260_503,
    evaluations_by_path: dict[str, tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Compute gene-bootstrap intervals for the fixed-harness model comparison.

    The point estimates match the paper's primary micro-averaged recall
    estimators. Each bootstrap sample resamples genes with replacement and
    recomputes the same pooled numerator/denominator estimator.
    """
    return evaluate_artifact_bootstrap(
        manifest_path,
        MODEL_COMPARISON_ARTIFACTS,
        n_bootstrap=n_bootstrap,
        seed=seed,
        evaluations_by_path=evaluations_by_path,
    )


def evaluate_cross_family_bootstrap(
    manifest_path: str | Path,
    *,
    n_bootstrap: int = 20_000,
    seed: int = 20_260_503,
    evaluations_by_path: dict[str, tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Compute gene-bootstrap intervals for the appendix cross-family table."""
    return evaluate_artifact_bootstrap(
        manifest_path,
        CROSS_FAMILY_ARTIFACTS,
        n_bootstrap=n_bootstrap,
        seed=seed,
        evaluations_by_path=evaluations_by_path,
    )


def evaluate_artifact_bootstrap(
    manifest_path: str | Path,
    artifact_paths: list[str],
    *,
    n_bootstrap: int = 20_000,
    seed: int = 20_260_503,
    evaluations_by_path: dict[str, tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Compute gene-bootstrap intervals for a fixed ordered artifact list."""
    if evaluations_by_path is None:
        evaluations_by_path = evaluate_prediction_artifacts(manifest_path)

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for artifact_path in artifact_paths:
        artifact, evaluation = evaluations_by_path[artifact_path]
        contributions = [_gene_contributions(row) for row in evaluation["genes"]]
        point = _micro_values(contributions)
        intervals = _bootstrap_intervals(contributions, n_bootstrap=n_bootstrap, rng=rng)
        display_name = _display_model_name(artifact)
        rows.append(_bootstrap_row(display_name, artifact, point, intervals, n_bootstrap))
    return rows


def _add_k_columns(row: dict[str, Any], prefix: str, values: dict[str, float]) -> None:
    for k_value in sorted(values, key=lambda value: int(value)):
        row[f"{prefix}_recall_at_{k_value}"] = values[k_value]


def _gene_contributions(row: dict[str, Any]) -> dict[str, float]:
    task1 = row["task1_function"]
    task2 = row["task2_expression"]
    task3 = row["task3_synonyms"]
    task1_k = int(task1.get("primary_k", PRIMARY_TASK1_K))
    task2_k = int(task2.get("primary_k", PRIMARY_TASK2_K))
    task3_k = int(task3.get("primary_k", PRIMARY_TASK3_K))
    return {
        "go_num": task1[f"semantic_recall_sum_at_{task1_k}"],
        "go_den": task1["gt_verified_count"],
        "expr_num": task2[f"anatomy_semantic_recall_sum_at_{task2_k}"],
        "expr_den": task2["gt_verified_count"],
        "syn_num": task3[f"combined_exact_hits_at_{task3_k}"],
        "syn_den": task3["gt_verified_combined_count"],
    }


def _micro_values(contributions: list[dict[str, float]]) -> dict[str, float]:
    values = {
        "go": _safe_divide_sum(contributions, "go_num", "go_den"),
        "expr": _safe_divide_sum(contributions, "expr_num", "expr_den"),
        "syn": _safe_divide_sum(contributions, "syn_num", "syn_den"),
    }
    values["avg"] = (values["go"] + values["expr"] + values["syn"]) / 3
    return values


def _safe_divide_sum(rows: list[dict[str, float]], num_key: str, den_key: str) -> float:
    denominator = sum(row[den_key] for row in rows)
    if denominator == 0:
        return 0.0
    return sum(row[num_key] for row in rows) / denominator


def _display_model_name(artifact: dict[str, Any]) -> str:
    model = artifact.get("model", "")
    return MODEL_DISPLAY_NAMES.get(model, model)


def _bootstrap_intervals(
    contributions: list[dict[str, float]],
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    n_genes = len(contributions)
    samples: dict[str, list[float]] = {"go": [], "expr": [], "syn": [], "avg": []}
    for _ in range(n_bootstrap):
        sample = [contributions[index] for index in rng.integers(0, n_genes, n_genes)]
        values = _micro_values(sample)
        for key, value in values.items():
            samples[key].append(value)
    return {
        key: (_percentile(values, 2.5), _percentile(values, 97.5))
        for key, values in samples.items()
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _bootstrap_row(
    display_name: str,
    artifact: dict[str, Any],
    point: dict[str, float],
    intervals: dict[str, tuple[float, float]],
    n_bootstrap: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": display_name,
        "artifact": artifact["path"],
        "baseline": artifact.get("baseline", ""),
        "provider": artifact.get("provider", ""),
        "paper_budget": artifact.get("paper_budget", ""),
        "n_bootstrap": n_bootstrap,
    }
    for metric in ("go", "expr", "syn", "avg"):
        low, high = intervals[metric]
        row[metric] = point[metric]
        row[f"{metric}_ci_low"] = low
        row[f"{metric}_ci_high"] = high
    return row


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
