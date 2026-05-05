"""Evaluate normalized FlyAOC predictions against verified benchmark labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from flyaoc.data import DatasetConfig, download_dataset_file, load_benchmark_records
from flyaoc.evaluation.constants import PRIMARY_TASK1_K, PRIMARY_TASK2_K, PRIMARY_TASK3_K
from flyaoc.evaluation.io import read_jsonl
from flyaoc.evaluation.ontology import AnatomySimilarity, GoSimilarity
from flyaoc.evaluation.recall import exact_recall_at_k, recall_series, semantic_recall_at_k


@dataclass(frozen=True)
class EvaluationConfig:
    dataset: DatasetConfig = DatasetConfig()
    task1_k: int = PRIMARY_TASK1_K
    task2_k: int = PRIMARY_TASK2_K
    task3_k: int = PRIMARY_TASK3_K
    go_obo_path: Path | None = None
    anatomy_obo_path: Path | None = None


def evaluate_prediction_file(
    prediction_file: str | Path,
    *,
    config: EvaluationConfig | None = None,
    benchmark_limit: int | None = None,
) -> dict[str, Any]:
    predictions = read_jsonl(prediction_file)
    return evaluate_prediction_rows(predictions, config=config, benchmark_limit=benchmark_limit)


def evaluate_prediction_rows(
    predictions: list[dict[str, Any]],
    *,
    config: EvaluationConfig | None = None,
    benchmark_limit: int | None = None,
) -> dict[str, Any]:
    cfg = config or EvaluationConfig()
    benchmark = {
        row["gene_id"]: row
        for row in load_benchmark_records(cfg.dataset, limit=benchmark_limit, streaming=False)
    }

    go_path = cfg.go_obo_path or download_dataset_file(cfg.dataset.go_obo_file, cfg.dataset)
    anatomy_path = cfg.anatomy_obo_path or download_dataset_file(cfg.dataset.anatomy_obo_file, cfg.dataset)
    go_sim = GoSimilarity(go_path)
    anatomy_sim = AnatomySimilarity(anatomy_path)

    gene_results = []
    for prediction in predictions:
        gene_id = prediction["gene_id"]
        if gene_id not in benchmark:
            raise KeyError(f"Prediction gene_id {gene_id} is not present in benchmark.jsonl")
        gene_results.append(
            evaluate_gene(
                prediction=prediction,
                benchmark_row=benchmark[gene_id],
                config=cfg,
                go_sim=go_sim,
                anatomy_sim=anatomy_sim,
            )
        )

    return {
        "n_genes": len(gene_results),
        "aggregate": aggregate_gene_results(gene_results),
        "genes": gene_results,
    }


def evaluate_gene(
    *,
    prediction: dict[str, Any],
    benchmark_row: dict[str, Any],
    config: EvaluationConfig,
    go_sim: GoSimilarity,
    anatomy_sim: AnatomySimilarity,
) -> dict[str, Any]:
    task1 = evaluate_task1(
        _prediction_list(prediction, "task1_function_predictions", "task1_function"),
        benchmark_row.get("task1_function", []),
        go_sim,
        config.task1_k,
    )
    task2 = evaluate_task2(
        _prediction_list(prediction, "task2_expression_predictions", "task2_expression"),
        benchmark_row.get("task2_expression", []),
        anatomy_sim,
        config.task2_k,
    )
    task3 = evaluate_task3(
        prediction.get("task3_synonym_predictions", prediction.get("task3_synonyms", {})),
        benchmark_row.get("task3_synonyms", {}),
        config.task3_k,
    )
    return {
        "gene_id": prediction["gene_id"],
        "gene_symbol": prediction.get("gene_symbol", benchmark_row.get("gene_symbol")),
        "task1_function": task1,
        "task2_expression": task2,
        "task3_synonyms": task3,
    }


def evaluate_task1(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    go_sim: GoSimilarity,
    k: int,
) -> dict[str, Any]:
    pred_ids = _unique_ordered(item.get("go_id") for item in predictions)
    gt_all = {item["go_id"] for item in ground_truth if item.get("go_id")}
    gt_verified = {
        item["go_id"]
        for item in ground_truth
        if item.get("go_id") and _is_verified_in_corpus(item)
    }
    series = recall_series(pred_ids, gt_verified, similarity_fn=go_sim.similarity)
    semantic_sum_at_k = _semantic_recall_sum_at_k(pred_ids, gt_verified, k, go_sim.similarity)
    return {
        "gt_total_count": len(gt_all),
        "gt_verified_count": len(gt_verified),
        "predicted_count": len(pred_ids),
        "primary_k": k,
        "exact_recall_at_k": series["exact_recall_at_k"],
        "semantic_recall_at_k": series["semantic_recall_at_k"],
        f"semantic_recall_sum_at_{k}": semantic_sum_at_k,
        f"semantic_recall_at_{k}": semantic_recall_at_k(pred_ids, gt_verified, k, go_sim.similarity),
    }


def evaluate_task2(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    anatomy_sim: AnatomySimilarity,
    k: int,
) -> dict[str, Any]:
    pred_anatomy = _unique_ordered(item.get("anatomy_id") for item in predictions)
    gt_all = {item["anatomy_id"] for item in ground_truth if item.get("anatomy_id")}
    gt_verified = {
        item["anatomy_id"]
        for item in ground_truth
        if item.get("anatomy_id") and _is_verified_in_corpus(item)
    }
    series = recall_series(pred_anatomy, gt_verified, similarity_fn=anatomy_sim.similarity)
    semantic_sum_at_k = _semantic_recall_sum_at_k(
        pred_anatomy, gt_verified, k, anatomy_sim.similarity
    )
    return {
        "gt_total_count": len(gt_all),
        "gt_verified_count": len(gt_verified),
        "predicted_count": len(pred_anatomy),
        "primary_k": k,
        "anatomy_exact_recall_at_k": series["exact_recall_at_k"],
        "anatomy_semantic_recall_at_k": series["semantic_recall_at_k"],
        f"anatomy_semantic_recall_sum_at_{k}": semantic_sum_at_k,
        f"anatomy_semantic_recall_at_{k}": semantic_recall_at_k(
            pred_anatomy, gt_verified, k, anatomy_sim.similarity
        ),
    }


def evaluate_task3(
    predictions: dict[str, Any],
    ground_truth: dict[str, Any],
    k: int,
) -> dict[str, Any]:
    pred_fullnames = _unique_ordered(predictions.get("fullname_synonyms", []))
    pred_symbols = _unique_ordered(predictions.get("symbol_synonyms", []))
    pred_combined = _unique_ordered([*pred_fullnames, *pred_symbols], normalize=True)

    gt_fullnames = _synonym_set(ground_truth.get("fullname_synonyms", []), verified_only=True)
    gt_symbols = _synonym_set(ground_truth.get("symbol_synonyms", []), verified_only=True)
    gt_combined = gt_fullnames | gt_symbols

    combined_hits_at_k = len(set(_normalize_synonym(x) for x in pred_combined[:k]) & gt_combined)
    return {
        "gt_verified_fullname_count": len(gt_fullnames),
        "gt_verified_symbol_count": len(gt_symbols),
        "gt_verified_combined_count": len(gt_combined),
        "predicted_fullname_count": len(pred_fullnames),
        "predicted_symbol_count": len(pred_symbols),
        "primary_k": k,
        "combined_exact_recall_at_k": {
            str(k_value): exact_recall_at_k(
                [_normalize_synonym(x) for x in pred_combined], gt_combined, k_value
            )
            for k_value in [1, 3, 5, 10, 20, 50]
        },
        f"combined_exact_hits_at_{k}": combined_hits_at_k,
        f"combined_exact_recall_at_{k}": exact_recall_at_k(
            [_normalize_synonym(x) for x in pred_combined], gt_combined, k
        ),
    }


def aggregate_gene_results(gene_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not gene_results:
        return {}
    task1_gt = sum(row["task1_function"]["gt_verified_count"] for row in gene_results)
    task2_gt = sum(row["task2_expression"]["gt_verified_count"] for row in gene_results)
    task3_gt = sum(row["task3_synonyms"]["gt_verified_combined_count"] for row in gene_results)
    task1_semantic_at_k = _aggregate_recall_at_k(
        gene_results,
        task_key="task1_function",
        series_key="semantic_recall_at_k",
        gt_count_key="gt_verified_count",
    )
    task2_semantic_at_k = _aggregate_recall_at_k(
        gene_results,
        task_key="task2_expression",
        series_key="anatomy_semantic_recall_at_k",
        gt_count_key="gt_verified_count",
    )
    task3_exact_at_k = _aggregate_recall_at_k(
        gene_results,
        task_key="task3_synonyms",
        series_key="combined_exact_recall_at_k",
        gt_count_key="gt_verified_combined_count",
    )
    task1_k = _primary_k(gene_results, "task1_function", PRIMARY_TASK1_K)
    task2_k = _primary_k(gene_results, "task2_expression", PRIMARY_TASK2_K)
    task3_k = _primary_k(gene_results, "task3_synonyms", PRIMARY_TASK3_K)
    return {
        "task1_verified_fact_count": task1_gt,
        "task2_verified_fact_count": task2_gt,
        "task3_verified_fact_count": task3_gt,
        "task1_primary_k": task1_k,
        "task2_primary_k": task2_k,
        "task3_primary_k": task3_k,
        "task1_semantic_recall_at_k_micro": task1_semantic_at_k["micro"],
        "task1_semantic_recall_at_k_macro": task1_semantic_at_k["macro"],
        "task2_anatomy_semantic_recall_at_k_micro": task2_semantic_at_k["micro"],
        "task2_anatomy_semantic_recall_at_k_macro": task2_semantic_at_k["macro"],
        "task3_combined_exact_recall_at_k_micro": task3_exact_at_k["micro"],
        "task3_combined_exact_recall_at_k_macro": task3_exact_at_k["macro"],
        f"task1_semantic_recall_at_{task1_k}_micro": _safe_divide(
            sum(
                row["task1_function"][f"semantic_recall_sum_at_{task1_k}"]
                for row in gene_results
            ),
            task1_gt,
        ),
        f"task2_anatomy_semantic_recall_at_{task2_k}_micro": _safe_divide(
            sum(
                row["task2_expression"][f"anatomy_semantic_recall_sum_at_{task2_k}"]
                for row in gene_results
            ),
            task2_gt,
        ),
        f"task3_combined_exact_recall_at_{task3_k}_micro": _safe_divide(
            sum(
                row["task3_synonyms"][f"combined_exact_hits_at_{task3_k}"]
                for row in gene_results
            ),
            task3_gt,
        ),
        f"task1_semantic_recall_at_{task1_k}_macro": mean(
            row["task1_function"][f"semantic_recall_at_{task1_k}"] for row in gene_results
        ),
        f"task2_anatomy_semantic_recall_at_{task2_k}_macro": mean(
            row["task2_expression"][f"anatomy_semantic_recall_at_{task2_k}"]
            for row in gene_results
        ),
        f"task3_combined_exact_recall_at_{task3_k}_macro": mean(
            row["task3_synonyms"][f"combined_exact_recall_at_{task3_k}"]
            for row in gene_results
        ),
    }


def _primary_k(gene_results: list[dict[str, Any]], task_key: str, default: int) -> int:
    return int(gene_results[0][task_key].get("primary_k", default))


def _aggregate_recall_at_k(
    gene_results: list[dict[str, Any]],
    *,
    task_key: str,
    series_key: str,
    gt_count_key: str,
) -> dict[str, dict[str, float]]:
    first_series = gene_results[0][task_key][series_key]
    k_values = sorted(first_series, key=lambda value: int(value))
    total_gt = sum(row[task_key][gt_count_key] for row in gene_results)
    micro: dict[str, float] = {}
    macro: dict[str, float] = {}
    for k_value in k_values:
        numerator = sum(
            row[task_key][series_key][k_value] * row[task_key][gt_count_key]
            for row in gene_results
        )
        micro[k_value] = _safe_divide(numerator, total_gt)
        macro[k_value] = mean(row[task_key][series_key][k_value] for row in gene_results)
    return {"micro": micro, "macro": macro}


def _semantic_recall_sum_at_k(
    predictions: list[str],
    ground_truth: set[str],
    k: int,
    similarity_fn: Any,
) -> float:
    top_k = predictions[:k]
    if not top_k or not ground_truth:
        return 0.0
    total = 0.0
    for gt_item in ground_truth:
        total += max((similarity_fn(pred, gt_item) for pred in top_k), default=0.0)
    return total


def _safe_divide(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _prediction_list(
    row: dict[str, Any],
    normalized_key: str,
    legacy_key: str,
) -> list[dict[str, Any]]:
    value = row.get(normalized_key, row.get(legacy_key, []))
    if not isinstance(value, list):
        raise TypeError(f"{normalized_key} must be a list")
    return value


def _is_verified_in_corpus(item: dict[str, Any]) -> bool:
    return bool(item.get("in_corpus_verified", item.get("in_corpus", False)))


def _unique_ordered(values: list[Any] | Any, *, normalize: bool = False) -> list[str]:
    if not isinstance(values, list):
        values = list(values)
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value:
            continue
        text = str(value)
        key = _normalize_synonym(text) if normalize else text
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _synonym_set(values: list[Any], *, verified_only: bool) -> set[str]:
    result: set[str] = set()
    for item in values:
        if isinstance(item, str):
            if not verified_only:
                result.add(_normalize_synonym(item))
            continue
        if not isinstance(item, dict):
            continue
        if verified_only and not _is_verified_in_corpus(item):
            continue
        synonym = item.get("synonym")
        if synonym:
            result.add(_normalize_synonym(synonym))
    return result


def _normalize_synonym(value: str) -> str:
    return value.strip().lower()
