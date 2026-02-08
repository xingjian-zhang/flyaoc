"""Statistical hypothesis testing for scaling experiment analysis."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy import stats

from .dataclasses import GeneTrajectory


def perform_statistical_tests(
    trajectories: list[GeneTrajectory],
    all_evals: dict[int, dict[str, Any]],
    gt: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Perform comprehensive statistical hypothesis tests."""
    _ = gt  # Unused but kept for API compatibility

    results: dict[str, Any] = {}

    # H1: Task 1 scales better than Task 2
    task1_gains = [t.scaling_gain("task1") for t in trajectories]
    task2_gains = [t.scaling_gain("task2") for t in trajectories]

    # Paired t-test (same genes across tasks)
    t_stat, p_value = stats.ttest_rel(task1_gains, task2_gains)
    results["h1_task1_vs_task2_scaling"] = {
        "hypothesis": "Task 1 (GO) scales better than Task 2 (Expression)",
        "test": "Paired t-test on scaling gains",
        "task1_mean_gain": float(np.mean(task1_gains)),
        "task2_mean_gain": float(np.mean(task2_gains)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "effect_size_cohens_d": float(
            (np.mean(task1_gains) - np.mean(task2_gains))
            / np.std(np.array(task1_gains) - np.array(task2_gains))
        ),
    }

    # H2: Genes with more in-corpus GT annotations have higher F1
    max_config = max(all_evals.keys())
    gt_counts: list[int] = []
    f1_scores: list[float] = []

    for gene in all_evals[max_config].get("genes", []):
        gene_id = gene["gene_id"]
        traj = next((t for t in trajectories if t.gene_id == gene_id), None)
        if traj:
            gt_counts.append(traj.gt_task1_in_corpus)
            f1_scores.append(gene.get("task1_go", {}).get("soft", {}).get("f1", 0))

    if len(gt_counts) > 5:
        spearman_result = stats.spearmanr(gt_counts, f1_scores)
        corr = float(cast(float, spearman_result[0]))
        corr_p = float(cast(float, spearman_result[1]))
        results["h2_gt_count_vs_f1"] = {
            "hypothesis": "More in-corpus GT annotations correlate with higher F1",
            "test": "Spearman correlation",
            "correlation": corr,
            "p_value": corr_p,
            "significant": bool(corr_p < 0.05),
            "interpretation": (
                "Positive correlation suggests richer ground truth enables better recall. "
                if corr > 0
                else "Negative correlation suggests harder genes have more GT but lower coverage."
            ),
        }

    # H3: False positive rate increases with paper count
    fp_rates: list[float] = []
    configs: list[int] = []
    for config, eval_data in sorted(all_evals.items()):
        genes = eval_data.get("genes", [])
        total_tp = sum(g.get("task1_go", {}).get("soft", {}).get("tp", 0) for g in genes)
        total_fp = sum(g.get("task1_go", {}).get("soft", {}).get("fp", 0) for g in genes)
        if total_tp + total_fp > 0:
            fp_rates.append(total_fp / (total_tp + total_fp))
            configs.append(config)

    if len(fp_rates) > 2:
        spearman_result = stats.spearmanr(configs, fp_rates)
        corr = float(cast(float, spearman_result[0]))
        corr_p = float(cast(float, spearman_result[1]))
        results["h3_fp_rate_vs_papers"] = {
            "hypothesis": "False positive rate increases with more papers (information overload)",
            "test": "Spearman correlation between paper count and FP rate",
            "fp_rates_by_config": dict(zip(configs, fp_rates, strict=False)),
            "correlation": corr,
            "p_value": corr_p,
            "significant": bool(corr_p < 0.05),
        }

    # H4: Precision decreases while recall increases (classic tradeoff)
    precisions: list[float] = []
    recalls: list[float] = []
    for _config, eval_data in sorted(all_evals.items()):
        genes = eval_data.get("genes", [])
        avg_p = float(
            np.mean([g.get("task1_go", {}).get("soft", {}).get("precision", 0) for g in genes])
        )
        avg_r = float(
            np.mean([g.get("task1_go", {}).get("soft", {}).get("recall", 0) for g in genes])
        )
        precisions.append(avg_p)
        recalls.append(avg_r)

    if len(precisions) > 2:
        # Test if precision decreases with config
        p_result = stats.spearmanr(list(range(len(precisions))), precisions)
        r_result = stats.spearmanr(list(range(len(recalls))), recalls)
        p_corr = float(cast(float, p_result[0]))
        p_p = float(cast(float, p_result[1]))
        r_corr = float(cast(float, r_result[0]))
        r_p = float(cast(float, r_result[1]))

        results["h4_precision_recall_tradeoff"] = {
            "hypothesis": "More papers increase recall at cost of precision",
            "precision_trend_correlation": p_corr,
            "precision_trend_p": p_p,
            "recall_trend_correlation": r_corr,
            "recall_trend_p": r_p,
            "precisions": precisions,
            "recalls": recalls,
            "interpretation": (
                f"Precision trend: {'decreasing' if p_corr < 0 else 'increasing'} (r={p_corr:.3f}). "
                f"Recall trend: {'increasing' if r_corr > 0 else 'decreasing'} (r={r_corr:.3f})."
            ),
        }

    return results


def deep_task_comparison(
    all_evals: dict[int, dict[str, Any]],
    gt: dict[str, dict[str, Any]],
    trajectories: list[GeneTrajectory],
) -> dict[str, Any]:
    """Deep comparison of why Task 1 and Task 2 scale differently."""
    _ = gt  # Unused but kept for API compatibility

    # Collect detailed statistics
    task1_gt_counts: list[int] = []
    task2_gt_counts: list[int] = []
    task1_f1_at_max: list[float] = []
    task2_f1_at_max: list[float] = []

    max_config = max(all_evals.keys())

    for traj in trajectories:
        task1_gt_counts.append(traj.gt_task1_in_corpus)
        task2_gt_counts.append(traj.gt_task2_in_corpus)
        task1_f1_at_max.append(traj.task1_f1.get(max_config, 0))
        task2_f1_at_max.append(traj.task2_f1.get(max_config, 0))

    # Calculate theoretical max recall
    task1_genes_with_gt = sum(1 for c in task1_gt_counts if c > 0)
    task2_genes_with_gt = sum(1 for c in task2_gt_counts if c > 0)

    # Analyze scaling patterns
    task1_improves = sum(1 for t in trajectories if t.category("task1") == "improves")
    task1_degrades = sum(1 for t in trajectories if t.category("task1") == "degrades")
    task2_improves = sum(1 for t in trajectories if t.category("task2") == "improves")
    task2_degrades = sum(1 for t in trajectories if t.category("task2") == "degrades")

    task2_mean = float(np.mean(task2_gt_counts))
    task1_mean = float(np.mean(task1_gt_counts))
    ratio = task1_mean / task2_mean if task2_mean > 0 else float("inf")

    return {
        "ground_truth_density": {
            "task1": {
                "mean": task1_mean,
                "median": float(np.median(task1_gt_counts)),
                "std": float(np.std(task1_gt_counts)),
                "max": int(max(task1_gt_counts)),
                "genes_with_gt": task1_genes_with_gt,
                "zero_gt_genes": sum(1 for c in task1_gt_counts if c == 0),
            },
            "task2": {
                "mean": task2_mean,
                "median": float(np.median(task2_gt_counts)),
                "std": float(np.std(task2_gt_counts)),
                "max": int(max(task2_gt_counts)),
                "genes_with_gt": task2_genes_with_gt,
                "zero_gt_genes": sum(1 for c in task2_gt_counts if c == 0),
            },
        },
        "performance_at_max_papers": {
            "task1": {
                "mean_f1": float(np.mean(task1_f1_at_max)),
                "std_f1": float(np.std(task1_f1_at_max)),
                "median_f1": float(np.median(task1_f1_at_max)),
            },
            "task2": {
                "mean_f1": float(np.mean(task2_f1_at_max)),
                "std_f1": float(np.std(task2_f1_at_max)),
                "median_f1": float(np.median(task2_f1_at_max)),
            },
        },
        "scaling_behavior": {
            "task1": {
                "improves": task1_improves,
                "degrades": task1_degrades,
                "stable": len(trajectories) - task1_improves - task1_degrades,
            },
            "task2": {
                "improves": task2_improves,
                "degrades": task2_degrades,
                "stable": len(trajectories) - task2_improves - task2_degrades,
            },
        },
        "key_differences": [
            f"Task 1 has {ratio:.1f}x more in-corpus GT annotations per gene on average.",
            f"Task 1: {task1_improves} genes improve with scaling vs {task1_degrades} degrade.",
            f"Task 2: {task2_improves} genes improve with scaling vs {task2_degrades} degrade.",
            f"Task 2 has {sum(1 for c in task2_gt_counts if c == 0)} genes with zero in-corpus GT "
            f"(vs {sum(1 for c in task1_gt_counts if c == 0)} for Task 1).",
        ],
        "hypothesis": (
            "Task 1 (GO annotation) scales better because: "
            "(1) Denser in-corpus ground truth provides more opportunities for correct extraction. "
            "(2) GO terms are more explicitly stated in papers (functional descriptions, mutant phenotypes). "
            "(3) Task 2 (Expression) requires spatial/temporal localization data which is sparser in text "
            "and often requires image interpretation that the agent cannot perform."
        ),
    }
