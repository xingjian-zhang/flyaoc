"""Gene trajectory analysis for scaling experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy import stats

from .dataclasses import GeneTrajectory


def compute_gene_trajectories(
    all_evals: dict[int, dict], gt: dict[str, dict]
) -> list[GeneTrajectory]:
    """Compute detailed trajectories for each gene across paper configs."""
    gene_data: dict[str, GeneTrajectory] = {}

    for config, eval_data in all_evals.items():
        for gene in eval_data.get("genes", []):
            gene_id = gene["gene_id"]
            gene_symbol = gene.get("gene_symbol", "")

            if gene_id not in gene_data:
                gene_data[gene_id] = GeneTrajectory(gene_id=gene_id, gene_symbol=gene_symbol)

            traj = gene_data[gene_id]

            # Task 1 - GO soft metrics
            t1 = gene.get("task1_go", {}).get("soft", {})
            traj.task1_f1[config] = t1.get("f1", 0.0)
            traj.task1_precision[config] = t1.get("precision", 0.0)
            traj.task1_recall[config] = t1.get("recall", 0.0)
            traj.task1_tp[config] = t1.get("tp", 0)
            traj.task1_fp[config] = t1.get("fp", 0)

            # Task 2 - Expression anatomy soft
            t2 = gene.get("task2_expression", {})
            anat = t2.get("anatomy_soft", t2.get("anatomy", {}))
            traj.task2_f1[config] = anat.get("f1", 0.0)
            traj.task2_precision[config] = anat.get("precision", 0.0)
            traj.task2_recall[config] = anat.get("recall", 0.0)

            # Task 3 - Synonym symbol
            t3 = gene.get("task3_synonyms", {}).get("symbol", {})
            traj.task3_f1[config] = t3.get("f1", 0.0)

    # Add ground truth counts
    for gene_id, traj in gene_data.items():
        if gene_id in gt:
            gt_entry = gt[gene_id]
            traj.gt_task1_total = len(gt_entry.get("task1_function", []))
            traj.gt_task1_in_corpus = sum(
                1 for ann in gt_entry.get("task1_function", []) if ann.get("in_corpus", False)
            )
            traj.gt_task2_total = len(gt_entry.get("task2_expression", []))
            traj.gt_task2_in_corpus = sum(
                1 for ann in gt_entry.get("task2_expression", []) if ann.get("in_corpus", False)
            )
            syn = gt_entry.get("task3_synonyms", {})
            traj.gt_task3_symbols = len(syn.get("symbol_synonyms", []))

    return list(gene_data.values())


def analyze_scaling_categories(trajectories: list[GeneTrajectory]) -> dict[str, Any]:
    """Categorize genes by scaling behavior with statistical analysis."""
    results = {}

    for task in ["task1", "task2", "task3"]:
        gains = [t.scaling_gain(task) for t in trajectories]

        categories = defaultdict(list)
        for traj in trajectories:
            cat = traj.category(task)
            categories[cat].append(
                {
                    "gene_id": traj.gene_id,
                    "gene_symbol": traj.gene_symbol,
                    "gain": traj.scaling_gain(task),
                    "best_config": traj.best_config(task),
                    "gt_in_corpus": getattr(traj, f"gt_{task}_in_corpus", 0)
                    if task != "task3"
                    else traj.gt_task3_symbols,
                }
            )

        # Statistical tests
        improves_gains = [g["gain"] for g in categories["improves"]]
        degrades_gains = [g["gain"] for g in categories["degrades"]]

        results[task] = {
            "improves_count": len(categories["improves"]),
            "stable_count": len(categories["stable"]),
            "degrades_count": len(categories["degrades"]),
            "mean_gain": float(np.mean(gains)),
            "std_gain": float(np.std(gains)),
            "median_gain": float(np.median(gains)),
            "improves": sorted(categories["improves"], key=lambda x: -x["gain"])[:15],
            "degrades": sorted(categories["degrades"], key=lambda x: x["gain"])[:15],
            # T-test: is mean gain significantly different from 0?
            "ttest_pvalue": float(stats.ttest_1samp(gains, 0).pvalue) if len(gains) > 1 else 1.0,
            # Effect size (Cohen's d)
            "cohens_d": float(np.mean(gains) / np.std(gains)) if np.std(gains) > 0 else 0.0,
        }

        # Compare improves vs degrades groups
        if len(improves_gains) > 1 and len(degrades_gains) > 1:
            results[task]["group_comparison"] = {
                "improves_mean_gt": float(
                    np.mean([g["gt_in_corpus"] for g in categories["improves"]])
                ),
                "degrades_mean_gt": float(
                    np.mean([g["gt_in_corpus"] for g in categories["degrades"]])
                ),
            }

    return results
