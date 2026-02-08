"""I/O functions for scaling analysis results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config_metrics import ConfigMetrics
from .reporting import find_knee_point


def save_results(metrics: list[ConfigMetrics], output_dir: Path) -> None:
    """Save aggregated results to JSON.

    Args:
        metrics: List of ConfigMetrics
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "configs": [
            {
                "max_papers": m.max_papers,
                "num_genes_attempted": m.num_genes_attempted,
                "num_genes_evaluated": m.num_genes_evaluated,
                "failure_rate": m.failure_rate,
                "avg_cost": m.avg_cost,
                "avg_cost_per_success": m.avg_cost_per_success,
                "total_cost": m.total_cost,
                "avg_papers_read": m.avg_papers_read,
                # Task 1 GO (macro-averaged exact)
                "go_exact_precision": m.go_exact_precision,
                "go_exact_recall": m.go_exact_recall,
                "go_full_recall": m.go_full_recall,
                "go_in_corpus_exact_recall": m.go_in_corpus_exact_recall,
                "go_exact_f1": m.go_exact_f1,
                "go_total_tp": m.go_total_tp,
                "go_total_fp": m.go_total_fp,
                "go_total_fn": m.go_total_fn,
                "go_fp_rate": m.go_fp_rate,
                # Task 1 GO (micro-averaged)
                "go_micro_precision": m.go_micro_precision,
                "go_micro_recall": m.go_micro_recall,
                "go_micro_f1": m.go_micro_f1,
                # Task 1 GO (semantic - partial credit)
                "go_semantic_precision": m.go_semantic_precision,
                "go_semantic_recall": m.go_semantic_recall,
                "go_semantic_f1": m.go_semantic_f1,
                "go_total_semantic_tp": m.go_total_semantic_tp,
                "go_full_semantic_recall": m.go_full_semantic_recall,
                # Task 2 Expression (macro-averaged exact)
                "expr_anatomy_precision": m.expr_anatomy_precision,
                "expr_anatomy_recall": m.expr_anatomy_recall,
                "expr_in_corpus_exact_recall": m.expr_in_corpus_exact_recall,
                "expr_full_recall": m.expr_full_recall,
                "expr_anatomy_f1": m.expr_anatomy_f1,
                # Task 2 Expression (micro-averaged)
                "expr_anatomy_micro_precision": m.expr_anatomy_micro_precision,
                "expr_anatomy_micro_recall": m.expr_anatomy_micro_recall,
                "expr_anatomy_micro_f1": m.expr_anatomy_micro_f1,
                # Task 2 Expression (semantic)
                "expr_semantic_precision": m.expr_semantic_precision,
                "expr_semantic_recall": m.expr_semantic_recall,
                "expr_semantic_f1": m.expr_semantic_f1,
                "expr_full_semantic_recall": m.expr_full_semantic_recall,
                # Task 2 Expression tuple
                "expr_tuple_precision": m.expr_tuple_precision,
                "expr_tuple_recall": m.expr_tuple_recall,
                "expr_tuple_f1": m.expr_tuple_f1,
                # Task 3 Synonyms (macro-averaged)
                "syn_fullname_precision": m.syn_fullname_precision,
                "syn_fullname_recall": m.syn_fullname_recall,
                "syn_fullname_in_corpus_recall": m.syn_fullname_in_corpus_recall,
                "syn_fullname_full_recall": m.syn_fullname_full_recall,
                "syn_fullname_f1": m.syn_fullname_f1,
                # Task 3 Synonyms (micro-averaged fullname)
                "syn_fullname_micro_precision": m.syn_fullname_micro_precision,
                "syn_fullname_micro_recall": m.syn_fullname_micro_recall,
                "syn_fullname_micro_f1": m.syn_fullname_micro_f1,
                "syn_symbol_precision": m.syn_symbol_precision,
                "syn_symbol_recall": m.syn_symbol_recall,
                "syn_symbol_in_corpus_recall": m.syn_symbol_in_corpus_recall,
                "syn_symbol_full_recall": m.syn_symbol_full_recall,
                "syn_symbol_f1": m.syn_symbol_f1,
                # Task 3 Synonyms (micro-averaged symbol)
                "syn_symbol_micro_precision": m.syn_symbol_micro_precision,
                "syn_symbol_micro_recall": m.syn_symbol_micro_recall,
                "syn_symbol_micro_f1": m.syn_symbol_micro_f1,
                # Task 3 Synonyms (combined)
                "syn_combined_precision": m.syn_combined_precision,
                "syn_combined_recall": m.syn_combined_recall,
                "syn_combined_in_corpus_recall": m.syn_combined_in_corpus_recall,
                "syn_combined_full_recall": m.syn_combined_full_recall,
                "syn_combined_f1": m.syn_combined_f1,
                "syn_combined_micro_precision": m.syn_combined_micro_precision,
                "syn_combined_micro_recall": m.syn_combined_micro_recall,
                "syn_combined_micro_f1": m.syn_combined_micro_f1,
                # Reference coverage (macro-averaged)
                "ref_precision": m.ref_precision,
                "ref_recall": m.ref_recall,
                "ref_f1": m.ref_f1,
                # Reference coverage (micro-averaged)
                "ref_micro_precision": m.ref_micro_precision,
                "ref_micro_recall": m.ref_micro_recall,
                "ref_micro_f1": m.ref_micro_f1,
                # Cost efficiency
                "cost_per_f1": m.avg_cost_per_success / m.go_exact_f1 if m.go_exact_f1 > 0 else None,
            }
            for m in metrics
        ]
    }

    # Find knee point
    knee = find_knee_point(metrics)
    if knee:
        results["recommended_config"] = {
            "max_papers": knee.max_papers,
            "reason": "Best cost/performance tradeoff based on F1 gain per dollar",
        }

    output_path = output_dir / "scaling_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
