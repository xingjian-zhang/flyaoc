#!/usr/bin/env python3
"""CLI entry point for deep analysis of scaling experiments.

This module provides the main function and argument parsing for running
in-depth analysis of scaling experiment results for academic publication.

Usage:
    uv run python -m scripts.deep_analysis --base-dir outputs/scaling/d37b178 --plot -v
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.data_loader import (
    load_all_scaling_eval_results,
    load_go_ontology,
    load_ground_truth,
    load_scaling_run_summary,
)

from .case_studies import generate_detailed_case_study
from .go_analysis import (
    analyze_go_aspects,
    analyze_per_paper_yield,
    categorize_false_positives_detailed,
    extract_false_positives_detailed,
)
from .reporting import PAPER_CONFIGS, generate_academic_report
from .statistics import deep_task_comparison, perform_statistical_tests
from .tool_analysis import (
    analyze_paper_selection_effectiveness,
    extract_detailed_tool_usage,
)
from .trajectories import analyze_scaling_categories, compute_gene_trajectories

# Plotting is optional - import will be attempted when --plot is used
# TODO: Extract generate_publication_plots to .plotting module
_generate_publication_plots = None


def main() -> None:
    """Main entry point for deep analysis CLI."""
    parser = argparse.ArgumentParser(
        description="In-depth analysis of scaling experiment for academic publication"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Base directory containing papers_N subdirectories",
    )
    parser.add_argument(
        "--configs",
        type=int,
        nargs="+",
        default=PAPER_CONFIGS,
        help="Paper configurations to analyze",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate publication-quality plots",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("DEEP ANALYSIS: Literature-Based Gene Annotation Scaling Experiment")
    print("=" * 70)
    print(f"\nAnalyzing: {args.base_dir}")

    # Load data
    print("\n[1/12] Loading evaluation results...")
    all_evals = load_all_scaling_eval_results(args.base_dir, args.configs)

    if not all_evals:
        print("Error: No evaluation results found", file=sys.stderr)
        sys.exit(1)

    configs_found = sorted(all_evals.keys())
    print(f"  Loaded {len(all_evals)} configs: {configs_found}")

    # Load run summaries
    run_summaries: dict[int, dict[str, Any]] = {}
    for config in args.configs:
        summary = load_scaling_run_summary(args.base_dir, config)
        if summary:
            run_summaries[config] = summary

    # Load ground truth (as dict for efficient lookups)
    print("[2/12] Loading ground truth...")
    gt_data = load_ground_truth(as_dict=True)
    # Type narrow: as_dict=True always returns a dict
    gt: dict[str, dict[str, Any]] = gt_data if isinstance(gt_data, dict) else {}
    print(f"  Loaded {len(gt)} genes")

    # Load GO ontology for term names
    print("[3/12] Loading GO ontology...")
    go_terms = load_go_ontology()
    print(f"  Loaded {len(go_terms)} GO terms")

    # Compute trajectories
    print("[4/12] Computing gene trajectories...")
    trajectories = compute_gene_trajectories(all_evals, gt)
    print(f"  Analyzed {len(trajectories)} genes")

    # Analyze scaling categories
    print("[5/12] Analyzing scaling behavior...")
    scaling_analysis = analyze_scaling_categories(trajectories)

    # Extract and analyze false positives
    print("[6/12] Deep false positive analysis...")
    fps = extract_false_positives_detailed(all_evals, args.base_dir, go_terms)
    fp_analysis = categorize_false_positives_detailed(fps, gt)
    print(f"  Analyzed {len(fps)} false positives")

    # Statistical hypothesis tests
    print("[7/12] Running statistical tests...")
    statistical_tests = perform_statistical_tests(trajectories, all_evals, gt)

    # Task comparison
    print("[8/12] Comparing Task 1 vs Task 2...")
    task_comparison = deep_task_comparison(all_evals, gt, trajectories)

    # Tool usage analysis
    print("[9/12] Analyzing tool usage patterns...")
    tool_usage = extract_detailed_tool_usage(args.base_dir, args.configs)

    # Per-paper yield
    print("[10/12] Computing per-paper yield...")
    paper_yield = analyze_per_paper_yield(args.base_dir, args.configs, all_evals)

    # Paper selection effectiveness
    print("[11/12] Analyzing paper selection strategy...")
    paper_selection = analyze_paper_selection_effectiveness(args.base_dir, args.configs, all_evals)

    # GO aspect analysis
    go_aspects = analyze_go_aspects(args.base_dir, args.configs, gt, go_terms)

    # Generate case studies
    print("[12/12] Generating detailed case studies...")

    # Select interesting genes for case studies
    case_gene_ids: list[str] = []

    # Best overall performer
    best_f1 = max(trajectories, key=lambda t: t.task1_f1.get(max(configs_found), 0))
    case_gene_ids.append(best_f1.gene_id)

    # Best scaler (most improvement)
    best_scaler = max(trajectories, key=lambda t: t.scaling_gain("task1"))
    if best_scaler.gene_id not in case_gene_ids:
        case_gene_ids.append(best_scaler.gene_id)

    # Worst scaler (most degradation)
    worst_scaler = min(trajectories, key=lambda t: t.scaling_gain("task1"))
    if worst_scaler.gene_id not in case_gene_ids:
        case_gene_ids.append(worst_scaler.gene_id)

    # Gene with most false positives
    fp_by_gene: dict[str, int] = defaultdict(int)
    for fp in fps:
        fp_by_gene[fp.gene_id] += 1
    if fp_by_gene:
        most_fp_gene = max(fp_by_gene.keys(), key=lambda x: fp_by_gene[x])
        if most_fp_gene not in case_gene_ids:
            case_gene_ids.append(most_fp_gene)

    # Reference gene (abd-A) - well-studied
    if "FBgn0000014" not in case_gene_ids:
        case_gene_ids.append("FBgn0000014")

    case_studies = [
        generate_detailed_case_study(gid, trajectories, all_evals, gt, args.base_dir, go_terms)
        for gid in case_gene_ids
    ]

    # Generate report
    print("\nGenerating academic report...")
    report = generate_academic_report(
        trajectories=trajectories,
        scaling_analysis=scaling_analysis,
        fp_analysis=fp_analysis,
        statistical_tests=statistical_tests,
        task_comparison=task_comparison,
        tool_usage=tool_usage,
        paper_yield=paper_yield,
        paper_selection=paper_selection,
        go_aspects=go_aspects,
        case_studies=case_studies,
    )

    # Save outputs
    output_dir = args.base_dir
    analysis_dir = output_dir / "analysis_data"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Save report
    report_path = output_dir / "deep_analysis_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # Save detailed analysis data
    with open(analysis_dir / "gene_trajectories.json", "w") as f:
        json.dump(
            [
                {
                    "gene_id": t.gene_id,
                    "gene_symbol": t.gene_symbol,
                    "task1_f1": t.task1_f1,
                    "task1_precision": t.task1_precision,
                    "task1_recall": t.task1_recall,
                    "task2_f1": t.task2_f1,
                    "task1_scaling_gain": t.scaling_gain("task1"),
                    "task2_scaling_gain": t.scaling_gain("task2"),
                    "task1_category": t.category("task1"),
                    "task2_category": t.category("task2"),
                    "gt_task1_in_corpus": t.gt_task1_in_corpus,
                    "gt_task2_in_corpus": t.gt_task2_in_corpus,
                }
                for t in trajectories
            ],
            f,
            indent=2,
        )

    with open(analysis_dir / "fp_analysis.json", "w") as f:
        # Remove non-serializable items
        fp_export = {k: v for k, v in fp_analysis.items() if k != "category_examples"}
        fp_export["category_examples"] = {
            cat: [{k: v for k, v in ex.items() if v is not None} for ex in examples]
            for cat, examples in fp_analysis.get("category_examples", {}).items()
        }
        json.dump(fp_export, f, indent=2)

    with open(analysis_dir / "statistical_tests.json", "w") as f:
        json.dump(statistical_tests, f, indent=2)

    with open(analysis_dir / "task_comparison.json", "w") as f:
        json.dump(task_comparison, f, indent=2)

    with open(analysis_dir / "paper_yield.json", "w") as f:
        json.dump(paper_yield, f, indent=2)

    with open(analysis_dir / "case_studies.json", "w") as f:
        json.dump(case_studies, f, indent=2)

    print(f"Analysis data saved to {analysis_dir}")

    # Generate plots
    if args.plot:
        # Lazy import plotting to avoid matplotlib dependency when not needed
        try:
            from .plotting import generate_publication_plots

            print("\nGenerating publication-quality plots...")
            generate_publication_plots(
                trajectories,
                all_evals,
                fp_analysis,
                paper_yield,
                task_comparison,
                output_dir,
            )
        except ImportError as e:
            print(
                f"\nWarning: Plotting module not available ({e}), skipping plots",
                file=sys.stderr,
            )

    # Print summary if verbose
    if args.verbose:
        print("\n" + "=" * 70)
        print(report[:5000] + "\n...[truncated]...")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
