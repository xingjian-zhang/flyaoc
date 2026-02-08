"""CLI entry point for scaling analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .collection import collect_all_metrics
from .io import save_results
from .plotting import (
    generate_comparison_plots,
    generate_pareto_plots,
    generate_plots,
    generate_recall_at_k_pareto_plots,
    generate_recall_at_k_plots,
)
from .reporting import find_knee_point, print_detailed_metrics, print_summary_table

# Paper configurations to analyze
PAPER_CONFIGS = [1, 2, 4, 8, 16]

# Available methods for comparison
AVAILABLE_METHODS = ["single", "multi", "pipeline", "memorization"]


def detect_paper_configs(method_dir: Path) -> list[int]:
    """Detect available paper configurations from directory structure.

    Args:
        method_dir: Directory containing papers_N subdirectories

    Returns:
        Sorted list of paper configurations found
    """
    configs = []
    for d in method_dir.iterdir():
        if d.is_dir() and d.name.startswith("papers_"):
            try:
                num = int(d.name.split("_")[1])
                configs.append(num)
            except (ValueError, IndexError):
                continue
    return sorted(configs)


def find_method_dirs(base_dir: Path, methods: list[str]) -> dict[str, Path]:
    """Find directories for each method.

    Looks for directories matching pattern: {method}-{commit} or just {method}

    Args:
        base_dir: Base directory to search
        methods: List of method names to find

    Returns:
        Dict mapping method name to directory path
    """
    method_dirs = {}

    if not base_dir.exists():
        return method_dirs

    for method in methods:
        # Look for exact match first
        exact = base_dir / method
        if exact.exists() and exact.is_dir():
            method_dirs[method] = exact
            continue

        # Look for pattern: {method}-{commit}
        for d in base_dir.iterdir():
            if d.is_dir() and d.name.startswith(f"{method}-"):
                method_dirs[method] = d
                break

    return method_dirs


def main() -> None:
    """Main entry point for analyze_scaling CLI."""
    parser = argparse.ArgumentParser(description="Analyze MCP agent scaling experiment results")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/scaling"),
        help="Base directory containing method subdirectories (default: outputs/scaling)",
    )
    parser.add_argument(
        "--papers",
        type=int,
        nargs="+",
        help="Specific paper configs to analyze (default: 1 2 4 8 16)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots (requires matplotlib)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed per-config metrics",
    )
    # New comparison options
    parser.add_argument(
        "--metric",
        type=str,
        choices=[
            "recall",
            "semantic_recall",
            "precision",
            "semantic_precision",
            "f1",
            "semantic_f1",
            "recall_at_k",
            "exact_recall_at_k",
        ],
        default="recall",
        help="Metric to plot (default: recall)",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["macro", "micro"],
        default="macro",
        help="Aggregation method (default: macro)",
    )
    parser.add_argument(
        "--in-corpus",
        action="store_true",
        help="Use in-corpus recall (only applies to recall metrics)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Generate single combined figure with all tasks",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=AVAILABLE_METHODS,
        help="Methods to compare (default: auto-detect from base-dir)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Enable comparison mode (plot multiple methods on same axes)",
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="Generate Pareto frontier plots (cost vs metric) instead of scaling curves",
    )
    parser.add_argument(
        "--recall-at-k",
        action="store_true",
        help="Generate recall@k scaling curve and/or Pareto plots (use with --plot)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="k value for recall_at_k / exact_recall_at_k metric (e.g. 10 or 20)",
    )
    # Explicit directory specification (overrides auto-detection)
    parser.add_argument(
        "--single-dir",
        type=Path,
        help="Explicit directory for single-agent method",
    )
    parser.add_argument(
        "--multi-dir",
        type=Path,
        help="Explicit directory for multi-agent method",
    )
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        help="Explicit directory for pipeline method",
    )
    parser.add_argument(
        "--memorization-dir",
        type=Path,
        help="Explicit directory for memorization baseline",
    )

    args = parser.parse_args()

    paper_configs = args.papers or PAPER_CONFIGS

    # Check if any explicit directory is provided
    explicit_dirs = {
        "single": args.single_dir,
        "multi": args.multi_dir,
        "pipeline": args.pipeline_dir,
        "memorization": args.memorization_dir,
    }
    has_explicit_dirs = any(d is not None for d in explicit_dirs.values())

    # Check if comparison mode
    if args.compare or args.methods or has_explicit_dirs:
        # Comparison mode: find and collect metrics from multiple method directories
        method_dirs = {}

        # Use explicit directories first
        for method, explicit_dir in explicit_dirs.items():
            if explicit_dir is not None:
                if explicit_dir.exists():
                    method_dirs[method] = explicit_dir
                else:
                    print(f"Warning: {method} directory not found: {explicit_dir}")

        # Auto-detect remaining methods if --methods specified or no explicit dirs
        if args.methods or (not has_explicit_dirs):
            methods_to_find = args.methods or AVAILABLE_METHODS
            # Only auto-detect methods not explicitly specified
            methods_to_find = [m for m in methods_to_find if m not in method_dirs]
            if methods_to_find:
                auto_dirs = find_method_dirs(args.base_dir, methods_to_find)
                method_dirs.update(auto_dirs)

        if not method_dirs:
            print(f"Error: No method directories found in {args.base_dir}")
            print(f"Looking for: {args.methods or AVAILABLE_METHODS}")
            print("\nExpected directory structure:")
            print("  outputs/scaling/")
            print("    single-abc123/papers_1/eval_results.json")
            print("    multi-abc123/papers_1/eval_results.json")
            print("    ...")
            sys.exit(1)

        print(f"Found method directories: {list(method_dirs.keys())}")

        # Collect metrics for each method
        methods_data = {}
        for method, method_dir in method_dirs.items():
            # Use user-specified configs or auto-detect from directory
            if args.papers:
                method_configs = paper_configs
            else:
                method_configs = detect_paper_configs(method_dir)
                if not method_configs:
                    method_configs = paper_configs  # Fallback to default

            metrics = collect_all_metrics(method_dir, method_configs)
            if metrics:
                methods_data[method] = metrics
                print(
                    f"  {method}: {len(metrics)} configs loaded (papers: {[m.max_papers for m in metrics]})"
                )
            else:
                print(f"  {method}: No metrics found")

        if not methods_data:
            print("Error: No metrics found for any method")
            sys.exit(1)

        # Generate plots
        if args.plot:
            output_dir = args.base_dir / "comparison_plots"

            if args.recall_at_k:
                # Generate recall@k comparison plot (always)
                generate_recall_at_k_plots(methods_data, output_dir)
                # Generate recall@k Pareto plots (if --pareto also specified)
                if args.pareto:
                    generate_recall_at_k_pareto_plots(methods_data, output_dir)
            elif args.pareto:
                # Generate Pareto frontier plots (cost vs in-corpus semantic recall)
                generate_pareto_plots(
                    methods_data,
                    output_dir,
                    metric="semantic_recall",
                    aggregation=args.aggregation,
                    in_corpus=True,
                    combined=args.combined,
                )
            else:
                # Generate comparison plots (in-corpus semantic recall, 1x4 layout)
                generate_comparison_plots(
                    methods_data,
                    output_dir,
                    metric="semantic_recall",
                    aggregation=args.aggregation,
                    in_corpus=True,
                    combined=True,
                )

        # Print summary for first method
        first_method = list(methods_data.keys())[0]
        print(f"\n=== Summary for {first_method} ===")
        print_summary_table(methods_data[first_method])

    else:
        # Single method mode (legacy behavior)
        metrics = collect_all_metrics(args.base_dir, paper_configs)

        if not metrics:
            print("Error: No metrics found. Have you run the experiment and evaluations?")
            print("\nExpected steps:")
            print("  1. Run scaling experiment: uv run python -m scripts.run_scaling_experiment")
            print("  2. Run evaluations for each config:")
            print("     for p in 1 2 4 8 16; do")
            print(
                "       uv run python -m eval.run_eval --batch "
                "--output-dir outputs/scaling/papers_$p "
                "-o outputs/scaling/papers_$p/eval_results.json"
            )
            print("     done")
            print("  3. Run analysis: uv run python -m scripts.analyze_scaling")
            sys.exit(1)

        # Print summary table
        print_summary_table(metrics)

        # Find and report knee point
        knee = find_knee_point(metrics)
        if knee:
            print(f"\n** RECOMMENDED CONFIG: max_papers={knee.max_papers} **")
            print(
                f"   Best cost/performance tradeoff (avg cost: ${knee.avg_cost:.3f}, "
                f"GO F1: {knee.go_exact_f1:.1%})"
            )

        # Detailed metrics if requested
        if args.verbose:
            print_detailed_metrics(metrics)

        # Generate plots if requested
        if args.plot:
            generate_plots(metrics, args.base_dir)

        # Save results
        save_results(metrics, args.base_dir)


if __name__ == "__main__":
    main()
