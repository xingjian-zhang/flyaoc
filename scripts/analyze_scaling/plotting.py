"""Comparison plotting functions for scaling analysis.

Generates plots comparing different methods (Single, Multi, Pipeline, Memorization)
on the same axes with consistent task colors.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config_metrics import ConfigMetrics

# Task colors from plot_task_sunburst.py
TASK_COLORS = {
    "task1": "#3498DB",  # Blue
    "task2": "#27AE60",  # Green
    "task3": "#9B59B6",  # Purple
    "reference": "#7F8C8D",  # Grey - for reference coverage
}

# Method colors and styles - distinct colors for each baseline method
METHOD_STYLES = {
    "single": {
        "linestyle": "-",
        "marker": "o",
        "label": "Single-Agent",
        "color": "#3498DB",
    },  # Blue
    "multi": {"linestyle": "-", "marker": "s", "label": "Multi-Agent", "color": "#E74C3C"},  # Red
    "pipeline": {"linestyle": "-", "marker": "^", "label": "Pipeline", "color": "#27AE60"},  # Green
    "memorization": {
        "linestyle": "--",
        "marker": "d",
        "label": "Memorization",
        "color": "#9B59B6",
    },  # Purple, dashed
}


def ci95(std: float, n: int) -> float:
    """Compute 95% confidence interval half-width."""
    if n < 2:
        return 0.0
    return 1.96 * std / (n**0.5)


def get_metric_getter(
    task: str,
    metric: str,
    aggregation: str,
    in_corpus: bool,
    k: int | None = None,
) -> Callable[[ConfigMetrics], tuple[float, float]]:
    """Get a function that extracts the specified metric from ConfigMetrics.

    Args:
        task: "task1", "task2", or "task3"
        metric: "recall", "semantic_recall", "precision", "semantic_precision", "f1", "semantic_f1",
                "recall_at_k", or "exact_recall_at_k"
        aggregation: "macro" or "micro"
        in_corpus: Whether to use in-corpus recall
        k: Required for recall_at_k / exact_recall_at_k metrics. The k value to look up.

    Returns:
        Function that takes ConfigMetrics and returns (value, std) tuple.
        std is 0.0 for micro-averaged or precision/recall metrics.
    """

    def getter(m: ConfigMetrics) -> tuple[float, float]:
        # Handle recall_at_k metrics (dict-based lookup)
        if metric in ("recall_at_k", "exact_recall_at_k"):
            if k is None:
                return (0.0, 0.0)
            k_str = str(k)
            if task == "task1":
                if metric == "recall_at_k":
                    d = (
                        m.go_micro_semantic_recall_at_k
                        if aggregation == "micro"
                        else m.go_macro_semantic_recall_at_k
                    )
                else:
                    d = (
                        m.go_micro_exact_recall_at_k
                        if aggregation == "micro"
                        else m.go_macro_exact_recall_at_k
                    )
                return (d.get(k_str, 0.0), 0.0)
            elif task == "task2":
                if metric == "recall_at_k":
                    d = (
                        m.expr_micro_semantic_recall_at_k
                        if aggregation == "micro"
                        else m.expr_macro_semantic_recall_at_k
                    )
                else:
                    d = (
                        m.expr_micro_exact_recall_at_k
                        if aggregation == "micro"
                        else m.expr_macro_exact_recall_at_k
                    )
                return (d.get(k_str, 0.0), 0.0)
            elif task == "task3":
                # Task 3 has no semantic, so both metric types use exact
                d = (
                    m.syn_micro_exact_recall_at_k
                    if aggregation == "micro"
                    else m.syn_macro_exact_recall_at_k
                )
                return (d.get(k_str, 0.0), 0.0)
            return (0.0, 0.0)

        if task == "task1":
            if metric == "recall":
                if in_corpus:
                    return (m.go_in_corpus_exact_recall, 0.0)
                if aggregation == "micro":
                    return (m.go_micro_recall, 0.0)
                return (m.go_full_recall, 0.0)
            elif metric == "semantic_recall":
                # Semantic recall: in-corpus vs full (against ALL GT)
                if in_corpus:
                    if aggregation == "micro":
                        return (m.go_semantic_micro_recall, 0.0)
                    return (m.go_semantic_recall, 0.0)
                return (m.go_full_semantic_recall, 0.0)
            elif metric == "precision":
                if aggregation == "micro":
                    return (m.go_micro_precision, 0.0)
                return (m.go_exact_precision, 0.0)
            elif metric == "semantic_precision":
                return (m.go_semantic_precision, 0.0)
            elif metric == "f1":
                if aggregation == "micro":
                    return (m.go_micro_f1, 0.0)
                return (m.go_exact_f1, m.go_exact_f1_std)
            elif metric == "semantic_f1":
                return (m.go_semantic_f1, m.go_semantic_f1_std)

        elif task == "task2":
            if metric == "recall":
                if in_corpus:
                    return (m.expr_in_corpus_exact_recall, 0.0)
                if aggregation == "micro":
                    return (m.expr_anatomy_micro_recall, 0.0)
                # Use full_recall for non-in-corpus (against ALL GT)
                return (m.expr_full_recall, 0.0)
            elif metric == "semantic_recall":
                # Semantic recall: in-corpus vs full (against ALL GT)
                if in_corpus:
                    if aggregation == "micro":
                        return (m.expr_semantic_micro_recall, 0.0)
                    return (m.expr_semantic_recall, 0.0)
                return (m.expr_full_semantic_recall, 0.0)
            elif metric == "precision":
                if aggregation == "micro":
                    return (m.expr_anatomy_micro_precision, 0.0)
                return (m.expr_anatomy_precision, 0.0)
            elif metric == "semantic_precision":
                return (m.expr_semantic_precision, 0.0)
            elif metric == "f1":
                if aggregation == "micro":
                    return (m.expr_anatomy_micro_f1, 0.0)
                return (m.expr_anatomy_f1, m.expr_anatomy_f1_std)
            elif metric == "semantic_f1":
                return (m.expr_semantic_f1, m.expr_semantic_f1_std)

        elif task == "task3":
            # Task 3: Use combined metrics (unified fullname + symbol)
            if metric == "recall":
                if in_corpus:
                    return (m.syn_combined_in_corpus_recall, 0.0)
                if aggregation == "micro":
                    return (m.syn_combined_micro_recall, 0.0)
                # Overall recall: use full_recall (matches against ALL GT)
                return (m.syn_combined_full_recall, 0.0)
            elif metric == "semantic_recall":
                # Synonyms use exact match, so semantic = regular
                if in_corpus:
                    return (m.syn_combined_in_corpus_recall, 0.0)
                if aggregation == "micro":
                    return (m.syn_combined_micro_recall, 0.0)
                return (m.syn_combined_full_recall, 0.0)
            elif metric == "precision":
                if aggregation == "micro":
                    return (m.syn_combined_micro_precision, 0.0)
                return (m.syn_combined_precision, 0.0)
            elif metric == "semantic_precision":
                # Synonyms use exact match, so semantic = regular
                if aggregation == "micro":
                    return (m.syn_combined_micro_precision, 0.0)
                return (m.syn_combined_precision, 0.0)
            elif metric == "f1":
                if aggregation == "micro":
                    return (m.syn_combined_micro_f1, 0.0)
                return (m.syn_combined_f1, m.syn_combined_f1_std)
            elif metric == "semantic_f1":
                # Synonyms use exact match, so semantic = regular
                return (m.syn_combined_f1, m.syn_combined_f1_std)

        elif task == "reference":
            # Reference coverage: How well agent retrieves effective papers
            if metric == "recall":
                if aggregation == "micro":
                    return (m.ref_micro_recall, 0.0)
                return (m.ref_recall, 0.0)
            elif metric == "precision":
                if aggregation == "micro":
                    return (m.ref_micro_precision, 0.0)
                return (m.ref_precision, 0.0)
            elif metric == "f1":
                if aggregation == "micro":
                    return (m.ref_micro_f1, 0.0)
                return (m.ref_f1, m.ref_f1_std)

        elif task == "overall":
            # Overall: Average of all three annotation tasks
            # Get individual task values using recursive calls
            t1_getter = get_metric_getter("task1", metric, aggregation, in_corpus, k=k)
            t2_getter = get_metric_getter("task2", metric, aggregation, in_corpus, k=k)
            t3_getter = get_metric_getter("task3", metric, aggregation, in_corpus, k=k)

            t1_val, t1_std = t1_getter(m)
            t2_val, t2_std = t2_getter(m)
            t3_val, t3_std = t3_getter(m)

            avg_val = (t1_val + t2_val + t3_val) / 3
            # Propagate std as average (conservative estimate)
            avg_std = (t1_std + t2_std + t3_std) / 3
            return (avg_val, avg_std)

        return (0.0, 0.0)

    return getter


def plot_method_comparison(
    methods_data: dict[str, list[ConfigMetrics]],
    task: str,
    metric_getter: Callable[[ConfigMetrics], tuple[float, float]],
    ax: Any,
    title: str,
    ylabel: str,
    show_legend: bool = True,
) -> None:
    """Plot comparison of methods for a single task.

    Args:
        methods_data: Dict mapping method name to list of ConfigMetrics
        task: Task name (unused, kept for API compatibility)
        metric_getter: Function to extract metric from ConfigMetrics
        ax: Matplotlib axis
        title: Plot title
        ylabel: Y-axis label
        show_legend: Whether to show legend on this plot
    """
    # Collect all paper values for x-axis range
    all_papers = set()
    for metrics_list in methods_data.values():
        for m in metrics_list:
            if m.max_papers > 0:  # Exclude papers=0 from x-axis range
                all_papers.add(m.max_papers)

    for method_name, metrics_list in methods_data.items():
        if not metrics_list:
            continue

        style = METHOD_STYLES.get(
            method_name, {"linestyle": "-", "marker": "o", "label": method_name, "color": "#666666"}
        )
        color = style.get("color", "#666666")

        papers = [m.max_papers for m in metrics_list]
        values = []
        errors = []

        for m in metrics_list:
            val, std = metric_getter(m)
            values.append(val * 100)  # Convert to percentage
            errors.append(ci95(std * 100, m.num_genes_evaluated))

        # Special case: memorization baseline (papers=0) - draw horizontal line
        if len(papers) == 1 and papers[0] == 0:
            ax.axhline(
                y=values[0],
                linestyle=style["linestyle"],
                color=color,
                label=style["label"],
                linewidth=2,
                alpha=0.8,
            )
        # Plot with error bars if we have std dev
        elif any(e > 0 for e in errors):
            ax.errorbar(
                papers,
                values,
                yerr=errors,
                linestyle=style["linestyle"],
                marker=style["marker"],
                color=color,
                label=style["label"],
                linewidth=2,
                capsize=3,
                markersize=6,
                alpha=0.8,
            )
        else:
            ax.plot(
                papers,
                values,
                linestyle=style["linestyle"],
                marker=style["marker"],
                color=color,
                label=style["label"],
                linewidth=2,
                markersize=6,
                alpha=0.8,
            )

    ax.set_xlabel("Max Papers")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    if methods_data:
        # Get paper configs from first method with data
        for metrics_list in methods_data.values():
            if metrics_list:
                papers = [m.max_papers for m in metrics_list]
                ax.set_xticks(papers)
                ax.set_xticklabels([str(p) for p in papers])
                break
    if show_legend:
        ax.legend()
    ax.grid(True, alpha=0.3)


def _place_annotations(
    ax: Any,
    annotations: list[tuple[str, tuple[float, float], str]],
    fontsize: int = 10,
) -> None:
    """Place annotations with per-label overlap avoidance.

    For each annotation, tries several offset candidates and picks the one
    whose display-coordinate position is farthest from all previously placed
    labels. Works correctly with log-scale axes by comparing in display space.

    Args:
        ax: Matplotlib axis.
        annotations: List of (label, (x_data, y_data), color) tuples.
        fontsize: Font size for annotations.
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    trans = ax.transData

    # Candidate offsets in points: (dx, dy, ha, va)
    candidates = [
        (7, 7, "left", "bottom"),  # upper-right
        (-7, 7, "right", "bottom"),  # upper-left
        (7, -7, "left", "top"),  # lower-right
        (-7, -7, "right", "top"),  # lower-left
        (0, 11, "center", "bottom"),  # above
        (0, -11, "center", "top"),  # below
        (12, 0, "left", "center"),  # right
        (-12, 0, "right", "center"),  # left
    ]

    placed: list[tuple[float, float]] = []

    for label, (xd, yd), color in annotations:
        disp_x, disp_y = trans.transform((xd, yd))

        best_offset = candidates[0]
        best_min_dist = -1.0
        for dx, dy, ha, va in candidates:
            cx, cy = disp_x + dx, disp_y + dy
            if placed:
                min_dist = min(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 for px, py in placed)
            else:
                min_dist = float("inf")
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_offset = (dx, dy, ha, va)

        dx, dy, ha, va = best_offset
        ax.annotate(
            label,
            (xd, yd),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=fontsize,
            color=color,
            ha=ha,
            va=va,
        )
        placed.append((disp_x + dx, disp_y + dy))


def plot_pareto_comparison(
    methods_data: dict[str, list[ConfigMetrics]],
    task: str,
    metric_getter: Callable[[ConfigMetrics], tuple[float, float]],
    ax: Any,
    title: str,
    ylabel: str,
    show_legend: bool = True,
) -> None:
    """Plot Pareto frontier (cost vs metric) comparing methods for a single task.

    Args:
        methods_data: Dict mapping method name to list of ConfigMetrics
        task: Task name (unused, kept for API compatibility)
        metric_getter: Function to extract metric from ConfigMetrics
        ax: Matplotlib axis
        title: Plot title
        ylabel: Y-axis label
        show_legend: Whether to show legend on this plot
    """
    # Collect all annotations for overlap avoidance: (label, (x, y), color)
    pending_annotations: list[tuple[str, tuple[float, float], str]] = []

    for method_name, metrics_list in methods_data.items():
        if not metrics_list:
            continue

        style = METHOD_STYLES.get(
            method_name, {"linestyle": "-", "marker": "o", "label": method_name, "color": "#666666"}
        )
        color = style.get("color", "#666666")

        costs = []
        values = []
        errors = []
        paper_labels = []

        for m in metrics_list:
            val, std = metric_getter(m)
            costs.append(m.avg_cost)
            values.append(val * 100)  # Convert to percentage
            errors.append(ci95(std * 100, m.num_genes_evaluated))
            paper_labels.append(f"{m.max_papers}p")

        # Check if this is memorization baseline (papers=0)
        is_memorization = len(metrics_list) == 1 and metrics_list[0].max_papers == 0

        # Special case: memorization baseline - just marker (no horizontal line)
        if is_memorization:
            ax.scatter(
                costs,
                values,
                marker=style["marker"],
                color=color,
                s=80,
                label=style["label"],
                zorder=5,
            )
            # No annotation for memorization — legend is sufficient
        # Plot with error bars if we have std dev
        elif any(e > 0 for e in errors):
            ax.errorbar(
                costs,
                values,
                yerr=errors,
                linestyle=style["linestyle"],
                marker=style["marker"],
                color=color,
                label=style["label"],
                linewidth=2,
                capsize=3,
                markersize=6,
                alpha=0.8,
            )
            for i, label in enumerate(paper_labels):
                pending_annotations.append((label, (costs[i], values[i]), color))
        else:
            ax.plot(
                costs,
                values,
                linestyle=style["linestyle"],
                marker=style["marker"],
                color=color,
                label=style["label"],
                linewidth=2,
                markersize=6,
                alpha=0.8,
            )
            for i, label in enumerate(paper_labels):
                pending_annotations.append((label, (costs[i], values[i]), color))

    # Place annotations with adjustText overlap avoidance
    _place_annotations(ax, pending_annotations)

    ax.set_xlabel("Average Cost per Gene ($)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log")
    if show_legend:
        ax.legend()
    ax.grid(True, alpha=0.3)


def generate_comparison_plots(
    methods_data: dict[str, list[ConfigMetrics]],
    output_dir: Path,
    metric: str = "recall",
    aggregation: str = "macro",
    in_corpus: bool = False,
    combined: bool = False,
) -> None:
    """Generate comparison plots for all tasks.

    Args:
        methods_data: Dict mapping method name to list of ConfigMetrics
        output_dir: Directory to save plots
        metric: Metric to plot
        aggregation: "macro" or "micro"
        in_corpus: Whether to use in-corpus recall
        combined: Whether to create a single combined figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plots", file=sys.stderr)
        print("  Install with: uv add matplotlib", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set font sizes for paper-ready plots
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    )

    # Build metric label
    metric_label = metric.replace("_", " ").title()
    if in_corpus:
        metric_label = f"In-Corpus {metric_label}"
    if aggregation == "micro":
        metric_label = f"{metric_label} (Micro)"

    tasks = [
        ("task1", "Task 1: Gene Ontology"),
        ("task2", "Task 2: Expression"),
        ("task3", "Task 3: Synonyms"),
        ("reference", "Reference Coverage"),
    ]

    if combined:
        # Single figure with 4 subplots (1x4 layout)
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        for i, (ax, (task, task_title)) in enumerate(zip(axes, tasks, strict=True)):
            # Reference always uses recall metric (not affected by in_corpus flag)
            if task == "reference":
                task_metric_getter = get_metric_getter(task, "recall", aggregation, in_corpus=False)
            else:
                task_metric_getter = get_metric_getter(task, metric, aggregation, in_corpus)
            # Only show ylabel and legend on first panel
            ylabel = f"{metric_label} (%)" if i == 0 else ""
            if task == "reference" and i == 0:
                ylabel = "Recall (%)"
            plot_method_comparison(
                methods_data,
                task,
                task_metric_getter,
                ax,
                title=task_title,
                ylabel=ylabel,
                show_legend=(i == 0),
            )

        plt.tight_layout()

        filename = f"method_comparison_{metric}"
        if in_corpus:
            filename += "_in_corpus"
        if aggregation == "micro":
            filename += "_micro"
        filename += ".pdf"

        plt.savefig(output_dir / filename, dpi=300, format="pdf")
        plt.close()
        print(f"Saved: {output_dir / filename}")

    else:
        # Separate files for each task
        for task, task_title in tasks:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Reference always uses recall metric
            if task == "reference":
                task_metric_getter = get_metric_getter(task, "recall", aggregation, in_corpus=False)
                ylabel = "Recall (%)"
            else:
                task_metric_getter = get_metric_getter(task, metric, aggregation, in_corpus)
                ylabel = f"{metric_label} (%)"

            plot_method_comparison(
                methods_data,
                task,
                task_metric_getter,
                ax,
                title=f"{task_title} - {metric_label}"
                if task != "reference"
                else "Reference Coverage - Recall",
                ylabel=ylabel,
            )

            plt.tight_layout()

            # Build filename
            task_short = {"task1": "go", "task2": "expr", "task3": "syn", "reference": "ref"}[task]
            filename = f"{task_short}_{metric}"
            if in_corpus:
                filename += "_in_corpus"
            if aggregation == "micro":
                filename += "_micro"
            filename += ".pdf"

            plt.savefig(output_dir / filename, dpi=300, format="pdf")
            plt.close()
            print(f"Saved: {output_dir / filename}")


def generate_pareto_plots(
    methods_data: dict[str, list[ConfigMetrics]],
    output_dir: Path,
    metric: str = "semantic_recall",
    aggregation: str = "macro",
    in_corpus: bool = False,
    combined: bool = False,
) -> None:
    """Generate Pareto frontier plots (cost vs metric) for all tasks.

    Args:
        methods_data: Dict mapping method name to list of ConfigMetrics
        output_dir: Directory to save plots
        metric: Metric to plot (default: semantic_recall)
        aggregation: "macro" or "micro"
        in_corpus: Whether to use in-corpus recall
        combined: Whether to create a single combined figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plots", file=sys.stderr)
        print("  Install with: uv add matplotlib", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set font sizes for paper-ready plots
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    )

    # Build metric label
    metric_label = metric.replace("_", " ").title()
    if in_corpus:
        metric_label = f"In-Corpus {metric_label}"
    if aggregation == "micro":
        metric_label = f"{metric_label} (Micro)"

    # Annotation tasks + overall
    tasks = [
        ("task1", "Task 1: Gene Ontology"),
        ("task2", "Task 2: Expression"),
        ("task3", "Task 3: Synonyms"),
        ("overall", "Overall"),
    ]

    if combined:
        # Single figure with 4 subplots (1x4 layout)
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        for i, (ax, (task, task_title)) in enumerate(zip(axes, tasks, strict=True)):
            task_metric_getter = get_metric_getter(task, metric, aggregation, in_corpus)
            # Only show ylabel and legend on first panel
            ylabel = f"{metric_label} (%)" if i == 0 else ""
            plot_pareto_comparison(
                methods_data,
                task if task != "overall" else "task1",
                task_metric_getter,
                ax,
                title=task_title,
                ylabel=ylabel,
                show_legend=(i == 0),
            )

        plt.tight_layout()

        filename = f"pareto_{metric}"
        if in_corpus:
            filename += "_in_corpus"
        if aggregation == "micro":
            filename += "_micro"
        filename += "_combined.pdf"

        plt.savefig(output_dir / filename, dpi=300, format="pdf")
        plt.close()
        print(f"Saved: {output_dir / filename}")

        # Also generate standalone overall plot
        fig, ax = plt.subplots(figsize=(8, 6))
        overall_getter = get_metric_getter("overall", metric, aggregation, in_corpus)
        plot_pareto_comparison(
            methods_data,
            "task1",  # Use task1 color for overall
            overall_getter,
            ax,
            title="",
            ylabel="Average Recall Across 3 Tasks (%)",
            show_legend=True,
        )
        plt.tight_layout()

        filename = f"pareto_overall_{metric}"
        if in_corpus:
            filename += "_in_corpus"
        if aggregation == "micro":
            filename += "_micro"
        filename += ".pdf"

        plt.savefig(output_dir / filename, dpi=300, format="pdf")
        plt.close()
        print(f"Saved: {output_dir / filename}")

    else:
        # Separate files for each task
        for task, task_title in tasks:
            fig, ax = plt.subplots(figsize=(10, 6))

            task_metric_getter = get_metric_getter(task, metric, aggregation, in_corpus)

            plot_pareto_comparison(
                methods_data,
                task if task != "overall" else "task1",  # Use task1 color for overall
                task_metric_getter,
                ax,
                title=f"{task_title} - Cost vs {metric_label}",
                ylabel=f"{metric_label} (%)",
            )

            plt.tight_layout()

            # Build filename
            task_short = {"task1": "go", "task2": "expr", "task3": "syn", "overall": "overall"}[
                task
            ]
            filename = f"pareto_{task_short}_{metric}"
            if in_corpus:
                filename += "_in_corpus"
            if aggregation == "micro":
                filename += "_micro"
            filename += ".pdf"

            plt.savefig(output_dir / filename, dpi=300, format="pdf")
            plt.close()
            print(f"Saved: {output_dir / filename}")


def generate_recall_at_k_plots(
    methods_data: dict[str, list[ConfigMetrics]],
    output_dir: Path,
) -> None:
    """Generate recall@k comparison plot with 4 subplots.

    Creates a single figure with 4 subplots comparing methods:
    - T1: micro semantic recall@20 (in-corpus)
    - T2: micro semantic recall@10 (in-corpus)
    - T3: micro exact recall@20 (in-corpus)
    - Ref: micro reference coverage recall

    Args:
        methods_data: Dict mapping method name to list of ConfigMetrics
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plots", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    )

    # Per-task metric configuration: (task, metric, aggregation, in_corpus, k, title, ylabel)
    subplot_configs = [
        (
            "task1",
            "recall_at_k",
            "micro",
            True,
            20,
            "T1: GO Semantic Recall@20",
            "Micro Recall (%)",
        ),
        ("task2", "recall_at_k", "micro", True, 10, "T2: Expr Semantic Recall@10", ""),
        ("task3", "exact_recall_at_k", "micro", True, 20, "T3: Syn Exact Recall@20", ""),
        ("reference", "recall", "micro", False, None, "Reference Coverage", ""),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for i, (ax, (task, metric, agg, ic, k, title, ylabel)) in enumerate(
        zip(axes, subplot_configs, strict=True)
    ):
        metric_getter = get_metric_getter(task, metric, agg, ic, k=k)
        plot_method_comparison(
            methods_data,
            task,
            metric_getter,
            ax,
            title=title,
            ylabel=ylabel,
            show_legend=(i == 0),
        )

    plt.tight_layout()
    filename = "recall_at_k_comparison_micro.pdf"
    plt.savefig(output_dir / filename, dpi=300, format="pdf")
    plt.close()
    print(f"Saved: {output_dir / filename}")


def generate_recall_at_k_pareto_plots(
    methods_data: dict[str, list[ConfigMetrics]],
    output_dir: Path,
) -> None:
    """Generate Pareto frontier plot using recall@k metrics.

    Creates a single figure with 4 subplots in 2x2 layout:
    - T1: micro semantic recall@20 (in-corpus) vs cost
    - T2: micro semantic recall@10 (in-corpus) vs cost
    - T3: micro exact recall@20 (in-corpus) vs cost
    - Ref: micro reference coverage recall vs cost

    Args:
        methods_data: Dict mapping method name to list of ConfigMetrics
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plots", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    )

    # Per-task configs: (task, metric, agg, k, title, ylabel)
    subplot_configs = [
        ("task1", "recall_at_k", "micro", 20, "Task 1: Gene Ontology", "Semantic Recall@20 (%)"),
        ("task2", "recall_at_k", "micro", 10, "Task 2: Expression", "Semantic Recall@10 (%)"),
        ("task3", "exact_recall_at_k", "micro", 20, "Task 3: Synonyms", "Exact Recall@20 (%)"),
        ("ref", "recall", "micro", None, "Reference Coverage", "Recall (%)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing

    for i, (ax, (task, metric, agg, k, title, ylabel)) in enumerate(
        zip(axes, subplot_configs, strict=True)
    ):
        if task == "ref":
            # Reference coverage: return (value, std_dev) - no error bars
            def ref_getter(m: ConfigMetrics) -> tuple[float, float]:
                return (m.ref_micro_recall, 0.0)
            metric_getter = ref_getter
        else:
            metric_getter = get_metric_getter(task, metric, agg, in_corpus=True, k=k)
        
        plot_pareto_comparison(
            methods_data,
            task if task != "ref" else "task1",  # Use task1 color scheme for ref
            metric_getter,
            ax,
            title=title,
            ylabel=ylabel,
            show_legend=False,
        )
        # Only show x-label on bottom row
        if i < 2:
            ax.set_xlabel("")

    # Shared legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=(0, 0.06, 1, 1.0))

    filename = "pareto_recall_at_k_micro.pdf"
    plt.savefig(output_dir / filename, dpi=300, format="pdf")
    plt.close()
    print(f"Saved: {output_dir / filename}")


def generate_plots(metrics: list[ConfigMetrics], output_dir: Path) -> None:
    """Generate legacy single-method visualization plots.

    This function maintains backward compatibility with the original interface.
    For comparison plots, use generate_comparison_plots directly.

    Args:
        metrics: List of ConfigMetrics for a single method
        output_dir: Directory to save plots
    """
    # Wrap single method as "single" for compatibility
    methods_data = {"single": metrics}

    # Generate default comparison plots (recall)
    generate_comparison_plots(
        methods_data,
        output_dir,
        metric="recall",
        aggregation="macro",
        in_corpus=False,
        combined=False,
    )

    # Also generate some legacy-style plots
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    papers = [m.max_papers for m in metrics]

    # Pareto frontier (cost vs F1)
    costs = [m.avg_cost for m in metrics]
    go_f1 = [m.go_exact_f1 * 100 for m in metrics]
    go_f1_ci = [ci95(m.go_exact_f1_std * 100, m.num_genes_evaluated) for m in metrics]

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        costs,
        go_f1,
        yerr=go_f1_ci,
        fmt="o-",
        capsize=4,
        capthick=1.5,
        color="blue",
        label="GO Exact F1",
        markersize=8,
        linewidth=1.5,
        alpha=0.8,
    )

    for i, m in enumerate(metrics):
        ax.annotate(
            f"{m.max_papers}p",
            (costs[i], go_f1[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    ax.set_xlabel("Average Cost per Gene ($)")
    ax.set_ylabel("GO Exact F1 (%)")
    ax.set_title("Cost vs Performance Pareto Frontier (+/-95% CI)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "pareto_frontier.pdf", dpi=300, format="pdf")
    plt.close()
    print(f"Saved: {output_dir / 'pareto_frontier.pdf'}")

    # Scaling curve (all tasks)
    _fig, ax = plt.subplots(figsize=(10, 6))

    # GO F1
    go_f1_vals = [m.go_exact_f1 * 100 for m in metrics]
    go_f1_cis = [ci95(m.go_exact_f1_std * 100, m.num_genes_evaluated) for m in metrics]
    ax.errorbar(
        papers,
        go_f1_vals,
        yerr=go_f1_cis,
        fmt="o-",
        color=TASK_COLORS["task1"],
        label="GO Exact F1",
        linewidth=2,
        capsize=3,
    )

    # Expression F1
    expr_f1_vals = [m.expr_anatomy_f1 * 100 for m in metrics]
    expr_f1_cis = [ci95(m.expr_anatomy_f1_std * 100, m.num_genes_evaluated) for m in metrics]
    ax.errorbar(
        papers,
        expr_f1_vals,
        yerr=expr_f1_cis,
        fmt="s-",
        color=TASK_COLORS["task2"],
        label="Expression Anatomy F1",
        linewidth=2,
        capsize=3,
    )

    # Synonym F1 (combined)
    syn_f1_vals = [m.syn_combined_f1 * 100 for m in metrics]
    syn_f1_cis = [ci95(m.syn_combined_f1_std * 100, m.num_genes_evaluated) for m in metrics]
    ax.errorbar(
        papers,
        syn_f1_vals,
        yerr=syn_f1_cis,
        fmt="^-",
        color=TASK_COLORS["task3"],
        label="Synonym Combined F1",
        linewidth=2,
        capsize=3,
    )

    ax.set_xlabel("Max Papers")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("All Tasks F1 Scaling with Paper Budget (+/-95% CI)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(papers)
    ax.set_xticklabels([str(p) for p in papers])
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "scaling_curve.pdf", dpi=300, format="pdf")
    plt.close()
    print(f"Saved: {output_dir / 'scaling_curve.pdf'}")
