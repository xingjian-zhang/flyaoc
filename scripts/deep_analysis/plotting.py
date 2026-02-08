"""Plotting functions for deep analysis of scaling experiments."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .dataclasses import GeneTrajectory


def generate_publication_plots(
    trajectories: list[GeneTrajectory],
    all_evals: dict[int, dict],
    fp_analysis: dict,
    paper_yield: dict,
    task_comparison: dict,
    output_dir: Path,
) -> None:
    """Generate publication-quality visualizations."""
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plots", file=sys.stderr)
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Set publication style
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
        }
    )

    # Figure 1: Task 1 vs Task 2 Scaling Comparison (Main Result)
    _fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    configs = sorted(all_evals.keys())

    # Task 1 metrics
    t1_recalls = []
    t1_precisions = []
    t1_f1s = []

    # Task 2 metrics
    t2_recalls = []
    t2_precisions = []
    t2_f1s = []

    for config in configs:
        genes = all_evals[config].get("genes", [])

        t1_recalls.append(
            np.mean([g.get("task1_go", {}).get("soft", {}).get("recall", 0) for g in genes])
        )
        t1_precisions.append(
            np.mean([g.get("task1_go", {}).get("soft", {}).get("precision", 0) for g in genes])
        )
        t1_f1s.append(np.mean([g.get("task1_go", {}).get("soft", {}).get("f1", 0) for g in genes]))

        t2 = [
            g.get("task2_expression", {}).get(
                "anatomy_soft", g.get("task2_expression", {}).get("anatomy", {})
            )
            for g in genes
        ]
        t2_recalls.append(np.mean([x.get("recall", 0) for x in t2]))
        t2_precisions.append(np.mean([x.get("precision", 0) for x in t2]))
        t2_f1s.append(np.mean([x.get("f1", 0) for x in t2]))

    # Left: F1 comparison
    ax = axes[0]
    ax.plot(
        configs, [f * 100 for f in t1_f1s], "b-o", label="Task 1: GO", linewidth=2, markersize=8
    )
    ax.plot(
        configs,
        [f * 100 for f in t2_f1s],
        "r-s",
        label="Task 2: Expression",
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Number of Papers")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("(a) F1 Score vs Paper Budget")
    ax.set_xscale("log", base=2)
    ax.set_xticks(configs)
    ax.set_xticklabels([str(c) for c in configs])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 60)

    # Right: Precision-Recall tradeoff
    ax = axes[1]
    ax.plot(configs, [p * 100 for p in t1_precisions], "b-o", label="Task 1 Precision", linewidth=2)
    ax.plot(
        configs,
        [r * 100 for r in t1_recalls],
        "b--^",
        label="Task 1 Recall",
        linewidth=2,
        alpha=0.7,
    )
    ax.plot(configs, [p * 100 for p in t2_precisions], "r-s", label="Task 2 Precision", linewidth=2)
    ax.plot(
        configs,
        [r * 100 for r in t2_recalls],
        "r--d",
        label="Task 2 Recall",
        linewidth=2,
        alpha=0.7,
    )
    ax.set_xlabel("Number of Papers")
    ax.set_ylabel("Score (%)")
    ax.set_title("(b) Precision vs Recall")
    ax.set_xscale("log", base=2)
    ax.set_xticks(configs)
    ax.set_xticklabels([str(c) for c in configs])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "fig1_task_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(figures_dir / "fig1_task_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fig1_task_comparison.png/pdf")

    # Figure 2: Per-paper yield analysis
    _fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    yield_data = paper_yield.get("cumulative_analysis", [])
    if yield_data:
        papers = [d["papers"] for d in yield_data]
        marginal_tp = [d["marginal_tp"] for d in yield_data]
        marginal_prec = [d["marginal_precision"] * 100 for d in yield_data]
        tp_per_paper = [d["tp_per_paper"] for d in yield_data]

        ax = axes[0]
        ax.bar(range(len(papers)), marginal_tp, color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(papers)))
        ax.set_xticklabels([str(p) for p in papers])
        ax.set_xlabel("Number of Papers")
        ax.set_ylabel("Marginal True Positives")
        ax.set_title("(a) New TPs from Additional Papers")
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1]
        ax.plot(papers, tp_per_paper, "g-o", linewidth=2, markersize=8)
        ax.set_xlabel("Number of Papers")
        ax.set_ylabel("TP per Paper")
        ax.set_title("(b) Annotation Efficiency")
        ax.set_xscale("log", base=2)
        ax.set_xticks(papers)
        ax.set_xticklabels([str(p) for p in papers])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "fig2_per_paper_yield.png", dpi=300, bbox_inches="tight")
    plt.savefig(figures_dir / "fig2_per_paper_yield.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fig2_per_paper_yield.png/pdf")

    # Figure 3: Gene scaling behavior heatmap
    _fig, ax = plt.subplots(figsize=(10, 12))

    # Sort genes by scaling gain
    sorted_trajs = sorted(trajectories, key=lambda t: -t.scaling_gain("task1"))[:50]

    matrix = []
    labels = []
    for traj in sorted_trajs:
        row = [traj.task1_f1.get(c, 0) for c in configs]
        matrix.append(row)
        labels.append(f"{traj.gene_symbol} ({traj.scaling_gain('task1'):+.0%})")

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([str(c) for c in configs])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Number of Papers")
    ax.set_ylabel("Gene (scaling gain)")
    ax.set_title("Task 1 F1 by Gene and Paper Budget")

    plt.colorbar(im, ax=ax, label="F1 Score", shrink=0.8)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig3_gene_heatmap.png", dpi=300, bbox_inches="tight")
    plt.savefig(figures_dir / "fig3_gene_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fig3_gene_heatmap.png/pdf")

    # Figure 4: False positive analysis
    _fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # FP categories pie chart
    ax = axes[0]
    categories = fp_analysis.get("category_counts", {})
    if categories:
        labels = [k.replace("_", " ").title() for k in categories.keys()]
        sizes = list(categories.values())
        colors = ["#ff6b6b", "#ffd93d", "#4ecdc4", "#45b7d1", "#96ceb4"]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors[: len(sizes)], autopct="%1.1f%%", startangle=90
        )
        ax.set_title("(a) False Positive Categories")

    # FP by config bar chart
    ax = axes[1]
    fp_by_config = fp_analysis.get("by_config", {})
    if fp_by_config:
        configs_fp = list(fp_by_config.keys())
        counts = list(fp_by_config.values())
        ax.bar(range(len(configs_fp)), counts, color="coral", alpha=0.8)
        ax.set_xticks(range(len(configs_fp)))
        ax.set_xticklabels([str(c) for c in configs_fp])
        ax.set_xlabel("Number of Papers")
        ax.set_ylabel("False Positive Count")
        ax.set_title("(b) False Positives by Paper Budget")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(figures_dir / "fig4_false_positives.png", dpi=300, bbox_inches="tight")
    plt.savefig(figures_dir / "fig4_false_positives.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fig4_false_positives.png/pdf")

    # Figure 5: GT density comparison
    _fig, ax = plt.subplots(figsize=(8, 6))

    t1_gt = [t.gt_task1_in_corpus for t in trajectories]
    t2_gt = [t.gt_task2_in_corpus for t in trajectories]

    positions = [1, 2]
    bp = ax.boxplot([t1_gt, t2_gt], positions=positions, widths=0.6, patch_artist=True)

    colors = ["steelblue", "coral"]
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Task 1: GO", "Task 2: Expression"])
    ax.set_ylabel("In-Corpus Ground Truth Annotations per Gene")
    ax.set_title("Ground Truth Density: Task 1 vs Task 2")
    ax.grid(True, alpha=0.3, axis="y")

    # Add mean markers
    means = [np.mean(t1_gt), np.mean(t2_gt)]
    ax.scatter(positions, means, marker="D", color="black", s=50, zorder=3, label="Mean")
    ax.legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "fig5_gt_density.png", dpi=300, bbox_inches="tight")
    plt.savefig(figures_dir / "fig5_gt_density.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fig5_gt_density.png/pdf")

    print(f"\nAll figures saved to {figures_dir}")
