#!/usr/bin/env python3
"""Generate comprehensive recall@k scaling tables in Markdown.

Produces tables indexed by method (rows) × paper budget (columns),
sectioned by task, with separate tables for each k value.
Covers micro/macro averaging and in-corpus/overall ground truth scopes.

Usage:
    uv run python -m scripts.generate_scaling_tables \
        --base-dir outputs/scaling \
        --methods single-950d202 multi-950d202 pipeline-950d202 \
        --output outputs/scaling/SCALING_TABLES.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RecallAtKData:
    """Recall@k data for one method × one paper budget."""

    # Aggregate-level (pre-computed in eval_results.json)
    mean_exact_at_k: dict[str, float] = field(default_factory=dict)
    mean_semantic_at_k: dict[str, float] = field(default_factory=dict)
    micro_exact_at_k: dict[str, float] = field(default_factory=dict)
    micro_semantic_at_k: dict[str, float] = field(default_factory=dict)

    # Overall (computed from per-gene data)
    mean_exact_at_k_overall: dict[str, float] = field(default_factory=dict)
    mean_semantic_at_k_overall: dict[str, float] = field(default_factory=dict)
    micro_exact_at_k_overall: dict[str, float] = field(default_factory=dict)
    micro_semantic_at_k_overall: dict[str, float] = field(default_factory=dict)


PAPER_BUDGETS = [1, 2, 4, 8, 16]
K_VALUES = [1, 3, 5, 10, 20, 50]

# Display names for methods
METHOD_LABELS = {
    "single": "Single-Agent",
    "multi": "Multi-Agent",
    "pipeline": "Pipeline",
    "memorization": "Memorization",
}


def method_label(method_dir: str) -> str:
    """Extract a display label from a directory name like 'single-950d202'."""
    prefix = method_dir.split("-")[0]
    return METHOD_LABELS.get(prefix, method_dir)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_eval_results(path: Path) -> dict[str, Any] | None:
    """Load eval_results.json, return None if missing."""
    fp = path / "eval_results.json"
    if not fp.exists():
        return None
    with open(fp) as f:
        return json.load(f)


def compute_overall_recall_at_k(
    genes: list[dict],
    task_key: str,
    recall_at_k_key: str,
    gt_count_key: str = "gt_count",
    gt_in_corpus_key: str = "gt_in_corpus_count",
) -> RecallAtKData:
    """Compute overall recall@k from per-gene data.

    'Overall' uses gt_count (all GT items) as denominator instead of
    gt_in_corpus_count. The numerator (matches) is derived from the
    in-corpus recall@k × gt_in_corpus_count.
    """
    data = RecallAtKData()

    # Also collect per-gene values for overall computation
    per_gene_exact_overall: dict[str, list[float]] = {str(k): [] for k in K_VALUES}
    per_gene_soft_overall: dict[str, list[float]] = {str(k): [] for k in K_VALUES}
    total_matches_exact: dict[str, float] = {str(k): 0.0 for k in K_VALUES}
    total_matches_soft: dict[str, float] = {str(k): 0.0 for k in K_VALUES}
    total_gt_all = 0.0
    total_gt_in_corpus = 0.0

    for gene in genes:
        task_data = gene.get(task_key)
        if task_data is None:
            continue

        rak = task_data.get(recall_at_k_key)
        if rak is None:
            continue

        gt_all = rak.get(gt_count_key, 0)
        gt_ic = rak.get(gt_in_corpus_key, 0)
        if gt_all == 0 or gt_ic == 0:
            continue

        total_gt_all += gt_all
        total_gt_in_corpus += gt_ic

        exact = rak.get("exact_recall_at_k", {})
        soft = rak.get("semantic_recall_at_k", {})

        for k_str in [str(k) for k in K_VALUES]:
            # In-corpus matches at k
            exact_val = exact.get(k_str, 0.0)
            soft_val = soft.get(k_str, 0.0)

            matches_exact = exact_val * gt_ic
            matches_soft = soft_val * gt_ic

            total_matches_exact[k_str] += matches_exact
            total_matches_soft[k_str] += matches_soft

            # Per-gene overall recall
            overall_exact = matches_exact / gt_all if gt_all > 0 else 0.0
            overall_soft = matches_soft / gt_all if gt_all > 0 else 0.0

            per_gene_exact_overall[k_str].append(overall_exact)
            per_gene_soft_overall[k_str].append(overall_soft)

    # Compute macro and micro overall
    for k_str in [str(k) for k in K_VALUES]:
        vals_exact = per_gene_exact_overall[k_str]
        vals_soft = per_gene_soft_overall[k_str]

        data.mean_exact_at_k_overall[k_str] = (
            sum(vals_exact) / len(vals_exact) if vals_exact else 0.0
        )
        data.mean_semantic_at_k_overall[k_str] = (
            sum(vals_soft) / len(vals_soft) if vals_soft else 0.0
        )
        data.micro_exact_at_k_overall[k_str] = (
            total_matches_exact[k_str] / total_gt_all if total_gt_all > 0 else 0.0
        )
        data.micro_semantic_at_k_overall[k_str] = (
            total_matches_soft[k_str] / total_gt_all if total_gt_all > 0 else 0.0
        )

    return data


def load_method_data(
    base_dir: Path,
    method_dir: str,
) -> dict[int, dict[str, RecallAtKData]]:
    """Load recall@k data for one method across all paper budgets.

    For memorization baselines (papers_0), the same data is replicated
    across all paper budget columns.

    Returns: {papers: {task_recall_key: RecallAtKData}}
    """
    method_path = base_dir / method_dir
    result: dict[int, dict[str, RecallAtKData]] = {}

    # Check for memorization baseline (papers_0 only)
    is_memorization = (method_path / "papers_0").exists() and not (
        method_path / "papers_1"
    ).exists()

    budgets_to_load = [0] if is_memorization else PAPER_BUDGETS

    for papers in budgets_to_load:
        papers_path = method_path / f"papers_{papers}"
        eval_data = load_eval_results(papers_path)
        if eval_data is None:
            continue

        agg = eval_data.get("aggregate", {})
        genes = eval_data.get("genes", [])
        task_data: dict[str, RecallAtKData] = {}

        # Task 1: GO
        t1_rak = agg.get("task1_go", {}).get("recall_at_k", {})
        t1 = RecallAtKData(
            mean_exact_at_k=t1_rak.get("mean_exact_at_k", {}),
            mean_semantic_at_k=t1_rak.get("mean_semantic_at_k", {}),
            micro_exact_at_k=t1_rak.get("micro_exact_at_k", {}),
            micro_semantic_at_k=t1_rak.get("micro_semantic_at_k", {}),
        )
        # Compute overall from per-gene
        t1_overall = compute_overall_recall_at_k(
            genes, "task1_go", "recall_at_k"
        )
        t1.mean_exact_at_k_overall = t1_overall.mean_exact_at_k_overall
        t1.mean_semantic_at_k_overall = t1_overall.mean_semantic_at_k_overall
        t1.micro_exact_at_k_overall = t1_overall.micro_exact_at_k_overall
        t1.micro_semantic_at_k_overall = t1_overall.micro_semantic_at_k_overall
        task_data["task1_go"] = t1

        # Task 2: Expression (anatomy)
        t2_rak = agg.get("task2_expression", {}).get("anatomy_recall_at_k", {})
        t2 = RecallAtKData(
            mean_exact_at_k=t2_rak.get("mean_exact_at_k", {}),
            mean_semantic_at_k=t2_rak.get("mean_semantic_at_k", {}),
            micro_exact_at_k=t2_rak.get("micro_exact_at_k", {}),
            micro_semantic_at_k=t2_rak.get("micro_semantic_at_k", {}),
        )
        t2_overall = compute_overall_recall_at_k(
            genes, "task2_expression", "anatomy_recall_at_k"
        )
        t2.mean_exact_at_k_overall = t2_overall.mean_exact_at_k_overall
        t2.mean_semantic_at_k_overall = t2_overall.mean_semantic_at_k_overall
        t2.micro_exact_at_k_overall = t2_overall.micro_exact_at_k_overall
        t2.micro_semantic_at_k_overall = t2_overall.micro_semantic_at_k_overall
        task_data["task2_expression"] = t2

        # Task 3: Synonyms (combined)
        t3_rak = agg.get("task3_synonyms", {}).get("combined_recall_at_k", {})
        t3 = RecallAtKData(
            mean_exact_at_k=t3_rak.get("mean_exact_at_k", {}),
            mean_semantic_at_k={},  # No soft matching for synonyms
            micro_exact_at_k=t3_rak.get("micro_exact_at_k", {}),
            micro_semantic_at_k={},
        )
        t3_overall = compute_overall_recall_at_k(
            genes, "task3_synonyms", "combined_recall_at_k"
        )
        t3.mean_exact_at_k_overall = t3_overall.mean_exact_at_k_overall
        t3.micro_exact_at_k_overall = t3_overall.micro_exact_at_k_overall
        # No soft for synonyms
        task_data["task3_synonyms"] = t3

        result[papers] = task_data

    # For memorization: replicate papers_0 data across all paper budgets
    if is_memorization and 0 in result:
        base_data = result[0]
        for budget in PAPER_BUDGETS:
            result[budget] = base_data

    return result


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def fmt(val: float | None) -> str:
    """Format a float as a percentage string."""
    if val is None:
        return "—"
    return f"{val * 100:.1f}"



def generate_combined_table(
    methods: list[str],
    all_data: dict[str, dict[int, dict[str, RecallAtKData]]],
    task_key: str,
    k: int,
    macro_attr: str,
    micro_attr: str,
) -> str:
    """Generate a table with macro block (top) and micro block (bottom).

    Rows: all methods (Macro) | separator | all methods (Micro)
    Columns: paper budgets
    """
    k_str = str(k)
    header = "| Method | " + " | ".join(f"{p} papers" for p in PAPER_BUDGETS) + " |"
    sep = "|" + "|".join(["---"] * (len(PAPER_BUDGETS) + 1)) + "|"
    rows = [header, sep]

    def _add_block(attr: str) -> None:
        for method in methods:
            label = method_label(method)
            cells = []
            for papers in PAPER_BUDGETS:
                task_data = all_data.get(method, {}).get(papers, {}).get(task_key)
                if task_data is None:
                    cells.append("—")
                    continue
                metric_dict = getattr(task_data, attr, {})
                val = metric_dict.get(k_str)
                cells.append(fmt(val))
            rows.append(f"| {label} | " + " | ".join(cells) + " |")

    # Macro block
    rows.append(f"| **Macro** | | | | | |")
    _add_block(macro_attr)
    # Micro block
    rows.append(f"| **Micro** | | | | | |")
    _add_block(micro_attr)

    return "\n".join(rows)


def generate_markdown(
    methods: list[str],
    all_data: dict[str, dict[int, dict[str, RecallAtKData]]],
) -> str:
    """Generate the full markdown document with all tables."""
    lines: list[str] = []
    lines.append("# Scaling Experiment: Recall@k Tables")
    lines.append("")
    lines.append("Tables indexed by **method** (rows) × **paper budget** (columns).")
    lines.append("Values are percentages (%).")
    lines.append("")
    lines.append("- **Macro**: per-gene average (each gene weighted equally)")
    lines.append("- **Micro**: pooled across genes (genes with more GT contribute more)")
    lines.append("- **In-corpus**: recall against GT items findable in the literature corpus")
    lines.append(
        "- **Overall**: recall against all GT items "
        "(includes annotations not in corpus; lower ceiling)"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Task configs: (title, key, has_semantic, include_exact)
    task_configs = [
        ("Task 1: GO Function Annotations", "task1_go", True, False),
        ("Task 2: Expression (Anatomy)", "task2_expression", True, False),
        ("Task 3: Synonyms (Combined)", "task3_synonyms", False, True),
    ]

    for task_title, task_key, has_semantic, include_exact in task_configs:
        lines.append(f"## {task_title}")
        lines.append("")

        for k in K_VALUES:
            lines.append(f"### k = {k}")
            lines.append("")

            if include_exact:
                # --- Exact Match, In-Corpus ---
                lines.append("#### Exact Recall@k — In-Corpus")
                lines.append("")
                lines.append(
                    generate_combined_table(
                        methods, all_data, task_key, k,
                        macro_attr="mean_exact_at_k",
                        micro_attr="micro_exact_at_k",
                    )
                )
                lines.append("")

                # --- Exact Match, Overall ---
                lines.append("#### Exact Recall@k — Overall")
                lines.append("")
                lines.append(
                    generate_combined_table(
                        methods, all_data, task_key, k,
                        macro_attr="mean_exact_at_k_overall",
                        micro_attr="micro_exact_at_k_overall",
                    )
                )
                lines.append("")

            if has_semantic:
                # --- Semantic, In-Corpus ---
                lines.append("#### Semantic Recall@k — In-Corpus")
                lines.append("")
                lines.append(
                    generate_combined_table(
                        methods, all_data, task_key, k,
                        macro_attr="mean_semantic_at_k",
                        micro_attr="micro_semantic_at_k",
                    )
                )
                lines.append("")

                # --- Semantic, Overall ---
                lines.append("#### Semantic Recall@k — Overall")
                lines.append("")
                lines.append(
                    generate_combined_table(
                        methods, all_data, task_key, k,
                        macro_attr="mean_semantic_at_k_overall",
                        micro_attr="micro_semantic_at_k_overall",
                    )
                )
                lines.append("")

            lines.append("---")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate recall@k scaling tables in Markdown."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/scaling"),
        help="Base directory containing method experiment folders.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "single-950d202",
            "multi-950d202",
            "pipeline-950d202",
            "memorization-950d202",
        ],
        help="Method directory names to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown file path. Prints to stdout if not specified.",
    )
    args = parser.parse_args()

    # Load all data
    all_data: dict[str, dict[int, dict[str, RecallAtKData]]] = {}
    for method in args.methods:
        method_path = args.base_dir / method
        if not method_path.exists():
            print(f"Warning: {method_path} does not exist, skipping.", file=sys.stderr)
            continue
        all_data[method] = load_method_data(args.base_dir, method)

    if not all_data:
        print("Error: No data loaded.", file=sys.stderr)
        sys.exit(1)

    # Generate markdown
    md = generate_markdown(args.methods, all_data)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(md)


if __name__ == "__main__":
    main()
