#!/usr/bin/env python3
"""Compare Claude Code baseline results against other FlyBench baselines.

Usage:
    python3 baselines/claude_code/compare_results.py \
        --cc-eval outputs/claude_code_p16/eval_results.json

    # Custom baseline paths
    python3 baselines/claude_code/compare_results.py \
        --cc-eval outputs/claude_code_p16/eval_results.json \
        --multi-eval outputs/scaling/multi-950d202/papers_16/eval_results.json \
        --single-eval outputs/scaling/single-950d202/papers_16/eval_results.json \
        --pipeline-eval outputs/scaling/pipeline-950d202/papers_16/eval_results.json \
        --memorization-eval outputs/scaling/memorization-950d202/papers_0/eval_results.json
"""

import argparse
import json
import sys
from pathlib import Path

# Default eval result paths (relative to drosophila-agent-benchmark/)
DEFAULT_BASELINES = {
    "Multi-Agent": "outputs/scaling/multi-950d202/papers_16/eval_results.json",
    "Single-Agent": "outputs/scaling/single-950d202/papers_16/eval_results.json",
    "Pipeline": "outputs/scaling/pipeline-950d202/papers_16/eval_results.json",
    "Memorization": "outputs/scaling/memorization-950d202/papers_0/eval_results.json",
}


def load_eval_results(path: str) -> dict:
    """Load eval_results.json and return {gene_id: gene_data} dict."""
    with open(path) as f:
        data = json.load(f)
    return {g["gene_id"]: g for g in data["genes"]}


def extract_metrics(gene_data: dict) -> dict:
    """Extract the three primary metrics from a gene's eval result."""
    go = gene_data.get("task1_go", {})
    expr = gene_data.get("task2_expression", {})
    syn = gene_data.get("task3_synonyms", {})

    go_score = (
        go.get("recall_at_k", {})
        .get("semantic_recall_at_k", {})
        .get("20", 0.0)
    )
    expr_score = (
        expr.get("anatomy_recall_at_k", {})
        .get("semantic_recall_at_k", {})
        .get("10", 0.0)
    )
    syn_score = (
        syn.get("combined_recall_at_k", {})
        .get("exact_recall_at_k", {})
        .get("20", 0.0)
    )

    avg = (go_score + expr_score + syn_score) / 3
    return {"GO": go_score, "Expr": expr_score, "Syn": syn_score, "Avg": avg}


def print_head_to_head(cc_results: dict, multi_results: dict, gene_ids: list[str]):
    """Print per-gene CC vs Multi-Agent comparison."""
    # Map gene_id -> symbol from CC results
    id_to_sym = {}
    for gid in gene_ids:
        if gid in cc_results:
            id_to_sym[gid] = cc_results[gid].get("gene_symbol", gid)
        elif gid in multi_results:
            id_to_sym[gid] = multi_results[gid].get("gene_symbol", gid)
        else:
            id_to_sym[gid] = gid

    print("\n" + "=" * 90)
    print("PER-GENE COMPARISON: Claude Code vs Multi-Agent (GPT-5-mini)")
    print("=" * 90)
    header = f"{'Gene':<12} {'CC GO':>7} {'MA GO':>7}  {'CC Expr':>8} {'MA Expr':>8}  {'CC Syn':>7} {'MA Syn':>7}  {'CC Avg':>7} {'MA Avg':>7}  {'Δ Avg':>7}"
    print(header)
    print("-" * 90)

    cc_totals = {"GO": 0, "Expr": 0, "Syn": 0, "Avg": 0}
    ma_totals = {"GO": 0, "Expr": 0, "Syn": 0, "Avg": 0}
    n = 0

    for gid in sorted(gene_ids, key=lambda x: id_to_sym.get(x, x)):
        sym = id_to_sym[gid]
        cc_m = extract_metrics(cc_results[gid]) if gid in cc_results else None
        ma_m = extract_metrics(multi_results[gid]) if gid in multi_results else None

        if cc_m is None or ma_m is None:
            continue

        delta = cc_m["Avg"] - ma_m["Avg"]
        delta_str = f"{delta:+.3f}"

        print(
            f"{sym:<12} {cc_m['GO']:7.3f} {ma_m['GO']:7.3f}  "
            f"{cc_m['Expr']:8.3f} {ma_m['Expr']:8.3f}  "
            f"{cc_m['Syn']:7.3f} {ma_m['Syn']:7.3f}  "
            f"{cc_m['Avg']:7.3f} {ma_m['Avg']:7.3f}  {delta_str:>7}"
        )

        for k in cc_totals:
            cc_totals[k] += cc_m[k]
            ma_totals[k] += ma_m[k]
        n += 1

    if n > 0:
        print("-" * 90)
        cc_avg = {k: v / n for k, v in cc_totals.items()}
        ma_avg = {k: v / n for k, v in ma_totals.items()}
        delta = cc_avg["Avg"] - ma_avg["Avg"]
        print(
            f"{'Macro Avg':<12} {cc_avg['GO']:7.3f} {ma_avg['GO']:7.3f}  "
            f"{cc_avg['Expr']:8.3f} {ma_avg['Expr']:8.3f}  "
            f"{cc_avg['Syn']:7.3f} {ma_avg['Syn']:7.3f}  "
            f"{cc_avg['Avg']:7.3f} {ma_avg['Avg']:7.3f}  {delta:+.3f}"
        )
        print(f"\nCC wins on {sum(1 for gid in gene_ids if gid in cc_results and gid in multi_results and extract_metrics(cc_results[gid])['Avg'] > extract_metrics(multi_results[gid])['Avg'])}/{n} genes")


def print_all_methods(
    all_results: dict[str, dict],
    gene_ids: list[str],
):
    """Print aggregate comparison across all methods, filtered to the same gene set."""
    print("\n" + "=" * 70)
    print(f"AGGREGATE COMPARISON (macro-averaged over {len(gene_ids)} genes)")
    print("=" * 70)
    header = f"{'Method':<20} {'GO':>7} {'Expr':>7} {'Syn':>7} {'Avg':>7}"
    print(header)
    print("-" * 70)

    method_avgs = []
    for method_name, results in all_results.items():
        totals = {"GO": 0, "Expr": 0, "Syn": 0, "Avg": 0}
        n = 0
        for gid in gene_ids:
            if gid in results:
                m = extract_metrics(results[gid])
                for k in totals:
                    totals[k] += m[k]
                n += 1

        if n == 0:
            print(f"{method_name:<20} {'(no data)':>7}")
            continue

        avg = {k: v / n for k, v in totals.items()}
        method_avgs.append((method_name, avg, n))

    # Sort by Avg descending
    method_avgs.sort(key=lambda x: x[1]["Avg"], reverse=True)

    for method_name, avg, n in method_avgs:
        genes_note = f" ({n}g)" if n < len(gene_ids) else ""
        print(
            f"{method_name:<20} {avg['GO']:7.3f} {avg['Expr']:7.3f} "
            f"{avg['Syn']:7.3f} {avg['Avg']:7.3f}{genes_note}"
        )

    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Claude Code baseline against other FlyBench baselines"
    )
    parser.add_argument(
        "--cc-eval",
        type=str,
        required=True,
        help="Path to Claude Code eval_results.json",
    )
    parser.add_argument(
        "--multi-eval",
        type=str,
        default=DEFAULT_BASELINES["Multi-Agent"],
        help="Path to Multi-Agent eval_results.json",
    )
    parser.add_argument(
        "--single-eval",
        type=str,
        default=DEFAULT_BASELINES["Single-Agent"],
        help="Path to Single-Agent eval_results.json",
    )
    parser.add_argument(
        "--pipeline-eval",
        type=str,
        default=DEFAULT_BASELINES["Pipeline"],
        help="Path to Pipeline eval_results.json",
    )
    parser.add_argument(
        "--memorization-eval",
        type=str,
        default=DEFAULT_BASELINES["Memorization"],
        help="Path to Memorization eval_results.json",
    )
    args = parser.parse_args()

    # Load CC results
    cc_path = Path(args.cc_eval)
    if not cc_path.exists():
        print(f"Error: CC eval results not found: {cc_path}", file=sys.stderr)
        sys.exit(1)
    cc_results = load_eval_results(args.cc_eval)
    cc_gene_ids = sorted(cc_results.keys())
    print(f"Claude Code: {len(cc_gene_ids)} genes evaluated")

    # Load baseline results
    all_results = {"Claude Code": cc_results}
    baseline_args = {
        "Multi-Agent": args.multi_eval,
        "Single-Agent": args.single_eval,
        "Pipeline": args.pipeline_eval,
        "Memorization": args.memorization_eval,
    }
    for name, path in baseline_args.items():
        p = Path(path)
        if p.exists():
            all_results[name] = load_eval_results(path)
            print(f"{name}: {len(all_results[name])} genes loaded")
        else:
            print(f"{name}: not found ({p}), skipping")

    # Head-to-head: CC vs Multi-Agent
    if "Multi-Agent" in all_results:
        print_head_to_head(cc_results, all_results["Multi-Agent"], cc_gene_ids)

    # Aggregate: all methods on the CC gene set
    print_all_methods(all_results, cc_gene_ids)

    # Print CC per-gene detail
    print("\n" + "=" * 60)
    print("CLAUDE CODE PER-GENE DETAIL")
    print("=" * 60)
    header = f"{'Gene':<12} {'GO':>7} {'Expr':>7} {'Syn':>7} {'Avg':>7}"
    print(header)
    print("-" * 60)
    for gid in sorted(cc_gene_ids, key=lambda x: cc_results[x].get("gene_symbol", x)):
        sym = cc_results[gid].get("gene_symbol", gid)
        m = extract_metrics(cc_results[gid])
        print(f"{sym:<12} {m['GO']:7.3f} {m['Expr']:7.3f} {m['Syn']:7.3f} {m['Avg']:7.3f}")


if __name__ == "__main__":
    main()
