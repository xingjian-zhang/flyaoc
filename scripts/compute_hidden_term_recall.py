#!/usr/bin/env python3
"""
Compute semantic recall@k breakdown for available vs hidden GO terms.

This script computes the semantic recall separately for:
- Available GT: GO terms that are searchable in the ontology
- Hidden GT: GO terms that were hidden from the agent's ontology search

Usage:
    uv run python scripts/compute_hidden_term_recall.py

Output:
    Prints semantic recall@k for available, hidden, and combined GT.
"""

import json
from pathlib import Path

from eval.task1_go import GOSimilarity


def main():
    # Configuration
    k = 20  # recall@k
    eval_path = Path("outputs/scaling/multi-hidden-950d202/eval_agentic_resolution.json")
    gt_path = Path("data/ground_truth_top100_hidden.jsonl")

    # Initialize similarity calculator
    print("Loading GO ontology...")
    go_sim = GOSimilarity()

    # Load ground truth
    print(f"Loading ground truth from {gt_path}...")
    gt_data = {}
    with open(gt_path) as f:
        for line in f:
            gene = json.loads(line)
            gt_data[gene["gene_id"]] = gene

    # Load eval results
    print(f"Loading eval results from {eval_path}...")
    eval_data = json.load(open(eval_path))

    # Compute recall separately for available vs hidden
    available_total_sim = 0.0
    available_count = 0
    hidden_total_sim = 0.0
    hidden_count = 0

    print(f"Computing semantic recall@{k}...")
    for gene in eval_data["genes"]:
        gene_id = gene["gene_id"]

        if gene_id not in gt_data:
            continue

        gt = gt_data[gene_id]
        task1_gt = gt.get("task1_function", [])

        # Get predictions (top k)
        matches = gene["task1_go"].get("matches", [])
        predictions = [m["predicted_go"] for m in matches[:k]]

        # Deduplicate in-corpus GT by GO ID (evaluator does this)
        seen_go_ids = {}  # go_id -> is_hidden
        for ann in task1_gt:
            if not ann.get("in_corpus", False):
                continue
            go_id = ann["go_id"]
            if go_id not in seen_go_ids:
                seen_go_ids[go_id] = ann.get("hidden", False)

        # For each unique in-corpus GT, find best similarity among predictions
        for gt_go, is_hidden in seen_go_ids.items():
            best_sim = 0.0
            for pred_go in predictions:
                sim = go_sim.compute_similarity(pred_go, gt_go)
                if sim > best_sim:
                    best_sim = sim

            if is_hidden:
                hidden_total_sim += best_sim
                hidden_count += 1
            else:
                available_total_sim += best_sim
                available_count += 1

    # Print results
    print("\n" + "=" * 50)
    print(f"Semantic Recall@{k} Breakdown (micro, deduplicated)")
    print("=" * 50)

    print(f"\nGround Truth Counts:")
    print(f"  Available GT (unique, in-corpus): {available_count}")
    print(f"  Hidden GT (unique, in-corpus): {hidden_count}")
    print(f"  Total: {available_count + hidden_count}")

    print(f"\nSemantic Recall@{k}:")
    if available_count > 0:
        available_recall = available_total_sim / available_count
        print(f"  Available: {available_recall:.1%} ({available_total_sim:.1f}/{available_count})")

    if hidden_count > 0:
        hidden_recall = hidden_total_sim / hidden_count
        print(f"  Hidden: {hidden_recall:.1%} ({hidden_total_sim:.1f}/{hidden_count})")

    combined = (available_total_sim + hidden_total_sim) / (available_count + hidden_count)
    print(f"  Combined: {combined:.1%}")

    # Compare to eval aggregate
    eval_recall = eval_data["aggregate"]["task1_go"]["recall_at_k"]["micro_semantic_at_k"]["20"]
    print(f"\nEval aggregate (for verification): {eval_recall:.1%}")

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    gap = available_recall - hidden_recall
    print(f"Gap (Available - Hidden): {gap:.1%} ({gap * 100:.1f} percentage points)")


if __name__ == "__main__":
    main()
