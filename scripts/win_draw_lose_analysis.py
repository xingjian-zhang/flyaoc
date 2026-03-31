#!/usr/bin/env python3
"""Win/Draw/Lose Analysis: Compare single-agent vs multi-agent at 16 paper budget.

For each ground truth annotation, computes which method found a better match
using semantic similarity (Tasks 1, 2) or exact match (Task 3).
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.task1_go import GOSimilarity
from eval.task2_expression import AnatomySimilarity


# =============================================================================
# Configuration
# =============================================================================

# k values per task (matching recall@k settings)
K_VALUES = {
    1: 20,  # GO terms
    2: 10,  # Expression
    3: 20,  # Synonyms (combined fullname + symbol)
}

# Draw threshold - scores within this range are considered ties
DEFAULT_THRESHOLD = 0.01


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Comparison:
    """Result of comparing single vs multi for one GT annotation."""

    gene_id: str
    task_id: int
    gt_item: dict  # Ground truth annotation
    single_score: float
    multi_score: float
    outcome: str  # "single_wins", "multi_wins", or "draw"


@dataclass
class TaskStats:
    """Aggregated stats for a task."""

    single_wins: int = 0
    multi_wins: int = 0
    draws: int = 0
    comparisons: list = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.single_wins + self.multi_wins + self.draws

    def pct(self, value: int) -> float:
        return 100.0 * value / self.total if self.total > 0 else 0.0


# =============================================================================
# Helper Functions
# =============================================================================


def normalize_synonym(s: str) -> str:
    """Normalize synonym for case-insensitive comparison."""
    return s.strip().lower()


def extract_gt_items(gt: dict, task_id: int, in_corpus_only: bool = True) -> list[dict]:
    """Extract ground truth items for a task, optionally filtering to in_corpus only."""
    if task_id == 1:
        items = gt.get("task1_function", [])
        if in_corpus_only:
            return [item for item in items if item.get("in_corpus", False)]
        return items
    elif task_id == 2:
        items = gt.get("task2_expression", [])
        if in_corpus_only:
            return [item for item in items if item.get("in_corpus", False)]
        return items
    elif task_id == 3:
        # For task 3, combine fullname and symbol synonyms
        synonyms = gt.get("task3_synonyms", {})
        items = []
        for syn in synonyms.get("fullname_synonyms", []):
            if isinstance(syn, dict):
                if in_corpus_only and not syn.get("in_corpus", False):
                    continue
                items.append({"synonym": syn.get("synonym", ""), "type": "fullname"})
            else:
                items.append({"synonym": syn, "type": "fullname"})
        for syn in synonyms.get("symbol_synonyms", []):
            if isinstance(syn, dict):
                if in_corpus_only and not syn.get("in_corpus", False):
                    continue
                items.append({"synonym": syn.get("synonym", ""), "type": "symbol"})
            else:
                items.append({"synonym": syn, "type": "symbol"})
        return items
    return []


def load_predictions(pred: dict, task_id: int, k: int) -> list:
    """Load top-k predictions for a task from prediction dict."""
    output = pred.get("output", pred)  # Handle both formats
    if task_id == 1:
        return output.get("task1_function", [])[:k]
    elif task_id == 2:
        return output.get("task2_expression", [])[:k]
    elif task_id == 3:
        # Combine fullname and symbol synonyms for task 3
        synonyms = output.get("task3_synonyms", {})
        combined = []
        seen = set()
        for s in synonyms.get("fullname_synonyms", []):
            norm = normalize_synonym(s)
            if norm not in seen:
                combined.append({"synonym": s, "type": "fullname"})
                seen.add(norm)
        for s in synonyms.get("symbol_synonyms", []):
            norm = normalize_synonym(s)
            if norm not in seen:
                combined.append({"synonym": s, "type": "symbol"})
                seen.add(norm)
        return combined[:k]
    return []


def best_similarity_task1(
    predictions: list[dict], gt_item: dict, go_sim: GOSimilarity
) -> float:
    """Find best similarity between any prediction and the GT GO term."""
    gt_go_id = gt_item.get("go_id", "")
    if not gt_go_id or not predictions:
        return 0.0

    best = 0.0
    for pred in predictions:
        pred_go_id = pred.get("go_id", "")
        if pred_go_id:
            sim = go_sim.compute_similarity(pred_go_id, gt_go_id)
            if sim > best:
                best = sim
    return best


def best_similarity_task2(
    predictions: list[dict], gt_item: dict, anatomy_sim: AnatomySimilarity
) -> float:
    """Find best similarity between any prediction and the GT anatomy term."""
    gt_anatomy_id = gt_item.get("anatomy_id", "")
    if not gt_anatomy_id or not predictions:
        return 0.0

    best = 0.0
    for pred in predictions:
        pred_anatomy_id = pred.get("anatomy_id", "")
        if pred_anatomy_id:
            sim = anatomy_sim.compute_similarity(pred_anatomy_id, gt_anatomy_id)
            if sim > best:
                best = sim
    return best


def best_similarity_task3(predictions: list[dict], gt_item: dict) -> float:
    """Check if any prediction exactly matches the GT synonym (case-insensitive)."""
    gt_synonym = normalize_synonym(gt_item.get("synonym", ""))
    if not gt_synonym:
        return 0.0

    for pred in predictions:
        pred_synonym = normalize_synonym(pred.get("synonym", ""))
        if pred_synonym == gt_synonym:
            return 1.0
    return 0.0


def determine_outcome(
    single_score: float, multi_score: float, threshold: float
) -> str:
    """Determine win/draw/lose outcome."""
    if single_score > multi_score + threshold:
        return "single_wins"
    elif multi_score > single_score + threshold:
        return "multi_wins"
    else:
        return "draw"


# =============================================================================
# Main Analysis
# =============================================================================


def run_analysis(
    single_dir: Path,
    multi_dir: Path,
    gt_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
    verbose: bool = False,
) -> dict[int, TaskStats]:
    """Run the win/draw/lose analysis.

    Args:
        single_dir: Path to single-agent predictions directory
        multi_dir: Path to multi-agent predictions directory
        gt_path: Path to ground truth JSONL file
        threshold: Draw threshold
        verbose: Print per-gene details

    Returns:
        Dict mapping task_id to TaskStats
    """
    # Load ground truth
    gt_by_gene = {}
    with open(gt_path) as f:
        for line in f:
            gt = json.loads(line)
            gt_by_gene[gt["gene_id"]] = gt

    print(f"Loaded {len(gt_by_gene)} genes from ground truth")

    # Initialize similarity calculators
    print("Initializing GO similarity calculator...")
    go_sim = GOSimilarity()
    print("Initializing Anatomy similarity calculator...")
    anatomy_sim = AnatomySimilarity()

    # Initialize stats
    stats: dict[int, TaskStats] = {1: TaskStats(), 2: TaskStats(), 3: TaskStats()}

    # Get common genes
    single_genes = {
        p.stem for p in single_dir.glob("FBgn*.json") if p.stem != "eval_results"
    }
    multi_genes = {
        p.stem for p in multi_dir.glob("FBgn*.json") if p.stem != "eval_results"
    }
    common_genes = single_genes & multi_genes & set(gt_by_gene.keys())
    print(f"Found {len(common_genes)} common genes to compare")

    # Process each gene
    for gene_id in sorted(common_genes):
        gt = gt_by_gene[gene_id]

        # Load predictions
        single_pred_path = single_dir / f"{gene_id}.json"
        multi_pred_path = multi_dir / f"{gene_id}.json"

        with open(single_pred_path) as f:
            single_pred = json.load(f)
        with open(multi_pred_path) as f:
            multi_pred = json.load(f)

        if verbose:
            print(f"\n{gene_id}:")

        # Compare each task
        for task_id, k in K_VALUES.items():
            gt_items = extract_gt_items(gt, task_id, in_corpus_only=True)
            single_preds = load_predictions(single_pred, task_id, k)
            multi_preds = load_predictions(multi_pred, task_id, k)

            if verbose and gt_items:
                print(f"  Task {task_id}: {len(gt_items)} GT items, "
                      f"{len(single_preds)} single preds, {len(multi_preds)} multi preds")

            for gt_item in gt_items:
                # Compute best similarity for each method
                if task_id == 1:
                    single_score = best_similarity_task1(single_preds, gt_item, go_sim)
                    multi_score = best_similarity_task1(multi_preds, gt_item, go_sim)
                elif task_id == 2:
                    single_score = best_similarity_task2(single_preds, gt_item, anatomy_sim)
                    multi_score = best_similarity_task2(multi_preds, gt_item, anatomy_sim)
                else:  # task_id == 3
                    single_score = best_similarity_task3(single_preds, gt_item)
                    multi_score = best_similarity_task3(multi_preds, gt_item)

                # Determine outcome
                outcome = determine_outcome(single_score, multi_score, threshold)

                # Record comparison
                comparison = Comparison(
                    gene_id=gene_id,
                    task_id=task_id,
                    gt_item=gt_item,
                    single_score=single_score,
                    multi_score=multi_score,
                    outcome=outcome,
                )
                stats[task_id].comparisons.append(comparison)

                # Update counts
                if outcome == "single_wins":
                    stats[task_id].single_wins += 1
                elif outcome == "multi_wins":
                    stats[task_id].multi_wins += 1
                else:
                    stats[task_id].draws += 1

    return stats


def print_summary(stats: dict[int, TaskStats], threshold: float):
    """Print summary table of results."""
    task_names = {
        1: "GO Functions",
        2: "Expression",
        3: "Synonyms",
    }

    print("\n" + "=" * 80)
    print("WIN/DRAW/LOSE ANALYSIS: Single-Agent vs Multi-Agent (16 papers)")
    print(f"Draw threshold: {threshold}")
    print("=" * 80)

    # Header
    print(f"\n{'Task':<20} {'Total':<8} {'Single Wins':<15} {'Multi Wins':<15} {'Draws':<15}")
    print("-" * 73)

    totals = TaskStats()

    for task_id in [1, 2, 3]:
        s = stats[task_id]
        name = task_names[task_id]
        print(
            f"{name:<20} {s.total:<8} "
            f"{s.single_wins:>4} ({s.pct(s.single_wins):>5.1f}%)   "
            f"{s.multi_wins:>4} ({s.pct(s.multi_wins):>5.1f}%)   "
            f"{s.draws:>4} ({s.pct(s.draws):>5.1f}%)"
        )
        totals.single_wins += s.single_wins
        totals.multi_wins += s.multi_wins
        totals.draws += s.draws

    print("-" * 73)
    print(
        f"{'TOTAL':<20} {totals.total:<8} "
        f"{totals.single_wins:>4} ({totals.pct(totals.single_wins):>5.1f}%)   "
        f"{totals.multi_wins:>4} ({totals.pct(totals.multi_wins):>5.1f}%)   "
        f"{totals.draws:>4} ({totals.pct(totals.draws):>5.1f}%)"
    )
    print()


def analyze_score_differences(stats: dict[int, TaskStats]):
    """Analyze the score differences for wins."""
    print("\n" + "=" * 80)
    print("SCORE DIFFERENCE ANALYSIS")
    print("=" * 80)

    for task_id in [1, 2, 3]:
        s = stats[task_id]
        single_diffs = []
        multi_diffs = []

        for c in s.comparisons:
            diff = c.single_score - c.multi_score
            if c.outcome == "single_wins":
                single_diffs.append(diff)
            elif c.outcome == "multi_wins":
                multi_diffs.append(-diff)  # Make positive

        print(f"\nTask {task_id}:")
        if single_diffs:
            avg_diff = sum(single_diffs) / len(single_diffs)
            print(f"  Single wins: avg margin = {avg_diff:.3f} (n={len(single_diffs)})")
        if multi_diffs:
            avg_diff = sum(multi_diffs) / len(multi_diffs)
            print(f"  Multi wins:  avg margin = {avg_diff:.3f} (n={len(multi_diffs)})")


def save_details_csv(stats: dict[int, TaskStats], output_path: Path):
    """Save per-annotation details to CSV."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gene_id", "task_id", "gt_item", "single_score", "multi_score",
            "score_diff", "outcome"
        ])

        for task_id in [1, 2, 3]:
            for c in stats[task_id].comparisons:
                writer.writerow([
                    c.gene_id,
                    c.task_id,
                    json.dumps(c.gt_item),
                    f"{c.single_score:.4f}",
                    f"{c.multi_score:.4f}",
                    f"{c.single_score - c.multi_score:.4f}",
                    c.outcome,
                ])

    print(f"Saved detailed results to: {output_path}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Compare single-agent vs multi-agent predictions using win/draw/lose analysis"
    )
    parser.add_argument(
        "--single-dir",
        type=Path,
        default=Path("outputs/scaling/single-950d202/papers_16"),
        help="Path to single-agent predictions directory",
    )
    parser.add_argument(
        "--multi-dir",
        type=Path,
        default=Path("outputs/scaling/multi-950d202/papers_16"),
        help="Path to multi-agent predictions directory",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path("data/ground_truth_top100.jsonl"),
        help="Path to ground truth JSONL file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Draw threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to save per-annotation details CSV",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-gene details",
    )
    parser.add_argument(
        "--analyze-diffs",
        action="store_true",
        help="Print score difference analysis",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.single_dir.exists():
        print(f"Error: Single-agent directory not found: {args.single_dir}")
        return 1
    if not args.multi_dir.exists():
        print(f"Error: Multi-agent directory not found: {args.multi_dir}")
        return 1
    if not args.gt.exists():
        print(f"Error: Ground truth file not found: {args.gt}")
        return 1

    # Run analysis
    stats = run_analysis(
        single_dir=args.single_dir,
        multi_dir=args.multi_dir,
        gt_path=args.gt,
        threshold=args.threshold,
        verbose=args.verbose,
    )

    # Print results
    print_summary(stats, args.threshold)

    if args.analyze_diffs:
        analyze_score_differences(stats)

    if args.output_csv:
        save_details_csv(stats, args.output_csv)

    return 0


if __name__ == "__main__":
    exit(main())
