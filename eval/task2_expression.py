"""Task 2: Expression evaluation with anatomy semantic similarity.

This module provides:
- AnatomySimilarity class for computing Wang semantic similarity between FBbt anatomy terms
- Expression annotation evaluation with exact and semantic metrics
"""

import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from goatools.obo_parser import GODag
from goatools.semsim.termwise.wang import SsWang

from eval.recall_at_k import RecallAtKResult, compute_recall_at_k_series

# =============================================================================
# Anatomy Similarity (merged from anatomy_similarity.py)
# =============================================================================


class AnatomySimilarity:
    """Compute semantic similarity between FBbt anatomy terms using Wang's method."""

    def __init__(self, obo_path: str = "ontologies/fly_anatomy.obo"):
        """Initialize with FBbt ontology.

        Args:
            obo_path: Path to fly_anatomy.obo file.
        """
        path = Path(obo_path)
        if not path.exists():
            raise FileNotFoundError(f"Anatomy ontology file not found: {path}")

        # Create filtered OBO file (removes GCI annotations that cause cycles)
        filtered_path = self._create_filtered_obo(path)
        self.godag = GODag(filtered_path, optional_attrs={"relationship"})
        self._sim_cache: dict[tuple[str, str], float] = {}

    def _create_filtered_obo(self, path: Path) -> str:
        """Create a filtered OBO file without GCI annotations.

        GCI (General Concept Inclusions) are conditional is_a relationships
        that create cycles in the hierarchy. We filter them out for
        similarity calculations.

        Args:
            path: Path to original OBO file.

        Returns:
            Path to filtered temporary OBO file.
        """
        with open(path) as f:
            content = f.read()

        lines = ["format-version: 1.2", "ontology: fbbt_filtered", ""]
        in_term = False
        current_lines: list[str] = []
        is_obsolete = False

        for line in content.split("\n"):
            stripped = line.strip()

            if stripped == "[Term]":
                # Save previous term if valid
                if in_term and current_lines and not is_obsolete:
                    if any("id: FBbt:" in l for l in current_lines):
                        lines.extend(current_lines)
                        lines.append("")
                current_lines = ["[Term]"]
                in_term = True
                is_obsolete = False

            elif stripped.startswith("[") and stripped.endswith("]"):
                # End of Term section (new stanza type)
                if in_term and current_lines and not is_obsolete:
                    if any("id: FBbt:" in l for l in current_lines):
                        lines.extend(current_lines)
                        lines.append("")
                in_term = False
                current_lines = []
                is_obsolete = False

            elif in_term:
                if stripped.startswith("is_obsolete: true"):
                    is_obsolete = True
                elif (
                    stripped.startswith("id: FBbt:")
                    or stripped.startswith("name:")
                    or stripped.startswith("namespace:")
                ):
                    current_lines.append(line)
                elif stripped.startswith("is_a: FBbt:") and "{" not in stripped:
                    # Only keep unconditional is_a relationships (no GCI)
                    current_lines.append(line)

        # Write to temporary file
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".obo", delete=False, prefix="fbbt_filtered_"
        )
        tmp.write("\n".join(lines))
        tmp.close()
        return tmp.name

    @lru_cache(maxsize=50000)
    def compute_similarity(self, term_a: str, term_b: str) -> float:
        """Compute Wang semantic similarity between two anatomy terms.

        Args:
            term_a: First FBbt term ID (e.g., "FBbt:00004729").
            term_b: Second FBbt term ID.

        Returns:
            Similarity score in [0, 1]. Returns 0 if terms not found.
        """
        # Exact match
        if term_a == term_b:
            return 1.0

        # Check both terms exist
        if term_a not in self.godag or term_b not in self.godag:
            return 0.0

        # Use Wang similarity
        try:
            wang = SsWang({term_a, term_b}, self.godag)
            sim = wang.get_sim(term_a, term_b)
            return sim if sim is not None else 0.0
        except Exception:
            # Fall back to 0 on any error
            return 0.0

    def find_best_match(
        self,
        query_term: str,
        candidates: list[str],
        threshold: float = 0.0,
    ) -> tuple[str | None, float]:
        """Find the best matching anatomy term from candidates.

        Args:
            query_term: The FBbt term to match.
            candidates: List of candidate FBbt terms.
            threshold: Minimum similarity to consider a match.

        Returns:
            Tuple of (best_match_id, similarity) or (None, best_sim) if no match above threshold.
        """
        best_match = None
        best_sim = 0.0

        for candidate in candidates:
            sim = self.compute_similarity(query_term, candidate)
            if sim > best_sim:
                best_sim = sim
                best_match = candidate

        if best_sim >= threshold:
            return best_match, best_sim
        return None, best_sim  # Return actual similarity for FP analysis


# Global instance for reuse
_anatomy_sim_instance: AnatomySimilarity | None = None


def get_anatomy_similarity(
    obo_path: str = "ontologies/fly_anatomy.obo",
) -> AnatomySimilarity:
    """Get or create global AnatomySimilarity instance."""
    global _anatomy_sim_instance
    if _anatomy_sim_instance is None:
        _anatomy_sim_instance = AnatomySimilarity(obo_path)
    return _anatomy_sim_instance


# =============================================================================
# Metrics and Result Classes
# =============================================================================


@dataclass
class Metrics:
    """Precision, recall, and F1 metrics."""

    precision: float
    recall: float
    f1: float
    tp: int  # True positives
    fp: int  # False positives
    fn: int  # False negatives

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def precision_recall_f1(tp: int, fp: int, fn: int) -> Metrics:
    """Calculate precision, recall, and F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return Metrics(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


@dataclass
class SemanticMetrics:
    """Semantic precision, recall, F1.

    Instead of binary TP counting, uses sum of similarities as weighted TP.
    This provides partial credit for semantically similar but non-exact matches.
    """

    precision: float
    recall: float
    f1: float
    semantic_tp: float  # Sum of similarities for matched GT items (for recall)
    precision_sum: float = 0.0  # Sum of best similarities for predictions (for micro precision)

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "semantic_tp": round(self.semantic_tp, 4),
        }


@dataclass
class Task2Result:
    """Results from Task 2 expression evaluation."""

    anatomy_metrics: Metrics  # Exact matching
    anatomy_semantic: SemanticMetrics  # Semantic metrics
    stage_metrics: Metrics
    tuple_metrics: Metrics  # Exact (anatomy, stage) tuple matching
    in_corpus_exact_recall: float  # Exact recall against in-corpus GT
    full_recall: float  # Exact recall against all GT
    full_semantic_recall: float  # Semantic recall against all GT
    predicted_count: int = 0
    gt_total_count: int = 0
    gt_in_corpus_count: int = 0
    anatomy_recall_at_k: RecallAtKResult | None = None  # Recall@k for anatomy terms

    def to_dict(self) -> dict:
        result = {
            "anatomy": self.anatomy_metrics.to_dict(),
            "anatomy_semantic": self.anatomy_semantic.to_dict(),
            "stage": self.stage_metrics.to_dict(),
            "tuple": self.tuple_metrics.to_dict(),
            "in_corpus_exact_recall": round(self.in_corpus_exact_recall, 4),
            "full_recall": round(self.full_recall, 4),
            "full_semantic_recall": round(self.full_semantic_recall, 4),
            "predicted_count": self.predicted_count,
            "gt_total_count": self.gt_total_count,
            "gt_in_corpus_count": self.gt_in_corpus_count,
        }
        if self.anatomy_recall_at_k is not None:
            result["anatomy_recall_at_k"] = self.anatomy_recall_at_k.to_dict()
        return result


# =============================================================================
# Helper Functions
# =============================================================================


def _stage_in_range(
    stage_id: str | None,
    start_id: str | None,
    end_id: str | None,
) -> bool:
    """Check if stage_id falls within the ground truth range.

    For simplicity, we check:
    - If no range (end_id is None), exact match on start_id
    - If range, check if stage_id matches either start or end (conservative)

    Note: A full implementation would parse FBdv IDs to compare stage numbers.
    """
    if stage_id is None:
        return False

    # If no start, can't match
    if start_id is None:
        return False

    # Exact match to start
    if stage_id == start_id:
        return True

    # If there's an end, check that too
    if end_id is not None and stage_id == end_id:
        return True

    # Simple heuristic: extract stage numbers and compare
    # FBdv:00005327 -> embryonic stage 12
    # For now, just check if the IDs are close numerically
    try:
        pred_num = int(stage_id.split(":")[1])
        start_num = int(start_id.split(":")[1])
        end_num = int(end_id.split(":")[1]) if end_id else start_num

        return start_num <= pred_num <= end_num
    except (ValueError, IndexError):
        return False


# =============================================================================
# Main Evaluation Function
# =============================================================================


def evaluate_expression(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> Task2Result:
    """Evaluate expression annotations.

    Computes metrics with both exact and semantic matching:
    - Anatomy-only: Match if anatomy_id matches (exact) or similar (semantic)
    - Stage-only: Match if stage_id falls within ground truth range
    - Tuple: Match if both anatomy AND stage match (exact only)

    Args:
        predicted: List of predicted expression records from agent (order matters for recall@k).
        ground_truth: List of ground truth expression records.

    Returns:
        Task2Result with anatomy, stage, and tuple metrics.
    """
    # Get anatomy similarity calculator
    anatomy_sim = get_anatomy_similarity()

    # Extract ordered list of anatomy IDs (preserving ranking order, deduplicating)
    seen_anatomy: set[str] = set()
    pred_anatomy_list: list[str] = []
    for p in predicted:
        anatomy_id = p.get("anatomy_id")
        if anatomy_id and anatomy_id not in seen_anatomy:
            pred_anatomy_list.append(anatomy_id)
            seen_anatomy.add(anatomy_id)

    # Extract predicted sets (for existing metrics)
    pred_anatomy = set(pred_anatomy_list)
    pred_stage = {p["stage_id"] for p in predicted if p.get("stage_id")}
    pred_tuples = {
        (p.get("anatomy_id"), p.get("stage_id")) for p in predicted if p.get("anatomy_id")
    }

    # Separate ground truth by in_corpus
    gt_all_anatomy = {gt["anatomy_id"] for gt in ground_truth if gt.get("anatomy_id")}
    gt_in_corpus_anatomy = {gt["anatomy_id"] for gt in ground_truth if gt.get("in_corpus", False)}

    gt_in_corpus_stage = {gt["stage_start_id"] for gt in ground_truth if gt.get("in_corpus", False)}

    # Build GT tuples for in-corpus
    gt_in_corpus_tuples = set()
    gt_in_corpus_records = [gt for gt in ground_truth if gt.get("in_corpus", False)]
    for gt in gt_in_corpus_records:
        anatomy = gt.get("anatomy_id")
        stage_start = gt.get("stage_start_id")
        if anatomy:
            gt_in_corpus_tuples.add((anatomy, stage_start))

    # --- Anatomy evaluation (exact) ---
    anatomy_tp = len(pred_anatomy & gt_in_corpus_anatomy)
    anatomy_fp = len(pred_anatomy - gt_all_anatomy)
    anatomy_fn = len(gt_in_corpus_anatomy - pred_anatomy)
    anatomy_metrics = precision_recall_f1(anatomy_tp, anatomy_fp, anatomy_fn)

    # --- Semantic metrics (uses ALL best similarities, no threshold) ---
    # Precision: For each PREDICTION, find best similarity to any GT
    # Recall: For each GT, find best similarity from any prediction
    # Both metrics bounded [0, 1]

    num_predictions = len(pred_anatomy)
    num_gt_in_corpus = len(gt_in_corpus_anatomy)

    # Precision: average best similarity for each prediction
    precision_sum = 0.0
    for pred_a in pred_anatomy:
        best_sim = 0.0
        for gt_a in gt_in_corpus_anatomy:
            sim = anatomy_sim.compute_similarity(pred_a, gt_a)
            if sim > best_sim:
                best_sim = sim
        precision_sum += best_sim

    sim_precision = precision_sum / num_predictions if num_predictions > 0 else 0.0

    # Recall: average best similarity for each GT item
    recall_sum = 0.0
    for gt_a in gt_in_corpus_anatomy:
        best_sim = 0.0
        for pred_a in pred_anatomy:
            sim = anatomy_sim.compute_similarity(gt_a, pred_a)
            if sim > best_sim:
                best_sim = sim
        recall_sum += best_sim

    sim_recall = recall_sum / num_gt_in_corpus if num_gt_in_corpus > 0 else 0.0

    sim_f1 = (
        2 * sim_precision * sim_recall / (sim_precision + sim_recall)
        if (sim_precision + sim_recall) > 0
        else 0.0
    )

    # semantic_tp stored for micro-averaging (sum of recall contributions)
    anatomy_semantic = SemanticMetrics(
        precision=sim_precision,
        recall=sim_recall,
        f1=sim_f1,
        semantic_tp=recall_sum,
        precision_sum=precision_sum,
    )

    # Full semantic recall (against ALL GT, not just in-corpus)
    full_recall_sum = 0.0
    for gt_a in gt_all_anatomy:
        best_sim = 0.0
        for pred_a in pred_anatomy:
            sim = anatomy_sim.compute_similarity(gt_a, pred_a)
            if sim > best_sim:
                best_sim = sim
        full_recall_sum += best_sim
    num_gt_all = len(gt_all_anatomy)
    full_semantic_recall = full_recall_sum / num_gt_all if num_gt_all > 0 else 0.0

    # --- Stage evaluation ---
    # For stage, we need to check if predicted stage falls in any GT range
    stage_tp = 0
    matched_gt_stages = set()

    for pred_s in pred_stage:
        for gt in gt_in_corpus_records:
            start = gt.get("stage_start_id")
            end = gt.get("stage_end_id")
            if _stage_in_range(pred_s, start, end):
                if start not in matched_gt_stages:
                    stage_tp += 1
                    matched_gt_stages.add(start)
                break

    stage_fp = len(pred_stage) - stage_tp
    stage_fn = len(gt_in_corpus_stage) - len(matched_gt_stages)
    stage_metrics = precision_recall_f1(stage_tp, stage_fp, stage_fn)

    # --- Tuple evaluation (exact) ---
    tuple_tp = 0
    matched_gt_tuples = set()

    for pred_anat, pred_stg in pred_tuples:
        for gt in gt_in_corpus_records:
            gt_anat = gt.get("anatomy_id")
            gt_start = gt.get("stage_start_id")
            gt_end = gt.get("stage_end_id")

            if pred_anat == gt_anat and _stage_in_range(pred_stg, gt_start, gt_end):
                if (gt_anat, gt_start) not in matched_gt_tuples:
                    tuple_tp += 1
                    matched_gt_tuples.add((gt_anat, gt_start))
                break

    tuple_fp = len(pred_tuples) - tuple_tp
    tuple_fn = len(gt_in_corpus_tuples) - len(matched_gt_tuples)
    tuple_metrics = precision_recall_f1(tuple_tp, tuple_fp, tuple_fn)

    # Calculate exact recalls
    in_corpus_exact_recall = anatomy_tp / len(gt_in_corpus_anatomy) if gt_in_corpus_anatomy else 0.0
    full_tp = len(pred_anatomy & gt_all_anatomy)
    full_recall = full_tp / len(gt_all_anatomy) if gt_all_anatomy else 0.0

    # Compute recall@k metrics for anatomy (using ordered predictions)
    anatomy_recall_at_k = compute_recall_at_k_series(
        predictions=pred_anatomy_list,
        gt_all=gt_all_anatomy,
        gt_in_corpus=gt_in_corpus_anatomy,
        similarity_fn=anatomy_sim.compute_similarity,
    )

    return Task2Result(
        anatomy_metrics=anatomy_metrics,
        anatomy_semantic=anatomy_semantic,
        stage_metrics=stage_metrics,
        tuple_metrics=tuple_metrics,
        in_corpus_exact_recall=in_corpus_exact_recall,
        full_recall=full_recall,
        full_semantic_recall=full_semantic_recall,
        predicted_count=len(predicted),
        gt_total_count=len(ground_truth),
        gt_in_corpus_count=len(gt_in_corpus_records),
        anatomy_recall_at_k=anatomy_recall_at_k,
    )
