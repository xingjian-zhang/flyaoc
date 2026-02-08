"""Task 1: GO annotation evaluation with Wang semantic similarity.

This module provides:
- GOSimilarity class for computing Wang semantic similarity between GO terms
- GO annotation evaluation with exact and semantic metrics
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from goatools.obo_parser import GODag

from eval.recall_at_k import RecallAtKResult, compute_recall_at_k_series

# =============================================================================
# GO Similarity (merged from go_similarity.py)
# =============================================================================


class GOSimilarity:
    """Compute semantic similarity between GO terms using Wang's method."""

    def __init__(self, obo_path: str = "ontologies/go-basic.obo"):
        """Initialize with GO DAG.

        Args:
            obo_path: Path to GO OBO file.
        """
        path = Path(obo_path)
        if not path.exists():
            raise FileNotFoundError(f"GO ontology file not found: {path}")

        self.godag = GODag(str(path), optional_attrs={"relationship"})
        self._sim_cache: dict[tuple[str, str], float] = {}

    def get_aspect(self, go_id: str) -> str | None:
        """Get the aspect (P, F, C) for a GO term."""
        if go_id not in self.godag:
            return None
        term = self.godag[go_id]
        ns = term.namespace
        if ns == "biological_process":
            return "P"
        elif ns == "molecular_function":
            return "F"
        elif ns == "cellular_component":
            return "C"
        return None

    @lru_cache(maxsize=10000)
    def compute_similarity(self, go_a: str, go_b: str) -> float:
        """Compute Wang semantic similarity between two GO terms.

        Args:
            go_a: First GO term ID (e.g., "GO:0008150").
            go_b: Second GO term ID.

        Returns:
            Similarity score in [0, 1]. Returns 0 if terms not found or different aspects.
        """
        # Exact match
        if go_a == go_b:
            return 1.0

        # Check both terms exist
        if go_a not in self.godag or go_b not in self.godag:
            return 0.0

        # Check same aspect (namespace)
        term_a = self.godag[go_a]
        term_b = self.godag[go_b]
        if term_a.namespace != term_b.namespace:
            return 0.0

        # Use Wang similarity implementation
        try:
            from goatools.semsim.termwise.wang import SsWang

            wang = SsWang({go_a, go_b}, self.godag, relationships={"part_of"})
            sim = wang.get_sim(go_a, go_b)
            return sim if sim is not None else 0.0
        except Exception:
            # Fall back to simple ancestor-based similarity
            return self._ancestor_similarity(go_a, go_b)

    def _ancestor_similarity(self, go_a: str, go_b: str) -> float:
        """Simple Jaccard similarity on ancestor sets."""
        ancestors_a = self._get_ancestors(go_a)
        ancestors_b = self._get_ancestors(go_b)

        if not ancestors_a or not ancestors_b:
            return 0.0

        intersection = len(ancestors_a & ancestors_b)
        union = len(ancestors_a | ancestors_b)

        return intersection / union if union > 0 else 0.0

    def _get_ancestors(self, go_id: str) -> set[str]:
        """Get all ancestors of a GO term."""
        if go_id not in self.godag:
            return set()

        ancestors = set()
        to_visit = [go_id]

        while to_visit:
            current = to_visit.pop()
            if current in ancestors:
                continue
            ancestors.add(current)

            term = self.godag.get(current)
            if term:
                for parent in term.parents:
                    to_visit.append(parent.id)

        return ancestors

    def find_best_match(
        self,
        query_go: str,
        candidates: list[str],
        threshold: float = 0.0,
    ) -> tuple[str | None, float]:
        """Find the best matching GO term from candidates.

        Args:
            query_go: The GO term to match.
            candidates: List of candidate GO terms.
            threshold: Minimum similarity to consider a match (default 0 = no threshold).

        Returns:
            Tuple of (best_match_id, similarity) or (None, best_sim) if no match above threshold.
        """
        best_match = None
        best_sim = 0.0

        for candidate in candidates:
            sim = self.compute_similarity(query_go, candidate)
            if sim > best_sim:
                best_sim = sim
                best_match = candidate

        if best_sim >= threshold:
            return best_match, best_sim
        return None, best_sim  # Return actual similarity for FP analysis


# Global instance for reuse
_go_sim_instance: GOSimilarity | None = None


def get_go_similarity(obo_path: str = "ontologies/go-basic.obo") -> GOSimilarity:
    """Get or create global GOSimilarity instance."""
    global _go_sim_instance
    if _go_sim_instance is None:
        _go_sim_instance = GOSimilarity(obo_path)
    return _go_sim_instance


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
class GOMatch:
    """A single GO annotation match result."""

    predicted_go: str
    matched_gt_go: str | None
    similarity: float
    is_exact: bool
    predicted_data: dict[str, Any]
    gt_data: dict[str, Any] | None


@dataclass
class SemanticMetrics:
    """Semantic precision, recall, F1.

    Instead of binary TP counting, uses sum of similarities as weighted TP.
    This provides partial credit for semantically similar but non-exact matches.

    - precision = sum(best_sim_per_pred) / num_predictions
    - recall = sum(best_sim_per_gt) / num_ground_truth
    - f1 = harmonic mean of precision and recall
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
class Task1Result:
    """Results from Task 1 GO evaluation."""

    exact_metrics: Metrics
    semantic: SemanticMetrics
    in_corpus_exact_recall: float  # Exact recall against in-corpus GT
    full_recall: float  # Exact recall against all GT
    full_semantic_recall: float  # Semantic recall against all GT
    matches: list[GOMatch] = field(default_factory=list)
    predicted_count: int = 0
    gt_total_count: int = 0
    gt_in_corpus_count: int = 0
    recall_at_k: RecallAtKResult | None = None  # Recall@k metrics

    def to_dict(self) -> dict:
        result = {
            "exact": self.exact_metrics.to_dict(),
            "semantic": self.semantic.to_dict(),
            "in_corpus_exact_recall": round(self.in_corpus_exact_recall, 4),
            "full_recall": round(self.full_recall, 4),
            "full_semantic_recall": round(self.full_semantic_recall, 4),
            "predicted_count": self.predicted_count,
            "gt_total_count": self.gt_total_count,
            "gt_in_corpus_count": self.gt_in_corpus_count,
        }
        if self.recall_at_k is not None:
            result["recall_at_k"] = self.recall_at_k.to_dict()
        if self.matches:
            result["matches"] = [
                {
                    "predicted_go": m.predicted_go,
                    "matched_gt_go": m.matched_gt_go,
                    "similarity": round(m.similarity, 4),
                    "is_exact": m.is_exact,
                }
                for m in self.matches
            ]
        return result


# =============================================================================
# Main Evaluation Function
# =============================================================================


def evaluate_go(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    sim: GOSimilarity | None = None,
) -> Task1Result:
    """Evaluate GO annotations.

    Args:
        predicted: List of predicted GO annotations from agent (order matters for recall@k).
        ground_truth: List of ground truth GO annotations.
        sim: GOSimilarity instance (uses global if None).

    Returns:
        Task1Result with exact and semantic metrics.
    """
    go_sim = sim if sim is not None else get_go_similarity()

    # Extract ordered list of GO IDs (preserving ranking order, deduplicating)
    seen: set[str] = set()
    predicted_go_list: list[str] = []
    for p in predicted:
        go_id = p.get("go_id")
        if go_id and go_id not in seen:
            predicted_go_list.append(go_id)
            seen.add(go_id)

    # Set for fast lookups
    predicted_gos = set(predicted_go_list)

    # Separate ground truth by in_corpus
    gt_all = {gt["go_id"] for gt in ground_truth if "go_id" in gt}
    gt_in_corpus = {gt["go_id"] for gt in ground_truth if gt.get("in_corpus", False)}

    # Create lookup dicts
    pred_by_go = {p["go_id"]: p for p in predicted if "go_id" in p}
    gt_by_go = {gt["go_id"]: gt for gt in ground_truth if "go_id" in gt}

    # Exact matching (against in-corpus)
    exact_tp = len(predicted_gos & gt_in_corpus)
    exact_fp = len(predicted_gos - gt_all)  # Predicted but not in any ground truth
    exact_fn = len(gt_in_corpus - predicted_gos)

    exact_metrics = precision_recall_f1(exact_tp, exact_fp, exact_fn)

    # Record matches for analysis
    matches: list[GOMatch] = []
    for pred_go in predicted_gos:
        if pred_go in gt_all:
            matches.append(
                GOMatch(
                    predicted_go=pred_go,
                    matched_gt_go=pred_go,
                    similarity=1.0,
                    is_exact=True,
                    predicted_data=pred_by_go[pred_go],
                    gt_data=gt_by_go.get(pred_go),
                )
            )
        else:
            # Find best match for analysis
            best_match, best_sim = go_sim.find_best_match(pred_go, list(gt_all))
            matches.append(
                GOMatch(
                    predicted_go=pred_go,
                    matched_gt_go=best_match,
                    similarity=best_sim,
                    is_exact=False,
                    predicted_data=pred_by_go[pred_go],
                    gt_data=gt_by_go.get(best_match) if best_match else None,
                )
            )

    # Semantic metrics
    # Uses "best match per item" approach to ensure bounded [0, 1]
    # - Precision: For each prediction, find best similarity to any in-corpus GT
    # - Recall: For each in-corpus GT, find best similarity from any prediction
    num_predictions = len(predicted_gos)
    num_gt_in_corpus = len(gt_in_corpus)

    # Precision: average best similarity for each prediction
    precision_sum = 0.0
    for pred_go in predicted_gos:
        best_score = 0.0
        for gt_go in gt_in_corpus:
            score = go_sim.compute_similarity(pred_go, gt_go)
            if score > best_score:
                best_score = score
        precision_sum += best_score

    sim_precision = precision_sum / num_predictions if num_predictions > 0 else 0.0

    # Recall: average best similarity for each in-corpus GT item
    recall_sum = 0.0
    for gt_go in gt_in_corpus:
        best_score = 0.0
        for pred_go in predicted_gos:
            score = go_sim.compute_similarity(gt_go, pred_go)
            if score > best_score:
                best_score = score
        recall_sum += best_score

    sim_recall = recall_sum / num_gt_in_corpus if num_gt_in_corpus > 0 else 0.0

    sim_f1 = (
        2 * sim_precision * sim_recall / (sim_precision + sim_recall)
        if (sim_precision + sim_recall) > 0
        else 0.0
    )

    # semantic_tp stored for micro-averaging (sum of recall contributions)
    semantic_tp = recall_sum

    # Full semantic recall (against ALL GT, not just in-corpus)
    num_gt_all = len(gt_all)
    full_recall_sum = 0.0
    for gt_go in gt_all:
        best_score = 0.0
        for pred_go in predicted_gos:
            score = go_sim.compute_similarity(gt_go, pred_go)
            if score > best_score:
                best_score = score
        full_recall_sum += best_score
    full_semantic_recall = full_recall_sum / num_gt_all if num_gt_all > 0 else 0.0

    semantic = SemanticMetrics(
        precision=sim_precision,
        recall=sim_recall,
        f1=sim_f1,
        semantic_tp=semantic_tp,
        precision_sum=precision_sum,
    )

    # Calculate exact recalls
    in_corpus_exact_recall = exact_tp / len(gt_in_corpus) if gt_in_corpus else 0.0
    full_recall = len(predicted_gos & gt_all) / len(gt_all) if gt_all else 0.0

    # Compute recall@k metrics (using ordered predictions)
    recall_at_k_result = compute_recall_at_k_series(
        predictions=predicted_go_list,
        gt_all=gt_all,
        gt_in_corpus=gt_in_corpus,
        similarity_fn=go_sim.compute_similarity,
    )

    return Task1Result(
        exact_metrics=exact_metrics,
        semantic=semantic,
        in_corpus_exact_recall=in_corpus_exact_recall,
        full_recall=full_recall,
        full_semantic_recall=full_semantic_recall,
        matches=matches,
        predicted_count=len(predicted_gos),
        gt_total_count=len(gt_all),
        gt_in_corpus_count=len(gt_in_corpus),
        recall_at_k=recall_at_k_result,
    )
