"""Recall@K evaluation metrics for ranked predictions.

Measures how many ground truth items appear in the top-k predictions,
rewarding agents that rank confident predictions first.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

# Default k values for evaluation
DEFAULT_K_VALUES: list[int] = [1, 3, 5, 10, 20, 50]


@dataclass
class RecallAtKResult:
    """Results from recall@k evaluation.

    Attributes:
        k_values: List of k values evaluated.
        exact_recall_at_k: Mapping from k to exact recall (binary matching).
        semantic_recall_at_k: Mapping from k to semantic recall (similarity-based).
        gt_count: Total ground truth count.
        gt_in_corpus_count: Ground truth items with in_corpus=True.
    """

    k_values: list[int] = field(default_factory=list)
    exact_recall_at_k: dict[int, float] = field(default_factory=dict)
    semantic_recall_at_k: dict[int, float] = field(default_factory=dict)
    gt_count: int = 0
    gt_in_corpus_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "k_values": self.k_values,
            "exact_recall_at_k": {str(k): round(v, 4) for k, v in self.exact_recall_at_k.items()},
            "semantic_recall_at_k": {str(k): round(v, 4) for k, v in self.semantic_recall_at_k.items()},
            "gt_count": self.gt_count,
            "gt_in_corpus_count": self.gt_in_corpus_count,
        }


def recall_at_k(predictions: list[str], ground_truth: set[str], k: int) -> float:
    """Compute exact recall@k.

    Args:
        predictions: Ordered list of predicted IDs (index 0 = most confident).
        ground_truth: Set of ground truth IDs.
        k: Number of top predictions to consider.

    Returns:
        Recall value: |predictions[:k] ∩ ground_truth| / |ground_truth|
    """
    if not ground_truth:
        return 0.0
    top_k = set(predictions[:k])
    hits = len(top_k & ground_truth)
    return hits / len(ground_truth)


def semantic_recall_at_k(
    predictions: list[str],
    ground_truth: set[str],
    k: int,
    similarity_fn: Callable[[str, str], float],
) -> float:
    """Compute semantic recall@k (similarity-based).

    For each GT item, finds the best similarity to any top-k prediction.
    Sum of best similarities divided by GT count.

    No threshold - similarity values contribute proportionally.
    Wang semantic similarity already encodes term relatedness.

    Args:
        predictions: Ordered list of predicted IDs (index 0 = most confident).
        ground_truth: Set of ground truth IDs.
        k: Number of top predictions to consider.
        similarity_fn: Function(pred, gt) -> similarity in [0, 1].

    Returns:
        Semantic recall in [0, 1].
    """
    if not ground_truth:
        return 0.0

    top_k = predictions[:k]
    if not top_k:
        return 0.0

    total_sim = 0.0
    for gt in ground_truth:
        best_sim = max((similarity_fn(pred, gt) for pred in top_k), default=0.0)
        total_sim += best_sim

    return total_sim / len(ground_truth)


def compute_recall_at_k_series(
    predictions: list[str],
    gt_all: set[str],
    gt_in_corpus: set[str],
    k_values: list[int] | None = None,
    similarity_fn: Callable[[str, str], float] | None = None,
) -> RecallAtKResult:
    """Compute recall@k for a series of k values.

    Args:
        predictions: Ordered list of predicted IDs (index 0 = most confident).
        gt_all: All ground truth IDs.
        gt_in_corpus: Ground truth IDs with in_corpus=True.
        k_values: List of k values to evaluate. Defaults to [1, 3, 5, 10, 20].
        similarity_fn: Optional function(pred, gt) -> similarity for soft matching.
            If None, semantic_recall_at_k will not be computed.

    Returns:
        RecallAtKResult with metrics for each k value.
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    result = RecallAtKResult(
        k_values=k_values,
        gt_count=len(gt_all),
        gt_in_corpus_count=len(gt_in_corpus),
    )

    for k in k_values:
        # Exact recall against in-corpus GT
        result.exact_recall_at_k[k] = recall_at_k(predictions, gt_in_corpus, k)

        # Soft recall (if similarity function provided)
        if similarity_fn is not None:
            result.semantic_recall_at_k[k] = semantic_recall_at_k(
                predictions, gt_in_corpus, k, similarity_fn
            )

    return result
