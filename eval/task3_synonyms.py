"""Task 3: Synonym evaluation."""

from dataclasses import dataclass
from typing import Any

from eval.recall_at_k import RecallAtKResult, compute_recall_at_k_series

# =============================================================================
# Metrics Classes
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
class Task3Result:
    """Results from Task 3 synonym evaluation."""

    fullname_metrics: Metrics
    symbol_metrics: Metrics
    combined_metrics: Metrics
    in_corpus_fullname_recall: float
    in_corpus_symbol_recall: float
    in_corpus_combined_recall: float  # Unified recall for fullname + symbol
    # Full recall: matches against ALL GT synonyms (not just in-corpus)
    full_fullname_recall: float
    full_symbol_recall: float
    full_combined_recall: float
    gt_fullname_total: int
    gt_fullname_in_corpus: int
    gt_symbol_total: int
    gt_symbol_in_corpus: int
    # Recall@k metrics (exact matching only, no soft similarity)
    fullname_recall_at_k: RecallAtKResult | None = None
    symbol_recall_at_k: RecallAtKResult | None = None
    combined_recall_at_k: RecallAtKResult | None = None

    def to_dict(self) -> dict:
        result = {
            "fullname": self.fullname_metrics.to_dict(),
            "symbol": self.symbol_metrics.to_dict(),
            "combined": self.combined_metrics.to_dict(),
            "in_corpus_fullname_recall": round(self.in_corpus_fullname_recall, 4),
            "in_corpus_symbol_recall": round(self.in_corpus_symbol_recall, 4),
            "in_corpus_combined_recall": round(self.in_corpus_combined_recall, 4),
            "full_fullname_recall": round(self.full_fullname_recall, 4),
            "full_symbol_recall": round(self.full_symbol_recall, 4),
            "full_combined_recall": round(self.full_combined_recall, 4),
            "gt_fullname_total": self.gt_fullname_total,
            "gt_fullname_in_corpus": self.gt_fullname_in_corpus,
            "gt_symbol_total": self.gt_symbol_total,
            "gt_symbol_in_corpus": self.gt_symbol_in_corpus,
        }
        if self.fullname_recall_at_k is not None:
            result["fullname_recall_at_k"] = self.fullname_recall_at_k.to_dict()
        if self.symbol_recall_at_k is not None:
            result["symbol_recall_at_k"] = self.symbol_recall_at_k.to_dict()
        if self.combined_recall_at_k is not None:
            result["combined_recall_at_k"] = self.combined_recall_at_k.to_dict()
        return result


# =============================================================================
# Helper Functions
# =============================================================================


def _normalize_synonym(s: str) -> str:
    """Normalize synonym for case-insensitive comparison."""
    return s.strip().lower()


def _extract_synonyms(syn_list: list, in_corpus_only: bool = False) -> set[str]:
    """Extract synonyms from list, handling both old and new formats.

    Old format: ["syn1", "syn2", ...]
    New format: [{"synonym": "syn1", "in_corpus": true}, ...]
    """
    result = set()
    for item in syn_list:
        if isinstance(item, str):
            # Old format - plain string
            result.add(_normalize_synonym(item))
        elif isinstance(item, dict):
            # New format - dict with in_corpus flag
            if in_corpus_only and not item.get("in_corpus", False):
                continue
            result.add(_normalize_synonym(item.get("synonym", "")))
    return result


def _count_in_corpus(syn_list: list) -> tuple[int, int]:
    """Count total and in_corpus synonyms. Returns (total, in_corpus)."""
    total = len(syn_list)
    in_corpus = 0
    for item in syn_list:
        if isinstance(item, dict) and item.get("in_corpus", False):
            in_corpus += 1
        elif isinstance(item, str):
            # Old format - assume all are in corpus (backwards compat)
            in_corpus += 1
    return total, in_corpus


# =============================================================================
# Main Evaluation Function
# =============================================================================


def evaluate_synonyms(
    predicted: dict[str, Any],
    ground_truth: dict[str, Any],
) -> Task3Result:
    """Evaluate synonym predictions.

    Ground truth synonyms now have in_corpus flags (labeled via text search).
    We evaluate against in-corpus synonyms for fair comparison.

    Args:
        predicted: Dict with fullname_synonyms and symbol_synonyms lists (order matters for recall@k).
        ground_truth: Dict with fullname_synonyms and symbol_synonyms lists
            (new format has in_corpus flags per synonym).

    Returns:
        Task3Result with metrics for fullname and symbol synonyms.
    """
    # Extract predicted synonyms as ordered lists (preserving ranking, deduplicating)
    pred_fullname_list: list[str] = []
    seen_fullname: set[str] = set()
    for s in predicted.get("fullname_synonyms", []):
        norm = _normalize_synonym(s)
        if norm not in seen_fullname:
            pred_fullname_list.append(norm)
            seen_fullname.add(norm)

    pred_symbol_list: list[str] = []
    seen_symbol: set[str] = set()
    for s in predicted.get("symbol_synonyms", []):
        norm = _normalize_synonym(s)
        if norm not in seen_symbol:
            pred_symbol_list.append(norm)
            seen_symbol.add(norm)

    # Combined list preserves fullname ordering, then symbol ordering, deduplicated
    pred_combined_list: list[str] = []
    seen_combined: set[str] = set()
    for s in pred_fullname_list + pred_symbol_list:
        if s not in seen_combined:
            pred_combined_list.append(s)
            seen_combined.add(s)

    # Sets for existing metrics
    pred_fullname = set(pred_fullname_list)
    pred_symbol = set(pred_symbol_list)

    # Extract ground truth - all and in-corpus only
    gt_fullname_list = ground_truth.get("fullname_synonyms", [])
    gt_symbol_list = ground_truth.get("symbol_synonyms", [])

    gt_fullname_all = _extract_synonyms(gt_fullname_list, in_corpus_only=False)
    gt_fullname_in_corpus = _extract_synonyms(gt_fullname_list, in_corpus_only=True)
    gt_symbol_all = _extract_synonyms(gt_symbol_list, in_corpus_only=False)
    gt_symbol_in_corpus = _extract_synonyms(gt_symbol_list, in_corpus_only=True)

    # Count stats
    fn_total, fn_in_corpus_count = _count_in_corpus(gt_fullname_list)
    sym_total, sym_in_corpus_count = _count_in_corpus(gt_symbol_list)

    # --- Fullname evaluation (against in-corpus) ---
    fn_tp = len(pred_fullname & gt_fullname_in_corpus)
    fn_fp = len(pred_fullname - gt_fullname_all)  # FP if not in ANY ground truth
    fn_fn = len(gt_fullname_in_corpus - pred_fullname)
    fullname_metrics = precision_recall_f1(fn_tp, fn_fp, fn_fn)

    # --- Symbol evaluation (against in-corpus) ---
    sym_tp = len(pred_symbol & gt_symbol_in_corpus)
    sym_fp = len(pred_symbol - gt_symbol_all)  # FP if not in ANY ground truth
    sym_fn = len(gt_symbol_in_corpus - pred_symbol)
    symbol_metrics = precision_recall_f1(sym_tp, sym_fp, sym_fn)

    # --- Combined (all synonyms, against in-corpus) ---
    pred_all = pred_fullname | pred_symbol
    gt_all_in_corpus = gt_fullname_in_corpus | gt_symbol_in_corpus
    gt_all_any = gt_fullname_all | gt_symbol_all
    all_tp = len(pred_all & gt_all_in_corpus)
    all_fp = len(pred_all - gt_all_any)
    all_fn = len(gt_all_in_corpus - pred_all)
    combined_metrics = precision_recall_f1(all_tp, all_fp, all_fn)

    # Calculate in-corpus recalls
    in_corpus_fn_recall = fn_tp / len(gt_fullname_in_corpus) if gt_fullname_in_corpus else 0.0
    in_corpus_sym_recall = sym_tp / len(gt_symbol_in_corpus) if gt_symbol_in_corpus else 0.0
    in_corpus_combined_recall = all_tp / len(gt_all_in_corpus) if gt_all_in_corpus else 0.0

    # Calculate full recalls (matches against ALL GT, not just in-corpus)
    # This shows what fraction of ALL annotations were found
    full_fn_tp = len(pred_fullname & gt_fullname_all)
    full_sym_tp = len(pred_symbol & gt_symbol_all)
    full_all_tp = len(pred_all & gt_all_any)

    full_fn_recall = full_fn_tp / len(gt_fullname_all) if gt_fullname_all else 0.0
    full_sym_recall = full_sym_tp / len(gt_symbol_all) if gt_symbol_all else 0.0
    full_combined_recall = full_all_tp / len(gt_all_any) if gt_all_any else 0.0

    # Compute recall@k metrics (exact matching only, no soft similarity)
    fullname_recall_at_k = compute_recall_at_k_series(
        predictions=pred_fullname_list,
        gt_all=gt_fullname_all,
        gt_in_corpus=gt_fullname_in_corpus,
        # No similarity_fn - exact matching only for synonyms
    )
    symbol_recall_at_k = compute_recall_at_k_series(
        predictions=pred_symbol_list,
        gt_all=gt_symbol_all,
        gt_in_corpus=gt_symbol_in_corpus,
    )
    combined_recall_at_k = compute_recall_at_k_series(
        predictions=pred_combined_list,
        gt_all=gt_all_any,
        gt_in_corpus=gt_all_in_corpus,
    )

    return Task3Result(
        fullname_metrics=fullname_metrics,
        symbol_metrics=symbol_metrics,
        combined_metrics=combined_metrics,
        in_corpus_fullname_recall=in_corpus_fn_recall,
        in_corpus_symbol_recall=in_corpus_sym_recall,
        in_corpus_combined_recall=in_corpus_combined_recall,
        full_fullname_recall=full_fn_recall,
        full_symbol_recall=full_sym_recall,
        full_combined_recall=full_combined_recall,
        gt_fullname_total=fn_total,
        gt_fullname_in_corpus=fn_in_corpus_count,
        gt_symbol_total=sym_total,
        gt_symbol_in_corpus=sym_in_corpus_count,
        fullname_recall_at_k=fullname_recall_at_k,
        symbol_recall_at_k=symbol_recall_at_k,
        combined_recall_at_k=combined_recall_at_k,
    )
