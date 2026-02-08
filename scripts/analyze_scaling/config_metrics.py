"""ConfigMetrics dataclass for scaling analysis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConfigMetrics:
    """Metrics for a single paper configuration.

    Simplified structure after eval codebase refactor:
    - Removed soft metrics (threshold-based binary matching at 0.7)
    - Kept exact metrics and semantic metrics
    - Primary metric: semantic (partial credit based on ontology similarity)
    """

    max_papers: int
    num_genes_attempted: int
    num_genes_evaluated: int
    failure_rate: float
    avg_cost: float
    avg_cost_per_success: float
    total_cost: float
    avg_papers_read: float

    # ==================== Task 1: GO ====================
    # Exact matching (binary: GO ID must match exactly)
    go_exact_precision: float
    go_exact_recall: float
    go_exact_f1: float
    go_exact_f1_std: float
    go_in_corpus_exact_recall: float  # Exact recall against in-corpus GT only
    go_full_recall: float  # Exact recall against all GT
    go_total_tp: int
    go_total_fp: int
    go_total_fn: int
    go_fp_rate: float
    # Micro-averaged exact (pooled across all genes)
    go_micro_precision: float
    go_micro_recall: float
    go_micro_f1: float
    # Semantic (primary metric: partial credit based on Wang similarity)
    go_semantic_precision: float
    go_semantic_recall: float  # Against in-corpus GT
    go_semantic_f1: float
    go_semantic_f1_std: float
    go_total_semantic_tp: float  # Sum of similarities for micro-averaging
    go_total_gt_in_corpus: int  # Total in-corpus GT items across all genes
    go_semantic_micro_recall: float  # Micro-averaged semantic recall
    go_full_semantic_recall: float  # Against ALL GT

    # ==================== Task 2: Expression ====================
    # Exact anatomy matching
    expr_anatomy_precision: float
    expr_anatomy_recall: float
    expr_anatomy_f1: float
    expr_anatomy_f1_std: float
    expr_in_corpus_exact_recall: float  # Exact recall against in-corpus GT only
    expr_full_recall: float  # Exact recall against all GT
    # Micro-averaged exact anatomy
    expr_anatomy_micro_precision: float
    expr_anatomy_micro_recall: float
    expr_anatomy_micro_f1: float
    # Semantic anatomy (primary metric)
    expr_semantic_precision: float
    expr_semantic_recall: float  # Against in-corpus GT
    expr_semantic_f1: float
    expr_semantic_f1_std: float
    expr_total_semantic_tp: float
    expr_total_gt_in_corpus: int  # Total in-corpus GT items across all genes
    expr_semantic_micro_recall: float  # Micro-averaged semantic recall
    expr_full_semantic_recall: float  # Against ALL GT
    # Exact tuple matching (anatomy + stage)
    expr_tuple_precision: float
    expr_tuple_recall: float
    expr_tuple_f1: float
    expr_tuple_f1_std: float

    # ==================== Task 3: Synonyms ====================
    # Fullname
    syn_fullname_precision: float
    syn_fullname_recall: float
    syn_fullname_f1: float
    syn_fullname_f1_std: float
    syn_fullname_in_corpus_recall: float
    syn_fullname_full_recall: float
    # Micro-averaged fullname
    syn_fullname_micro_precision: float
    syn_fullname_micro_recall: float
    syn_fullname_micro_f1: float
    # Symbol
    syn_symbol_precision: float
    syn_symbol_recall: float
    syn_symbol_f1: float
    syn_symbol_f1_std: float
    syn_symbol_in_corpus_recall: float
    syn_symbol_full_recall: float
    # Micro-averaged symbol
    syn_symbol_micro_precision: float
    syn_symbol_micro_recall: float
    syn_symbol_micro_f1: float
    # Combined (fullname + symbol)
    syn_combined_precision: float
    syn_combined_recall: float
    syn_combined_f1: float
    syn_combined_f1_std: float
    syn_combined_in_corpus_recall: float
    syn_combined_full_recall: float
    # Micro-averaged combined
    syn_combined_micro_precision: float
    syn_combined_micro_recall: float
    syn_combined_micro_f1: float

    # ==================== Reference Coverage ====================
    ref_recall: float
    ref_precision: float
    ref_f1: float
    ref_f1_std: float
    # Micro-averaged
    ref_micro_precision: float
    ref_micro_recall: float
    ref_micro_f1: float

    # ==================== Recall@k (dict: str(k) -> float) ====================
    # Task 1: GO
    go_macro_exact_recall_at_k: dict[str, float] = field(default_factory=dict)
    go_macro_semantic_recall_at_k: dict[str, float] = field(default_factory=dict)
    go_micro_exact_recall_at_k: dict[str, float] = field(default_factory=dict)
    go_micro_semantic_recall_at_k: dict[str, float] = field(default_factory=dict)
    # Task 2: Expression
    expr_macro_exact_recall_at_k: dict[str, float] = field(default_factory=dict)
    expr_macro_semantic_recall_at_k: dict[str, float] = field(default_factory=dict)
    expr_micro_exact_recall_at_k: dict[str, float] = field(default_factory=dict)
    expr_micro_semantic_recall_at_k: dict[str, float] = field(default_factory=dict)
    # Task 3: Synonyms (exact only, no semantic similarity)
    syn_macro_exact_recall_at_k: dict[str, float] = field(default_factory=dict)
    syn_micro_exact_recall_at_k: dict[str, float] = field(default_factory=dict)
