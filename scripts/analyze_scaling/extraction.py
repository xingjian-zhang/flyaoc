"""Metrics extraction functions for scaling analysis.

Updated for simplified eval output format:
- Reads "exact" instead of "soft" for binary matching
- Reads "in_corpus_exact_recall" instead of "in_corpus_recall"
- Removed soft metrics extraction
"""

from __future__ import annotations

from typing import Any

from .config_metrics import ConfigMetrics


def extract_metrics(
    max_papers: int, run_summary: dict[str, Any], eval_results: dict[str, Any]
) -> ConfigMetrics:
    """Extract metrics from run summary and evaluation results.

    Computes all metrics consistently from per-gene results rather than
    relying on aggregate values, ensuring mean_f1 = mean(per-gene F1s).

    Args:
        max_papers: Paper configuration
        run_summary: Agent run summary
        eval_results: Evaluation results

    Returns:
        ConfigMetrics for this configuration
    """

    def safe_mean(values: list[float]) -> float:
        """Compute mean, returning 0 for empty lists."""
        return sum(values) / len(values) if values else 0.0

    def safe_std(values: list[float]) -> float:
        """Compute standard deviation, returning 0 for lists with < 2 elements."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)  # Sample std
        return variance**0.5

    # Calculate cost metrics from run summary
    genes = run_summary.get("genes", [])
    num_genes_attempted = len(genes) or run_summary.get("total", 0)
    total_cost = run_summary.get("total_cost", 0)
    avg_cost = total_cost / num_genes_attempted if num_genes_attempted > 0 else 0
    avg_papers = (
        sum(g.get("papers_read", 0) for g in genes) / num_genes_attempted
        if num_genes_attempted > 0
        else 0
    )

    # Get evaluated genes from eval_results
    eval_genes = eval_results.get("genes", [])
    num_genes_evaluated = len(eval_genes)

    # Get agent failure count (genes with null output, now scored as zero)
    aggregate = eval_results.get("aggregate", {})
    num_genes_failed = aggregate.get("genes_failed", 0)
    failure_rate = num_genes_failed / num_genes_attempted if num_genes_attempted > 0 else 0

    # Cost per success excludes failed genes
    num_genes_succeeded = num_genes_evaluated - num_genes_failed
    avg_cost_per_success = total_cost / num_genes_succeeded if num_genes_succeeded > 0 else 0

    # ==================== Task 1: GO (Exact) ====================
    go_precisions = []
    go_recalls = []
    go_f1s = []
    go_in_corpus_exact_recalls = []
    go_full_recalls = []
    go_total_tp = 0
    go_total_fp = 0
    go_total_fn = 0

    for gene in eval_genes:
        t1 = gene.get("task1_go", {})
        go_exact = t1.get("exact", {})
        go_precisions.append(go_exact.get("precision", 0))
        go_recalls.append(go_exact.get("recall", 0))
        go_f1s.append(go_exact.get("f1", 0))
        go_in_corpus_exact_recalls.append(t1.get("in_corpus_exact_recall", 0))
        go_full_recalls.append(t1.get("full_recall", 0))
        go_total_tp += go_exact.get("tp", 0)
        go_total_fp += go_exact.get("fp", 0)
        go_total_fn += go_exact.get("fn", 0)

    # Macro-averaged (per-gene average)
    go_exact_precision = safe_mean(go_precisions)
    go_exact_recall = safe_mean(go_recalls)
    go_exact_f1 = safe_mean(go_f1s)
    go_exact_f1_std = safe_std(go_f1s)
    go_in_corpus_exact_recall = safe_mean(go_in_corpus_exact_recalls)
    go_full_recall = safe_mean(go_full_recalls)
    go_fp_rate = go_total_fp / (go_total_tp + go_total_fp) if (go_total_tp + go_total_fp) > 0 else 0

    # Micro-averaged (pooled across all genes)
    go_micro_precision = (
        go_total_tp / (go_total_tp + go_total_fp) if (go_total_tp + go_total_fp) > 0 else 0
    )
    go_micro_recall = (
        go_total_tp / (go_total_tp + go_total_fn) if (go_total_tp + go_total_fn) > 0 else 0
    )
    go_micro_f1 = (
        2 * go_micro_precision * go_micro_recall / (go_micro_precision + go_micro_recall)
        if (go_micro_precision + go_micro_recall) > 0
        else 0
    )

    # Semantic metrics (primary metric: partial credit)
    go_sim_precisions = []
    go_sim_recalls = []
    go_sim_f1s = []
    go_full_semantic_recalls = []
    go_total_semantic_tp = 0.0
    go_total_gt_in_corpus = 0

    for gene in eval_genes:
        t1 = gene.get("task1_go", {})
        sim = t1.get("semantic", t1.get("similarity_weighted", {}))
        go_sim_precisions.append(sim.get("precision", 0))
        go_sim_recalls.append(sim.get("recall", 0))
        go_sim_f1s.append(sim.get("f1", 0))
        go_full_semantic_recalls.append(
            t1.get("full_semantic_recall", t1.get("full_sim_weighted_recall", 0))
        )
        go_total_semantic_tp += sim.get("semantic_tp", sim.get("weighted_tp", 0))
        go_total_gt_in_corpus += t1.get("gt_in_corpus_count", 0)

    # Macro-averaged semantic
    go_semantic_precision = safe_mean(go_sim_precisions)
    go_semantic_recall = safe_mean(go_sim_recalls)
    go_semantic_f1 = safe_mean(go_sim_f1s)
    go_semantic_f1_std = safe_std(go_sim_f1s)
    go_full_semantic_recall = safe_mean(go_full_semantic_recalls)

    # Micro-averaged semantic recall
    go_semantic_micro_recall = (
        go_total_semantic_tp / go_total_gt_in_corpus if go_total_gt_in_corpus > 0 else 0
    )

    # ==================== Task 2: Expression (Exact Anatomy) ====================
    expr_anat_precisions = []
    expr_anat_recalls = []
    expr_anat_f1s = []
    expr_in_corpus_exact_recalls = []
    expr_full_recalls = []
    expr_anat_total_tp = 0
    expr_anat_total_fp = 0
    expr_anat_total_fn = 0

    for gene in eval_genes:
        t2 = gene.get("task2_expression", {})
        anat = t2.get("anatomy", {})
        expr_anat_precisions.append(anat.get("precision", 0))
        expr_anat_recalls.append(anat.get("recall", 0))
        expr_anat_f1s.append(anat.get("f1", 0))
        expr_in_corpus_exact_recalls.append(t2.get("in_corpus_exact_recall", 0))
        expr_full_recalls.append(t2.get("full_recall", 0))
        expr_anat_total_tp += anat.get("tp", 0)
        expr_anat_total_fp += anat.get("fp", 0)
        expr_anat_total_fn += anat.get("fn", 0)

    # Macro-averaged (per-gene average)
    expr_anatomy_precision = safe_mean(expr_anat_precisions)
    expr_anatomy_recall = safe_mean(expr_anat_recalls)
    expr_anatomy_f1 = safe_mean(expr_anat_f1s)
    expr_anatomy_f1_std = safe_std(expr_anat_f1s)
    expr_in_corpus_exact_recall = safe_mean(expr_in_corpus_exact_recalls)
    expr_full_recall = safe_mean(expr_full_recalls)

    # Micro-averaged (pooled across all genes)
    expr_anatomy_micro_precision = (
        expr_anat_total_tp / (expr_anat_total_tp + expr_anat_total_fp)
        if (expr_anat_total_tp + expr_anat_total_fp) > 0
        else 0
    )
    expr_anatomy_micro_recall = (
        expr_anat_total_tp / (expr_anat_total_tp + expr_anat_total_fn)
        if (expr_anat_total_tp + expr_anat_total_fn) > 0
        else 0
    )
    expr_anatomy_micro_f1 = (
        2
        * expr_anatomy_micro_precision
        * expr_anatomy_micro_recall
        / (expr_anatomy_micro_precision + expr_anatomy_micro_recall)
        if (expr_anatomy_micro_precision + expr_anatomy_micro_recall) > 0
        else 0
    )

    # Task 2: Semantic metrics (primary metric)
    expr_sim_precisions = []
    expr_sim_recalls = []
    expr_sim_f1s = []
    expr_full_semantic_recalls = []
    expr_total_semantic_tp = 0.0
    expr_total_gt_in_corpus = 0

    for gene in eval_genes:
        t2 = gene.get("task2_expression", {})
        sim = t2.get("anatomy_semantic", t2.get("anatomy_similarity_weighted", {}))
        expr_sim_precisions.append(sim.get("precision", 0))
        expr_sim_recalls.append(sim.get("recall", 0))
        expr_sim_f1s.append(sim.get("f1", 0))
        expr_full_semantic_recalls.append(
            t2.get("full_semantic_recall", t2.get("full_sim_weighted_recall", 0))
        )
        expr_total_semantic_tp += sim.get("semantic_tp", sim.get("weighted_tp", 0))
        expr_total_gt_in_corpus += t2.get("gt_in_corpus_count", 0)

    # Macro-averaged semantic
    expr_semantic_precision = safe_mean(expr_sim_precisions)
    expr_semantic_recall = safe_mean(expr_sim_recalls)
    expr_semantic_f1 = safe_mean(expr_sim_f1s)
    expr_semantic_f1_std = safe_std(expr_sim_f1s)
    expr_full_semantic_recall = safe_mean(expr_full_semantic_recalls)

    # Micro-averaged semantic recall
    expr_semantic_micro_recall = (
        expr_total_semantic_tp / expr_total_gt_in_corpus if expr_total_gt_in_corpus > 0 else 0
    )

    # ==================== Task 2: Expression (Tuple) ====================
    expr_tuple_precisions = []
    expr_tuple_recalls = []
    expr_tuple_f1s = []

    for gene in eval_genes:
        t2 = gene.get("task2_expression", {})
        tup = t2.get("tuple", {})
        expr_tuple_precisions.append(tup.get("precision", 0))
        expr_tuple_recalls.append(tup.get("recall", 0))
        expr_tuple_f1s.append(tup.get("f1", 0))

    expr_tuple_precision = safe_mean(expr_tuple_precisions)
    expr_tuple_recall = safe_mean(expr_tuple_recalls)
    expr_tuple_f1 = safe_mean(expr_tuple_f1s)
    expr_tuple_f1_std = safe_std(expr_tuple_f1s)

    # ==================== Task 3: Synonyms (Fullname) ====================
    syn_fn_precisions = []
    syn_fn_recalls = []
    syn_fn_f1s = []
    syn_fn_in_corpus_recalls = []
    syn_fn_full_recalls = []
    syn_fn_total_tp = 0
    syn_fn_total_fp = 0
    syn_fn_total_fn = 0

    for gene in eval_genes:
        t3 = gene.get("task3_synonyms", {})
        fn = t3.get("fullname", {})
        syn_fn_precisions.append(fn.get("precision", 0))
        syn_fn_recalls.append(fn.get("recall", 0))
        syn_fn_f1s.append(fn.get("f1", 0))
        syn_fn_in_corpus_recalls.append(t3.get("in_corpus_fullname_recall", 0))
        syn_fn_full_recalls.append(t3.get("full_fullname_recall", 0))
        syn_fn_total_tp += fn.get("tp", 0)
        syn_fn_total_fp += fn.get("fp", 0)
        syn_fn_total_fn += fn.get("fn", 0)

    # Macro-averaged (per-gene average)
    syn_fullname_precision = safe_mean(syn_fn_precisions)
    syn_fullname_recall = safe_mean(syn_fn_recalls)
    syn_fullname_f1 = safe_mean(syn_fn_f1s)
    syn_fullname_f1_std = safe_std(syn_fn_f1s)
    syn_fullname_in_corpus_recall = safe_mean(syn_fn_in_corpus_recalls)
    syn_fullname_full_recall = safe_mean(syn_fn_full_recalls)

    # Micro-averaged (pooled across all genes)
    syn_fullname_micro_precision = (
        syn_fn_total_tp / (syn_fn_total_tp + syn_fn_total_fp)
        if (syn_fn_total_tp + syn_fn_total_fp) > 0
        else 0
    )
    syn_fullname_micro_recall = (
        syn_fn_total_tp / (syn_fn_total_tp + syn_fn_total_fn)
        if (syn_fn_total_tp + syn_fn_total_fn) > 0
        else 0
    )
    syn_fullname_micro_f1 = (
        2
        * syn_fullname_micro_precision
        * syn_fullname_micro_recall
        / (syn_fullname_micro_precision + syn_fullname_micro_recall)
        if (syn_fullname_micro_precision + syn_fullname_micro_recall) > 0
        else 0
    )

    # ==================== Task 3: Synonyms (Symbol) ====================
    syn_sym_precisions = []
    syn_sym_recalls = []
    syn_sym_f1s = []
    syn_sym_in_corpus_recalls = []
    syn_sym_full_recalls = []
    syn_sym_total_tp = 0
    syn_sym_total_fp = 0
    syn_sym_total_fn = 0

    for gene in eval_genes:
        t3 = gene.get("task3_synonyms", {})
        sym = t3.get("symbol", {})
        syn_sym_precisions.append(sym.get("precision", 0))
        syn_sym_recalls.append(sym.get("recall", 0))
        syn_sym_f1s.append(sym.get("f1", 0))
        syn_sym_in_corpus_recalls.append(t3.get("in_corpus_symbol_recall", 0))
        syn_sym_full_recalls.append(t3.get("full_symbol_recall", 0))
        syn_sym_total_tp += sym.get("tp", 0)
        syn_sym_total_fp += sym.get("fp", 0)
        syn_sym_total_fn += sym.get("fn", 0)

    # Macro-averaged (per-gene average)
    syn_symbol_precision = safe_mean(syn_sym_precisions)
    syn_symbol_recall = safe_mean(syn_sym_recalls)
    syn_symbol_f1 = safe_mean(syn_sym_f1s)
    syn_symbol_f1_std = safe_std(syn_sym_f1s)
    syn_symbol_in_corpus_recall = safe_mean(syn_sym_in_corpus_recalls)
    syn_symbol_full_recall = safe_mean(syn_sym_full_recalls)

    # Micro-averaged (pooled across all genes)
    syn_symbol_micro_precision = (
        syn_sym_total_tp / (syn_sym_total_tp + syn_sym_total_fp)
        if (syn_sym_total_tp + syn_sym_total_fp) > 0
        else 0
    )
    syn_symbol_micro_recall = (
        syn_sym_total_tp / (syn_sym_total_tp + syn_sym_total_fn)
        if (syn_sym_total_tp + syn_sym_total_fn) > 0
        else 0
    )
    syn_symbol_micro_f1 = (
        2
        * syn_symbol_micro_precision
        * syn_symbol_micro_recall
        / (syn_symbol_micro_precision + syn_symbol_micro_recall)
        if (syn_symbol_micro_precision + syn_symbol_micro_recall) > 0
        else 0
    )

    # ==================== Task 3: Synonyms (Combined) ====================
    syn_combined_precisions = []
    syn_combined_recalls = []
    syn_combined_f1s = []
    syn_combined_in_corpus_recalls = []
    syn_combined_full_recalls = []
    syn_combined_total_tp = 0
    syn_combined_total_fp = 0
    syn_combined_total_fn = 0

    for gene in eval_genes:
        t3 = gene.get("task3_synonyms", {})
        combined = t3.get("combined", {})
        syn_combined_precisions.append(combined.get("precision", 0))
        syn_combined_recalls.append(combined.get("recall", 0))
        syn_combined_f1s.append(combined.get("f1", 0))
        syn_combined_in_corpus_recalls.append(t3.get("in_corpus_combined_recall", 0))
        syn_combined_full_recalls.append(t3.get("full_combined_recall", 0))
        syn_combined_total_tp += combined.get("tp", 0)
        syn_combined_total_fp += combined.get("fp", 0)
        syn_combined_total_fn += combined.get("fn", 0)

    # Macro-averaged (per-gene average)
    syn_combined_precision = safe_mean(syn_combined_precisions)
    syn_combined_recall = safe_mean(syn_combined_recalls)
    syn_combined_f1 = safe_mean(syn_combined_f1s)
    syn_combined_f1_std = safe_std(syn_combined_f1s)
    syn_combined_in_corpus_recall = safe_mean(syn_combined_in_corpus_recalls)
    syn_combined_full_recall = safe_mean(syn_combined_full_recalls)

    # Micro-averaged (pooled across all genes)
    syn_combined_micro_precision = (
        syn_combined_total_tp / (syn_combined_total_tp + syn_combined_total_fp)
        if (syn_combined_total_tp + syn_combined_total_fp) > 0
        else 0
    )
    syn_combined_micro_recall = (
        syn_combined_total_tp / (syn_combined_total_tp + syn_combined_total_fn)
        if (syn_combined_total_tp + syn_combined_total_fn) > 0
        else 0
    )
    syn_combined_micro_f1 = (
        2
        * syn_combined_micro_precision
        * syn_combined_micro_recall
        / (syn_combined_micro_precision + syn_combined_micro_recall)
        if (syn_combined_micro_precision + syn_combined_micro_recall) > 0
        else 0
    )

    # ==================== Reference Coverage ====================
    ref_precisions = []
    ref_recalls = []
    ref_f1s = []
    ref_total_tp = 0
    ref_total_agent_refs = 0  # TP + FP (agent's predicted references)
    ref_total_gt_refs = 0  # TP + FN (ground truth references)

    for gene in eval_genes:
        ref = gene.get("reference_coverage", {})
        p = ref.get("agent_precision", 0)
        r = ref.get("recall", 0)
        ref_precisions.append(p)
        ref_recalls.append(r)
        # Compute per-gene F1 for reference coverage
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        ref_f1s.append(f1)
        # For micro-averaging: collect counts
        ref_total_tp += ref.get("hits", 0)
        ref_total_agent_refs += ref.get("agent_cited", 0)
        ref_total_gt_refs += ref.get("effective_papers", 0)

    # Macro-averaged (per-gene average)
    ref_precision = safe_mean(ref_precisions)
    ref_recall = safe_mean(ref_recalls)
    ref_f1 = safe_mean(ref_f1s)
    ref_f1_std = safe_std(ref_f1s)

    # Micro-averaged (pooled across all genes)
    ref_micro_precision = ref_total_tp / ref_total_agent_refs if ref_total_agent_refs > 0 else 0
    ref_micro_recall = ref_total_tp / ref_total_gt_refs if ref_total_gt_refs > 0 else 0
    ref_micro_f1 = (
        2 * ref_micro_precision * ref_micro_recall / (ref_micro_precision + ref_micro_recall)
        if (ref_micro_precision + ref_micro_recall) > 0
        else 0
    )

    # ==================== Recall@k ====================
    # Extract recall@k dicts from aggregate (computed by _aggregate_recall_at_k in evaluator)
    t1_rak = aggregate.get("task1_go", {}).get("recall_at_k", {})
    go_macro_exact_recall_at_k = t1_rak.get("mean_exact_at_k", {})
    go_macro_semantic_recall_at_k = t1_rak.get("mean_semantic_at_k", {})
    go_micro_exact_recall_at_k = t1_rak.get("micro_exact_at_k", {})
    go_micro_semantic_recall_at_k = t1_rak.get("micro_semantic_at_k", {})

    t2_rak = aggregate.get("task2_expression", {}).get("anatomy_recall_at_k", {})
    expr_macro_exact_recall_at_k = t2_rak.get("mean_exact_at_k", {})
    expr_macro_semantic_recall_at_k = t2_rak.get("mean_semantic_at_k", {})
    expr_micro_exact_recall_at_k = t2_rak.get("micro_exact_at_k", {})
    expr_micro_semantic_recall_at_k = t2_rak.get("micro_semantic_at_k", {})

    t3_rak = aggregate.get("task3_synonyms", {}).get("combined_recall_at_k", {})
    syn_macro_exact_recall_at_k = t3_rak.get("mean_exact_at_k", {})
    syn_micro_exact_recall_at_k = t3_rak.get("micro_exact_at_k", {})

    return ConfigMetrics(
        max_papers=max_papers,
        num_genes_attempted=num_genes_attempted,
        num_genes_evaluated=num_genes_evaluated,
        failure_rate=failure_rate,
        avg_cost=avg_cost,
        avg_cost_per_success=avg_cost_per_success,
        total_cost=total_cost,
        avg_papers_read=avg_papers,
        # Task 1: GO
        go_exact_precision=go_exact_precision,
        go_exact_recall=go_exact_recall,
        go_exact_f1=go_exact_f1,
        go_exact_f1_std=go_exact_f1_std,
        go_in_corpus_exact_recall=go_in_corpus_exact_recall,
        go_full_recall=go_full_recall,
        go_total_tp=go_total_tp,
        go_total_fp=go_total_fp,
        go_total_fn=go_total_fn,
        go_fp_rate=go_fp_rate,
        go_micro_precision=go_micro_precision,
        go_micro_recall=go_micro_recall,
        go_micro_f1=go_micro_f1,
        go_semantic_precision=go_semantic_precision,
        go_semantic_recall=go_semantic_recall,
        go_semantic_f1=go_semantic_f1,
        go_semantic_f1_std=go_semantic_f1_std,
        go_total_semantic_tp=go_total_semantic_tp,
        go_total_gt_in_corpus=go_total_gt_in_corpus,
        go_semantic_micro_recall=go_semantic_micro_recall,
        go_full_semantic_recall=go_full_semantic_recall,
        # Task 2: Expression
        expr_anatomy_precision=expr_anatomy_precision,
        expr_anatomy_recall=expr_anatomy_recall,
        expr_anatomy_f1=expr_anatomy_f1,
        expr_anatomy_f1_std=expr_anatomy_f1_std,
        expr_in_corpus_exact_recall=expr_in_corpus_exact_recall,
        expr_full_recall=expr_full_recall,
        expr_anatomy_micro_precision=expr_anatomy_micro_precision,
        expr_anatomy_micro_recall=expr_anatomy_micro_recall,
        expr_anatomy_micro_f1=expr_anatomy_micro_f1,
        expr_semantic_precision=expr_semantic_precision,
        expr_semantic_recall=expr_semantic_recall,
        expr_semantic_f1=expr_semantic_f1,
        expr_semantic_f1_std=expr_semantic_f1_std,
        expr_total_semantic_tp=expr_total_semantic_tp,
        expr_total_gt_in_corpus=expr_total_gt_in_corpus,
        expr_semantic_micro_recall=expr_semantic_micro_recall,
        expr_full_semantic_recall=expr_full_semantic_recall,
        expr_tuple_precision=expr_tuple_precision,
        expr_tuple_recall=expr_tuple_recall,
        expr_tuple_f1=expr_tuple_f1,
        expr_tuple_f1_std=expr_tuple_f1_std,
        # Task 3: Synonyms
        syn_fullname_precision=syn_fullname_precision,
        syn_fullname_recall=syn_fullname_recall,
        syn_fullname_f1=syn_fullname_f1,
        syn_fullname_f1_std=syn_fullname_f1_std,
        syn_fullname_in_corpus_recall=syn_fullname_in_corpus_recall,
        syn_fullname_full_recall=syn_fullname_full_recall,
        syn_fullname_micro_precision=syn_fullname_micro_precision,
        syn_fullname_micro_recall=syn_fullname_micro_recall,
        syn_fullname_micro_f1=syn_fullname_micro_f1,
        syn_symbol_precision=syn_symbol_precision,
        syn_symbol_recall=syn_symbol_recall,
        syn_symbol_f1=syn_symbol_f1,
        syn_symbol_f1_std=syn_symbol_f1_std,
        syn_symbol_in_corpus_recall=syn_symbol_in_corpus_recall,
        syn_symbol_full_recall=syn_symbol_full_recall,
        syn_symbol_micro_precision=syn_symbol_micro_precision,
        syn_symbol_micro_recall=syn_symbol_micro_recall,
        syn_symbol_micro_f1=syn_symbol_micro_f1,
        syn_combined_precision=syn_combined_precision,
        syn_combined_recall=syn_combined_recall,
        syn_combined_f1=syn_combined_f1,
        syn_combined_f1_std=syn_combined_f1_std,
        syn_combined_in_corpus_recall=syn_combined_in_corpus_recall,
        syn_combined_full_recall=syn_combined_full_recall,
        syn_combined_micro_precision=syn_combined_micro_precision,
        syn_combined_micro_recall=syn_combined_micro_recall,
        syn_combined_micro_f1=syn_combined_micro_f1,
        # Reference coverage
        ref_recall=ref_recall,
        ref_precision=ref_precision,
        ref_f1=ref_f1,
        ref_f1_std=ref_f1_std,
        ref_micro_precision=ref_micro_precision,
        ref_micro_recall=ref_micro_recall,
        ref_micro_f1=ref_micro_f1,
        # Recall@k
        go_macro_exact_recall_at_k=go_macro_exact_recall_at_k,
        go_macro_semantic_recall_at_k=go_macro_semantic_recall_at_k,
        go_micro_exact_recall_at_k=go_micro_exact_recall_at_k,
        go_micro_semantic_recall_at_k=go_micro_semantic_recall_at_k,
        expr_macro_exact_recall_at_k=expr_macro_exact_recall_at_k,
        expr_macro_semantic_recall_at_k=expr_macro_semantic_recall_at_k,
        expr_micro_exact_recall_at_k=expr_micro_exact_recall_at_k,
        expr_micro_semantic_recall_at_k=expr_micro_semantic_recall_at_k,
        syn_macro_exact_recall_at_k=syn_macro_exact_recall_at_k,
        syn_micro_exact_recall_at_k=syn_micro_exact_recall_at_k,
    )
