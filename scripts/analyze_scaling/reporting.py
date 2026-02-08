"""Reporting functions for scaling analysis output."""

from __future__ import annotations

from .config_metrics import ConfigMetrics


def print_summary_table(metrics: list[ConfigMetrics]) -> None:
    """Print formatted summary table of all configurations."""
    print("\n" + "=" * 120)
    print("SCALING EXPERIMENT RESULTS")
    print("=" * 120)

    # Header
    print(
        f"\n{'Papers':>7} | {'Cost':>8} | {'Read':>5} | "
        f"{'GO-F1':>7} | {'TP':>4} | {'FP':>4} | {'FP%':>6} | "
        f"{'Fail%':>6} | {'$/F1':>8}"
    )
    print("-" * 120)

    # Data rows
    for m in metrics:
        cost_per_f1 = m.avg_cost_per_success / m.go_exact_f1 if m.go_exact_f1 > 0 else float("inf")
        print(
            f"{m.max_papers:>7} | ${m.avg_cost_per_success:>6.3f} | {m.avg_papers_read:>5.1f} | "
            f"{m.go_exact_f1:>6.1%} | {m.go_total_tp:>4} | {m.go_total_fp:>4} | {m.go_fp_rate:>5.1%} | "
            f"{m.failure_rate:>5.1%} | ${cost_per_f1:>6.2f}"
        )

    print("-" * 120)


def print_detailed_metrics(metrics: list[ConfigMetrics]) -> None:
    """Print detailed metrics for each configuration."""
    print("\n" + "=" * 100)
    print("DETAILED METRICS BY CONFIGURATION")
    print("(Macro = per-gene average, Micro = pooled across all annotations)")
    print("=" * 100)

    for m in metrics:
        n = m.num_genes_evaluated
        print(f"\n--- max_papers={m.max_papers} ---")
        print(f"  Genes attempted: {m.num_genes_attempted}")
        print(f"  Genes evaluated: {m.num_genes_evaluated}")
        print(f"  Failure rate: {m.failure_rate:.1%}")
        print(f"  Avg cost per attempt: ${m.avg_cost:.4f}")
        print(f"  Avg cost per success: ${m.avg_cost_per_success:.4f}")
        print(f"  Total cost: ${m.total_cost:.4f}")
        print(f"  Avg papers read: {m.avg_papers_read:.1f}")
        print()
        print("  Task 1 GO (Exact):")
        print("    Macro-averaged (per-gene):")
        print(f"      Precision:        {m.go_exact_precision:.1%}")
        print(f"      Recall (full GT): {m.go_full_recall:.1%}")
        print(f"      Recall (in-corpus): {m.go_in_corpus_exact_recall:.1%}")
        print(f"      F1:               {m.go_exact_f1:.1%} +/- {m.go_exact_f1_std:.1%} (n={n})")
        print("    Micro-averaged (pooled):")
        print(f"      Precision:        {m.go_micro_precision:.1%}")
        print(f"      Recall:           {m.go_micro_recall:.1%}")
        print(f"      F1:               {m.go_micro_f1:.1%}")
        print(
            f"    Counts: TP={m.go_total_tp}, FP={m.go_total_fp}, FN={m.go_total_fn}, FP Rate={m.go_fp_rate:.1%}"
        )
        print("    Semantic (partial credit):")
        print(f"      Precision:        {m.go_semantic_precision:.1%}")
        print(f"      Recall:           {m.go_semantic_recall:.1%}")
        print(
            f"      F1:               {m.go_semantic_f1:.1%} +/- {m.go_semantic_f1_std:.1%} (n={n})"
        )
        print(f"      Semantic TP:      {m.go_total_semantic_tp:.1f}")
        print(f"      Full Recall (all GT): {m.go_full_semantic_recall:.1%}")
        print()
        print("  Task 2 Expression (Anatomy Exact):")
        print("    Macro-averaged (per-gene):")
        print(f"      Precision:        {m.expr_anatomy_precision:.1%}")
        print(f"      Recall (full GT): {m.expr_full_recall:.1%}")
        print(f"      Recall (in-corpus): {m.expr_in_corpus_exact_recall:.1%}")
        print(
            f"      F1:               {m.expr_anatomy_f1:.1%} +/- {m.expr_anatomy_f1_std:.1%} (n={n})"
        )
        print("    Micro-averaged (pooled):")
        print(f"      Precision:        {m.expr_anatomy_micro_precision:.1%}")
        print(f"      Recall:           {m.expr_anatomy_micro_recall:.1%}")
        print(f"      F1:               {m.expr_anatomy_micro_f1:.1%}")
        print("    Semantic (partial credit):")
        print(f"      Precision:        {m.expr_semantic_precision:.1%}")
        print(f"      Recall:           {m.expr_semantic_recall:.1%}")
        print(
            f"      F1:               {m.expr_semantic_f1:.1%} +/- {m.expr_semantic_f1_std:.1%} (n={n})"
        )
        print(f"      Full Recall (all GT): {m.expr_full_semantic_recall:.1%}")
        print("  Task 2 Expression (Tuple Exact):")
        print(f"    Precision: {m.expr_tuple_precision:.1%}")
        print(f"    Recall:    {m.expr_tuple_recall:.1%}")
        print(f"    F1:        {m.expr_tuple_f1:.1%} +/- {m.expr_tuple_f1_std:.1%} (n={n})")
        print()
        print("  Task 3 Synonyms (Fullname):")
        print("    Macro-averaged (per-gene):")
        print(f"      Precision:        {m.syn_fullname_precision:.1%}")
        print(f"      Recall:           {m.syn_fullname_recall:.1%}")
        print(f"      In-corpus Recall: {m.syn_fullname_in_corpus_recall:.1%}")
        print(f"      Full Recall:      {m.syn_fullname_full_recall:.1%}")
        print(
            f"      F1:               {m.syn_fullname_f1:.1%} +/- {m.syn_fullname_f1_std:.1%} (n={n})"
        )
        print("    Micro-averaged (pooled):")
        print(f"      Precision:        {m.syn_fullname_micro_precision:.1%}")
        print(f"      Recall:           {m.syn_fullname_micro_recall:.1%}")
        print(f"      F1:               {m.syn_fullname_micro_f1:.1%}")
        print("  Task 3 Synonyms (Symbol):")
        print("    Macro-averaged (per-gene):")
        print(f"      Precision:        {m.syn_symbol_precision:.1%}")
        print(f"      Recall:           {m.syn_symbol_recall:.1%}")
        print(f"      In-corpus Recall: {m.syn_symbol_in_corpus_recall:.1%}")
        print(f"      Full Recall:      {m.syn_symbol_full_recall:.1%}")
        print(
            f"      F1:               {m.syn_symbol_f1:.1%} +/- {m.syn_symbol_f1_std:.1%} (n={n})"
        )
        print("    Micro-averaged (pooled):")
        print(f"      Precision:        {m.syn_symbol_micro_precision:.1%}")
        print(f"      Recall:           {m.syn_symbol_micro_recall:.1%}")
        print(f"      F1:               {m.syn_symbol_micro_f1:.1%}")
        print("  Task 3 Synonyms (Combined):")
        print("    Macro-averaged (per-gene):")
        print(f"      Precision:        {m.syn_combined_precision:.1%}")
        print(f"      Recall:           {m.syn_combined_recall:.1%}")
        print(f"      In-corpus Recall: {m.syn_combined_in_corpus_recall:.1%}")
        print(f"      Full Recall:      {m.syn_combined_full_recall:.1%}")
        print(
            f"      F1:               {m.syn_combined_f1:.1%} +/- {m.syn_combined_f1_std:.1%} (n={n})"
        )
        print("    Micro-averaged (pooled):")
        print(f"      Precision:        {m.syn_combined_micro_precision:.1%}")
        print(f"      Recall:           {m.syn_combined_micro_recall:.1%}")
        print(f"      F1:               {m.syn_combined_micro_f1:.1%}")
        print()
        print("  Reference Coverage:")
        print("    Macro-averaged (per-gene):")
        print(f"      Precision: {m.ref_precision:.1%}")
        print(f"      Recall:    {m.ref_recall:.1%}")
        print(f"      F1:        {m.ref_f1:.1%} +/- {m.ref_f1_std:.1%} (n={n})")
        print("    Micro-averaged (pooled):")
        print(f"      Precision: {m.ref_micro_precision:.1%}")
        print(f"      Recall:    {m.ref_micro_recall:.1%}")
        print(f"      F1:        {m.ref_micro_f1:.1%}")


def find_knee_point(metrics: list[ConfigMetrics]) -> ConfigMetrics | None:
    """Find the 'knee' point where diminishing returns begin.

    Uses the F1 gain per dollar spent to identify the optimal config.

    Args:
        metrics: List of ConfigMetrics sorted by max_papers

    Returns:
        The optimal ConfigMetrics or None if insufficient data
    """
    if len(metrics) < 2:
        return None

    best_efficiency = 0.0
    best_config = None

    for i in range(1, len(metrics)):
        prev = metrics[i - 1]
        curr = metrics[i]

        f1_gain = curr.go_exact_f1 - prev.go_exact_f1
        cost_increase = curr.avg_cost - prev.avg_cost

        if cost_increase > 0:
            efficiency = f1_gain / cost_increase
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_config = prev  # The config before gains flatten

    return best_config
