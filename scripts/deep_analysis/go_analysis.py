"""GO analysis and false positive analysis functions for deep analysis.

This module contains functions for analyzing GO annotation performance,
including aspect-level analysis and detailed false positive categorization.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from eval.data_loader import load_agent_output_scaling

from .dataclasses import FalsePositive

# =============================================================================
# Per-Paper Yield Analysis
# =============================================================================


def analyze_per_paper_yield(
    base_dir: Path, configs: list[int], all_evals: dict[int, dict]
) -> dict[str, Any]:
    """Analyze annotation yield by paper position (1st, 2nd, 3rd paper read).

    We infer yield by comparing configs:
    - papers=1 gives yield from 1st paper
    - papers=2 - papers=1 gives marginal yield from 2nd paper
    - etc.

    Args:
        base_dir: Base directory containing papers_N subdirectories
        configs: List of paper configurations to analyze
        all_evals: Dict mapping config -> evaluation results

    Returns:
        Dict containing cumulative analysis with marginal TP/FP counts
    """
    results: dict[str, Any] = {
        "by_position": {},
        "marginal_tp": {},
        "marginal_fp": {},
        "cumulative_analysis": [],
    }

    sorted_configs = sorted(configs)

    for i, config in enumerate(sorted_configs):
        eval_data = all_evals.get(config)
        if not eval_data:
            continue

        genes = eval_data.get("genes", [])
        total_tp = sum(g.get("task1_go", {}).get("soft", {}).get("tp", 0) for g in genes)
        total_fp = sum(g.get("task1_go", {}).get("soft", {}).get("fp", 0) for g in genes)
        avg_f1 = float(np.mean([g.get("task1_go", {}).get("soft", {}).get("f1", 0) for g in genes]))

        if i == 0:
            marginal_tp = total_tp
            marginal_fp = total_fp
        else:
            prev_config = sorted_configs[i - 1]
            prev_eval = all_evals.get(prev_config)
            if prev_eval:
                prev_genes = prev_eval.get("genes", [])
                prev_tp = sum(
                    g.get("task1_go", {}).get("soft", {}).get("tp", 0) for g in prev_genes
                )
                prev_fp = sum(
                    g.get("task1_go", {}).get("soft", {}).get("fp", 0) for g in prev_genes
                )
                marginal_tp = total_tp - prev_tp
                marginal_fp = total_fp - prev_fp
            else:
                marginal_tp = 0
                marginal_fp = 0

        results["cumulative_analysis"].append(
            {
                "papers": config,
                "total_tp": total_tp,
                "total_fp": total_fp,
                "marginal_tp": marginal_tp,
                "marginal_fp": marginal_fp,
                "avg_f1": avg_f1,
                "tp_per_paper": total_tp / config if config > 0 else 0,
                "marginal_precision": (
                    marginal_tp / (marginal_tp + marginal_fp)
                    if (marginal_tp + marginal_fp) > 0
                    else 0
                ),
            }
        )

    return results


# =============================================================================
# GO Aspect Analysis
# =============================================================================


def analyze_go_aspects(
    base_dir: Path, configs: list[int], gt: dict[str, dict], go_terms: dict[str, dict]
) -> dict[str, Any]:
    """Analyze performance by GO aspect (Biological Process, Molecular Function, Cellular Component).

    Args:
        base_dir: Base directory containing papers_N subdirectories
        configs: List of paper configurations
        gt: Ground truth dict indexed by gene_id
        go_terms: GO ontology dict mapping GO ID -> term info

    Returns:
        Dict containing ground truth counts, prediction counts, and hypothesis by aspect
    """
    # Count GT by aspect
    gt_by_aspect: dict[str, int] = {"P": 0, "F": 0, "C": 0}
    gt_in_corpus_by_aspect: dict[str, int] = {"P": 0, "F": 0, "C": 0}

    for gene_entry in gt.values():
        for ann in gene_entry.get("task1_function", []):
            aspect = ann.get("aspect", "")
            if aspect in gt_by_aspect:
                gt_by_aspect[aspect] += 1
                if ann.get("in_corpus", False):
                    gt_in_corpus_by_aspect[aspect] += 1

    # Analyze predictions by aspect from agent outputs
    aspect_results: dict[str, list[str]] = {"P": [], "F": [], "C": []}

    max_config = max(configs)
    config_dir = base_dir / f"papers_{max_config}"

    for output_file in config_dir.glob("FBgn*.json"):
        try:
            with open(output_file) as f:
                agent_output = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue

        for ann in agent_output.get("task1_function", []):
            go_id = ann.get("go_id", "")
            aspect = ann.get("aspect", "")
            if not aspect and go_id in go_terms:
                ns = go_terms[go_id].get("namespace", "")
                if "biological_process" in ns:
                    aspect = "P"
                elif "molecular_function" in ns:
                    aspect = "F"
                elif "cellular_component" in ns:
                    aspect = "C"

            if aspect in aspect_results:
                aspect_results[aspect].append(go_id)

    return {
        "ground_truth": {
            "total": gt_by_aspect,
            "in_corpus": gt_in_corpus_by_aspect,
        },
        "predictions": {
            "P": len(aspect_results["P"]),
            "F": len(aspect_results["F"]),
            "C": len(aspect_results["C"]),
        },
        "aspect_difficulty_hypothesis": (
            "Molecular Function (F) typically requires biochemical assay evidence "
            "which is more explicitly stated in papers. Biological Process (P) often "
            "requires inference from phenotypes. Cellular Component (C) needs localization data."
        ),
    }


# =============================================================================
# False Positive Analysis
# =============================================================================


def extract_false_positives_detailed(
    all_evals: dict[int, dict],
    base_dir: Path,
    go_terms: dict[str, dict],
) -> list[FalsePositive]:
    """Extract all false positives with full context.

    Args:
        all_evals: Dict mapping config -> evaluation results
        base_dir: Base directory containing papers_N subdirectories
        go_terms: GO ontology dict mapping GO ID -> term info

    Returns:
        List of FalsePositive objects with full context
    """
    fps: list[FalsePositive] = []

    for config, eval_data in all_evals.items():
        for gene in eval_data.get("genes", []):
            gene_id = gene["gene_id"]
            gene_symbol = gene.get("gene_symbol", "")

            # Load agent output to get evidence
            agent_output = load_agent_output_scaling(base_dir, config, gene_id)
            evidence_by_go: dict[str, dict[str, Any]] = {}
            if agent_output:
                for ann in agent_output.get("task1_function", []):
                    go_id = ann.get("go_id", "")
                    evidence = ann.get("evidence", {})
                    evidence_by_go[go_id] = {
                        "pmcid": evidence.get("pmcid", ""),
                        "text": (evidence.get("text", "")[:500] if evidence.get("text") else ""),
                        "aspect": ann.get("aspect", ""),
                    }

            # Task 1 GO FPs
            for match in gene.get("task1_go", {}).get("matches", []):
                if match.get("matched_gt_go") is None:
                    pred_go = match["predicted_go"]
                    term_info = go_terms.get(pred_go, {})
                    evidence = evidence_by_go.get(pred_go, {})

                    fps.append(
                        FalsePositive(
                            gene_id=gene_id,
                            gene_symbol=gene_symbol,
                            predicted_id=pred_go,
                            predicted_term_name=term_info.get("name", ""),
                            task="task1_go",
                            similarity=match.get("similarity", 0.0),
                            paper_config=config,
                            evidence_pmcid=evidence.get("pmcid"),
                            evidence_text=evidence.get("text"),
                            aspect=evidence.get("aspect"),
                        )
                    )

    return fps


def categorize_false_positives_detailed(
    fps: list[FalsePositive], gt: dict[str, dict]
) -> dict[str, Any]:
    """Deep categorization of false positive patterns.

    Categorizes FPs by likely cause based on similarity to ground truth:
    - hallucinated: Very low similarity (<0.1), unrelated or non-existent terms
    - overgeneralized: Related but too general (0.1-0.3 sim)
    - wrong_aspect: Right concept, wrong GO aspect (0.3-0.5 sim)
    - near_miss: Close but not matching (0.5-0.7 sim)
    - wrong_gene: Correct term but for different gene (>0.7 sim)

    Args:
        fps: List of FalsePositive objects
        gt: Ground truth dict indexed by gene_id

    Returns:
        Dict containing category counts, examples, and statistics
    """
    # Group by various dimensions
    by_config: dict[int, list[FalsePositive]] = defaultdict(list)
    by_gene: dict[str, list[FalsePositive]] = defaultdict(list)
    by_aspect: dict[str, list[FalsePositive]] = defaultdict(list)
    by_term: dict[str, int] = defaultdict(int)

    for fp in fps:
        by_config[fp.paper_config].append(fp)
        by_gene[fp.gene_id].append(fp)
        if fp.aspect:
            by_aspect[fp.aspect].append(fp)
        if fp.predicted_term_name:
            by_term[fp.predicted_term_name] += 1

    # Identify most common FP terms (potential systematic errors)
    common_fp_terms = sorted(by_term.items(), key=lambda x: -x[1])[:20]

    # Analyze FP characteristics
    sims = [fp.similarity for fp in fps]

    # Categorize by likely cause
    categories: dict[str, list[FalsePositive]] = {
        "hallucinated": [],  # Very low similarity, term doesn't exist or unrelated
        "overgeneralized": [],  # Related but too general
        "wrong_gene": [],  # Correct term but for wrong gene
        "wrong_aspect": [],  # Right concept but wrong GO aspect
        "near_miss": [],  # High similarity but not exact match
    }

    for fp in fps:
        if fp.similarity < 0.1:
            categories["hallucinated"].append(fp)
        elif fp.similarity < 0.3:
            categories["overgeneralized"].append(fp)
        elif fp.similarity < 0.5:
            categories["wrong_aspect"].append(fp)
        elif fp.similarity < 0.7:
            categories["near_miss"].append(fp)
        else:
            # High similarity but still FP - might be wrong gene
            categories["wrong_gene"].append(fp)

    # Concrete examples for each category
    examples: dict[str, list[dict[str, Any]]] = {}
    for cat, fps_list in categories.items():
        if fps_list:
            # Pick diverse examples
            example_fps = fps_list[:5]
            examples[cat] = [
                {
                    "gene": f"{fp.gene_symbol} ({fp.gene_id})",
                    "predicted": f"{fp.predicted_id} ({fp.predicted_term_name})",
                    "similarity": fp.similarity,
                    "evidence_pmcid": fp.evidence_pmcid,
                    "evidence_excerpt": (
                        fp.evidence_text[:200] + "..."
                        if fp.evidence_text and len(fp.evidence_text) > 200
                        else fp.evidence_text
                    ),
                }
                for fp in example_fps
            ]

    return {
        "total_fps": len(fps),
        "by_config": {k: len(v) for k, v in sorted(by_config.items())},
        "by_aspect": {k: len(v) for k, v in by_aspect.items()},
        "most_common_fp_terms": common_fp_terms,
        "genes_with_most_fps": sorted(
            [(gene_id, len(fp_list)) for gene_id, fp_list in by_gene.items()],
            key=lambda x: -x[1],
        )[:10],
        "similarity_distribution": {
            "mean": float(np.mean(sims)) if sims else 0.0,
            "std": float(np.std(sims)) if sims else 0.0,
            "quartiles": (
                [float(np.percentile(sims, q)) for q in [25, 50, 75]] if sims else [0, 0, 0]
            ),
        },
        "category_counts": {k: len(v) for k, v in categories.items()},
        "category_examples": examples,
    }
