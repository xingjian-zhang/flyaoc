"""Case study generation for deep analysis of scaling experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eval.data_loader import load_scaling_trace

from .dataclasses import GeneTrajectory


def generate_detailed_case_study(
    gene_id: str,
    trajectories: list[GeneTrajectory],
    all_evals: dict[int, dict[str, Any]],
    gt: dict[str, dict[str, Any]],
    base_dir: Path,
    go_terms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Generate comprehensive case study for a single gene."""

    traj = next((t for t in trajectories if t.gene_id == gene_id), None)
    if not traj:
        return {"error": f"Gene {gene_id} not found"}

    gt_entry = gt.get(gene_id, {})

    case: dict[str, Any] = {
        "gene_id": gene_id,
        "gene_symbol": traj.gene_symbol,
        "ground_truth_summary": {
            "task1_total": traj.gt_task1_total,
            "task1_in_corpus": traj.gt_task1_in_corpus,
            "task2_total": traj.gt_task2_total,
            "task2_in_corpus": traj.gt_task2_in_corpus,
        },
        "performance_trajectory": {
            "task1_f1_by_config": traj.task1_f1,
            "task1_precision_by_config": traj.task1_precision,
            "task1_recall_by_config": traj.task1_recall,
            "task1_scaling_gain": traj.scaling_gain("task1"),
            "task1_best_config": traj.best_config("task1"),
            "task1_category": traj.category("task1"),
        },
    }

    # Get detailed evaluation from max config
    max_config = max(all_evals.keys())
    for gene in all_evals[max_config].get("genes", []):
        if gene["gene_id"] == gene_id:
            matches = gene.get("task1_go", {}).get("matches", [])

            # Classify matches
            true_positives = []
            false_positives = []

            for match in matches:
                pred_go = match["predicted_go"]
                term_name = go_terms.get(pred_go, {}).get("name", "Unknown")

                if match.get("matched_gt_go"):
                    true_positives.append(
                        {
                            "predicted": f"{pred_go} ({term_name})",
                            "matched": match["matched_gt_go"],
                            "similarity": match["similarity"],
                            "is_exact": match["is_exact"],
                        }
                    )
                else:
                    false_positives.append(
                        {
                            "predicted": f"{pred_go} ({term_name})",
                            "similarity": match["similarity"],
                        }
                    )

            case["annotation_details"] = {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "tp_count": len(true_positives),
                "fp_count": len(false_positives),
            }
            break

    # Get ground truth annotations for comparison
    gt_annotations = []
    for ann in gt_entry.get("task1_function", [])[:10]:
        go_id = ann.get("go_id", "")
        term_name = go_terms.get(go_id, {}).get("name", "Unknown")
        gt_annotations.append(
            {
                "go_id": go_id,
                "term_name": term_name,
                "aspect": ann.get("aspect", ""),
                "in_corpus": ann.get("in_corpus", False),
                "reference": ann.get("reference", ""),
            }
        )
    case["ground_truth_annotations"] = gt_annotations

    # Analyze trace for this gene
    trace = load_scaling_trace(base_dir, max_config, gene_id)
    if trace:
        papers_read = []
        for event in trace.get("events", []):
            if event.get("event_type") == "tool_call" and event.get("tool") == "get_paper_text":
                pmcid = event.get("args", {}).get("pmcid", "")
                papers_read.append(pmcid)
        case["papers_read"] = papers_read
        case["papers_read_count"] = len(papers_read)

    # Narrative analysis
    narratives = []

    if traj.category("task1") == "improves":
        narratives.append(
            f"This gene shows positive scaling: F1 improves from "
            f"{traj.task1_f1.get(min(traj.task1_f1.keys()), 0):.1%} to "
            f"{traj.task1_f1.get(max(traj.task1_f1.keys()), 0):.1%} as more papers are read."
        )
    elif traj.category("task1") == "degrades":
        narratives.append(
            f"This gene shows negative scaling: F1 degrades from "
            f"{traj.task1_f1.get(min(traj.task1_f1.keys()), 0):.1%} to "
            f"{traj.task1_f1.get(max(traj.task1_f1.keys()), 0):.1%}. "
            f"More papers may introduce noise or off-topic annotations."
        )

    fp_count = case.get("annotation_details", {}).get("fp_count", 0)
    if fp_count > 3:
        narratives.append(
            f"High false positive count ({fp_count}) suggests the agent may be "
            f"over-extracting or misattributing functions from related genes in the papers."
        )

    case["narrative_analysis"] = narratives

    return case
