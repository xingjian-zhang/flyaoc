"""Data collection functions for scaling analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from agent.agentic.hooks import MODEL_PRICING
from eval.data_loader import load_eval_results, load_run_summary

from .config_metrics import ConfigMetrics
from .extraction import extract_metrics


def _infer_model_from_dir(dir_name: str) -> str:
    """Infer model name from experiment directory name.

    Looks for patterns like 'multi-gpt4o-950d202' or 'single-gpt5-950d202'.
    Defaults to 'gpt-5-mini' if no model pattern found.
    """
    # Check for explicit model in directory name
    if "gpt5-" in dir_name or "-gpt5" in dir_name:
        return "gpt-5"
    if "gpt4o-mini" in dir_name:
        return "gpt-4o-mini"
    if "gpt4o" in dir_name:
        return "gpt-4o"
    if "gpt4-turbo" in dir_name:
        return "gpt-4-turbo"
    if "gpt4" in dir_name:
        return "gpt-4"
    # Default to gpt-5-mini (the project default)
    return "gpt-5-mini"


def _calculate_cost_from_tokens(
    cached_tokens: int, uncached_tokens: int, output_tokens: int, model: str
) -> float:
    """Calculate cost from token counts using current MODEL_PRICING."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
    uncached_cost = (uncached_tokens / 1000) * pricing["input"]
    cached_cost = (cached_tokens / 1000) * pricing.get("cached", pricing["input"] * 0.5)
    output_cost = (output_tokens / 1000) * pricing["output"]
    return uncached_cost + cached_cost + output_cost


def _recalculate_costs(
    output_dir: Path, run_summary: dict[str, Any], experiment_name: str
) -> dict[str, Any]:
    """Recalculate costs from token counts in individual gene output files.

    Args:
        output_dir: Directory containing gene output JSON files
        run_summary: Original run summary with potentially stale costs
        experiment_name: Name of experiment directory for inferring model

    Returns:
        Updated run_summary with recalculated costs
    """
    model = _infer_model_from_dir(experiment_name)
    total_cost = 0.0
    updated_genes = []

    # Get gene list from run_summary, or scan directory for gene files
    genes_list = run_summary.get("genes", [])
    if not genes_list:
        # Scan directory for FBgn*.json files
        gene_files = sorted(output_dir.glob("FBgn*.json"))
        genes_list = [{"gene_id": f.stem} for f in gene_files]

    for gene_info in genes_list:
        gene_id = gene_info.get("gene_id")
        gene_file = output_dir / f"{gene_id}.json"

        # Copy gene info and try to recalculate cost
        updated_gene = dict(gene_info)

        if gene_file.exists():
            try:
                with open(gene_file) as f:
                    gene_data = json.load(f)
                usage = gene_data.get("usage", {})
                cached = usage.get("cached_tokens", 0)
                uncached = usage.get("uncached_tokens", 0)
                output = usage.get("output_tokens", 0)

                if cached > 0 or uncached > 0 or output > 0:
                    new_cost = _calculate_cost_from_tokens(cached, uncached, output, model)
                    updated_gene["cost_usd"] = new_cost
                    total_cost += new_cost
                else:
                    # No token data, use original cost
                    total_cost += gene_info.get("cost_usd", 0)
            except (json.JSONDecodeError, OSError):
                # Fall back to original cost
                total_cost += gene_info.get("cost_usd", 0)
        else:
            # No gene file, use original cost
            total_cost += gene_info.get("cost_usd", 0)

        updated_genes.append(updated_gene)

    # Create updated run_summary
    updated_summary = dict(run_summary)
    updated_summary["genes"] = updated_genes
    updated_summary["total_cost"] = total_cost

    return updated_summary

def collect_all_metrics(
    base_dir: Path, paper_configs: list[int], recalculate_costs: bool = True
) -> list[ConfigMetrics]:
    """Collect metrics for all paper configurations.

    Args:
        base_dir: Base directory containing papers_N subdirectories
        paper_configs: List of paper configurations to analyze
        recalculate_costs: If True, recalculate costs from token counts using
            current MODEL_PRICING (default). If False, use stored cost_usd values.

    Returns:
        List of ConfigMetrics, one per configuration
    """
    metrics = []

    for max_papers in paper_configs:
        output_dir = base_dir / f"papers_{max_papers}"

        run_summary = load_run_summary(output_dir)
        eval_results = load_eval_results(output_dir)

        if run_summary is None:
            print(f"Warning: No run_summary.json found for papers={max_papers}", file=sys.stderr)
            continue

        if eval_results is None:
            print(f"Warning: No eval_results.json found for papers={max_papers}", file=sys.stderr)
            print(
                f"  Run: uv run python -m eval.run_eval --batch "
                f"--output-dir {output_dir} -o {output_dir}/eval_results.json",
                file=sys.stderr,
            )
            continue

        # Recalculate costs from token counts if requested
        if recalculate_costs:
            run_summary = _recalculate_costs(output_dir, run_summary, base_dir.name)

        config_metrics = extract_metrics(max_papers, run_summary, eval_results)
        metrics.append(config_metrics)

    return metrics
