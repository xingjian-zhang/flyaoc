"""Unified data loading utilities for evaluation and analysis.

This module consolidates data loading functions that were previously duplicated
across eval/loader.py, scripts/deep_analysis.py, and scripts/analyze_scaling.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DataLoadError(Exception):
    """Raised when data loading fails."""

    pass


# =============================================================================
# Ground Truth Loading
# =============================================================================


def load_ground_truth(
    gene_id: str | None = None,
    ground_truth_path: str | Path = "data/ground_truth_top100.jsonl",
    as_dict: bool = False,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Load ground truth data from JSONL file.

    Args:
        gene_id: Optional gene ID to filter to single gene
        ground_truth_path: Path to JSONL file
        as_dict: If True, return dict indexed by gene_id (useful for lookups)

    Returns:
        - If gene_id provided: single gene dict
        - If as_dict=True: dict mapping gene_id -> gene data
        - Otherwise: list of all gene dicts

    Raises:
        FileNotFoundError: If file doesn't exist
        DataLoadError: If gene_id provided but not found
    """
    filepath = Path(ground_truth_path)
    if not filepath.exists():
        raise FileNotFoundError(f"Ground truth file not found: {filepath}")

    genes: list[dict[str, Any]] = []
    with open(filepath) as f:
        for line in f:
            gene = json.loads(line)
            if gene_id is not None and gene["gene_id"] == gene_id:
                return gene
            genes.append(gene)

    if gene_id is not None:
        raise DataLoadError(f"Gene {gene_id} not found in ground truth")

    if as_dict:
        return {g["gene_id"]: g for g in genes}

    return genes


def get_all_gene_ids(
    ground_truth_path: str | Path = "data/ground_truth_top100.jsonl",
) -> list[str]:
    """Get all gene IDs from ground truth.

    Args:
        ground_truth_path: Path to ground truth JSONL file

    Returns:
        List of gene IDs
    """
    genes = load_ground_truth(ground_truth_path=ground_truth_path)
    if isinstance(genes, dict):
        # Single gene was returned (shouldn't happen without gene_id)
        return [genes["gene_id"]]
    return [g["gene_id"] for g in genes]


# =============================================================================
# Agent Output Loading
# =============================================================================


def load_agent_output(
    gene_id: str,
    output_dir: str | Path = "outputs/",
    extract_output: bool = True,
) -> dict[str, Any]:
    """Load agent output for a gene.

    Args:
        gene_id: The gene ID (FBgn...) to load
        output_dir: Directory containing agent output files
        extract_output: If True, extract nested ["output"] key if present

    Returns:
        Agent output dict with task1_function, task2_expression, task3_synonyms

    Raises:
        FileNotFoundError: If no output file found for gene
        DataLoadError: If output is null (run failed)
    """
    output_path = Path(output_dir)

    # Try different naming patterns
    patterns = [
        f"{gene_id}.json",
        f"{gene_id}_output.json",
        f"*{gene_id}*.json",
    ]

    for pattern in patterns:
        matches = list(output_path.glob(pattern))
        if matches:
            with open(matches[0]) as f:
                data = json.load(f)

            if extract_output and "output" in data:
                output = data["output"]
                # Handle failed runs where output is null
                if output is None:
                    raise DataLoadError(
                        f"Gene {gene_id} has null output (run likely failed). "
                        f"Error: {data.get('error', 'unknown')}"
                    )
                return output
            return data

    # Also try loading by gene symbol from filenames
    for file in output_path.glob("*.json"):
        with open(file) as f:
            data = json.load(f)
        output = data.get("output", data)
        if output is not None and output.get("gene_id") == gene_id:
            return output

    raise FileNotFoundError(f"No output file found for gene {gene_id} in {output_dir}")


def load_agent_output_scaling(
    base_dir: Path,
    config: int,
    gene_id: str,
) -> dict[str, Any] | None:
    """Load agent output for a gene in a scaling experiment.

    Args:
        base_dir: Base directory of scaling experiment
        config: Paper configuration (e.g., 1, 3, 5, 10)
        gene_id: Gene ID to load

    Returns:
        Agent output dict or None if not found/invalid
    """
    path = base_dir / f"papers_{config}" / f"{gene_id}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


# =============================================================================
# Evaluation Results Loading
# =============================================================================


def load_eval_results(
    path: Path,
    allow_missing: bool = True,
) -> dict[str, Any] | None:
    """Load evaluation results JSON.

    Args:
        path: Path to eval_results.json (or directory containing it)
        allow_missing: If True, return None for missing files; if False, raise

    Returns:
        Evaluation results dict or None if missing and allow_missing=True

    Raises:
        FileNotFoundError: If file missing and allow_missing=False
    """
    # Handle both file path and directory path
    if path.is_dir():
        path = path / "eval_results.json"

    if not path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(f"Evaluation results not found: {path}")

    with open(path) as f:
        return json.load(f)


def load_run_summary(
    path: Path,
    allow_missing: bool = True,
) -> dict[str, Any] | None:
    """Load run summary JSON.

    Args:
        path: Path to run_summary.json (or directory containing it)
        allow_missing: If True, return None for missing files; if False, raise

    Returns:
        Run summary dict or None if missing and allow_missing=True

    Raises:
        FileNotFoundError: If file missing and allow_missing=False
    """
    # Handle both file path and directory path
    if path.is_dir():
        path = path / "run_summary.json"

    if not path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(f"Run summary not found: {path}")

    with open(path) as f:
        return json.load(f)


def load_trace(
    path: Path,
    allow_missing: bool = True,
) -> dict[str, Any] | None:
    """Load agent trace JSON.

    Args:
        path: Path to trace JSON file
        allow_missing: If True, return None for missing files; if False, raise

    Returns:
        Trace dict or None if missing and allow_missing=True

    Raises:
        FileNotFoundError: If file missing and allow_missing=False
    """
    if not path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(f"Trace not found: {path}")

    with open(path) as f:
        return json.load(f)


# =============================================================================
# Scaling Experiment Loading
# =============================================================================


def load_scaling_eval_results(
    base_dir: Path,
    config: int,
) -> dict[str, Any] | None:
    """Load evaluation results for a paper config in a scaling experiment.

    Args:
        base_dir: Base directory of scaling experiment
        config: Paper configuration (e.g., 1, 3, 5, 10)

    Returns:
        Evaluation results dict or None if not found
    """
    path = base_dir / f"papers_{config}" / "eval_results.json"
    return load_eval_results(path, allow_missing=True)


def load_scaling_run_summary(
    base_dir: Path,
    config: int,
) -> dict[str, Any] | None:
    """Load run summary for a paper config in a scaling experiment.

    Args:
        base_dir: Base directory of scaling experiment
        config: Paper configuration (e.g., 1, 3, 5, 10)

    Returns:
        Run summary dict or None if not found
    """
    path = base_dir / f"papers_{config}" / "run_summary.json"
    return load_run_summary(path, allow_missing=True)


def load_scaling_trace(
    base_dir: Path,
    config: int,
    gene_id: str,
) -> dict[str, Any] | None:
    """Load trace for a gene in a scaling experiment.

    Args:
        base_dir: Base directory of scaling experiment
        config: Paper configuration (e.g., 1, 3, 5, 10)
        gene_id: Gene ID to load trace for

    Returns:
        Trace dict or None if not found
    """
    path = base_dir / f"papers_{config}" / "traces" / f"{gene_id}_trace.json"
    return load_trace(path, allow_missing=True)


def load_all_scaling_eval_results(
    base_dir: Path,
    configs: list[int],
) -> dict[int, dict[str, Any]]:
    """Load evaluation results for all configs in a scaling experiment.

    Args:
        base_dir: Base directory of scaling experiment
        configs: List of paper configurations to load

    Returns:
        Dict mapping config -> eval_results (only configs with data)
    """
    results = {}
    for config in configs:
        data = load_scaling_eval_results(base_dir, config)
        if data:
            results[config] = data
    return results


# =============================================================================
# Ontology Loading
# =============================================================================


def load_go_ontology(
    path: str | Path = "ontologies/go-basic.obo",
) -> dict[str, dict[str, str]]:
    """Load GO ontology for term name lookup.

    Args:
        path: Path to GO OBO file

    Returns:
        Dict mapping GO ID -> {"name": str, "namespace": str}
        Empty dict if file doesn't exist
    """
    go_path = Path(path)
    if not go_path.exists():
        return {}

    terms: dict[str, dict[str, str]] = {}
    current_id: str | None = None
    current_name: str | None = None
    current_namespace: str | None = None

    with open(go_path) as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                if current_id and current_name:
                    terms[current_id] = {
                        "name": current_name,
                        "namespace": current_namespace or "",
                    }
                current_id = None
                current_name = None
                current_namespace = None
            elif line.startswith("id: GO:"):
                current_id = line[4:]
            elif line.startswith("name: "):
                current_name = line[6:]
            elif line.startswith("namespace: "):
                current_namespace = line[11:]

        # Don't forget the last term
        if current_id and current_name:
            terms[current_id] = {
                "name": current_name,
                "namespace": current_namespace or "",
            }

    return terms
