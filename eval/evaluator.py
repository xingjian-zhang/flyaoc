"""Unified evaluation framework for agent annotation quality assessment.

This module provides:
- Core metric calculations (precision, recall, F1)
- Reference coverage evaluation
- NL description resolution for hidden GO terms
- Main evaluation orchestration (single gene and batch)
- CLI entry point

Simplification: Removed threshold-based soft metrics (binary 0.7 cutoff).
Kept: exact matching + semantic metrics (primary metric).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
from pydantic import BaseModel

from agent.core.ontology import search_go_core

import asyncio

from .data_loader import DataLoadError, get_all_gene_ids, load_agent_output, load_ground_truth
from .recall_at_k import DEFAULT_K_VALUES


def _load_gene_ground_truth(gene_id: str, ground_truth_path: str) -> dict[str, Any]:
    """Load ground truth for a single gene with proper typing.

    This is a type-narrowing wrapper around load_ground_truth.
    """
    gt = load_ground_truth(gene_id, ground_truth_path)
    if isinstance(gt, list):
        raise DataLoadError(f"Expected single gene data for {gene_id}, got list")
    return gt


from .task1_go import Task1Result, evaluate_go
from .task2_expression import Task2Result, evaluate_expression
from .task3_synonyms import Task3Result, evaluate_synonyms

load_dotenv()


# =============================================================================
# Core Metrics
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
    """Calculate precision, recall, and F1 from counts.

    Args:
        tp: True positives (correctly predicted).
        fp: False positives (predicted but not in ground truth).
        fn: False negatives (in ground truth but not predicted).

    Returns:
        Metrics dataclass with precision, recall, F1.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def compute_set_metrics(predicted: set, ground_truth: set) -> Metrics:
    """Compute metrics from two sets.

    Args:
        predicted: Set of predicted items.
        ground_truth: Set of ground truth items.

    Returns:
        Metrics computed from set overlap.
    """
    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)

    return precision_recall_f1(tp, fp, fn)


# =============================================================================
# Semantic Metrics
# =============================================================================


@dataclass
class SemanticMetrics:
    """Semantic precision, recall, F1.

    Instead of binary TP counting, uses sum of ontology similarities as semantic TP.
    This provides partial credit for semantically similar but non-exact matches.

    - precision = sum(best_sim_per_pred) / num_predictions
    - recall = sum(best_sim_per_gt) / num_ground_truth
    - f1 = harmonic mean of precision and recall
    """

    precision: float
    recall: float
    f1: float
    semantic_tp: float  # Sum of similarities for matched GT items (for recall)
    precision_sum: float = 0.0  # Sum of best similarities for predictions (for precision)

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "semantic_tp": round(self.semantic_tp, 4),
        }


# =============================================================================
# Reference Coverage
# =============================================================================


@dataclass
class ReferenceCoverageResult:
    """Reference coverage metrics for a single gene."""

    gene_id: str
    corpus_size: int  # Total papers in corpus for this gene
    effective_papers: int  # Papers containing in-corpus ground truth annotations
    effective_pmcids: set[str]  # The actual PMCIDs that are effective
    agent_cited: int  # Papers cited by agent
    agent_pmcids: set[str]  # PMCIDs cited by agent
    hits: int  # Overlap between agent cited and effective papers
    hit_pmcids: set[str]  # The PMCIDs that were hits

    @property
    def recall(self) -> float:
        """Fraction of effective papers that were cited by agent."""
        return self.hits / self.effective_papers if self.effective_papers > 0 else 0.0

    @property
    def agent_precision(self) -> float:
        """Fraction of agent-cited papers that were effective."""
        return self.hits / self.agent_cited if self.agent_cited > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_size": self.corpus_size,
            "effective_papers": self.effective_papers,
            "effective_pmcids": sorted(self.effective_pmcids),
            "agent_cited": self.agent_cited,
            "agent_pmcids": sorted(self.agent_pmcids),
            "hits": self.hits,
            "hit_pmcids": sorted(self.hit_pmcids),
            "recall": round(self.recall, 4),
            "agent_precision": round(self.agent_precision, 4),
        }


class PMIDtoPMCIDMapper:
    """Maps PMIDs to PMCIDs using FlyBase reference mapping."""

    def __init__(self, mapping_path: str | Path | None = None):
        """Initialize mapper.

        Args:
            mapping_path: Path to FlyBase reference mapping TSV file.
                Defaults to ../dataset_zy/data/fbrf_pmid_pmcid_doi_fb_2025_04.tsv
        """
        self._mapping: dict[str, str] = {}
        self._loaded = False
        self._mapping_path = mapping_path

    def _load(self) -> None:
        """Lazy-load the mapping file."""
        if self._loaded:
            return

        if self._mapping_path is None:
            # Default path relative to this file
            default_path = (
                Path(__file__).parent.parent.parent
                / "dataset_zy"
                / "data"
                / "fbrf_pmid_pmcid_doi_fb_2025_04.tsv"
            )
            self._mapping_path = default_path

        mapping_path = Path(self._mapping_path)
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"PMID-PMCID mapping file not found: {mapping_path}. "
                "Reference coverage evaluation requires this file."
            )

        with open(mapping_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                pmid = row.get("PMID", "")
                pmcid = row.get("PMCID", "")
                if pmid and pmcid:
                    self._mapping[pmid] = pmcid

        self._loaded = True

    def get_pmcid(self, pmid: str) -> str | None:
        """Get PMCID for a PMID.

        Args:
            pmid: PMID string (with or without 'PMID:' prefix)

        Returns:
            PMCID string or None if not found
        """
        self._load()
        # Strip prefix if present
        if pmid.startswith("PMID:"):
            pmid = pmid[5:]
        return self._mapping.get(pmid)

    def __len__(self) -> int:
        self._load()
        return len(self._mapping)


# Global mapper instance
_mapper: PMIDtoPMCIDMapper | None = None


def get_pmid_mapper() -> PMIDtoPMCIDMapper:
    """Get or create the global PMID-PMCID mapper."""
    global _mapper
    if _mapper is None:
        _mapper = PMIDtoPMCIDMapper()
    return _mapper


def load_gene_corpus(
    gene_id: str,
    corpus_path: str | Path = "data/gene_to_pmcids_top100.json",
) -> set[str]:
    """Load the corpus PMCIDs for a gene.

    Args:
        gene_id: FlyBase gene ID
        corpus_path: Path to gene-to-PMCIDs mapping JSON

    Returns:
        Set of PMCIDs in corpus for this gene
    """
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus mapping not found: {corpus_path}")

    with open(corpus_path) as f:
        corpus = json.load(f)

    return set(corpus.get(gene_id, []))


def get_effective_pmcids(
    ground_truth: dict[str, Any],
    corpus_pmcids: set[str],
    mapper: PMIDtoPMCIDMapper | None = None,
) -> set[str]:
    """Get PMCIDs that contain in-corpus ground truth annotations.

    Args:
        ground_truth: Ground truth dict for a gene
        corpus_pmcids: Set of PMCIDs in corpus for this gene
        mapper: PMID-PMCID mapper instance

    Returns:
        Set of PMCIDs that are "effective" (contain findable annotations)
    """
    if mapper is None:
        mapper = get_pmid_mapper()

    effective = set()

    # Check GO annotations
    for ann in ground_truth.get("task1_function", []):
        if ann.get("in_corpus"):
            ref = ann.get("reference", "")
            pmcid = mapper.get_pmcid(ref)
            if pmcid and pmcid in corpus_pmcids:
                effective.add(pmcid)

    # Check expression annotations
    for ann in ground_truth.get("task2_expression", []):
        if ann.get("in_corpus"):
            ref = ann.get("reference", "")
            pmcid = mapper.get_pmcid(ref)
            if pmcid and pmcid in corpus_pmcids:
                effective.add(pmcid)

    return effective


def get_agent_pmcids(agent_output: dict[str, Any]) -> set[str]:
    """Extract PMCIDs cited by agent in its annotations.

    Args:
        agent_output: Agent output dict (the 'output' field from result)

    Returns:
        Set of PMCIDs cited in evidence fields
    """
    pmcids = set()

    # Check GO annotations
    for ann in agent_output.get("task1_function", []):
        pmcid = ann.get("evidence", {}).get("pmcid")
        if pmcid:
            pmcids.add(pmcid)

    # Check expression annotations
    for ann in agent_output.get("task2_expression", []):
        pmcid = ann.get("evidence", {}).get("pmcid")
        if pmcid:
            pmcids.add(pmcid)

    return pmcids


def evaluate_reference_coverage(
    gene_id: str,
    agent_output: dict[str, Any],
    ground_truth: dict[str, Any],
    corpus_path: str | Path = "data/gene_to_pmcids_top100.json",
    mapper: PMIDtoPMCIDMapper | None = None,
) -> ReferenceCoverageResult:
    """Evaluate reference coverage for a single gene.

    Args:
        gene_id: FlyBase gene ID
        agent_output: Agent output dict
        ground_truth: Ground truth dict
        corpus_path: Path to gene-to-PMCIDs mapping
        mapper: PMID-PMCID mapper instance

    Returns:
        ReferenceCoverageResult with coverage metrics
    """
    if mapper is None:
        mapper = get_pmid_mapper()

    # Load corpus for this gene
    corpus_pmcids = load_gene_corpus(gene_id, corpus_path)

    # Get effective and agent PMCIDs
    effective = get_effective_pmcids(ground_truth, corpus_pmcids, mapper)
    agent = get_agent_pmcids(agent_output)

    # Compute overlap
    hits = agent & effective

    return ReferenceCoverageResult(
        gene_id=gene_id,
        corpus_size=len(corpus_pmcids),
        effective_papers=len(effective),
        effective_pmcids=effective,
        agent_cited=len(agent),
        agent_pmcids=agent,
        hits=len(hits),
        hit_pmcids=hits,
    )


# =============================================================================
# NL Description Resolution
# =============================================================================

# Configurable model via environment variable
DEFAULT_RESOLVER_MODEL = os.getenv("NL_RESOLVER_MODEL", "gpt-5-mini")


@dataclass
class ResolverResult:
    """Result of NL→GO resolution."""

    selected_go_id: str | None
    candidates_searched: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_go_id": self.selected_go_id,
            "candidates_searched": self.candidates_searched,
            "error": self.error,
        }


class _AgentOutput(BaseModel):
    """Pydantic model for structured agent output."""

    selected_go_id: str | None = None


# Tool for the agentic resolver
@function_tool
def search_go_ontology(query: str, aspect: str | None = None, limit: int = 10) -> str:
    """Search the Gene Ontology for terms matching the query.

    Args:
        query: Search term (e.g., "transcription", "cell migration", "kinase activity").
               Try different phrasings if initial search returns no results.
        aspect: Optional filter - "P" (Biological Process), "F" (Molecular Function),
                or "C" (Cellular Component). Leave empty to search all.
        limit: Maximum number of results to return.

    Returns:
        JSON array of matching GO terms with go_id, name, definition, and synonyms.
        Returns empty array [] if no matches found.
    """
    results = search_go_core(query, aspect=aspect, limit=limit)
    return json.dumps(results, indent=2)


_AGENTIC_RESOLVER_INSTRUCTIONS = """
You are a Gene Ontology (GO) expert. Your task is to find the best GO term that matches
a given biological description.

INSTRUCTIONS:
1. Use the search_go_ontology tool to search for relevant GO terms
2. If the first search returns no results, try alternative phrasings:
   - Remove gene-specific names (e.g., "rhomboid expression" → "gene expression")
   - Use standard GO vocabulary (e.g., "positive regulation of", "negative regulation of")
   - Try broader terms if specific ones fail
3. You may search up to 3 times with different queries
4. From all candidates found, select the ONE best matching GO term
5. Prefer more specific terms over general ones when both fit

IMPORTANT: You MUST try at least one search. Do your best to find a match.

When done, return the best GO ID, or null if truly nothing matches.
"""


# Cache the agent instance for reuse
_agentic_resolver: Agent | None = None


def _get_agentic_resolver() -> Agent:
    """Lazy-create agentic resolver with search tool."""
    global _agentic_resolver
    if _agentic_resolver is None:
        _agentic_resolver = Agent(
            name="AgenticGOResolver",
            instructions=_AGENTIC_RESOLVER_INSTRUCTIONS,
            model=DEFAULT_RESOLVER_MODEL,
            tools=[search_go_ontology],
            output_type=_AgentOutput,
        )
    return _agentic_resolver


# In-memory cache for resolved descriptions
_resolution_cache: dict[tuple[str, str | None], str | None] = {}


async def resolve_description(
    description: str,
    aspect: str | None = None,
    use_cache: bool = True,
) -> ResolverResult:
    """Resolve NL description to GO ID using an agentic LLM with search tool.

    The agent can search the GO ontology multiple times with different queries
    to find the best matching term.

    Args:
        description: Natural language description of GO term
        aspect: Optional GO aspect hint ("P", "F", "C") included in prompt
        use_cache: Whether to use cached results (default True)

    Returns:
        ResolverResult with selected_go_id or error
    """
    cache_key = (description.strip().lower(), aspect)

    # Check cache
    if use_cache and cache_key in _resolution_cache:
        cached_id = _resolution_cache[cache_key]
        return ResolverResult(
            selected_go_id=cached_id,
            candidates_searched=0,  # Indicates cache hit
            error=None,
        )

    # Build prompt
    aspect_hint = ""
    if aspect:
        aspect_map = {"P": "Biological Process", "F": "Molecular Function", "C": "Cellular Component"}
        aspect_hint = f"\nHint: This is likely a {aspect_map.get(aspect, aspect)} term."

    prompt = f"""Find the best GO term for this biological description:

"{description}"{aspect_hint}

Search the ontology and return the best matching GO ID."""

    # Run agentic resolver
    try:
        agent = _get_agentic_resolver()
        result = await Runner.run(starting_agent=agent, input=prompt)
        output = result.final_output_as(_AgentOutput)

        # Cache the result
        if use_cache:
            _resolution_cache[cache_key] = output.selected_go_id

        return ResolverResult(
            selected_go_id=output.selected_go_id,
            candidates_searched=1,  # Agent searched at least once
        )
    except Exception as e:
        return ResolverResult(
            selected_go_id=None,
            candidates_searched=0,
            error=str(e),
        )


def clear_resolution_cache() -> None:
    """Clear the resolution cache."""
    _resolution_cache.clear()


async def resolve_descriptions_in_output(
    agent_output: dict[str, Any],
    verbose: bool = False,
) -> dict[str, Any]:
    """Resolve NL descriptions to GO IDs in agent output.

    Finds annotations with 'description' but no 'go_id' and resolves them
    using the agentic LLM resolver. Modifies the output in place and returns it.

    Args:
        agent_output: Agent output dict with task1_function, etc.
        verbose: Print progress messages.

    Returns:
        Modified agent_output with resolved GO IDs.
    """
    task1 = agent_output.get("task1_function", [])
    resolved_count = 0
    failed_count = 0

    for ann in task1:
        # Skip if already has go_id
        if ann.get("go_id"):
            continue

        description = ann.get("description")
        if not description:
            continue

        # Resolve description to GO ID
        aspect = ann.get("aspect")
        result = await resolve_description(description, aspect=aspect)

        if result.selected_go_id:
            ann["go_id"] = result.selected_go_id
            resolved_count += 1
            if verbose:
                print(f"  Resolved: '{description[:50]}...' → {result.selected_go_id}")
        else:
            failed_count += 1
            if verbose:
                print(f"  Failed: '{description[:50]}...' ({result.error})")

    if verbose and (resolved_count > 0 or failed_count > 0):
        print(f"  Resolution: {resolved_count} resolved, {failed_count} failed")

    return agent_output


# =============================================================================
# Main Evaluation
# =============================================================================


@dataclass
class EvaluationResult:
    """Complete evaluation result for a gene."""

    gene_id: str
    gene_symbol: str
    task1: Task1Result
    task2: Task2Result
    task3: Task3Result
    reference_coverage: ReferenceCoverageResult | None = None

    def to_dict(self) -> dict:
        result = {
            "gene_id": self.gene_id,
            "gene_symbol": self.gene_symbol,
            "task1_go": self.task1.to_dict(),
            "task2_expression": self.task2.to_dict(),
            "task3_synonyms": self.task3.to_dict(),
        }
        if self.reference_coverage:
            result["reference_coverage"] = self.reference_coverage.to_dict()
        return result


def create_zero_score_result(
    gene_id: str,
    ground_truth_path: str = "data/ground_truth_top100.jsonl",
) -> EvaluationResult:
    """Create an EvaluationResult with zero scores for a failed gene.

    Used when agent run failed (e.g., context overflow). All metrics are zero,
    with FN counts reflecting the full ground truth (everything was missed).

    Args:
        gene_id: Gene ID that failed.
        ground_truth_path: Path to ground truth JSONL.

    Returns:
        EvaluationResult with zero P/R/F1 and proper FN counts.
    """
    from .task1_go import Metrics as T1Metrics
    from .task1_go import SemanticMetrics as T1SimMetrics
    from .task2_expression import Metrics as T2Metrics
    from .task2_expression import SemanticMetrics as T2SimMetrics
    from .task3_synonyms import Metrics as T3Metrics

    gt = _load_gene_ground_truth(gene_id, ground_truth_path)

    # Task 1: GO terms
    gt_go = gt.get("task1_function", [])
    gt_go_in_corpus = [g for g in gt_go if g.get("in_corpus", False)]
    task1 = Task1Result(
        exact_metrics=T1Metrics(
            precision=0.0, recall=0.0, f1=0.0, tp=0, fp=0, fn=len(gt_go_in_corpus)
        ),
        semantic=T1SimMetrics(
            precision=0.0, recall=0.0, f1=0.0, semantic_tp=0.0, precision_sum=0.0
        ),
        in_corpus_exact_recall=0.0,
        full_recall=0.0,
        full_semantic_recall=0.0,
        matches=[],
        predicted_count=0,
        gt_total_count=len(gt_go),
        gt_in_corpus_count=len(gt_go_in_corpus),
    )

    # Task 2: Expression
    gt_expr = gt.get("task2_expression", [])
    gt_expr_in_corpus = [e for e in gt_expr if e.get("in_corpus", False)]
    task2 = Task2Result(
        anatomy_metrics=T2Metrics(
            precision=0.0, recall=0.0, f1=0.0, tp=0, fp=0, fn=len(gt_expr_in_corpus)
        ),
        anatomy_semantic=T2SimMetrics(
            precision=0.0, recall=0.0, f1=0.0, semantic_tp=0.0, precision_sum=0.0
        ),
        stage_metrics=T2Metrics(
            precision=0.0, recall=0.0, f1=0.0, tp=0, fp=0, fn=len(gt_expr_in_corpus)
        ),
        tuple_metrics=T2Metrics(
            precision=0.0, recall=0.0, f1=0.0, tp=0, fp=0, fn=len(gt_expr_in_corpus)
        ),
        in_corpus_exact_recall=0.0,
        full_recall=0.0,
        full_semantic_recall=0.0,
        predicted_count=0,
        gt_total_count=len(gt_expr),
        gt_in_corpus_count=len(gt_expr_in_corpus),
    )

    # Task 3: Synonyms
    gt_syn = gt.get("task3_synonyms", {})
    gt_fullnames = gt_syn.get("fullname_synonyms", [])
    gt_symbols = gt_syn.get("symbol_synonyms", [])
    gt_fn_in_corpus = [f for f in gt_fullnames if isinstance(f, dict) and f.get("in_corpus", False)]
    gt_sym_in_corpus = [s for s in gt_symbols if isinstance(s, dict) and s.get("in_corpus", False)]
    task3 = Task3Result(
        fullname_metrics=T3Metrics(
            precision=0.0, recall=0.0, f1=0.0, tp=0, fp=0, fn=len(gt_fn_in_corpus)
        ),
        symbol_metrics=T3Metrics(
            precision=0.0, recall=0.0, f1=0.0, tp=0, fp=0, fn=len(gt_sym_in_corpus)
        ),
        combined_metrics=T3Metrics(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            tp=0,
            fp=0,
            fn=len(gt_fn_in_corpus) + len(gt_sym_in_corpus),
        ),
        in_corpus_fullname_recall=0.0,
        in_corpus_symbol_recall=0.0,
        in_corpus_combined_recall=0.0,
        full_fullname_recall=0.0,
        full_symbol_recall=0.0,
        full_combined_recall=0.0,
        gt_fullname_total=len(gt_fullnames),
        gt_fullname_in_corpus=len(gt_fn_in_corpus),
        gt_symbol_total=len(gt_symbols),
        gt_symbol_in_corpus=len(gt_sym_in_corpus),
    )

    return EvaluationResult(
        gene_id=gene_id,
        gene_symbol=gt.get("gene_symbol", ""),
        task1=task1,
        task2=task2,
        task3=task3,
        reference_coverage=None,  # No predictions, no coverage
    )


def evaluate_gene(
    gene_id: str,
    output_dir: str = "outputs/",
    ground_truth_path: str = "data/ground_truth_top100.jsonl",
    pmid_mapper: PMIDtoPMCIDMapper | None = None,
    corpus_path: str = "data/gene_to_pmcids_top100.json",
    include_reference_coverage: bool = True,
    resolve_descriptions: bool = False,
    verbose: bool = False,
) -> EvaluationResult:
    """Evaluate agent output for a single gene.

    Args:
        gene_id: Gene ID to evaluate (FBgn...).
        output_dir: Directory containing agent outputs.
        ground_truth_path: Path to ground truth JSONL.
        pmid_mapper: PMID-PMCID mapper for reference coverage.
        corpus_path: Path to gene-to-PMCIDs mapping.
        include_reference_coverage: Whether to evaluate reference coverage.
        resolve_descriptions: If True, resolve NL descriptions to GO IDs before eval.
        verbose: Print progress messages.

    Returns:
        EvaluationResult with metrics for all three tasks.
    """
    # Load data
    gt = _load_gene_ground_truth(gene_id, ground_truth_path)
    agent_output = load_agent_output(gene_id, output_dir)

    # Resolve descriptions if requested
    if resolve_descriptions:
        if verbose:
            print(f"Resolving descriptions for {gene_id}...")
        agent_output = asyncio.run(
            resolve_descriptions_in_output(agent_output, verbose=verbose)
        )

    # Evaluate each task
    task1 = evaluate_go(
        predicted=agent_output.get("task1_function", []),
        ground_truth=gt.get("task1_function", []),
    )

    task2 = evaluate_expression(
        predicted=agent_output.get("task2_expression", []),
        ground_truth=gt.get("task2_expression", []),
    )

    task3 = evaluate_synonyms(
        predicted=agent_output.get("task3_synonyms", {}),
        ground_truth=gt.get("task3_synonyms", {}),
    )

    # Evaluate reference coverage
    ref_coverage = None
    if include_reference_coverage:
        try:
            if pmid_mapper is None:
                pmid_mapper = get_pmid_mapper()
            ref_coverage = evaluate_reference_coverage(
                gene_id=gene_id,
                agent_output=agent_output,
                ground_truth=gt,
                corpus_path=corpus_path,
                mapper=pmid_mapper,
            )
        except FileNotFoundError:
            # PMID mapping file not available, skip reference coverage
            pass

    return EvaluationResult(
        gene_id=gene_id,
        gene_symbol=gt.get("gene_symbol", ""),
        task1=task1,
        task2=task2,
        task3=task3,
        reference_coverage=ref_coverage,
    )


@dataclass
class BatchResult:
    """Results from batch evaluation."""

    results: list[EvaluationResult]
    aggregate: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "genes": [r.to_dict() for r in self.results],
            "aggregate": self.aggregate,
        }


def _mean(values: list[float]) -> float:
    """Compute mean of a list, returning 0 if empty."""
    return sum(values) / len(values) if values else 0.0


def _aggregate_recall_at_k(
    results: list,
    get_recall_at_k,
    get_gt_in_corpus_count=None,
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """Aggregate recall@k metrics across genes (macro and micro averaged).

    Macro-average: mean of per-gene recall@k values (each gene weighted equally).
    Micro-average: pools numerators and denominators across genes before dividing,
        so genes with more GT items contribute proportionally. This avoids bias
        from genes with small GT sets achieving trivially high recall.

    Args:
        results: List of evaluation results.
        get_recall_at_k: Function to extract RecallAtKResult from a result.
        get_gt_in_corpus_count: Optional fallback function to extract GT in-corpus count
            from a result when recall_at_k is None. If not provided, genes with
            None recall_at_k are excluded from micro-average.
        k_values: List of k values to aggregate. Defaults to DEFAULT_K_VALUES.

    Returns:
        Dict with mean (macro) and micro exact/soft recall@k for each k.
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    # Macro-average accumulators
    exact_at_k: dict[int, list[float]] = {k: [] for k in k_values}
    soft_at_k: dict[int, list[float]] = {k: [] for k in k_values}

    # Micro-average accumulators
    exact_num: dict[int, float] = dict.fromkeys(k_values, 0.0)
    soft_num: dict[int, float] = dict.fromkeys(k_values, 0.0)
    total_gt = 0

    for r in results:
        recall_at_k = get_recall_at_k(r)
        if recall_at_k is None:
            # Macro: include 0.0 to match mean_recall denominator
            for k in k_values:
                exact_at_k[k].append(0.0)
                soft_at_k[k].append(0.0)
            # Micro: include GT count in denominator (0 numerator)
            if get_gt_in_corpus_count is not None:
                total_gt += get_gt_in_corpus_count(r)
        else:
            gt_count = recall_at_k.gt_in_corpus_count
            total_gt += gt_count
            for k in k_values:
                exact_val = recall_at_k.exact_recall_at_k.get(k, 0.0)
                soft_val = recall_at_k.semantic_recall_at_k.get(k, 0.0)
                # Macro
                exact_at_k[k].append(exact_val)
                soft_at_k[k].append(soft_val)
                # Micro: recover numerator = recall * gt_count
                exact_num[k] += exact_val * gt_count
                soft_num[k] += soft_val * gt_count

    return {
        "k_values": k_values,
        "mean_exact_at_k": {str(k): round(_mean(exact_at_k[k]), 4) for k in k_values},
        "mean_semantic_at_k": {str(k): round(_mean(soft_at_k[k]), 4) for k in k_values},
        "micro_exact_at_k": {
            str(k): round(exact_num[k] / total_gt, 4) if total_gt > 0 else 0.0
            for k in k_values
        },
        "micro_semantic_at_k": {
            str(k): round(soft_num[k] / total_gt, 4) if total_gt > 0 else 0.0
            for k in k_values
        },
    }


def evaluate_batch(
    gene_ids: list[str] | None = None,
    output_dir: str = "outputs/",
    ground_truth_path: str = "data/ground_truth_top100.jsonl",
    corpus_path: str = "data/gene_to_pmcids_top100.json",
    include_reference_coverage: bool = True,
    resolve_descriptions: bool = False,
    verbose: bool = False,
) -> BatchResult:
    """Evaluate agent outputs for multiple genes.

    Args:
        gene_ids: List of gene IDs to evaluate. If None, evaluates all genes
            that have output files.
        output_dir: Directory containing agent outputs.
        ground_truth_path: Path to ground truth JSONL.
        corpus_path: Path to gene-to-PMCIDs mapping.
        include_reference_coverage: Whether to evaluate reference coverage.
        resolve_descriptions: If True, resolve NL descriptions to GO IDs before eval.
        verbose: Print progress messages.

    Returns:
        BatchResult with per-gene and aggregate metrics.
    """
    if gene_ids is None:
        gene_ids = get_all_gene_ids(ground_truth_path)

    # Initialize PMID mapper once for efficiency
    pmid_mapper = None
    if include_reference_coverage:
        try:
            pmid_mapper = get_pmid_mapper()
        except FileNotFoundError:
            include_reference_coverage = False

    results: list[EvaluationResult] = []
    skipped: list[str] = []
    failed: list[str] = []

    for gene_id in gene_ids:
        try:
            result = evaluate_gene(
                gene_id=gene_id,
                output_dir=output_dir,
                ground_truth_path=ground_truth_path,
                pmid_mapper=pmid_mapper,
                corpus_path=corpus_path,
                include_reference_coverage=include_reference_coverage,
                resolve_descriptions=resolve_descriptions,
                verbose=verbose,
            )
            results.append(result)
        except FileNotFoundError:
            skipped.append(gene_id)
        except DataLoadError:
            # Agent run failed (e.g., context overflow) - count as zero-score
            failed.append(gene_id)
            result = create_zero_score_result(gene_id, ground_truth_path)
            results.append(result)

    # Compute aggregate metrics
    aggregate = compute_aggregate_metrics(results, skipped, failed)

    return BatchResult(results=results, aggregate=aggregate)


def compute_aggregate_metrics(
    results: list[EvaluationResult],
    skipped: list[str],
    failed: list[str] | None = None,
) -> dict[str, Any]:
    """Compute aggregate metrics across all evaluated genes.

    Args:
        results: List of evaluation results.
        skipped: Gene IDs with no output files.
        failed: Gene IDs with failed runs (null output), included with zero scores.
    """
    if failed is None:
        failed = []

    if not results:
        return {"error": "No genes evaluated", "skipped": skipped, "failed": failed}

    # ==================== Task 1: GO ====================
    # Exact metrics
    t1_exact_p = [r.task1.exact_metrics.precision for r in results]
    t1_exact_r = [r.task1.exact_metrics.recall for r in results]
    t1_exact_f1 = [r.task1.exact_metrics.f1 for r in results]
    t1_in_corpus_exact_recall = [r.task1.in_corpus_exact_recall for r in results]
    t1_full_recall = [r.task1.full_recall for r in results]

    # Micro metrics (sum tp/fp/fn across all genes)
    t1_exact_tp = sum(r.task1.exact_metrics.tp for r in results)
    t1_exact_fp = sum(r.task1.exact_metrics.fp for r in results)
    t1_exact_fn = sum(r.task1.exact_metrics.fn for r in results)

    # Semantic metrics
    t1_semantic_p = [r.task1.semantic.precision for r in results]
    t1_semantic_r = [r.task1.semantic.recall for r in results]
    t1_semantic_f1 = [r.task1.semantic.f1 for r in results]
    t1_full_semantic_recall = [r.task1.full_semantic_recall for r in results]

    # For micro-averaged semantic metrics
    t1_total_semantic_tp = sum(r.task1.semantic.semantic_tp for r in results)
    t1_total_precision_sum = sum(r.task1.semantic.precision_sum for r in results)
    t1_total_predictions = sum(r.task1.predicted_count for r in results)
    t1_total_gt_in_corpus = sum(r.task1.gt_in_corpus_count for r in results)

    # ==================== Task 2: Expression ====================
    # Exact anatomy metrics
    t2_anat_p = [r.task2.anatomy_metrics.precision for r in results]
    t2_anat_r = [r.task2.anatomy_metrics.recall for r in results]
    t2_anat_f1 = [r.task2.anatomy_metrics.f1 for r in results]
    t2_in_corpus_exact_recall = [r.task2.in_corpus_exact_recall for r in results]
    t2_full_recall = [r.task2.full_recall for r in results]

    # Micro metrics (sum tp/fp/fn across all genes)
    t2_anat_tp = sum(r.task2.anatomy_metrics.tp for r in results)
    t2_anat_fp = sum(r.task2.anatomy_metrics.fp for r in results)
    t2_anat_fn = sum(r.task2.anatomy_metrics.fn for r in results)

    # Tuple metrics
    t2_tuple_p = [r.task2.tuple_metrics.precision for r in results]
    t2_tuple_r = [r.task2.tuple_metrics.recall for r in results]
    t2_tuple_f1 = [r.task2.tuple_metrics.f1 for r in results]
    t2_tuple_tp = sum(r.task2.tuple_metrics.tp for r in results)
    t2_tuple_fp = sum(r.task2.tuple_metrics.fp for r in results)
    t2_tuple_fn = sum(r.task2.tuple_metrics.fn for r in results)

    # Semantic metrics
    t2_semantic_p = [r.task2.anatomy_semantic.precision for r in results]
    t2_semantic_r = [r.task2.anatomy_semantic.recall for r in results]
    t2_semantic_f1 = [r.task2.anatomy_semantic.f1 for r in results]
    t2_full_semantic_recall = [r.task2.full_semantic_recall for r in results]
    t2_total_semantic_tp = sum(r.task2.anatomy_semantic.semantic_tp for r in results)
    t2_total_predictions = sum(r.task2.predicted_count for r in results)
    t2_total_gt_in_corpus = sum(r.task2.gt_in_corpus_count for r in results)

    # ==================== Task 3: Synonyms ====================
    # Fullname metrics
    t3_fn_p = [r.task3.fullname_metrics.precision for r in results]
    t3_fn_r = [r.task3.fullname_metrics.recall for r in results]
    t3_fn_f1 = [r.task3.fullname_metrics.f1 for r in results]
    t3_fn_in_corpus_recall = [r.task3.in_corpus_fullname_recall for r in results]
    t3_fn_full_recall = [r.task3.full_fullname_recall for r in results]
    t3_fn_tp = sum(r.task3.fullname_metrics.tp for r in results)
    t3_fn_fp = sum(r.task3.fullname_metrics.fp for r in results)
    t3_fn_fn = sum(r.task3.fullname_metrics.fn for r in results)

    # Symbol metrics
    t3_sym_p = [r.task3.symbol_metrics.precision for r in results]
    t3_sym_r = [r.task3.symbol_metrics.recall for r in results]
    t3_sym_f1 = [r.task3.symbol_metrics.f1 for r in results]
    t3_sym_in_corpus_recall = [r.task3.in_corpus_symbol_recall for r in results]
    t3_sym_full_recall = [r.task3.full_symbol_recall for r in results]
    t3_sym_tp = sum(r.task3.symbol_metrics.tp for r in results)
    t3_sym_fp = sum(r.task3.symbol_metrics.fp for r in results)
    t3_sym_fn = sum(r.task3.symbol_metrics.fn for r in results)

    # Combined metrics
    t3_combined_p = [r.task3.combined_metrics.precision for r in results]
    t3_combined_r = [r.task3.combined_metrics.recall for r in results]
    t3_combined_f1 = [r.task3.combined_metrics.f1 for r in results]
    t3_combined_in_corpus_recall = [r.task3.in_corpus_combined_recall for r in results]
    t3_combined_full_recall = [r.task3.full_combined_recall for r in results]
    t3_combined_tp = sum(r.task3.combined_metrics.tp for r in results)
    t3_combined_fp = sum(r.task3.combined_metrics.fp for r in results)
    t3_combined_fn = sum(r.task3.combined_metrics.fn for r in results)

    # ==================== Reference Coverage ====================
    ref_results = [r.reference_coverage for r in results if r.reference_coverage]
    ref_coverage_agg = None
    if ref_results:
        total_effective = sum(r.effective_papers for r in ref_results)
        total_cited = sum(r.agent_cited for r in ref_results)
        total_hits = sum(r.hits for r in ref_results)
        recalls = [r.recall for r in ref_results if r.effective_papers > 0]
        precisions = [r.agent_precision for r in ref_results if r.agent_cited > 0]

        ref_coverage_agg = {
            "total_effective_papers": total_effective,
            "total_agent_cited": total_cited,
            "total_hits": total_hits,
            "overall_recall": round(total_hits / total_effective if total_effective > 0 else 0, 4),
            "overall_agent_precision": round(total_hits / total_cited if total_cited > 0 else 0, 4),
            "mean_recall": round(_mean(recalls), 4),
            "mean_agent_precision": round(_mean(precisions), 4),
        }

    aggregate: dict[str, Any] = {
        "genes_evaluated": len(results),
        "genes_skipped": len(skipped),
        "skipped_ids": skipped,
        "genes_failed": len(failed),
        "failed_ids": failed,
        "task1_go": {
            "exact": {
                "mean_precision": round(_mean(t1_exact_p), 4),
                "mean_recall": round(_mean(t1_exact_r), 4),
                "mean_f1": round(_mean(t1_exact_f1), 4),
                "micro_precision": round(
                    t1_exact_tp / (t1_exact_tp + t1_exact_fp)
                    if (t1_exact_tp + t1_exact_fp) > 0
                    else 0,
                    4,
                ),
                "micro_recall": round(
                    t1_exact_tp / (t1_exact_tp + t1_exact_fn)
                    if (t1_exact_tp + t1_exact_fn) > 0
                    else 0,
                    4,
                ),
                "total_tp": t1_exact_tp,
                "total_fp": t1_exact_fp,
                "total_fn": t1_exact_fn,
            },
            "semantic": {
                "mean_precision": round(_mean(t1_semantic_p), 4),
                "mean_recall": round(_mean(t1_semantic_r), 4),
                "mean_f1": round(_mean(t1_semantic_f1), 4),
                "micro_precision": round(
                    t1_total_precision_sum / t1_total_predictions
                    if t1_total_predictions > 0
                    else 0,
                    4,
                ),
                "micro_recall": round(
                    t1_total_semantic_tp / t1_total_gt_in_corpus
                    if t1_total_gt_in_corpus > 0
                    else 0,
                    4,
                ),
                "total_semantic_tp": round(t1_total_semantic_tp, 4),
                "total_predictions": t1_total_predictions,
                "total_gt_in_corpus": t1_total_gt_in_corpus,
            },
            "mean_in_corpus_exact_recall": round(_mean(t1_in_corpus_exact_recall), 4),
            "mean_full_recall": round(_mean(t1_full_recall), 4),
            "mean_full_semantic_recall": round(_mean(t1_full_semantic_recall), 4),
            "recall_at_k": _aggregate_recall_at_k(
                results,
                lambda r: r.task1.recall_at_k,
                lambda r: r.task1.gt_in_corpus_count,
            ),
        },
        "task2_expression": {
            "anatomy": {
                "mean_precision": round(_mean(t2_anat_p), 4),
                "mean_recall": round(_mean(t2_anat_r), 4),
                "mean_f1": round(_mean(t2_anat_f1), 4),
                "micro_precision": round(
                    t2_anat_tp / (t2_anat_tp + t2_anat_fp) if (t2_anat_tp + t2_anat_fp) > 0 else 0,
                    4,
                ),
                "micro_recall": round(
                    t2_anat_tp / (t2_anat_tp + t2_anat_fn) if (t2_anat_tp + t2_anat_fn) > 0 else 0,
                    4,
                ),
                "total_tp": t2_anat_tp,
                "total_fp": t2_anat_fp,
                "total_fn": t2_anat_fn,
            },
            "anatomy_semantic": {
                "mean_precision": round(_mean(t2_semantic_p), 4),
                "mean_recall": round(_mean(t2_semantic_r), 4),
                "mean_f1": round(_mean(t2_semantic_f1), 4),
                "micro_precision": round(
                    sum(r.task2.anatomy_semantic.precision_sum for r in results)
                    / t2_total_predictions
                    if t2_total_predictions > 0
                    else 0,
                    4,
                ),
                "micro_recall": round(
                    t2_total_semantic_tp / t2_total_gt_in_corpus
                    if t2_total_gt_in_corpus > 0
                    else 0,
                    4,
                ),
                "total_semantic_tp": round(t2_total_semantic_tp, 4),
                "total_predictions": t2_total_predictions,
                "total_gt_in_corpus": t2_total_gt_in_corpus,
            },
            "tuple": {
                "mean_precision": round(_mean(t2_tuple_p), 4),
                "mean_recall": round(_mean(t2_tuple_r), 4),
                "mean_f1": round(_mean(t2_tuple_f1), 4),
                "micro_precision": round(
                    t2_tuple_tp / (t2_tuple_tp + t2_tuple_fp)
                    if (t2_tuple_tp + t2_tuple_fp) > 0
                    else 0,
                    4,
                ),
                "micro_recall": round(
                    t2_tuple_tp / (t2_tuple_tp + t2_tuple_fn)
                    if (t2_tuple_tp + t2_tuple_fn) > 0
                    else 0,
                    4,
                ),
                "total_tp": t2_tuple_tp,
                "total_fp": t2_tuple_fp,
                "total_fn": t2_tuple_fn,
            },
            "mean_in_corpus_exact_recall": round(_mean(t2_in_corpus_exact_recall), 4),
            "mean_full_recall": round(_mean(t2_full_recall), 4),
            "mean_full_semantic_recall": round(_mean(t2_full_semantic_recall), 4),
            "anatomy_recall_at_k": _aggregate_recall_at_k(
                results,
                lambda r: r.task2.anatomy_recall_at_k,
                lambda r: r.task2.gt_in_corpus_count,
            ),
        },
        "task3_synonyms": {
            "fullname": {
                "mean_precision": round(_mean(t3_fn_p), 4),
                "mean_recall": round(_mean(t3_fn_r), 4),
                "mean_f1": round(_mean(t3_fn_f1), 4),
                "mean_in_corpus_recall": round(_mean(t3_fn_in_corpus_recall), 4),
                "mean_full_recall": round(_mean(t3_fn_full_recall), 4),
                "micro_precision": round(
                    t3_fn_tp / (t3_fn_tp + t3_fn_fp) if (t3_fn_tp + t3_fn_fp) > 0 else 0, 4
                ),
                "micro_recall": round(
                    t3_fn_tp / (t3_fn_tp + t3_fn_fn) if (t3_fn_tp + t3_fn_fn) > 0 else 0, 4
                ),
                "total_tp": t3_fn_tp,
                "total_fp": t3_fn_fp,
                "total_fn": t3_fn_fn,
            },
            "symbol": {
                "mean_precision": round(_mean(t3_sym_p), 4),
                "mean_recall": round(_mean(t3_sym_r), 4),
                "mean_f1": round(_mean(t3_sym_f1), 4),
                "mean_in_corpus_recall": round(_mean(t3_sym_in_corpus_recall), 4),
                "mean_full_recall": round(_mean(t3_sym_full_recall), 4),
                "micro_precision": round(
                    t3_sym_tp / (t3_sym_tp + t3_sym_fp) if (t3_sym_tp + t3_sym_fp) > 0 else 0, 4
                ),
                "micro_recall": round(
                    t3_sym_tp / (t3_sym_tp + t3_sym_fn) if (t3_sym_tp + t3_sym_fn) > 0 else 0, 4
                ),
                "total_tp": t3_sym_tp,
                "total_fp": t3_sym_fp,
                "total_fn": t3_sym_fn,
            },
            "combined": {
                "mean_precision": round(_mean(t3_combined_p), 4),
                "mean_recall": round(_mean(t3_combined_r), 4),
                "mean_f1": round(_mean(t3_combined_f1), 4),
                "mean_in_corpus_recall": round(_mean(t3_combined_in_corpus_recall), 4),
                "mean_full_recall": round(_mean(t3_combined_full_recall), 4),
                "micro_precision": round(
                    t3_combined_tp / (t3_combined_tp + t3_combined_fp)
                    if (t3_combined_tp + t3_combined_fp) > 0
                    else 0,
                    4,
                ),
                "micro_recall": round(
                    t3_combined_tp / (t3_combined_tp + t3_combined_fn)
                    if (t3_combined_tp + t3_combined_fn) > 0
                    else 0,
                    4,
                ),
                "total_tp": t3_combined_tp,
                "total_fp": t3_combined_fp,
                "total_fn": t3_combined_fn,
            },
            "fullname_recall_at_k": _aggregate_recall_at_k(
                results,
                lambda r: r.task3.fullname_recall_at_k,
                lambda r: r.task3.gt_fullname_in_corpus,
            ),
            "symbol_recall_at_k": _aggregate_recall_at_k(
                results,
                lambda r: r.task3.symbol_recall_at_k,
                lambda r: r.task3.gt_symbol_in_corpus,
            ),
            "combined_recall_at_k": _aggregate_recall_at_k(
                results,
                lambda r: r.task3.combined_recall_at_k,
            ),
        },
    }

    if ref_coverage_agg:
        aggregate["reference_coverage"] = ref_coverage_agg

    return aggregate


# =============================================================================
# CLI
# =============================================================================


def print_single_result(result: EvaluationResult) -> None:
    """Print formatted single gene result."""
    print(f"\n{'=' * 60}")
    print(f"Evaluation: {result.gene_symbol} ({result.gene_id})")
    print(f"{'=' * 60}")

    # Task 1
    t1 = result.task1
    print("\nTask 1: GO Annotations")
    print(f"  Predicted: {t1.predicted_count}, Ground Truth (in-corpus): {t1.gt_in_corpus_count}")
    print("  Exact Match:")
    print(f"    Precision: {t1.exact_metrics.precision:.2%}")
    print(f"    Recall:    {t1.exact_metrics.recall:.2%}")
    print(f"    F1:        {t1.exact_metrics.f1:.2%}")
    print("  Semantic (partial credit):")
    print(f"    Precision: {t1.semantic.precision:.2%}")
    print(f"    Recall:    {t1.semantic.recall:.2%}")
    print(f"    F1:        {t1.semantic.f1:.2%}")

    # Task 2
    t2 = result.task2
    print("\nTask 2: Expression")
    print(f"  Predicted: {t2.predicted_count}, Ground Truth (in-corpus): {t2.gt_in_corpus_count}")
    print("  Anatomy Exact:")
    print(f"    Precision: {t2.anatomy_metrics.precision:.2%}")
    print(f"    Recall:    {t2.anatomy_metrics.recall:.2%}")
    print(f"    F1:        {t2.anatomy_metrics.f1:.2%}")
    print("  Anatomy Semantic:")
    print(f"    Precision: {t2.anatomy_semantic.precision:.2%}")
    print(f"    Recall:    {t2.anatomy_semantic.recall:.2%}")
    print(f"    F1:        {t2.anatomy_semantic.f1:.2%}")
    print("  Tuple Exact:")
    print(f"    Precision: {t2.tuple_metrics.precision:.2%}")
    print(f"    Recall:    {t2.tuple_metrics.recall:.2%}")

    # Task 3
    t3 = result.task3
    print("\nTask 3: Synonyms")
    print(f"  Fullname (in-corpus: {t3.gt_fullname_in_corpus}/{t3.gt_fullname_total}):")
    print(f"    Precision: {t3.fullname_metrics.precision:.2%}")
    print(f"    Recall:    {t3.fullname_metrics.recall:.2%}")
    print(f"  Symbol (in-corpus: {t3.gt_symbol_in_corpus}/{t3.gt_symbol_total}):")
    print(f"    Precision: {t3.symbol_metrics.precision:.2%}")
    print(f"    Recall:    {t3.symbol_metrics.recall:.2%}")

    # Reference Coverage
    ref = result.reference_coverage
    if ref:
        print("\nReference Coverage")
        print(f"  Corpus size: {ref.corpus_size} papers")
        print(f"  Effective papers: {ref.effective_papers} (contain findable annotations)")
        print(f"  Agent cited: {ref.agent_cited} papers")
        print(f"  Hits: {ref.hits}/{ref.effective_papers} ({ref.recall:.1%} of effective)")
        print(
            f"  Agent precision: {ref.agent_precision:.1%} (fraction of cited that were effective)"
        )

    print()


def print_batch_result(batch_result: BatchResult) -> None:
    """Print formatted batch result summary."""
    agg = batch_result.aggregate

    print(f"\n{'=' * 60}")
    print("Batch Evaluation Summary")
    print(f"{'=' * 60}")
    print(f"Genes evaluated: {agg['genes_evaluated']}")
    print(f"Genes skipped:   {agg['genes_skipped']}")

    if agg.get("skipped_ids"):
        print(f"  Skipped: {', '.join(agg['skipped_ids'][:5])}")
        if len(agg["skipped_ids"]) > 5:
            print(f"  ... and {len(agg['skipped_ids']) - 5} more")

    t1 = agg["task1_go"]
    print("\nTask 1: GO Annotations (mean across genes)")
    print(
        f"  Exact:       P={t1['exact']['mean_precision']:.2%}, R={t1['exact']['mean_recall']:.2%}, F1={t1['exact']['mean_f1']:.2%}"
    )
    print(
        f"  Semantic:  P={t1['semantic']['mean_precision']:.2%}, R={t1['semantic']['mean_recall']:.2%}, F1={t1['semantic']['mean_f1']:.2%}"
    )

    t2 = agg["task2_expression"]
    print("\nTask 2: Expression (mean across genes)")
    print(
        f"  Anatomy:     P={t2['anatomy']['mean_precision']:.2%}, R={t2['anatomy']['mean_recall']:.2%}, F1={t2['anatomy']['mean_f1']:.2%}"
    )
    print(
        f"  Semantic:  P={t2['anatomy_semantic']['mean_precision']:.2%}, R={t2['anatomy_semantic']['mean_recall']:.2%}, F1={t2['anatomy_semantic']['mean_f1']:.2%}"
    )
    print(
        f"  Tuple:       P={t2['tuple']['mean_precision']:.2%}, R={t2['tuple']['mean_recall']:.2%}, F1={t2['tuple']['mean_f1']:.2%}"
    )
    print(f"  In-corpus exact recall: {t2['mean_in_corpus_exact_recall']:.2%}")

    t3 = agg["task3_synonyms"]
    print("\nTask 3: Synonyms (mean across genes)")
    print(
        f"  Fullname: P={t3['fullname']['mean_precision']:.2%}, R={t3['fullname']['mean_recall']:.2%}, In-corpus R={t3['fullname']['mean_in_corpus_recall']:.2%}"
    )
    print(
        f"  Symbol:   P={t3['symbol']['mean_precision']:.2%}, R={t3['symbol']['mean_recall']:.2%}, In-corpus R={t3['symbol']['mean_in_corpus_recall']:.2%}"
    )

    # Reference Coverage
    ref = agg.get("reference_coverage")
    if ref:
        print("\nReference Coverage")
        print(f"  Total effective papers: {ref['total_effective_papers']}")
        print(f"  Total agent cited: {ref['total_agent_cited']}")
        print(f"  Total hits: {ref['total_hits']}")
        print(f"  Overall recall: {ref['overall_recall']:.1%} (hits / effective)")
        print(f"  Overall precision: {ref['overall_agent_precision']:.1%} (hits / cited)")
        print(f"  Mean recall: {ref['mean_recall']:.1%}")
        print(f"  Mean precision: {ref['mean_agent_precision']:.1%}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent annotations against ground truth")
    parser.add_argument(
        "--gene-id",
        type=str,
        help="Single gene ID to evaluate (e.g., FBgn0000014)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch evaluation on all genes with outputs",
    )
    parser.add_argument(
        "--gene-ids",
        type=str,
        nargs="+",
        help="List of gene IDs to evaluate",
    )
    parser.add_argument(
        "--dir",
        "--output-dir",
        type=str,
        default="outputs/",
        dest="output_dir",
        help="Directory containing agent output files",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="data/ground_truth_top100.jsonl",
        help="Path to ground truth JSONL file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for results (JSON). Prints to stdout if not specified.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed per-gene results",
    )
    parser.add_argument(
        "--resolve-descriptions",
        action="store_true",
        help="Resolve NL descriptions to GO IDs before evaluation (for hidden-terms runs)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.gene_id and not args.batch and not args.gene_ids:
        parser.error("Must specify --gene-id, --batch, or --gene-ids")

    try:
        if args.gene_id:
            # Single gene evaluation
            result = evaluate_gene(
                gene_id=args.gene_id,
                output_dir=args.output_dir,
                ground_truth_path=args.ground_truth,
                resolve_descriptions=args.resolve_descriptions,
                verbose=args.verbose,
            )
            output = result.to_dict()

            if args.verbose:
                print_single_result(result)

        else:
            # Batch evaluation
            gene_ids = args.gene_ids if args.gene_ids else None
            batch_result = evaluate_batch(
                gene_ids=gene_ids,
                output_dir=args.output_dir,
                ground_truth_path=args.ground_truth,
                resolve_descriptions=args.resolve_descriptions,
                verbose=args.verbose,
            )
            output = batch_result.to_dict()

            if args.verbose:
                print_batch_result(batch_result)

        # Output results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Results written to {args.output}")
        elif not args.verbose:
            print(json.dumps(output, indent=2))

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
