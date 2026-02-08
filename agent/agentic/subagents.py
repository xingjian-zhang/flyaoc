"""Paper reader agent with full ontology tool access for context isolation.

This module implements the "Agent as Tool" pattern from OpenAI Agents SDK,
enabling the Multi-Agent method. The PaperReaderAgent reads paper text,
extracts annotations, AND resolves them to ontology IDs using search tools -
all within an isolated context.

Workflow:
1. Main agent calls analyze_papers_batch(pmcids=[...], gene_symbol=...)
2. Papers are fetched and analyzed in parallel via asyncio.gather
3. Each paper is processed by PaperReaderAgent with ontology tools
4. Returns RESOLVED annotations (with ontology IDs, not just text)
5. Main agent aggregates, deduplicates, and submits

Benefits:
- Main agent context stays bounded regardless of paper count
- Each paper reader agent sees only ONE paper (~30K tokens) - no O(n^2) growth
- LLM-driven resolution preserves quality (vs programmatic lookup)
- Handles hidden terms benchmark correctly (uses description field)
"""

import asyncio
import json
import os

from agents import Agent, ModelSettings, Runner, function_tool
from agents.result import RunResult
from pydantic import BaseModel

from ..config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from ..core.ontology import search_anatomy_core, search_go_core, search_stage_core
from ..core.papers import get_paper_text_core


def extract_run_usage(result: RunResult) -> dict[str, int]:
    """Extract total token usage from a RunResult.

    The OpenAI Agents SDK stores usage in raw_responses from each LLM call.
    We aggregate across all responses to get total subagent usage, including
    cached token breakdown for accurate cost calculation.
    """
    total_input = 0
    total_output = 0
    total_cached = 0

    for response in getattr(result, "raw_responses", []):
        usage = getattr(response, "usage", None)
        if usage:
            total_input += getattr(usage, "input_tokens", 0) or 0
            total_output += getattr(usage, "output_tokens", 0) or 0

            # Get cached token breakdown from input_tokens_details
            input_details = getattr(usage, "input_tokens_details", None)
            if input_details:
                total_cached += getattr(input_details, "cached_tokens", 0) or 0

    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cached_tokens": total_cached,
    }


# =============================================================================
# Data Models - Resolved Annotations (with ontology IDs)
# =============================================================================


class ResolvedFunctionAnnotation(BaseModel):
    """Function annotation with resolved GO ID."""

    go_id: str | None = None  # GO:XXXXXXX or None if no match
    description: str | None = None  # Natural language if no GO term found
    qualifier: str  # "involved_in" | "enables" | "located_in" | "part_of" | "is_active_in"
    aspect: str  # "P" | "F" | "C"
    is_negated: bool = False
    evidence_text: str  # Quote from paper


class ResolvedExpressionAnnotation(BaseModel):
    """Expression annotation with resolved FBbt/FBdv IDs."""

    expression_type: str  # "polypeptide" | "transcript"
    anatomy_id: str | None = None  # FBbt:XXXXXXXX
    anatomy_description: str | None = None  # Fallback if no match
    stage_id: str | None = None  # FBdv:XXXXXXXX
    stage_description: str | None = None  # Fallback if no match
    evidence_text: str  # Quote from paper


class ResolvedPaperExtractions(BaseModel):
    """Paper extractions with resolved ontology IDs."""

    pmcid: str
    function_annotations: list[ResolvedFunctionAnnotation]
    expression_annotations: list[ResolvedExpressionAnnotation]
    synonyms: list[str]  # Gene name variants found in paper
    key_findings: str  # Brief summary for main agent context


class BatchExtractions(BaseModel):
    """Combined extractions from multiple papers."""

    papers: list[ResolvedPaperExtractions]
    errors: list[str]  # Any papers that failed
    usage: dict[str, int]  # Aggregated subagent token usage


# =============================================================================
# Ontology Search Tools (function tools for subagent)
# =============================================================================


def _should_hide_go_terms() -> bool:
    """Check if GO terms should be hidden (reads from env each time).

    This is called at runtime when analyze_papers_batch is invoked.
    The runner.py sets HIDE_GO_TERMS env var based on --hide-terms flag.
    """
    return os.environ.get("HIDE_GO_TERMS") == "1"


@function_tool
def search_go_terms(query: str, aspect: str | None = None, limit: int = 5) -> str:
    """Search Gene Ontology for terms matching the query.

    Args:
        query: Search term (e.g., "DNA binding", "transcription factor")
        aspect: Optional filter - "P" (Biological Process), "F" (Molecular Function),
                or "C" (Cellular Component)
        limit: Maximum number of results (default 5)

    Returns:
        JSON string with matching GO terms including go_id, name, definition, and parents.
        If no matching terms found, returns empty list.

    Example:
        search_go_terms("wing development", aspect="P")
        -> [{"go_id": "GO:0035220", "name": "wing disc development", ...}]
    """
    results = search_go_core(query, aspect=aspect, limit=limit)
    return json.dumps(results, indent=2)


@function_tool
def search_anatomy_terms(query: str, limit: int = 5) -> str:
    """Search FBbt ontology for Drosophila anatomy terms.

    Args:
        query: Search term (e.g., "wing disc", "neuron", "muscle")
        limit: Maximum number of results (default 5)

    Returns:
        JSON string with matching anatomy terms including fbbt_id, name, and definition.

    Example:
        search_anatomy_terms("wing imaginal disc")
        -> [{"fbbt_id": "FBbt:00001760", "name": "wing disc", ...}]
    """
    results = search_anatomy_core(query, limit=limit)
    return json.dumps(results, indent=2)


@function_tool
def search_stage_terms(query: str, limit: int = 5) -> str:
    """Search FBdv ontology for Drosophila developmental stage terms.

    Args:
        query: Search term (e.g., "embryo", "larval stage", "pupal")
        limit: Maximum number of results (default 5)

    Returns:
        JSON string with matching stage terms including fbdv_id, name, and definition.

    Example:
        search_stage_terms("third instar larva")
        -> [{"fbdv_id": "FBdv:00005339", "name": "third instar larval stage", ...}]
    """
    results = search_stage_core(query, limit=limit)
    return json.dumps(results, indent=2)


# =============================================================================
# Paper Reader Instructions
# =============================================================================

PAPER_READER_INSTRUCTIONS = """You are a specialized reader extracting and resolving gene annotations from scientific papers.

**Input Format**: You receive paper data as JSON with fields like "title", "abstract", "introduction", "results", etc.
Extract information from ALL sections, especially "results" and "introduction" which contain experimental findings.

## Your Task (Two Phases)

### Phase 1: Extract annotations from the paper
Read the paper and identify:
1. **Function annotations**: What molecular functions, biological processes, or cellular components?
2. **Expression annotations**: Where and when is the gene expressed?
3. **Synonyms**: Any alternative names for the gene

### Phase 2: Resolve to ontology IDs
For EACH annotation found, use the search tools to find the correct ontology ID:
- `search_go_terms(query, aspect)` for GO terms (function/process/component)
- `search_anatomy_terms(query)` for FBbt anatomy terms
- `search_stage_terms(query)` for FBdv developmental stage terms

## Output Format

Return JSON with RESOLVED ontology IDs:

```json
{
  "pmcid": "PMC123456",
  "function_annotations": [
    {
      "go_id": "GO:0035220",  // Use ID from search results
      "qualifier": "involved_in",
      "aspect": "P",
      "is_negated": false,
      "evidence_text": "quote from paper"
    }
  ],
  "expression_annotations": [
    {
      "expression_type": "polypeptide",
      "anatomy_id": "FBbt:00001760",  // Use ID from search results
      "stage_id": "FBdv:00005339",    // Use ID from search results
      "evidence_text": "quote from paper"
    }
  ],
  "synonyms": ["alternative-name"],
  "key_findings": "Brief summary"
}
```

## When No Suitable Term Exists

If search returns no suitable term (only overly general ones), use `description` instead:
- For GO: `"go_id": null, "description": "specific wing patterning process"`
- For anatomy: `"anatomy_id": null, "anatomy_description": "specific muscle structure"`
- For stage: `"stage_id": null, "stage_description": "late embryonic stage"`

Do NOT force a general parent term just to have an ID. Use `description` for specificity.

## Annotation Guidelines

### Good Evidence Patterns:
- **Mutant phenotypes**: "X mutants fail to / have defects in / are required for Y"
- **Biochemical assays**: "X binds/phosphorylates/cleaves Y" or "X has Y activity"
- **Localization**: "X localizes to / is detected in / accumulates in Y"
- **Expression**: "X is expressed in / detected in [tissue] at [stage]"

### Avoid Annotating From:
- Speculation: "X may play a role in..."
- Background statements: "X is known to..." without new evidence
- Marker usage: "we used X-GFP to label cells"
- Homology claims: "X is similar to Y which..."

### Qualifiers:
- **enables** (F): Gene directly performs this molecular activity
- **involved_in** (P): Gene participates in this biological process
- **located_in** (C): Gene product is found in this location
- **part_of** (C): Structural component of a protein complex
- **is_active_in** (C): Functions in this location

### Expression Types:
- **polypeptide**: Protein detection (immunostaining, Western blot)
- **transcript**: RNA detection (in situ, RNA-seq, Northern)

## Important
- Be THOROUGH - a single paper can yield 5-10 annotations
- Include verbatim quotes as evidence_text
- Search tools return the best matches - pick the most specific appropriate term
- If the paper doesn't mention the gene experimentally, return empty lists
"""

PAPER_READER_HIDDEN_ADDENDUM = """

## Hidden Terms Mode (Specificity Gap Benchmark)

Some GO terms are intentionally hidden in this benchmark. If `search_go_terms` returns
no results or only overly general terms for a concept you're confident exists:

1. Do NOT waste calls searching with variations - the term is likely hidden
2. Do NOT fall back to overly general parent terms just to have an ID
3. DO use `description` field with a proposed term name:

```json
{
  "go_id": null,
  "description": "oenocyte differentiation",  // Short noun phrase, 2-6 words
  "qualifier": "involved_in",
  "aspect": "P",
  ...
}
```

The `description` should be a **proposed GO term name** - a short noun phrase that
another person could search for to find the correct term.
"""


# =============================================================================
# Paper Reader Agent (with ontology tools)
# =============================================================================


def _create_paper_reader_agent(hide_terms: bool = False) -> Agent:
    """Create PaperReaderAgent with appropriate instructions.

    Args:
        hide_terms: If True, append hidden terms addendum to instructions.

    Returns:
        Configured Agent instance
    """
    instructions = PAPER_READER_INSTRUCTIONS
    if hide_terms:
        instructions += PAPER_READER_HIDDEN_ADDENDUM

    return Agent(
        name="PaperReaderAgent",
        instructions=instructions,
        model=DEFAULT_MODEL,
        model_settings=ModelSettings(temperature=DEFAULT_TEMPERATURE),
        output_type=ResolvedPaperExtractions,
        tools=[search_go_terms, search_anatomy_terms, search_stage_terms],
    )


# Default agent (without hidden terms mode)
paper_reader_agent = _create_paper_reader_agent(hide_terms=False)


# =============================================================================
# Public API - Analyze Papers
# =============================================================================


async def _analyze_single_paper(
    pmcid: str, gene_symbol: str, hide_terms: bool = False
) -> tuple[ResolvedPaperExtractions | str, dict[str, int]]:
    """Fetch and analyze a single paper. Returns (extraction_or_error, usage)."""
    zero_usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    try:
        # Fetch paper (sync call, but fast - just dict lookup from cache)
        paper_data = get_paper_text_core(pmcid)
        if "error" in paper_data:
            return f"{pmcid}: {paper_data['error']}", zero_usage

        paper_text = json.dumps(paper_data)

        # Use appropriate agent based on hide_terms
        agent = _create_paper_reader_agent(hide_terms=hide_terms)

        # Run subagent
        result = await Runner.run(
            starting_agent=agent,
            input=f"Extract and resolve all annotations for gene '{gene_symbol}' from this paper (PMCID: {pmcid}):\n\n{paper_text}",
        )
        usage = extract_run_usage(result)
        return result.final_output_as(ResolvedPaperExtractions), usage
    except Exception as e:
        return f"{pmcid}: {e}", zero_usage


@function_tool
async def analyze_paper(pmcid: str, paper_text: str, gene_symbol: str) -> str:
    """Analyze a paper and extract RESOLVED annotations for the target gene.

    This tool delegates paper reading to a specialized subagent with full
    ontology search capability. The subagent reads the paper, extracts annotations,
    AND resolves them to ontology IDs - all in isolated context.

    Args:
        pmcid: PubMed Central ID of the paper
        paper_text: Full text of the paper to analyze
        gene_symbol: The gene symbol to extract annotations for

    Returns:
        JSON string with RESOLVED extractions including:
        - function_annotations: With go_id or description
        - expression_annotations: With anatomy_id/stage_id or descriptions
        - synonyms: Alternative gene names found
        - key_findings: Brief summary
    """
    hide_terms = _should_hide_go_terms()
    agent = _create_paper_reader_agent(hide_terms=hide_terms)

    result = await Runner.run(
        starting_agent=agent,
        input=f"Extract and resolve all annotations for gene '{gene_symbol}' from this paper (PMCID: {pmcid}):\n\n{paper_text}",
    )
    return result.final_output_as(ResolvedPaperExtractions).model_dump_json()


@function_tool
async def analyze_papers_batch(pmcids: list[str], gene_symbol: str) -> str:
    """Analyze multiple papers in parallel and extract RESOLVED annotations.

    This tool fetches all papers and runs subagent analysis concurrently.
    Each subagent has full ontology search capability to resolve annotations
    to ontology IDs within its isolated context.

    Args:
        pmcids: List of PubMed Central IDs to analyze
        gene_symbol: The gene symbol to extract annotations for

    Returns:
        JSON with combined RESOLVED extractions from all papers:
        {
            "papers": [
                {
                    "pmcid": "PMC123",
                    "function_annotations": [{"go_id": "GO:0035220", ...}],
                    "expression_annotations": [{"anatomy_id": "FBbt:...", ...}],
                    "synonyms": [...],
                    "key_findings": "..."
                },
                ...
            ],
            "errors": ["PMC789: error message"],
            "usage": {"input_tokens": N, "output_tokens": M, "cached_tokens": C}
        }
    """
    hide_terms = _should_hide_go_terms()

    # Run all paper analyses in parallel
    tasks = [_analyze_single_paper(pmcid, gene_symbol, hide_terms=hide_terms) for pmcid in pmcids]
    results = await asyncio.gather(*tasks)

    # Separate successes from errors, aggregate usage
    papers: list[ResolvedPaperExtractions] = []
    errors: list[str] = []
    total_usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

    for extraction, usage in results:
        total_usage["input_tokens"] += usage["input_tokens"]
        total_usage["output_tokens"] += usage["output_tokens"]
        total_usage["cached_tokens"] += usage.get("cached_tokens", 0)

        if isinstance(extraction, ResolvedPaperExtractions):
            papers.append(extraction)
        else:
            errors.append(extraction)

    batch_result = BatchExtractions(papers=papers, errors=errors, usage=total_usage)
    return batch_result.model_dump_json()


# =============================================================================
# Legacy Models (for backwards compatibility if needed)
# =============================================================================


class FunctionAnnotation(BaseModel):
    """Text-based function annotation (legacy - use ResolvedFunctionAnnotation)."""

    function_description: str
    qualifier: str
    aspect: str
    is_negated: bool
    evidence_text: str


class ExpressionAnnotation(BaseModel):
    """Text-based expression annotation (legacy - use ResolvedExpressionAnnotation)."""

    expression_type: str
    anatomy_description: str
    stage_description: str
    evidence_text: str


class PaperExtractions(BaseModel):
    """Text-based paper extractions (legacy - use ResolvedPaperExtractions)."""

    pmcid: str
    function_annotations: list[FunctionAnnotation]
    expression_annotations: list[ExpressionAnnotation]
    synonyms: list[str]
    key_findings: str
