"""Agent state definition for the LangGraph workflow."""

import operator
from typing import Annotated, TypedDict

# =============================================================================
# Text-Based Annotations (extracted from papers, before ontology resolution)
# =============================================================================


class TextFunctionAnnotation(TypedDict):
    """Text-based function annotation (before ontology resolution).

    The resolve node converts function_description → go_id.
    """

    function_description: str  # e.g., "transcription factor activity"
    qualifier: str  # "involved_in" | "enables" | "located_in" | "part_of" | "is_active_in"
    aspect: str  # "P" | "F" | "C"
    is_negated: bool
    evidence_text: str  # Quote from paper
    pmcid: str  # Source paper


class TextExpressionAnnotation(TypedDict):
    """Text-based expression annotation (before ontology resolution).

    The resolve node converts anatomy_description → anatomy_id and
    stage_description → stage_id.
    """

    expression_type: str  # "polypeptide" | "transcript"
    anatomy_description: str  # e.g., "wing imaginal disc"
    stage_description: str  # e.g., "third instar larva"
    evidence_text: str  # Quote from paper
    pmcid: str  # Source paper


# =============================================================================
# ID-Based Annotations (after ontology resolution)
# =============================================================================


class GOAnnotation(TypedDict):
    """A Gene Ontology annotation (resolved to GO ID)."""

    go_id: str
    qualifier: str
    aspect: str  # P, F, or C
    is_negated: bool | None
    evidence: dict | None


class ExpressionRecord(TypedDict):
    """An expression record (resolved to FBbt/FBdv IDs)."""

    expression_type: str  # "polypeptide" or "transcript"
    anatomy_id: str | None  # FBbt ID
    stage_id: str | None  # FBdv ID
    evidence: dict | None


class Synonyms(TypedDict):
    """Gene synonyms."""

    fullname_synonyms: list[str]
    symbol_synonyms: list[str]


class AgentOutput(TypedDict):
    """Final agent output matching the benchmark schema."""

    gene_id: str
    gene_symbol: str
    task1_function: list[GOAnnotation]
    task2_expression: list[ExpressionRecord]
    task3_synonyms: Synonyms


def merge_lists(a: list, b: list) -> list:
    """Merge two lists, avoiding duplicates based on key fields."""
    return a + b


class AgentConfig(TypedDict, total=False):
    """Configuration for agent execution."""

    model: str  # OpenAI model name (default: gpt-5-mini)
    verbose: bool  # Enable verbose logging
    hide_terms: bool  # Enable specificity gap benchmark (hide GO terms)


class AgentState(TypedDict):
    """State for the gene annotation agent workflow.

    Attributes:
        gene_id: FlyBase gene ID (e.g., FBgn0000014)
        gene_symbol: Gene symbol (e.g., abd-A)
        summary: Expert-written gene summary from FlyBase
        config: Agent configuration (model, verbose)

        relevant_papers: PMCIDs found by corpus search
        paper_texts: Mapping of PMCID -> paper content

        go_annotations: Accumulated GO term annotations
        expression_records: Accumulated expression records
        synonyms_found: Accumulated synonyms (fullname and symbol)

        messages: Chat messages for LLM interactions
        final_output: Complete validated output (set at end)
        error: Error message if something went wrong
    """

    # Input
    gene_id: str
    gene_symbol: str
    summary: str
    config: AgentConfig

    # Intermediate - paper discovery
    relevant_papers: Annotated[list[str], merge_lists]
    paper_texts: dict[str, dict]

    # Intermediate - text-based annotations (from extraction, before resolution)
    text_function_annotations: Annotated[list[TextFunctionAnnotation], merge_lists]
    text_expression_annotations: Annotated[list[TextExpressionAnnotation], merge_lists]
    synonyms_found: Annotated[list[str], merge_lists]

    # Intermediate - resolved annotations (after ontology resolution)
    go_annotations: Annotated[list[GOAnnotation], merge_lists]
    expression_records: Annotated[list[ExpressionRecord], merge_lists]

    # LLM interaction
    messages: Annotated[list, operator.add]

    # Output
    final_output: AgentOutput | None
    error: str | None
