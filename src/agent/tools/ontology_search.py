"""Ontology search tools for GO, FBbt (anatomy), and FBdv (developmental stage).

This module provides LangChain tool wrappers around the core ontology search functionality.
"""

from typing import Any

from langchain_core.tools import tool

from ..core.ontology import (
    OntologyTerm,
    get_fbbt_index,
    get_fbdv_index,
    get_go_index,
    search_anatomy_core,
    search_go_core,
    search_stage_core,
)

# Re-export for backwards compatibility
__all__ = [
    "OntologyTerm",
    "get_go_index",
    "get_fbbt_index",
    "get_fbdv_index",
    "search_go_terms",
    "search_anatomy_terms",
    "search_stage_terms",
    "GOSearchTool",
    "AnatomySearchTool",
    "StageSearchTool",
]


@tool
def search_go_terms(query: str, aspect: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    """Search Gene Ontology for matching terms.

    Args:
        query: Search term (e.g., "DNA binding", "transcription factor")
        aspect: Optional filter - "P" (Biological Process), "F" (Molecular Function),
                or "C" (Cellular Component)
        limit: Maximum number of results (default 10)

    Returns:
        List of matching GO terms with go_id, name, namespace, and definition
    """
    return search_go_core(query, aspect=aspect, limit=limit)


@tool
def search_anatomy_terms(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search FBbt ontology for Drosophila anatomy terms.

    Args:
        query: Search term (e.g., "wing disc", "neuron", "muscle")
        limit: Maximum number of results (default 10)

    Returns:
        List of matching anatomy terms with fbbt_id, name, and definition
    """
    return search_anatomy_core(query, limit=limit)


@tool
def search_stage_terms(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search FBdv ontology for Drosophila developmental stage terms.

    Args:
        query: Search term (e.g., "embryo", "larval stage", "pupal")
        limit: Maximum number of results (default 10)

    Returns:
        List of matching stage terms with fbdv_id, name, and definition
    """
    return search_stage_core(query, limit=limit)


# Tool classes for LangChain integration
GOSearchTool = search_go_terms
AnatomySearchTool = search_anatomy_terms
StageSearchTool = search_stage_terms
