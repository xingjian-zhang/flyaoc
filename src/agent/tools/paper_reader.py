"""Paper reader tool for fetching full text from the corpus by PMCID.

This module provides a LangChain tool wrapper around the core paper reading functionality.
"""

from typing import Any

from langchain_core.tools import tool

from ..core.papers import PaperCache, get_paper_cache, get_paper_text_core

# Re-export for backwards compatibility
__all__ = ["PaperCache", "get_paper_cache", "get_paper_text", "PaperReaderTool"]


@tool
def get_paper_text(pmcid: str, sections: list[str] | None = None) -> dict[str, Any]:
    """Retrieve the full text of a specific paper from the corpus.

    Use this tool to read the detailed content of a paper found via search.
    Papers contain title, abstract, and multiple sections (INTRO, METHODS,
    RESULTS, DISCUSS, CONCL).

    Args:
        pmcid: PubMed Central ID (e.g., "PMC1234567")
        sections: Optional list of sections to retrieve. If not specified,
                  returns all sections. Valid sections: "abstract", "INTRO",
                  "METHODS", "RESULTS", "DISCUSS", "CONCL"

    Returns:
        Dictionary with paper content:
        - pmcid: PubMed Central ID
        - title: Paper title
        - abstract: Paper abstract
        - sections: Dict of section name -> list of paragraphs

    Raises:
        ValueError: If paper not found in corpus
    """
    return get_paper_text_core(pmcid, sections)


# Alias for LangChain integration
PaperReaderTool = get_paper_text
