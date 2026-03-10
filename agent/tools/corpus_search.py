"""Corpus search tool using BM25 over the HuggingFace literature corpus.

This module provides a LangChain tool wrapper around the core corpus search functionality.
"""

from langchain_core.tools import tool

from ..core.corpus import CorpusIndex, get_corpus_index, search_corpus_core

# Re-export for backwards compatibility
__all__ = ["CorpusIndex", "get_corpus_index", "search_corpus", "CorpusSearchTool"]


@tool
def search_corpus(query: str, limit: int = 20) -> list[dict]:
    """Search the Drosophila literature corpus for papers matching a query.

    Use this tool to find scientific papers that discuss a specific gene,
    biological process, or research topic.

    Args:
        query: Search query - can be a gene symbol (e.g., "abd-A"),
               keyword (e.g., "transcription factor"), or phrase
        limit: Maximum number of results to return (default 20)

    Returns:
        List of matching papers with:
        - pmcid: PubMed Central ID
        - title: Paper title
        - abstract: Paper abstract (truncated to 500 chars)
        - relevance_score: BM25 relevance score
    """
    return search_corpus_core(query, limit=limit)


# Alias for LangChain integration
CorpusSearchTool = search_corpus
