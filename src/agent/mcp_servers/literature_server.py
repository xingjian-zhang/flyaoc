#!/usr/bin/env python3
"""MCP server for literature corpus search and paper reading.

This server provides two tools:
- search_corpus: Search the Drosophila literature corpus using BM25
- get_paper_text: Retrieve full text of a paper by PMCID

Run as:
    python -m agent.mcp_servers.literature_server
"""

import json
import os

from mcp.server.fastmcp import FastMCP

from ..core.corpus import search_corpus_core
from ..core.papers import get_paper_text_core

# Create MCP server
mcp = FastMCP("Drosophila Literature Server")

# Paper limit tracking (set via MAX_PAPERS env var)
_max_papers = int(os.environ.get("MAX_PAPERS", "0"))  # 0 = no limit
_papers_read = 0

# Hide get_paper_text tool when using subagent mode (set via HIDE_GET_PAPER_TEXT env var)
_hide_get_paper_text = os.environ.get("HIDE_GET_PAPER_TEXT", "") == "1"

# Oracle retrieval mode: provide ground truth papers instead of BM25 search
# ORACLE_PMCIDS should be a JSON-encoded list of PMCIDs
_oracle_pmcids_raw = os.environ.get("ORACLE_PMCIDS", "")
_oracle_pmcids: list[str] = json.loads(_oracle_pmcids_raw) if _oracle_pmcids_raw else []


@mcp.tool()
def search_corpus(query: str, limit: int = 20) -> str:
    """Search the Drosophila literature corpus for papers matching a query.

    Use this tool to find scientific papers that discuss a specific gene,
    biological process, or research topic.

    Args:
        query: Search query - can be a gene symbol (e.g., "abd-A"),
               keyword (e.g., "transcription factor"), or phrase
        limit: Maximum number of results to return (default 20)

    Returns:
        JSON string with list of matching papers containing:
        - pmcid: PubMed Central ID
        - title: Paper title
        - abstract: Paper abstract (truncated to 500 chars)
        - relevance_score: BM25 relevance score
        - gene_in_title: True if query appears in title (HIGH relevance signal -
          papers with gene in title are typically focused studies)
    """
    if _oracle_pmcids:
        # Oracle mode: prioritize ground truth papers, then backfill with BM25
        import logging

        logger = logging.getLogger(__name__)
        results = []
        oracle_set = set(_oracle_pmcids)

        # First, add oracle papers (up to limit)
        for pmcid in _oracle_pmcids:
            if len(results) >= limit:
                break
            paper = get_paper_text_core(pmcid)
            if paper and not paper.get("error"):
                results.append(
                    {
                        "pmcid": pmcid,
                        "title": paper.get("title", ""),
                        "abstract": (paper.get("abstract", "") or "")[:500],
                        "relevance_score": 100.0,
                        "oracle": True,
                    }
                )
            else:
                logger.warning(
                    "Oracle paper %s not found in corpus (skipped)", pmcid
                )

        # Then backfill remaining slots with BM25 results (excluding oracle papers)
        if len(results) < limit:
            bm25_results = search_corpus_core(query, limit=limit + len(oracle_set))
            for bm25_paper in bm25_results:
                if len(results) >= limit:
                    break
                if bm25_paper["pmcid"] not in oracle_set:
                    results.append(bm25_paper)

        return json.dumps(results, indent=2)

    results = search_corpus_core(query, limit=limit)
    return json.dumps(results, indent=2)


# Only register get_paper_text if not hidden (subagent mode uses analyze_papers_batch instead)
if not _hide_get_paper_text:

    @mcp.tool()
    def get_paper_text(pmcid: str, sections: list[str] | None = None) -> str:
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
            JSON string with paper content containing:
            - pmcid: PubMed Central ID
            - title: Paper title
            - abstract: Paper abstract
            - sections: Dict of section name -> list of paragraphs
        """
        global _papers_read

        # Check paper limit before reading
        if _max_papers > 0 and _papers_read >= _max_papers:
            return json.dumps(
                {
                    "error": f"Paper limit reached ({_papers_read}/{_max_papers}). "
                    "You have read the maximum number of papers allowed. "
                    "Please submit your annotations now using submit_annotations."
                }
            )

        # Increment counter and read paper
        _papers_read += 1
        result = get_paper_text_core(pmcid, sections)
        return json.dumps(result, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
