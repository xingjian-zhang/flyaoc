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
