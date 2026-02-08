#!/usr/bin/env python3
"""MCP server for ontology search (GO, FBbt, FBdv).

This server provides three tools:
- search_go_terms: Search Gene Ontology
- search_anatomy_terms: Search FBbt (anatomy) ontology
- search_stage_terms: Search FBdv (developmental stage) ontology

Run as:
    python -m agent.mcp_servers.ontology_server
"""

import json

from mcp.server.fastmcp import FastMCP

from ..core.ontology import (
    get_term_children_core,
    search_anatomy_core,
    search_go_core,
    search_stage_core,
)

# Create MCP server
mcp = FastMCP("Drosophila Ontology Server")


@mcp.tool()
def search_go_terms(query: str, aspect: str | None = None, limit: int = 10) -> str:
    """Search Gene Ontology for matching terms.

    Use this tool to find GO term IDs for biological functions, processes,
    or cellular components mentioned in the literature.

    Args:
        query: Search term (e.g., "DNA binding", "transcription factor")
        aspect: Optional filter - "P" (Biological Process), "F" (Molecular Function),
                or "C" (Cellular Component)
        limit: Maximum number of results (default 10)

    Returns:
        JSON string with list of matching GO terms containing:
        - go_id: GO term ID (e.g., "GO:0003700")
        - name: Term name
        - namespace: Ontology namespace
        - definition: Term definition
        - synonyms: List of alternative names (up to 5)
        - parents: Parent term IDs with names (helps understand hierarchy)
        - children_count: Number of more specific child terms
    """
    results = search_go_core(query, aspect=aspect, limit=limit)
    return json.dumps(results, indent=2)


@mcp.tool()
def search_anatomy_terms(query: str, limit: int = 10) -> str:
    """Search FBbt ontology for Drosophila anatomy terms.

    Use this tool to find FBbt term IDs for anatomical structures,
    tissues, or cell types mentioned in expression data.

    Args:
        query: Search term (e.g., "wing disc", "neuron", "muscle")
        limit: Maximum number of results (default 10)

    Returns:
        JSON string with list of matching anatomy terms containing:
        - fbbt_id: FBbt term ID (e.g., "FBbt:00004729")
        - name: Term name
        - definition: Term definition
        - synonyms: List of alternative names (up to 5)
        - parents: Parent term IDs with names (e.g., "FBbt:00005333 (imaginal disc)")
        - children_count: Number of more specific child terms (0 = leaf term)
    """
    results = search_anatomy_core(query, limit=limit)
    return json.dumps(results, indent=2)


@mcp.tool()
def search_stage_terms(query: str, limit: int = 10) -> str:
    """Search FBdv ontology for Drosophila developmental stage terms.

    Use this tool to find FBdv term IDs for developmental stages
    mentioned in expression data. IMPORTANT: Always search for stages
    when extracting expression data - every expression annotation needs
    both anatomy AND stage.

    Args:
        query: Search term (e.g., "embryo", "larval stage", "pupal")
        limit: Maximum number of results (default 10)

    Returns:
        JSON string with list of matching stage terms containing:
        - fbdv_id: FBdv term ID (e.g., "FBdv:00005289")
        - name: Term name
        - definition: Term definition
        - synonyms: List of alternative names
        - parents: Parent term IDs with names (helps understand stage hierarchy)
        - children_count: Number of more specific child stages
    """
    results = search_stage_core(query, limit=limit)
    return json.dumps(results, indent=2)


@mcp.tool()
def get_term_children(term_id: str, limit: int = 20) -> str:
    """Get direct children of any ontology term (GO, FBbt, or FBdv).

    Use this tool to explore the ontology hierarchy - find more specific
    terms under a general term. Useful when:
    - You found a general term and want to find a more specific match
    - You want to see what subcategories exist under a term
    - You need to drill down from "muscle" to "alary muscle cell"

    Args:
        term_id: Any ontology term ID:
            - GO:XXXXXXX (Gene Ontology)
            - FBbt:XXXXXXXX (anatomy)
            - FBdv:XXXXXXXX (developmental stage)
        limit: Maximum children to return (default 20)

    Returns:
        JSON string with:
        - term_id: The queried term
        - term_name: Name of the queried term
        - children_count: Total number of direct children
        - children: List of child terms with id, name, and their children_count
        - truncated: True if more children exist beyond limit
    """
    result = get_term_children_core(term_id, limit=limit)
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
