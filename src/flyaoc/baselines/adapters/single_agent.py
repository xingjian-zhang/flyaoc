"""Single-agent baseline adapter.

Maps the public `single_agent` baseline to the original MCP/OpenAI Agents SDK
runner with literature, ontology, and validation MCP tools.
"""

from __future__ import annotations

from flyaoc.baselines.adapters.common import agentic_execution_config, raw_result_from_mapping
from flyaoc.baselines.types import BaselineRunConfig, GeneInput, RawBaselineResult


async def run_agentic_mcp(gene: GeneInput, config: BaselineRunConfig) -> dict:
    from agent.agentic.runner import run_agent_mcp

    result = await run_agent_mcp(
        gene.gene_id,
        gene.gene_symbol,
        summary=gene.summary,
        config=agentic_execution_config(
            config,
            multi_agent=False,
            no_literature=False,
        ),
    )
    return result.to_dict()


def run_gene(config: BaselineRunConfig, gene: GeneInput) -> RawBaselineResult:
    import asyncio

    return raw_result_from_mapping(config, gene, asyncio.run(run_agentic_mcp(gene, config)))
