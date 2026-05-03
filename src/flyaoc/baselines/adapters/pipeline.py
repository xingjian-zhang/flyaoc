"""Pipeline baseline adapter.

Maps the public `pipeline` baseline to the original LangGraph fixed DAG.
"""

from __future__ import annotations

from flyaoc.baselines.adapters.common import raw_result_from_mapping
from flyaoc.baselines.types import BaselineRunConfig, GeneInput, RawBaselineResult


async def run_pipeline_graph(gene: GeneInput, config: BaselineRunConfig) -> dict:
    from agent.pipeline.agent import run_agent

    return await run_agent(
        gene.gene_id,
        gene.gene_symbol,
        summary=gene.summary,
        model=config.model,
        verbose=config.verbose,
        max_papers=config.paper_budget,
    )


def run_gene(config: BaselineRunConfig, gene: GeneInput) -> RawBaselineResult:
    import asyncio

    return raw_result_from_mapping(config, gene, asyncio.run(run_pipeline_graph(gene, config)))
