"""HF-first benchmark inputs for baseline reruns."""

from __future__ import annotations

from flyaoc.baselines.types import GeneInput
from flyaoc.data import DatasetConfig, load_benchmark_records


def load_gene_inputs(
    *,
    gene_id: str | None = None,
    limit: int | None = None,
    dataset: DatasetConfig | None = None,
) -> list[GeneInput]:
    rows = load_benchmark_records(dataset, limit=None, streaming=False)
    genes = [
        GeneInput(
            gene_id=row["gene_id"],
            gene_symbol=row["gene_symbol"],
            summary=row.get("summary", ""),
            pmcids=list(row.get("pmcids", [])),
        )
        for row in rows
    ]
    if gene_id is not None:
        genes = [gene for gene in genes if gene.gene_id == gene_id]
        if not genes:
            raise KeyError(f"Gene {gene_id} is not present in benchmark.jsonl")
    if limit is not None:
        genes = genes[:limit]
    return genes
