"""Tool usage and behavioral analysis functions for deep analysis."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .dataclasses import PaperRead, ToolCall


def extract_detailed_tool_usage(base_dir: Path, configs: list[int]) -> dict[str, Any]:
    """Extract detailed tool usage patterns from traces."""

    all_tool_calls: list[ToolCall] = []
    paper_reads: list[PaperRead] = []
    ontology_searches = defaultdict(list)

    for config in configs:
        traces_dir = base_dir / f"papers_{config}" / "traces"
        if not traces_dir.exists():
            continue

        for trace_file in traces_dir.glob("*_trace.json"):
            gene_id = trace_file.stem.replace("_trace", "")

            try:
                with open(trace_file) as f:
                    trace = json.load(f)
            except json.JSONDecodeError:
                continue

            paper_read_order = 0
            corpus_search_results = []

            for event in trace.get("events", []):
                if event.get("event_type") == "tool_call":
                    tool = event.get("tool", "")
                    args = event.get("args", {})

                    all_tool_calls.append(
                        ToolCall(
                            tool=tool,
                            args=args,
                            result=None,
                            timestamp=event.get("timestamp", ""),
                            gene_id=gene_id,
                            paper_config=config,
                        )
                    )

                    if tool == "get_paper_text":
                        paper_read_order += 1
                        pmcid = args.get("pmcid", "")

                        # Find if this paper was in search results with gene_in_title
                        gene_in_title = False
                        relevance_score = 0.0
                        for sr in corpus_search_results:
                            if sr.get("pmcid") == pmcid:
                                gene_in_title = sr.get("gene_in_title", False)
                                relevance_score = sr.get("relevance_score", 0.0)
                                break

                        paper_reads.append(
                            PaperRead(
                                pmcid=pmcid,
                                gene_id=gene_id,
                                paper_config=config,
                                read_order=paper_read_order,
                                gene_in_title=gene_in_title,
                                relevance_score=relevance_score,
                            )
                        )

                    elif tool.startswith("search_"):
                        query = args.get("query", args.get("term", ""))
                        ontology_searches[tool].append(
                            {
                                "query": query,
                                "gene_id": gene_id,
                                "config": config,
                            }
                        )

                elif event.get("event_type") == "tool_result":
                    tool = event.get("tool", "")
                    if tool == "search_corpus":
                        # Parse search results
                        result = event.get("result", {})
                        if isinstance(result, dict) and "text" in result:
                            try:
                                corpus_search_results = json.loads(result["text"])
                            except (json.JSONDecodeError, TypeError):
                                pass

    # Analyze tool call patterns
    tool_counts = Counter(tc.tool for tc in all_tool_calls)

    # Analyze paper selection strategy
    gene_in_title_reads = [pr for pr in paper_reads if pr.gene_in_title]

    # Analyze by read order
    reads_by_order = defaultdict(list)
    for pr in paper_reads:
        reads_by_order[pr.read_order].append(pr)

    # Ontology search analysis
    search_analysis = {}
    for tool, searches in ontology_searches.items():
        queries = [s["query"] for s in searches]
        search_analysis[tool] = {
            "total_searches": len(searches),
            "unique_queries": len(set(queries)),
            "avg_per_gene": len(searches) / len(configs) / 100,  # Approximate
            "sample_queries": list(set(queries))[:10],
        }

    return {
        "tool_call_counts": dict(tool_counts),
        "total_tool_calls": len(all_tool_calls),
        "paper_selection": {
            "total_papers_read": len(paper_reads),
            "gene_in_title_fraction": len(gene_in_title_reads) / len(paper_reads)
            if paper_reads
            else 0,
            "avg_relevance_score": float(np.mean([pr.relevance_score for pr in paper_reads]))
            if paper_reads
            else 0,
        },
        "reads_by_position": {
            order: {
                "count": len(reads),
                "gene_in_title_fraction": sum(1 for r in reads if r.gene_in_title) / len(reads)
                if reads
                else 0,
            }
            for order, reads in sorted(reads_by_order.items())[:10]
        },
        "ontology_searches": search_analysis,
    }


def analyze_paper_selection_effectiveness(
    base_dir: Path, configs: list[int], all_evals: dict[int, dict]
) -> dict[str, Any]:
    """Analyze whether paper selection strategy affects success."""

    # For each gene, track which papers were read and whether they led to TPs
    gene_paper_effectiveness = []

    max_config = max(configs)
    traces_dir = base_dir / f"papers_{max_config}" / "traces"

    if not traces_dir.exists():
        return {"error": "No traces found"}

    eval_data = all_evals.get(max_config, {})
    eval_by_gene = {g["gene_id"]: g for g in eval_data.get("genes", [])}

    for trace_file in traces_dir.glob("*_trace.json"):
        gene_id = trace_file.stem.replace("_trace", "")

        try:
            with open(trace_file) as f:
                trace = json.load(f)
        except json.JSONDecodeError:
            continue

        # Extract papers read in order
        papers_read = []
        corpus_results = []

        for event in trace.get("events", []):
            if event.get("event_type") == "tool_result" and event.get("tool") == "search_corpus":
                result = event.get("result", {})
                if isinstance(result, dict) and "text" in result:
                    try:
                        corpus_results = json.loads(result["text"])
                    except (json.JSONDecodeError, TypeError):
                        pass

            elif event.get("event_type") == "tool_call" and event.get("tool") == "get_paper_text":
                pmcid = event.get("args", {}).get("pmcid", "")

                # Find paper info
                paper_info = {"pmcid": pmcid, "gene_in_title": False, "relevance_score": 0}
                for cr in corpus_results:
                    if cr.get("pmcid") == pmcid:
                        paper_info["gene_in_title"] = cr.get("gene_in_title", False)
                        paper_info["relevance_score"] = cr.get("relevance_score", 0)
                        break

                papers_read.append(paper_info)

        # Get evaluation results
        gene_eval = eval_by_gene.get(gene_id, {})
        tp = gene_eval.get("task1_go", {}).get("soft", {}).get("tp", 0)
        fp = gene_eval.get("task1_go", {}).get("soft", {}).get("fp", 0)

        gene_paper_effectiveness.append(
            {
                "gene_id": gene_id,
                "papers_count": len(papers_read),
                "gene_in_title_count": sum(1 for p in papers_read if p["gene_in_title"]),
                "avg_relevance": float(np.mean([p["relevance_score"] for p in papers_read]))
                if papers_read
                else 0,
                "tp": tp,
                "fp": fp,
                "f1": gene_eval.get("task1_go", {}).get("soft", {}).get("f1", 0),
            }
        )

    # Correlation analysis
    if len(gene_paper_effectiveness) > 5:
        git_counts = [g["gene_in_title_count"] for g in gene_paper_effectiveness]
        f1s = [g["f1"] for g in gene_paper_effectiveness]
        relevances = [g["avg_relevance"] for g in gene_paper_effectiveness]

        git_f1_corr = float(np.corrcoef(git_counts, f1s)[0, 1]) if len(set(git_counts)) > 1 else 0
        rel_f1_corr = float(np.corrcoef(relevances, f1s)[0, 1]) if len(set(relevances)) > 1 else 0
    else:
        git_f1_corr = 0
        rel_f1_corr = 0

    return {
        "gene_count": len(gene_paper_effectiveness),
        "avg_gene_in_title_papers": float(
            np.mean([g["gene_in_title_count"] for g in gene_paper_effectiveness])
        ),
        "avg_relevance_score": float(
            np.mean([g["avg_relevance"] for g in gene_paper_effectiveness])
        ),
        "correlation_gene_in_title_vs_f1": git_f1_corr,
        "correlation_relevance_vs_f1": rel_f1_corr,
        "interpretation": (
            f"Gene-in-title correlation with F1: {git_f1_corr:.3f}. "
            f"{'Prioritizing gene-focused papers helps.' if git_f1_corr > 0.1 else 'Paper selection strategy has limited impact on performance.'}"
        ),
    }
