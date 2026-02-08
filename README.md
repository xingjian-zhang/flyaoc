# FlyAOC: Agentic Ontology Curation Benchmark

A benchmark for evaluating AI agents on literature-based gene annotation tasks in *Drosophila melanogaster*. Agents extract GO terms, expression data, and synonyms from scientific literature given a gene symbol.

## Getting Started

```bash
# Prerequisites: Python 3.11+, uv package manager
git clone <repo-url>
cd FlyAOC
uv sync

# Configure API key
cp .env.example .env
# Edit .env to add your OPENAI_API_KEY

# Run on a single gene
uv run python -m agent.run_agent --gene-id FBgn0000014 --gene-symbol abd-A
```

## Usage

```bash
# Single-Agent mode (default)
uv run python -m agent.run_agent --gene-id FBgn0000014 --gene-symbol abd-A

# Multi-Agent mode (hierarchical delegation, bounded context)
uv run python -m agent.run_agent --gene-id FBgn0000014 --gene-symbol abd-A --multi-agent

# With budget constraints (--max-turns 30, --max-papers 3, --max-cost 0.50)
uv run python -m agent.run_agent --gene-id FBgn0000014 --gene-symbol abd-A --max-turns 30 --max-papers 3

# Production model (default: gpt-5-mini ~$0.40/gene, gpt-5 ~$2/gene at 16 papers)
uv run python -m agent.run_agent --gene-id FBgn0000014 --gene-symbol abd-A --model gpt-5

# Verbose with trace output
uv run python -m agent.run_agent --gene-id FBgn0000014 --gene-symbol abd-A -v

# Specificity gap benchmark (hides 4,389 GO terms)
uv run python -m agent.run_agent --gene-id FBgn0000014 --gene-symbol abd-A --hide-terms
```

Output saved to `outputs/single_agent/{gene_id}.json` (or `outputs/multi_agent/` with `--multi-agent`).

**Alternative:** Pipeline agent available via `uv run python -m agent.run_pipeline`.

## Tasks

Given a gene symbol, agents must extract:

| Task | Output | Ontology |
|------|--------|----------|
| 1. Gene Function | GO terms with evidence | Gene Ontology |
| 2. Expression | Anatomy + developmental stage | FBbt, FBdv |
| 3. Synonyms | Historical names and aliases | - |

See [schemas/OUTPUT_SPEC.md](schemas/OUTPUT_SPEC.md) for field specifications.

## Architecture

Three agent methods:

**Single-Agent** (default): One agent with sequential tool use
- `uv run python -m agent.run_agent --gene-id ...`
- Context grows linearly with papers read
- OpenAI Agents SDK with MCP tool servers

**Multi-Agent**: Hierarchical delegation with bounded context
- `uv run python -m agent.run_agent --gene-id ... --multi-agent`
- Delegates paper reading to specialized paper reader agents
- Main agent context stays bounded regardless of paper count

**Pipeline**: Fixed parallel DAG
- `uv run python -m agent.run_pipeline --gene-id ...`
- Uses LangGraph with fan-out/fan-in pattern
- Faster but less flexible (~$0.10/gene at 16 papers)

**MCP Servers** (`agent/mcp_servers/`):
- `literature_server.py` - `search_corpus`, `get_paper_text`
- `ontology_server.py` - `search_go_terms`, `search_anatomy_terms`, `search_stage_terms`
- `validation_server.py` - `submit_annotations`

**Core modules** (`agent/core/`): BM25 corpus search, paper retrieval, Whoosh-indexed ontology search

## Data

**Literature corpus:** Available on HuggingFace (see paper for dataset link).

**Benchmark files** (`data/`):
- `genes_top100.csv` - 100 benchmark genes
- `ground_truth_top100.jsonl` - Ground truth annotations (JSONL)
- `gene_to_pmcids_top100.json` - Gene to PMCID mapping

**Ontologies** (`ontologies/`): `go-basic.obo`, `fly_anatomy.obo`, `fly_development.obo`

## Evaluation

Metrics: Precision, Recall, F1 against `in_corpus=True` annotations.

**Maximum achievable recall** (ground truth includes annotations not in corpus):
- Task 1 (GO): ~29%
- Task 2 (Expression): ~37%

```bash
uv run python -m eval.run_eval --gene-id FBgn0000014 -v  # Single gene
uv run python -m eval.run_eval --batch -v                # All genes
```

## Specificity Gap Benchmark

Tests whether models recognize when available GO terms lack sufficient specificity. When `--hide-terms` is enabled, 4,389 GO terms are hidden from ontology search results (116 primary terms + all descendants).

**Expected behavior:** When no suitable GO term exists, output a `description` instead of `go_id`:
```json
{"description": "compound eye morphogenesis", "qualifier": "involved_in", "aspect": "P"}
```

**Data files:**
- `data/ground_truth_top100_hidden.jsonl` - Ground truth with `hidden: true/false` flags
- `data/hidden_go_terms.json` - List of hidden terms (primary + descendants)

## Project Structure

```
agent/
  core/           # Corpus search, paper retrieval, ontology indices
  agentic/        # Single-Agent and Multi-Agent runner, budget hooks, prompts
  mcp_servers/    # MCP tool servers
  pipeline/       # Pipeline (LangGraph) implementation
  tools/          # LangChain tool wrappers
data/             # Benchmark genes and ground truth
eval/             # Evaluation scripts
ontologies/       # GO, FBbt, FBdv OBO files
schemas/          # Output specifications
```
