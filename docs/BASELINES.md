# Baseline Reruns

The default paper reproduction path uses frozen predictions and does not need
API keys. This page is for anyone who wants to rerun the four main paper
baselines, evaluate generated outputs, or use the baseline harness as a
template for adding a new method.

The rerun harness ports the original paper code:

- memorization: MCP/OpenAI Agents SDK runner with literature access disabled
- pipeline: LangGraph fixed DAG
- single-agent: MCP/OpenAI Agents SDK runner with literature, ontology, and validation tools
- multi-agent: MCP/OpenAI Agents SDK runner with paper-reader delegation

Because provider APIs and stochastic decoding can change, API reruns reproduce
the experimental procedure. The exact paper tables are reproduced from the
frozen prediction files under `artifacts/predictions/`.

## Setup

```bash
uv sync --extra baselines
```

For direct OpenAI:

```bash
export OPENAI_API_KEY=...
```

For OpenAI-compatible routes, including Bedrock proxy routes:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint
```

The runner loads benchmark records, corpus papers, and ontology files from
`anonymous-042/flyaoc`. Local indexes and downloaded files are cached under
`.flyaoc_cache/`.

## Implementation Map

The baseline layer is split into two parts:

- `src/agent/`: a close port of the original paper harness.
- `src/flyaoc/baselines/`: thin public adapters, CLI, and output normalization.

The public baselines map to the original implementation as follows:

| CLI baseline | Adapter | Original harness call |
| --- | --- | --- |
| `memorization` | `flyaoc.baselines.adapters.memorization` | `agent.agentic.runner.run_agent_mcp` with `no_literature=True` |
| `single_agent` | `flyaoc.baselines.adapters.single_agent` | `agent.agentic.runner.run_agent_mcp` with `multi_agent=False` |
| `multi_agent` | `flyaoc.baselines.adapters.multi_agent` | `agent.agentic.runner.run_agent_mcp` with `multi_agent=True` |
| `pipeline` | `flyaoc.baselines.adapters.pipeline` | `agent.pipeline.agent.run_agent` |

This structure keeps the scientific behavior close to the paper code while
making the public rerun path easy to inspect.

## Commands

Memorization:

```bash
flyaoc-run-baseline \
  --baseline memorization \
  --provider openai \
  --model gpt-5-mini \
  --paper-budget 0 \
  --gene-id FBgn0000014 \
  --output-dir runs/memorization-smoke
```

Pipeline:

```bash
flyaoc-run-baseline \
  --baseline pipeline \
  --provider openai \
  --model gpt-5-mini \
  --paper-budget 16 \
  --limit 5 \
  --output-dir runs/pipeline-gpt5mini-budget16
```

Single-agent:

```bash
flyaoc-run-baseline \
  --baseline single_agent \
  --provider openai \
  --model gpt-5-mini \
  --paper-budget 16 \
  --limit 5 \
  --output-dir runs/single-agent-gpt5mini-budget16
```

Multi-agent:

```bash
flyaoc-run-baseline \
  --baseline multi_agent \
  --provider openai \
  --model gpt-5-mini \
  --paper-budget 16 \
  --limit 5 \
  --output-dir runs/multi-agent-gpt5mini-budget16
```

Cross-family models use the same commands with `--provider bedrock_proxy` once
`OPENAI_BASE_URL` points at the OpenAI-compatible proxy.

Each run writes:

- `raw/*.json`: raw per-gene outputs, usage, and errors for debugging
- `predictions.jsonl`: normalized evaluator-ready predictions
- `run_summary.json`: compact run metadata and status counts

Evaluate generated predictions:

```bash
flyaoc-evaluate-predictions runs/single-agent-gpt5mini-budget16/predictions.jsonl
```

`--resume` is enabled by default, so rerunning the same output directory skips
existing raw per-gene files. Use `--no-resume` to rerun all genes.

Missing-term experiments and retrieval ablations are intentionally excluded
from this v1 clean harness.
