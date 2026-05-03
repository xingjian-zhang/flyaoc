# Baseline Reruns

The default paper reproduction path uses frozen predictions and does not need
API keys. This page is for reviewers who want to rerun the four main paper
baselines with their own credentials and then evaluate the generated outputs.

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
