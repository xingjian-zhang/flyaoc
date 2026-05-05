# FlyAOC

FlyAOC is a benchmark for literature-grounded gene annotation in Drosophila.
This repository is designed for anyone who wants to reproduce our paper
results, add a new method, or run the same evaluation on their own predictions.

The benchmark data is hosted as an anonymous Hugging Face dataset:

<https://huggingface.co/datasets/anonymous-042/flyaoc>

The repository includes frozen predictions for the paper experiments, scripts
that regenerate the reported tables, evaluator code for new systems, and the
actual baseline harness used by the paper. Running the default reproduction
path does not require model API credentials. Rerunning baselines is optional
and requires provider credentials and API budget.

## Quickstart

```bash
uv sync
uv run python scripts/smoke_test.py
uv run python scripts/reproduce_tables.py
```

The smoke test downloads the benchmark labels and ontology files, evaluates a
small prediction fixture, and prints the resulting metrics. The table
reproduction command then recomputes the frozen-result tables used by the
paper.

## Common Workflows

To reproduce the paper results, run `scripts/reproduce_tables.py`. This loads
the released predictions from `artifacts/predictions/`, evaluates them against
the verified labels, and writes regenerated tables to `artifacts/tables/`,
including bootstrap confidence intervals for the primary baseline and model
comparisons.

To evaluate a new method, produce predictions in the normalized FlyAOC
prediction format and run the evaluator against the HF benchmark data. See
`docs/DATA.md` for the dataset schema and `artifacts/predictions/` for example
prediction files.

To rerun an existing baseline, install the optional baseline dependencies and
use the public runner. This invokes the original MCP/OpenAI Agents SDK or
LangGraph implementations, with data loaded from the anonymous HF dataset.
This path is separate from table reproduction because provider APIs can drift
over time; exact paper numbers are reproduced from frozen predictions.

```bash
uv sync --extra baselines
flyaoc-run-baseline --baseline memorization --provider openai --model gpt-5-mini \
  --paper-budget 0 --gene-id FBgn0000014 --output-dir runs/memorization-smoke
flyaoc-evaluate-predictions runs/memorization-smoke/predictions.jsonl
```

See `docs/BASELINES.md` for baseline rerun details.

## Official Metrics

The paper reports micro-averaged recall over verified, corpus-grounded facts:

- Task 1: GO semantic recall@30.
- Task 2: anatomy semantic recall@10.
- Task 3: synonym exact recall@20.

Failed or empty runs are counted as zero recall rather than removed from the
evaluation. This means the reported numbers reflect both annotation quality and
whether a system produced usable outputs. Macro averages and additional
recall@k cutoffs are included as secondary fields in the regenerated tables.

## Repository Map

```text
configs/                 Provider and experiment config templates.
src/agent/               Original paper baseline harness port.
src/flyaoc/data/          HF-first benchmark loaders.
src/flyaoc/evaluation/    Verified-label task evaluation.
src/flyaoc/reporting/     Table/figure reproduction helpers.
src/flyaoc/baselines/     Public baseline CLI and normalization.
scripts/                  Reproduction commands.
artifacts/predictions/    Frozen normalized model predictions.
artifacts/evaluations/    Derived metric summaries.
artifacts/tables/         Regenerated paper tables.
artifacts/figures/        Regenerated paper figures.
docs/                     Data and reproducibility documentation.
runs/                     Ignored local baseline reruns.
```

## Reproduction Path

The default reproduction path does not require OpenAI, Anthropic, Bedrock, or
other model-provider credentials. It performs the following steps:

1. Load benchmark labels and ontology files from HF.
2. Load normalized frozen predictions from `artifacts/predictions/`.
3. Recompute verified-label metrics.
4. Regenerate paper-facing tables and figures.

The optional API rerun path is separate from this workflow. It reproduces the
experimental procedure with current model-provider behavior; it should not be
treated as a bit-identical route to the frozen paper tables.

## Data Source

The code treats the HF dataset layout as canonical:

- `benchmark.jsonl`: gene metadata, PMCID list, and verified labels.
- `corpus.jsonl`: full-text paper corpus.
- `ontologies/go-basic.obo`: Gene Ontology.
- `ontologies/fly_anatomy.obo`: FlyBase anatomy ontology.
- `ontologies/fly_development.obo`: FlyBase developmental-stage ontology.

See `docs/DATA.md` for schemas.

## Included Results

The released prediction files cover the main architecture comparison,
GPT-family model scaling, cross-family model comparison, and the budget-32
multi-agent run. The prediction manifest records the baseline, provider, model,
paper budget, source run, and run-status counts for each file.
