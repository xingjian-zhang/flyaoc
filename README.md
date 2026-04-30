# FlyAOC NeurIPS Review Artifact

This repository is the anonymous review artifact for FlyAOC, a benchmark for
literature-grounded gene annotation in Drosophila.

The benchmark data source is the anonymous Hugging Face dataset:

https://huggingface.co/datasets/anonymous-042/flyaoc

This GitHub artifact contains the code and frozen result structure needed to
reproduce paper tables and figures without rerunning model APIs. Agent reruns
are documented as optional and require provider credentials.

## Quickstart

```bash
uv sync
uv run python scripts/smoke_test.py
uv run python scripts/reproduce_tables.py
```

The smoke test downloads `benchmark.jsonl` and the ontology files from the
anonymous HF dataset, evaluates a small normalized prediction fixture, and
prints the computed metrics.

## Repository Map

```text
configs/                 Provider and experiment config templates.
src/flyaoc/data/          HF-first benchmark loaders.
src/flyaoc/evaluation/    Verified-label task evaluation.
src/flyaoc/reporting/     Table/figure reproduction helpers.
src/flyaoc/baselines/     Baseline runner entry points and documentation.
src/flyaoc/providers/     Provider adapter interfaces and config schemas.
scripts/                  Reviewer-facing reproduction commands.
artifacts/predictions/    Frozen normalized model predictions.
artifacts/evaluations/    Derived metric summaries.
artifacts/tables/         Regenerated paper tables.
artifacts/figures/        Regenerated paper figures.
docs/                     Data and reproducibility documentation.
```

## Default Reproduction Path

Reviewers do not need OpenAI, Bedrock, or other model-provider credentials for
the default reproduction path. The intended path is:

1. Load benchmark labels and ontology files from HF.
2. Load normalized frozen predictions from `artifacts/predictions/`.
3. Recompute verified-label metrics.
4. Regenerate paper-facing tables and figures.

The optional API rerun path is intentionally separate from the default path.

## Data Source

The code treats the HF dataset layout as canonical:

- `benchmark.jsonl`: gene metadata, PMCID list, and verified labels.
- `corpus.jsonl`: full-text paper corpus.
- `ontologies/go-basic.obo`: Gene Ontology.
- `ontologies/fly_anatomy.obo`: FlyBase anatomy ontology.
- `ontologies/fly_development.obo`: FlyBase developmental-stage ontology.

See `docs/DATA.md` for schemas.

## Current Artifact Status

This branch establishes the review artifact structure, evaluator, scripts, and
smoke fixture. The remaining normalization pass is to import the final frozen
paper-result predictions into `artifacts/predictions/` and update
`artifacts/predictions/manifest.json` so every main-body table maps to a
specific prediction file.
