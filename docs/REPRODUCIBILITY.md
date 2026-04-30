# Reproducibility

The required review path is designed to be credential-free.

## No-API Path

```bash
uv sync
uv run python scripts/smoke_test.py
uv run python scripts/reproduce_tables.py
```

This path loads labels from HF and predictions from `artifacts/predictions/`,
then recomputes verified-label metrics.

## Optional API Reruns

Agent reruns are optional and require provider credentials. The review branch
keeps this separate from table reproduction so reviewers can verify reported
numbers without spending API budget.

The supported paper baselines are:

- Memorization
- Pipeline
- Single-Agent
- Multi-Agent

The supported paper providers/models are documented in
`configs/providers.example.yaml`.

## Excluded from v1

The missing-term experiment is excluded from the clean review branch v1 because
it requires a separate hidden-label setup. The main-body verified-label
benchmark path remains fully represented.
