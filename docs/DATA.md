# Data Layout

FlyAOC uses the anonymous Hugging Face dataset as the canonical data source.
The review code should not depend on legacy local files such as
`genes_top100.csv`, `ground_truth_top100_as_verified.jsonl`, or
`gene_to_pmcids_top100.json`.

## HF Files

- `benchmark.jsonl`: one row per benchmark gene.
- `corpus.jsonl`: one row per literature article.
- `pmc_license_manifest.jsonl`: per-article provenance and license metadata.
- `ontologies/go-basic.obo`: GO ontology for Task 1 semantic matching.
- `ontologies/fly_anatomy.obo`: anatomy ontology for Task 2 semantic matching.
- `ontologies/fly_development.obo`: developmental-stage ontology.

## `benchmark.jsonl` Schema

Each benchmark row has:

- `gene_id`: FlyBase gene identifier.
- `gene_symbol`: gene symbol.
- `summary`: gene summary used as input context.
- `pmcids`: article IDs available for retrieval/evaluation.
- `task1_function`: verified GO labels.
- `task2_expression`: verified expression labels.
- `task3_synonyms`: verified synonym labels.

The evaluation code treats `in_corpus_verified` as the preferred corpus-grounded
label flag. If a row uses `in_corpus` instead, the loader accepts it as a
backward-compatible alias.

## Prediction Artifact Schema

Each row in `artifacts/predictions/**/*.jsonl` is one model output for one gene:

```json
{
  "gene_id": "FBgn0000014",
  "gene_symbol": "abd-A",
  "baseline": "single_agent",
  "provider": "openai",
  "model": "gpt-5-mini",
  "paper_budget": 8,
  "run_status": "ok",
  "task1_function_predictions": [{"go_id": "GO:0003700"}],
  "task2_expression_predictions": [{"anatomy_id": "FBbt:00001919", "stage_id": "FBdv:00005332"}],
  "task3_synonym_predictions": {
    "fullname_synonyms": ["abdominal A"],
    "symbol_synonyms": ["Abd-A"]
  }
}
```

Do not include prompts, traces, raw API responses, provider metadata, local
paths, or credentials in prediction artifacts.

`run_status` is one of:

- `ok`: the run produced a structured model output.
- `failed`: the run failed validation or execution and is represented as an
  empty submission for evaluation.
- `empty_output`: no structured output was present and no publishable error
  detail is included.
