# Results Manifest

`artifacts/predictions/manifest.json` is the machine-readable source of truth
for mapping frozen prediction files to paper tables and figures.

The manifest contains one prediction JSONL file per baseline, provider, model,
and paper budget used by the main paper-result groups. Each file contains the
fields defined in `docs/DATA.md`.

Included result groups:

- Memorization, OpenAI GPT-family model used in the paper.
- Pipeline, OpenAI GPT-family model used in the paper.
- Single-Agent, OpenAI GPT-family model used in the paper.
- Multi-Agent, OpenAI GPT-family model used in the paper.
- Single-Agent cross-provider comparison: Claude Sonnet 4.6, MiniMax M2.5,
  DeepSeek V3.2.
- Cross-family memorization baselines for Claude Sonnet 4, MiniMax M2.5, and
  DeepSeek V3.2.
- Multi-Agent GPT-5-mini budget-32 extended scaling.

Do not copy raw run directories wholesale. Normalize only the model output
objects needed by evaluation.

`artifacts/tables/main_results.csv` is generated from this manifest and is the
canonical no-API metric table. It reports primary micro metrics, secondary
macro metrics, denominator counts, and additional recall@k columns. Failed and
empty-output rows remain in the evaluated gene set and score zero recall.
The primary metrics are GO semantic recall@30, anatomy semantic recall@10, and
synonym exact recall@20.

The generated `artifacts/tables/*_bootstrap_ci.csv` files cover the primary
paper comparisons: main architecture scaling, fixed-harness Multi-Agent model
comparison, and cross-family harness comparison. Each bootstrap sample resamples
genes with replacement and recomputes the primary micro-averaged recall
estimator, so the intervals quantify variation across benchmark genes rather
than repeated provider/API runs.
