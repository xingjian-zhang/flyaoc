# Results Manifest

`artifacts/predictions/manifest.json` is the machine-readable source of truth
for mapping frozen prediction files to paper tables and figures.

Current status:

- The smoke fixture is present and validates the evaluator.
- The final paper-result prediction files still need to be normalized from the
  internal output directories.

The normalization pass should add one prediction JSONL file per baseline,
provider, model, and paper budget. Each file should contain exactly the fields
defined in `docs/DATA.md`.

Expected main-body groups:

- Memorization, OpenAI GPT-family model used in the paper.
- Pipeline, OpenAI GPT-family model used in the paper.
- Single-Agent, OpenAI GPT-family model used in the paper.
- Multi-Agent, OpenAI GPT-family model used in the paper.
- Single-Agent cross-provider comparison: Claude Sonnet 4.6, MiniMax M2.5,
  DeepSeek V3.2.

Do not copy raw run directories wholesale. Normalize only the model output
objects needed by evaluation.
