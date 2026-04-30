# Prediction Artifacts

This directory contains normalized frozen predictions used for reproduction.

Rules:

- One JSONL row per gene.
- Include only model outputs needed by evaluation.
- Strip traces, prompts, logs, raw API responses, local paths, and credentials.
- Map every paper-result file in `manifest.json`.

The `smoke/` fixture is intentionally small and is not a paper-result artifact.
