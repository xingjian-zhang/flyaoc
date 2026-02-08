# Evaluation Framework

Evaluate agent annotation quality against FlyBase ground truth.

## Usage

```bash
# Single gene evaluation
uv run python -m eval.run_eval --gene-id FBgn0000014 -v

# Batch evaluation
uv run python -m eval.run_eval --batch -v

# Save results to file
uv run python -m eval.run_eval --gene-id FBgn0000014 -o results.json
```

## Metrics

### Task 1: GO Annotations
- **Exact match**: GO ID matches exactly (qualifier ignored)
- **Soft match**: Wang semantic similarity ≥ 0.7 (same aspect only)
- Reports both in-corpus and full ground truth recall

### Task 2: Expression
- **Anatomy-only**: Match if anatomy_id (FBbt) matches
- **Stage-only**: Match if stage_id (FBdv) falls within ground truth range
- **Tuple**: Match if both anatomy AND stage match

### Task 3: Synonyms
- Case-insensitive string matching
- Separate metrics for fullname and symbol synonyms

## Ground Truth `in_corpus` Flag

The `in_corpus` flag indicates whether an annotation can theoretically be found in the literature corpus.

| Task | How `in_corpus` is determined |
|------|------------------------------|
| Task 1 (GO) | Direct reference mapping: annotation has PMID → map to PMCID → check if PMCID in corpus |
| Task 2 (Expression) | Direct reference mapping: annotation has FBrf → map to PMCID → check if PMCID in corpus |
| Task 3 (Synonyms) | **Text search**: synonym file has NO references; must search paper texts to determine if synonym appears in corpus |

**Note on Task 3**: Unlike Tasks 1 & 2 where FlyBase provides per-annotation references, the synonym file (`fb_synonym_fb_2025_04.tsv.gz`) is a flat list of all known synonyms without paper attribution. The `in_corpus` flag for synonyms is determined by searching corpus paper texts for each synonym string, which is a fundamentally different labeling approach.

## Files

- `evaluator.py` - Main orchestrator
- `loader.py` - Load ground truth and agent outputs
- `metrics.py` - Precision/recall/F1 calculations
- `go_similarity.py` - Wang semantic similarity using goatools
- `task1_go.py` - GO annotation evaluation
- `task2_expression.py` - Expression evaluation
- `task3_synonyms.py` - Synonym evaluation
- `run_eval.py` - CLI entry point
