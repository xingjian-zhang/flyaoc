# Agent Output Specification

## Overview

Given a **gene symbol**, agents search the literature corpus and return structured annotations:

| Task | Output | Ontology |
|------|--------|----------|
| Task 1 | Gene functions | GO terms |
| Task 2 | Expression patterns | FBbt (anatomy) + FBdv (stage) |
| Task 3 | Synonyms | Free text |

## Input

```json
{
  "gene_id": "FBgn0000014",
  "gene_symbol": "abd-A",
  "summary": "abdominal A (abd-A) encodes a homeobox-containing transcription factor..."
}
```

The `summary` field contains the expert-written FlyBase Gene Snapshot, providing context about the gene's known function.

## Output Structure

```json
{
  "gene_id": "FBgn0000014",
  "gene_symbol": "abd-A",
  "task1_function": [...],
  "task2_expression": [...],
  "task3_synonyms": {...}
}
```

---

## Task 1: Gene Function

Extract GO annotations with experimental evidence.

```json
{
  "task1_function": [
    {
      "go_id": "GO:0005634",
      "qualifier": "located_in",
      "aspect": "C",
      "evidence": {
        "pmcid": "PMC1234567",
        "text": "abd-A protein localized to the nucleus..."
      }
    }
  ]
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `go_id` | One of `go_id` or `description` | GO term ID (`GO:XXXXXXX`) |
| `description` | One of `go_id` or `description` | Proposed GO term name when no suitable term exists |
| `qualifier` | Yes | Relationship type |
| `aspect` | Yes | `P` (process), `F` (function), `C` (component) |
| `is_negated` | No | `true` for NOT annotations (default: `false`) |
| `evidence` | No | Source citation |

### Using `description` (Specificity Gap Mode)

When the ontology lacks a suitable GO term, use `description` instead of `go_id` to propose what the term name should be. This simulates a scientist proposing a new GO term.

**Format requirements:**
- Short noun phrase (2-6 words typical)
- Must be searchable: another person should find the correct term using this description
- Do NOT include regulatory relationships (use `qualifier` for that)

**Correct:**
```json
{
  "description": "compound eye morphogenesis",
  "qualifier": "involved_in",
  "aspect": "P"
}
```

**Incorrect:**
```json
{
  "description": "regulates programmed cell death during wing development",
  "qualifier": "involved_in",
  "aspect": "P"
}
```

The incorrect example conflates the biological concept with the regulatory relationship. Instead, use `"description": "wing programmed cell death"` with `"qualifier": "acts_upstream_of"`.

### Qualifiers

- **Component (C):** `located_in`, `part_of`, `is_active_in`, `colocalizes_with`
- **Function (F):** `enables`, `contributes_to`
- **Process (P):** `involved_in`, `acts_upstream_of`, `acts_upstream_of_positive_effect`, `acts_upstream_of_negative_effect`, `acts_upstream_of_or_within`, `acts_upstream_of_or_within_positive_effect`, `acts_upstream_of_or_within_negative_effect`

---

## Task 2: Expression

Extract where/when the gene is expressed.

```json
{
  "task2_expression": [
    {
      "expression_type": "polypeptide",
      "anatomy_id": "FBbt:00000146",
      "stage_id": "FBdv:00005327",
      "evidence": {
        "pmcid": "PMC5678901",
        "text": "abd-A protein detected in parasegment 8 at stage 12..."
      }
    }
  ]
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `expression_type` | Yes | `polypeptide` or `transcript` |
| `anatomy_id` | No | FBbt term (`FBbt:XXXXXXXX`) |
| `stage_id` | No | FBdv term (`FBdv:XXXXXXXX`) |
| `evidence` | No | Source citation |

At least one of `anatomy_id` or `stage_id` should be provided.

---

## Task 3: Synonyms

Extract historical names and aliases.

```json
{
  "task3_synonyms": {
    "fullname_synonyms": ["Abdominal A", "Contrabithoraxoid"],
    "symbol_synonyms": ["Abd-A", "abdA", "iab-2", "CG10325"]
  }
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `fullname_synonyms` | Yes | Full name variants (array) |
| `symbol_synonyms` | Yes | Symbol/alias variants (array) |

- Case-sensitive matching
- Do NOT include the current official symbol

---

## Evidence (Optional)

```json
{
  "evidence": {
    "pmcid": "PMC1234567",
    "text": "Supporting excerpt (max 500 chars)..."
  }
}
```

Evidence aids interpretability but is not scored.

---

## Complete Example

```json
{
  "gene_id": "FBgn0000014",
  "gene_symbol": "abd-A",
  "task1_function": [
    {
      "go_id": "GO:0005634",
      "qualifier": "located_in",
      "aspect": "C",
      "evidence": { "pmcid": "PMC2345678", "text": "abd-A localized to nuclei" }
    },
    {
      "go_id": "GO:0003700",
      "qualifier": "enables",
      "aspect": "F"
    },
    {
      "go_id": "GO:0009952",
      "qualifier": "involved_in",
      "aspect": "P"
    }
  ],
  "task2_expression": [
    {
      "expression_type": "polypeptide",
      "anatomy_id": "FBbt:00000146",
      "stage_id": "FBdv:00005327",
      "evidence": { "pmcid": "PMC5678901", "text": "detected in PS8 at stage 12" }
    },
    {
      "expression_type": "transcript",
      "anatomy_id": "FBbt:00001663",
      "stage_id": "FBdv:00005336"
    }
  ],
  "task3_synonyms": {
    "fullname_synonyms": ["Abdominal A", "Contrabithoraxoid"],
    "symbol_synonyms": ["Abd-A", "abdA", "iab-2", "CG10325"]
  }
}
```
