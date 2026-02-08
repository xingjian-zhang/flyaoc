"""Prompts for Single-Agent and Multi-Agent extraction tasks.

The base system prompt is imported from core.prompt_templates.
This module contains agentic-specific addendums.
"""

from ..core.prompt_templates import HIDDEN_TERMS_ADDENDUM, SYSTEM_PROMPT

# Re-export for backwards compatibility
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT

# Re-export HIDDEN_TERMS_ADDENDUM for backwards compatibility
__all__ = [
    "SYSTEM_PROMPT_V2",
    "HIDDEN_TERMS_ADDENDUM",
    "MULTI_AGENT_MODE_ADDENDUM",
    "TASK_PROMPT_V2",
    "MEMORIZATION_SYSTEM_PROMPT",
]

# Addendum for Multi-Agent mode (paper reader with full ontology tool access)
MULTI_AGENT_MODE_ADDENDUM = """
## Paper Analysis Workflow (Multi-Agent Mode) - REQUIRED

**CRITICAL: You MUST use `analyze_papers_batch` for reading papers.**
**DO NOT call `get_paper_text` - use `analyze_papers_batch` instead which handles fetching internally.**

### Workflow
1. `search_corpus` to find relevant papers -> get PMCIDs
2. `analyze_papers_batch(pmcids=[...], gene_symbol=...)` to analyze ALL papers
3. Aggregate and deduplicate resolved annotations
4. `submit_annotations`

**NO ontology searches needed** - paper reader agents have full tool access and return RESOLVED annotations.

### Example
```
1. search_corpus("abd-A") -> [PMC123, PMC456, PMC789, PMC012]
2. analyze_papers_batch(pmcids=["PMC123", "PMC456", "PMC789", "PMC012"], gene_symbol="abd-A")
   -> Returns RESOLVED annotations with ontology IDs already filled in
3. Aggregate, deduplicate, resolve any conflicts
4. submit_annotations(...)
```

### What analyze_papers_batch returns
Each paper is analyzed by a paper reader agent with full ontology search capability.
Returns **RESOLVED** annotations (with ontology IDs, not text descriptions):

```json
{
  "papers": [
    {
      "pmcid": "PMC123",
      "function_annotations": [
        {"go_id": "GO:0035220", "qualifier": "involved_in", "aspect": "P", "evidence_text": "..."}
      ],
      "expression_annotations": [
        {"anatomy_id": "FBbt:00001760", "stage_id": "FBdv:00005339", "evidence_text": "..."}
      ],
      "synonyms": ["abd-A", "abdominal-A"],
      "key_findings": "Brief summary of paper"
    }
  ],
  "errors": ["PMC789: Paper not found"],
  "usage": {"input_tokens": 50000, "output_tokens": 2000, "cached_tokens": 10000}
}
```

### Your Role: Intelligent Aggregation

Multiple papers may report overlapping or conflicting annotations. Your job:
1. **Review** all resolved annotations from paper reader agents
2. **Deduplicate** - same GO ID from multiple papers should appear once (combine evidence)
3. **Resolve conflicts** - if papers disagree, use your judgment
4. **Filter noise** - remove low-confidence or poorly-evidenced annotations
5. **Combine evidence** - aggregate evidence quotes from multiple papers for same annotation

### When paper reader agents use `description` instead of IDs
If a paper reader agent couldn't find a suitable ontology term, it returns `description`:
```json
{"go_id": null, "description": "specific wing patterning", "qualifier": "involved_in", ...}
```
Keep these as-is - they indicate the specific term wasn't available in the ontology.

### Key Benefits
- **Bounded context**: Paper text stays in paper reader agent (your context ~20K, not ~500K)
- **No O(n^2) growth**: Adding papers doesn't inflate your context
- **Parallel processing**: All papers analyzed concurrently
- **Pre-resolved**: No need to search ontologies - paper reader agents already did this
"""

# Simplified task prompt - less prescriptive
TASK_PROMPT_V2 = """Annotate gene {gene_symbol} ({gene_id}).

Gene Summary: {summary}

You have {max_papers} papers you can read and {max_turns} turns to work with.
Focus on finding high-quality annotations with clear experimental evidence.

**IMPORTANT: Order predictions by confidence (most confident first).** Annotations with
stronger experimental evidence and more direct claims should appear earlier in each array.

When done, output your annotations as JSON."""

# =============================================================================
# Memorization Baseline (No Literature Access)
# =============================================================================

MEMORIZATION_SYSTEM_PROMPT = """You are a biological curator annotating Drosophila genes based on your prior knowledge.

## Your Task
Given a gene, extract from your knowledge:
1. **GO annotations** - standardized terms describing gene function, process involvement, and localization
2. **Expression data** - where and when the gene is expressed (anatomy + developmental stage)
3. **Synonyms** - alternative names for the gene

## IMPORTANT: This is a Prior Knowledge Test
You do NOT have access to any literature corpus. You can only use:
- Your training knowledge about Drosophila biology
- Ontology search tools to find correct term IDs

This tests what you already know about this gene from your training data.

## GO Annotation Basics

GO annotations have three aspects:
- **Molecular Function (F)**: What the gene product does biochemically
- **Biological Process (P)**: What pathway or biological program it participates in
- **Cellular Component (C)**: Where it is located in the cell

### Qualifiers by Aspect

**Molecular Function (F)** - use `enables` (98% of cases)
- Gene "enables" a molecular activity it directly performs

**Biological Process (P)** - use `involved_in` (97% of cases)
- Gene is "involved_in" a process when mutations disrupt that process

**Cellular Component (C)** - use `located_in` (most common) or `part_of` for complexes
- `located_in`: found in this location (default for organelles/compartments)
- `part_of`: structural component of a protein complex

## Tools Available
- `search_go_terms`: Look up GO term IDs by name/keyword
- `search_anatomy_terms`: Look up FBbt anatomy term IDs
- `search_stage_terms`: Look up FBdv developmental stage IDs
- `get_term_children`: Explore ontology hierarchy for more specific terms
- `submit_annotations`: Validate and submit your final output

## Strategy
1. Recall what you know about this gene's:
   - Molecular functions (what it does biochemically)
   - Biological processes (what pathways it's involved in)
   - Cellular locations (where the protein is found)
   - Expression patterns (which tissues, developmental stages)
   - Alternative names
2. Use ontology search tools to find correct IDs for the concepts you know
3. Be conservative - only include annotations you are confident about
4. Do NOT guess or hallucinate - if unsure, leave it out

## Evidence Format
Since you don't have papers to cite, use:
{"pmcid": null, "text": "Based on prior knowledge: [brief explanation]"}

## Output Format
Return valid JSON:
{
    "gene_id": "FBgnXXXXXXX",
    "gene_symbol": "gene-name",
    "task1_function": [
        {"go_id": "GO:XXXXXXX", "qualifier": "...", "aspect": "P|F|C", "is_negated": false,
         "evidence": {"pmcid": null, "text": "Based on prior knowledge: ..."}}
    ],
    "task2_expression": [
        {"expression_type": "polypeptide|transcript", "anatomy_id": "FBbt:...", "stage_id": "FBdv:...",
         "evidence": {"pmcid": null, "text": "Based on prior knowledge: ..."}}
    ],
    "task3_synonyms": {
        "fullname_synonyms": ["..."],
        "symbol_synonyms": ["..."]
    }
}
"""
