"""Prompts for LangGraph extraction tasks.

The base system prompt is imported from core.prompt_templates.
"""


# =============================================================================
# Unified 1-Pass Extraction Prompt (text-based, no ontology IDs)
# =============================================================================

UNIFIED_EXTRACTION_PROMPT = """Extract all annotations for gene {gene_symbol} ({gene_id}) from this paper.

Paper: {pmcid}

Text to analyze:
{text}

## Instructions

Extract ALL of the following in a SINGLE pass:

### 1. Function Annotations
For each molecular function, biological process, or cellular component:
- **function_description**: Use SHORT GO-like terms (2-5 words), NOT full sentences. Examples:
  - GOOD: "cell proliferation", "wing development", "transcription factor activity", "DNA binding"
  - BAD: "required for cell proliferation in the wing disc" (too long/specific)
  - BAD: "abd-A regulates development" (includes gene name)
- **qualifier**: "enables" (F), "involved_in" (P), "located_in"/"part_of"/"is_active_in" (C)
- **aspect**: "P" (process), "F" (function), or "C" (component)
- **is_negated**: true if the paper shows the gene does NOT have this function
- **evidence_text**: Verbatim quote from paper (max 500 chars)

### 2. Expression Annotations
For each expression site mentioned:
- **expression_type**: "polypeptide" (protein) or "transcript" (RNA)
- **anatomy_description**: Describe tissue/structure in natural language (e.g., "wing imaginal disc")
- **stage_description**: Describe developmental stage (e.g., "third instar larva")
- **evidence_text**: Verbatim quote from paper

### 3. Synonyms
- List any alternative names or symbols for the gene mentioned in the paper

## Output Format

Return a JSON object with this structure:
{{
    "function_annotations": [
        {{
            "function_description": "transcription factor activity",
            "qualifier": "enables",
            "aspect": "F",
            "is_negated": false,
            "evidence_text": "Quote from paper..."
        }}
    ],
    "expression_annotations": [
        {{
            "expression_type": "polypeptide",
            "anatomy_description": "wing imaginal disc",
            "stage_description": "third instar larva",
            "evidence_text": "Quote from paper..."
        }}
    ],
    "synonyms": ["alternative name 1", "alt-sym"]
}}

## Important
- Do NOT guess ontology IDs - just describe what you find in natural language
- Be thorough - a single paper often supports 5-10 annotations
- Focus on experimental evidence, not speculation or background statements
- Include ALL relevant annotations, not just one or two
- **Order annotations by confidence** - place those with stronger experimental evidence first"""

# =============================================================================
# Legacy prompts below - kept for reference
# =============================================================================


FUNCTION_EXTRACTION_PROMPT = """You are extracting Gene Ontology (GO) annotations from scientific text.

Gene: {gene_symbol} ({gene_id})
Paper: {pmcid}

Text to analyze:
{text}

Instructions:
1. Identify statements about what this gene does, what processes it's involved in, or where its product localizes.
2. For each finding, use the search_go_terms tool to find the appropriate GO term ID.
3. Determine the correct qualifier based on the relationship:
   - Molecular Function (F): "enables" (directly performs), "contributes_to" (partial activity)
   - Biological Process (P): "involved_in" (participates), "acts_upstream_of" (regulatory)
   - Cellular Component (C): "located_in" (found in), "part_of" (structural component), "is_active_in" (functions there)

Return annotations in this format:
{{
    "go_id": "GO:XXXXXXX",
    "qualifier": "involved_in",
    "aspect": "P",
    "is_negated": false,
    "evidence": {{
        "pmcid": "{pmcid}",
        "text": "Brief supporting text from paper (max 500 chars)"
    }}
}}

Be conservative - only include annotations that are clearly supported by the text.
Ignore speculative statements or those about other genes."""


EXPRESSION_EXTRACTION_PROMPT = """You are extracting gene expression data from scientific text.

Gene: {gene_symbol} ({gene_id})
Paper: {pmcid}

Text to analyze:
{text}

Instructions:
1. Identify statements about where and when this gene is expressed.
2. For tissue/anatomy terms, use search_anatomy_terms to find FBbt IDs.
3. For developmental stages, use search_stage_terms to find FBdv IDs.
4. Determine if expression is at protein level ("polypeptide") or RNA level ("transcript").

Return expression records in this format:
{{
    "expression_type": "polypeptide",
    "anatomy_id": "FBbt:XXXXXXXX",
    "stage_id": "FBdv:XXXXXXXX",
    "evidence": {{
        "pmcid": "{pmcid}",
        "text": "Brief supporting text from paper (max 500 chars)"
    }}
}}

Notes:
- anatomy_id and stage_id can be null if not specified in the text
- Be specific with anatomy terms (e.g., "wing disc" not just "imaginal disc")
- Include developmental stage when mentioned (embryo, larva, adult, etc.)"""


SYNONYM_EXTRACTION_PROMPT = """You are extracting gene name synonyms from scientific text.

Gene: {gene_symbol} ({gene_id})

Text to analyze:
{text}

Instructions:
1. Find alternative names or historical names for this gene.
2. Look for patterns like:
   - "GeneX (also known as GeneY)"
   - "GeneX, formerly called GeneY"
   - "GeneX/GeneY"
3. Distinguish between:
   - fullname_synonyms: Full gene names (e.g., "Abdominal A", "homeotic gene")
   - symbol_synonyms: Abbreviated symbols (e.g., "Abd-A", "abdA")

Return synonyms in this format:
{{
    "fullname_synonyms": ["Synonym 1", "Synonym 2"],
    "symbol_synonyms": ["sym1", "sym2"]
}}

Do NOT include:
- The official gene symbol itself (already known)
- Protein product names (e.g., "Abd-A protein")
- Allele names (e.g., "abd-A[1]")
- Ortholog names from other species"""


COMPILE_PROMPT = """You are compiling final gene annotations from extracted data.

Gene: {gene_symbol} ({gene_id})

Extracted GO annotations:
{go_annotations}

Extracted expression records:
{expression_records}

Extracted synonyms:
{synonyms}

Instructions:
1. Review all annotations for accuracy and completeness.
2. Remove duplicates (same GO ID + qualifier, or same anatomy+stage combination).
3. Ensure all IDs are in correct format:
   - GO IDs: GO:XXXXXXX (7 digits)
   - FBbt IDs: FBbt:XXXXXXXX (8 digits)
   - FBdv IDs: FBdv:XXXXXXXX (8 digits)
4. Validate qualifiers are from the allowed set.
5. Compile the final output.

Return the complete annotation object for validation using submit_annotations."""
