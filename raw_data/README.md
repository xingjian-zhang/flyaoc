# Raw Data Files

Source files used to generate ground truth labels for the benchmark.

## Files

### FlyBase Source Files (FB2025_04 release)

| File | Description | Format | Task |
|------|-------------|--------|------|
| `gene_association.fb.gz` | GO annotations | GAF 2.2 | Task 1: Function |
| `curated_expression_fb_2025_04.tsv.gz` | Expression data | TSV | Task 2: Expression |
| `fb_synonym_fb_2025_04.tsv.gz` | Gene synonyms | TSV | Task 3: Synonyms |
| `fbrf_pmid_pmcid_doi_fb_2025_04.tsv` | Reference ID mapping | TSV | Coverage analysis |

Downloaded from [FlyBase Bulk Data Downloads](https://flybase.org/downloads/bulkdata).

### Benchmark Input Files

| File | Description | Format |
|------|-------------|--------|
| `genes.csv` | 3,446 benchmark genes with expert summaries | CSV |
| `gene_to_pmcids.json` | Gene to literature corpus mapping | JSON |

## Format Details

### gene_association.fb.gz (GAF 2.2)

Gene Ontology Annotation File format. Key columns:
- Column 2: Gene ID (FBgn)
- Column 4: Qualifier (enables, involved_in, located_in)
- Column 5: GO ID
- Column 6: Reference (PMID or GO_REF)
- Column 7: Evidence code
- Column 9: Aspect (P/F/C)

### curated_expression_fb_2025_04.tsv.gz

Expression data with ontology IDs in parenthetical format:
- `stage_start/end`: e.g., "embryonic stage 4 (FBdv:00005306)"
- `anatomical_structure_term`: e.g., "organism (FBbt:00000001)"
- `reference_id`: FBrf IDs

### fb_synonym_fb_2025_04.tsv.gz

Gene synonyms with pipe-separated values:
- Column 2: organism_abbreviation (filter for "Dmel")
- Column 5: fullname_synonym(s)
- Column 6: symbol_synonym(s)

### fbrf_pmid_pmcid_doi_fb_2025_04.tsv

Reference ID mapping between FlyBase and external databases:
- Column 1: FBrf (FlyBase reference ID)
- Column 2: PMID
- Column 3: PMCID
- Column 4: DOI

Used to check which ground truth references are available in the literature corpus.
