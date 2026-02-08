# Ontologies

This directory contains the ontology files used for mapping agent predictions to standardized terms.

## Required Files

| File | Description | Source |
|------|-------------|--------|
| `go-basic.obo` | Gene Ontology (basic) | [GO Downloads](http://geneontology.org/docs/download-ontology/) |
| `fly_anatomy.obo` | FlyBase anatomy (FBbt) | [OBO Foundry](http://purl.obolibrary.org/obo/fbbt.obo) |
| `fly_development.obo` | FlyBase dev stages (FBdv) | [OBO Foundry](http://purl.obolibrary.org/obo/fbdv.obo) |

## Download

Run the download script:

```bash
python scripts/download_ontologies.py
```

Or download manually from the URLs above.
