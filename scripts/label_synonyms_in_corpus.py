"""Label synonyms with in_corpus flag by searching paper texts.

Unlike Tasks 1 & 2 (which have direct reference mappings), the synonym file
has no paper attribution. This script searches corpus paper texts to determine
which synonyms appear in the literature.

Usage:
    uv run python scripts/label_synonyms_in_corpus.py
"""

import json
import re
from pathlib import Path

from datasets import load_dataset


def load_corpus_texts(gene_pmcids: list[str]) -> dict[str, str]:
    """Load paper texts from the HuggingFace dataset."""
    print(f"Loading corpus for {len(gene_pmcids)} papers...")

    # Load the dataset
    dataset = load_dataset("jimmyzxj/drosophila-literature-corpus", split="train")

    # Filter to relevant papers and extract text
    texts = {}
    for paper in dataset:
        pmcid = paper.get("pmcid")
        if pmcid in gene_pmcids:
            # Combine title, abstract, and body text
            text_parts = []
            if paper.get("title"):
                text_parts.append(paper["title"])
            if paper.get("abstract"):
                text_parts.append(paper["abstract"])
            if paper.get("body"):
                text_parts.append(paper["body"])
            texts[pmcid] = " ".join(text_parts)

    print(f"Loaded {len(texts)} papers")
    return texts


def synonym_in_text(synonym: str, text: str) -> bool:
    """Check if synonym appears in text (case-insensitive, word boundary)."""
    # Escape special regex characters
    escaped = re.escape(synonym)
    # Word boundary match, case-insensitive
    pattern = rf"\b{escaped}\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


def process_gene_synonyms(
    gene_data: dict,
    gene_pmcids: list[str],
    corpus_texts: dict[str, str],
) -> dict:
    """Add in_corpus flags to synonyms for a gene."""

    # Combine all paper texts for this gene
    combined_text = " ".join(corpus_texts.get(pmcid, "") for pmcid in gene_pmcids)

    task3 = gene_data.get("task3_synonyms", {})

    # Process fullname synonyms
    fullname_with_flags = []
    for syn in task3.get("fullname_synonyms", []):
        in_corpus = synonym_in_text(syn, combined_text)
        fullname_with_flags.append(
            {
                "synonym": syn,
                "in_corpus": in_corpus,
            }
        )

    # Process symbol synonyms
    symbol_with_flags = []
    for syn in task3.get("symbol_synonyms", []):
        in_corpus = synonym_in_text(syn, combined_text)
        symbol_with_flags.append(
            {
                "synonym": syn,
                "in_corpus": in_corpus,
            }
        )

    # Update task3_synonyms structure
    new_task3 = {
        "current_fullname": task3.get("current_fullname"),
        "fullname_synonyms": fullname_with_flags,
        "symbol_synonyms": symbol_with_flags,
    }

    gene_data["task3_synonyms"] = new_task3
    return gene_data


def main():
    # Load ground truth
    gt_path = Path("data/ground_truth_top100.jsonl")
    pmcid_path = Path("data/gene_to_pmcids_top100.json")
    output_path = Path("data/ground_truth_top100_v2.jsonl")

    print("Loading ground truth...")
    with open(gt_path) as f:
        genes = [json.loads(line) for line in f]

    print("Loading gene-to-PMCID mapping...")
    with open(pmcid_path) as f:
        gene_to_pmcids = json.load(f)

    # Collect all PMCIDs we need
    all_pmcids = set()
    for gene_id, pmcids in gene_to_pmcids.items():
        all_pmcids.update(pmcids)

    # Load corpus texts
    corpus_texts = load_corpus_texts(list(all_pmcids))

    # Process each gene
    print("Processing synonyms...")
    updated_genes = []
    stats = {"total_fullname": 0, "in_corpus_fullname": 0, "total_symbol": 0, "in_corpus_symbol": 0}

    for gene in genes:
        gene_id = gene["gene_id"]
        gene_pmcids = gene_to_pmcids.get(gene_id, [])

        updated = process_gene_synonyms(gene, gene_pmcids, corpus_texts)
        updated_genes.append(updated)

        # Collect stats
        task3 = updated["task3_synonyms"]
        for s in task3.get("fullname_synonyms", []):
            stats["total_fullname"] += 1
            if s["in_corpus"]:
                stats["in_corpus_fullname"] += 1
        for s in task3.get("symbol_synonyms", []):
            stats["total_symbol"] += 1
            if s["in_corpus"]:
                stats["in_corpus_symbol"] += 1

    # Write output
    print(f"Writing to {output_path}...")
    with open(output_path, "w") as f:
        for gene in updated_genes:
            f.write(json.dumps(gene) + "\n")

    # Print stats
    print("\nSynonym coverage statistics:")
    print(
        f"  Fullname: {stats['in_corpus_fullname']}/{stats['total_fullname']} "
        f"({100 * stats['in_corpus_fullname'] / stats['total_fullname']:.1f}%) in corpus"
    )
    print(
        f"  Symbol: {stats['in_corpus_symbol']}/{stats['total_symbol']} "
        f"({100 * stats['in_corpus_symbol'] / stats['total_symbol']:.1f}%) in corpus"
    )

    print(f"\nDone! Updated ground truth saved to {output_path}")
    print("Review and rename to ground_truth_top100.jsonl when ready.")


if __name__ == "__main__":
    main()
