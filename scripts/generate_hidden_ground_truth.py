#!/usr/bin/env python3
"""Generate ground truth with hidden GO term flags.

This script creates `data/ground_truth_top100_hidden.jsonl` by:
1. Counting GO term frequency across ALL ground truth annotations
2. Identifying leaf nodes (terms with no children in GO hierarchy)
3. Selecting terms to hide until 15% of in-corpus annotations are affected
4. Expanding to include ALL descendants of hidden terms (so agent can't find children)
5. Adding `hidden: true/false` flag to each annotation

The selection uses impact-based targeting: we greedily select terms (prioritizing
singleton+leaf, then non-singleton+leaf, then non-leaf by cascade potential) until
the target percentage of in-corpus annotations are affected.

Usage:
    python -m scripts.generate_hidden_ground_truth
"""

import json
from collections import Counter
from pathlib import Path

# Import ontology module to access GO hierarchy
from agent.core.ontology import OntologyHierarchy, parse_obo_file

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
ONTOLOGY_DIR = Path(__file__).parent.parent / "ontologies"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth_top100.jsonl"
OUTPUT_PATH = DATA_DIR / "ground_truth_top100_hidden.jsonl"

# Target: hide terms until this fraction of in-corpus annotations are affected
# 15% provides meaningful challenge while preserving 85% for baseline measurement
TARGET_IN_CORPUS_IMPACT = 0.15


def load_ground_truth() -> list[dict]:
    """Load ground truth JSONL file."""
    genes = []
    with open(GROUND_TRUTH_PATH) as f:
        for line in f:
            genes.append(json.loads(line))
    return genes


def count_go_term_frequency(genes: list[dict]) -> Counter:
    """Count how many times each GO term appears across ALL annotations.

    Counts ALL annotations, not just in_corpus ones, since we want to know
    true term frequency across the benchmark.
    """
    counter = Counter()
    for gene in genes:
        for annot in gene.get("task1_function", []):
            go_id = annot.get("go_id")
            if go_id:
                counter[go_id] += 1
    return counter


def get_in_corpus_terms(genes: list[dict]) -> set[str]:
    """Get GO terms that have at least one in_corpus annotation.

    We prioritize hiding these since they're theoretically findable.
    """
    terms = set()
    for gene in genes:
        for annot in gene.get("task1_function", []):
            if annot.get("in_corpus"):
                go_id = annot.get("go_id")
                if go_id:
                    terms.add(go_id)
    return terms


def count_in_corpus_by_term(genes: list[dict]) -> tuple[Counter, int]:
    """Count in-corpus annotations per GO term.

    Returns:
        Tuple of (Counter mapping term -> in_corpus count, total in_corpus count)
    """
    counter = Counter()
    total = 0
    for gene in genes:
        for annot in gene.get("task1_function", []):
            if annot.get("in_corpus"):
                go_id = annot.get("go_id")
                if go_id:
                    counter[go_id] += 1
                    total += 1
    return counter, total


def load_go_hierarchy() -> OntologyHierarchy:
    """Load GO ontology hierarchy to identify leaf nodes."""
    obo_path = ONTOLOGY_DIR / "go-basic.obo"
    terms = parse_obo_file(obo_path)
    return OntologyHierarchy(terms)


def get_all_descendants(term_id: str, hierarchy: OntologyHierarchy) -> set[str]:
    """Get all descendants (children, grandchildren, etc.) of a term.

    Uses BFS to traverse the hierarchy downward.
    """
    descendants = set()
    queue = list(hierarchy.children.get(term_id, []))

    while queue:
        child = queue.pop(0)
        if child not in descendants:
            descendants.add(child)
            # Add this child's children to the queue
            queue.extend(hierarchy.children.get(child, []))

    return descendants


def expand_with_descendants(
    primary_terms: set[str],
    hierarchy: OntologyHierarchy,
) -> tuple[set[str], dict]:
    """Expand a set of terms to include all their descendants.

    Args:
        primary_terms: The initially selected terms to hide
        hierarchy: GO ontology hierarchy

    Returns:
        Tuple of (expanded set including descendants, stats dict)
    """
    all_hidden = set(primary_terms)
    descendant_count = 0

    for term_id in primary_terms:
        descendants = get_all_descendants(term_id, hierarchy)
        new_descendants = descendants - all_hidden
        descendant_count += len(new_descendants)
        all_hidden.update(descendants)

    stats = {
        "primary_hidden_terms": len(primary_terms),
        "descendants_added": descendant_count,
        "total_hidden_terms": len(all_hidden),
    }

    return all_hidden, stats


def select_hidden_terms(
    term_counts: Counter,
    in_corpus_terms: set[str],
    in_corpus_by_term: Counter,
    total_in_corpus: int,
    hierarchy: OntologyHierarchy,
) -> tuple[set[str], dict]:
    """Select terms to hide using impact-based selection.

    Greedily selects terms until TARGET_IN_CORPUS_IMPACT fraction of in-corpus
    annotations are affected. Uses prioritized ordering:
    1. Singleton (freq == 1) + leaf (no children) + in_corpus
    2. Non-singleton + leaf + in_corpus
    3. Non-leaf + in_corpus (sorted by children count for cascade potential)

    Returns:
        Tuple of (set of hidden term IDs, selection stats dict)
    """
    # Identify leaf terms (no children)
    all_go_terms = set(term_counts.keys())
    leaf_terms = {t for t in all_go_terms if hierarchy.get_children_count(t) == 0}
    nonleaf_terms = all_go_terms - leaf_terms

    # Identify singletons
    singleton_terms = {t for t, count in term_counts.items() if count == 1}

    # Prioritize in_corpus terms for hiding
    singleton_leaf_in_corpus = singleton_terms & leaf_terms & in_corpus_terms
    nonsingleton_leaf_in_corpus = (leaf_terms - singleton_terms) & in_corpus_terms
    nonleaf_in_corpus = nonleaf_terms & in_corpus_terms

    # Build priority queue: singleton+leaf, non-singleton+leaf, non-leaf (by children)
    priority_queue = []
    priority_queue.extend(sorted(singleton_leaf_in_corpus))
    priority_queue.extend(sorted(nonsingleton_leaf_in_corpus))
    priority_queue.extend(
        sorted(nonleaf_in_corpus, key=lambda t: hierarchy.get_children_count(t), reverse=True)
    )

    # Greedy selection until we hit target impact
    target_annotations = int(total_in_corpus * TARGET_IN_CORPUS_IMPACT)
    hidden_terms = set()
    current_impact = 0
    selected_by_category = {"singleton_leaf": 0, "nonsingleton_leaf": 0, "nonleaf": 0}

    for term in priority_queue:
        if current_impact >= target_annotations:
            break

        # Add term and count its impact
        hidden_terms.add(term)
        current_impact += in_corpus_by_term.get(term, 0)

        # Track which category this term came from
        if term in singleton_leaf_in_corpus:
            selected_by_category["singleton_leaf"] += 1
        elif term in nonsingleton_leaf_in_corpus:
            selected_by_category["nonsingleton_leaf"] += 1
        else:
            selected_by_category["nonleaf"] += 1

    stats = {
        "total_go_terms_in_benchmark": len(all_go_terms),
        "leaf_terms_in_benchmark": len(leaf_terms),
        "nonleaf_terms_in_benchmark": len(nonleaf_terms),
        "singleton_terms": len(singleton_terms),
        "in_corpus_terms": len(in_corpus_terms),
        "total_in_corpus_annotations": total_in_corpus,
        "target_impact_ratio": TARGET_IN_CORPUS_IMPACT,
        "target_annotations": target_annotations,
        "singleton_leaf_in_corpus_available": len(singleton_leaf_in_corpus),
        "nonsingleton_leaf_in_corpus_available": len(nonsingleton_leaf_in_corpus),
        "nonleaf_in_corpus_available": len(nonleaf_in_corpus),
        "selected_singleton_leaf": selected_by_category["singleton_leaf"],
        "selected_nonsingleton_leaf": selected_by_category["nonsingleton_leaf"],
        "selected_nonleaf": selected_by_category["nonleaf"],
        "total_primary_hidden": len(hidden_terms),
        "actual_in_corpus_impact": current_impact,
        "actual_impact_ratio": current_impact / total_in_corpus,
    }

    return hidden_terms, stats


def add_hidden_flags(genes: list[dict], hidden_terms: set[str]) -> list[dict]:
    """Add hidden flag to each GO annotation.

    Returns new list with modified gene records.
    """
    result = []
    for gene in genes:
        gene_copy = dict(gene)
        new_annotations = []
        for annot in gene.get("task1_function", []):
            annot_copy = dict(annot)
            go_id = annot_copy.get("go_id")
            annot_copy["hidden"] = go_id in hidden_terms if go_id else False
            new_annotations.append(annot_copy)
        gene_copy["task1_function"] = new_annotations
        result.append(gene_copy)
    return result


def count_hidden_annotations(genes: list[dict]) -> dict:
    """Count hidden annotations for reporting."""
    total_hidden = 0
    in_corpus_hidden = 0
    for gene in genes:
        for annot in gene.get("task1_function", []):
            if annot.get("hidden"):
                total_hidden += 1
                if annot.get("in_corpus"):
                    in_corpus_hidden += 1
    return {
        "total_hidden_annotations": total_hidden,
        "in_corpus_hidden_annotations": in_corpus_hidden,
    }


def main():
    print("Loading ground truth...")
    genes = load_ground_truth()
    print(f"  Loaded {len(genes)} genes")

    print("Counting GO term frequency...")
    term_counts = count_go_term_frequency(genes)
    print(f"  Found {len(term_counts)} unique GO terms")

    print("Identifying in_corpus terms...")
    in_corpus_terms = get_in_corpus_terms(genes)
    print(f"  Found {len(in_corpus_terms)} terms with in_corpus annotations")

    print("Counting in-corpus annotations per term...")
    in_corpus_by_term, total_in_corpus = count_in_corpus_by_term(genes)
    print(f"  Total in-corpus annotations: {total_in_corpus}")
    print(
        f"  Target impact ({TARGET_IN_CORPUS_IMPACT:.0%}): {int(total_in_corpus * TARGET_IN_CORPUS_IMPACT)} annotations"
    )

    print("Loading GO hierarchy...")
    hierarchy = load_go_hierarchy()
    print(f"  Loaded hierarchy with {len(hierarchy.term_names)} terms")

    print("Selecting primary terms to hide (impact-based)...")
    primary_hidden_terms, selection_stats = select_hidden_terms(
        term_counts, in_corpus_terms, in_corpus_by_term, total_in_corpus, hierarchy
    )

    print("\nPrimary selection statistics:")
    for key, value in selection_stats.items():
        print(f"  {key}: {value}")

    print("\nExpanding to include all descendants...")
    hidden_terms_expanded, expansion_stats = expand_with_descendants(
        primary_hidden_terms, hierarchy
    )

    print("\nExpansion statistics:")
    for key, value in expansion_stats.items():
        print(f"  {key}: {value}")

    # Ground truth flags only mark the primary hidden terms (not descendants)
    # because descendants may not be in the ground truth at all
    print("\nAdding hidden flags to ground truth (primary terms only)...")
    genes_with_flags = add_hidden_flags(genes, primary_hidden_terms)

    annotation_stats = count_hidden_annotations(genes_with_flags)
    print("\nAnnotation statistics (primary terms):")
    for key, value in annotation_stats.items():
        print(f"  {key}: {value}")

    print(f"\nWriting to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        for gene in genes_with_flags:
            f.write(json.dumps(gene) + "\n")

    # Save the EXPANDED hidden terms list for the ontology server
    # This includes all descendants so agents can't find children of hidden terms
    hidden_terms_path = DATA_DIR / "hidden_go_terms.json"
    print(f"Writing expanded hidden terms list to {hidden_terms_path}...")
    with open(hidden_terms_path, "w") as f:
        json.dump(
            {
                "hidden_terms": sorted(hidden_terms_expanded),
                "primary_terms": sorted(primary_hidden_terms),
                "stats": {**selection_stats, **expansion_stats, **annotation_stats},
            },
            f,
            indent=2,
        )

    print("\nDone!")
    print(f"  Ground truth with flags: {OUTPUT_PATH}")
    print(f"  Hidden terms list: {hidden_terms_path}")
    print(f"  Primary terms: {len(primary_hidden_terms)}")
    print(f"  Total hidden (with descendants): {len(hidden_terms_expanded)}")


if __name__ == "__main__":
    main()
