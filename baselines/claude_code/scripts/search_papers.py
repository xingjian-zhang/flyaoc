#!/usr/bin/env python3
"""BM25 corpus search over the Drosophila literature corpus.

Usage:
    python3 scripts/search_papers.py "query" [--limit 20]

Output: JSON array to stdout with pmcid, title, abstract snippet, score.

Requires: rank_bm25
Data:     /data/corpus_cache/bm25_index.pkl, /data/corpus_cache/corpus_metadata.json
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

DATA_DIR = Path("/data/corpus_cache")


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def check_query_in_text(text: str, query: str) -> bool:
    text_lower = text.lower()
    query_lower = query.lower()
    if query_lower in text_lower:
        return True
    query_words = query_lower.split()
    return all(word in text_lower for word in query_words)


def search(query: str, limit: int = 20) -> list[dict]:
    index_path = DATA_DIR / "bm25_index.pkl"
    corpus_path = DATA_DIR / "corpus_metadata.json"

    if not index_path.exists():
        print(f"Error: BM25 index not found at {index_path}", file=sys.stderr)
        sys.exit(1)
    if not corpus_path.exists():
        print(f"Error: Corpus metadata not found at {corpus_path}", file=sys.stderr)
        sys.exit(1)

    with open(index_path, "rb") as f:
        bm25 = pickle.load(f)
    with open(corpus_path) as f:
        corpus = json.load(f)

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            paper = corpus[idx]
            title = paper["title"]
            abstract = paper["abstract"]
            results.append({
                "pmcid": paper["pmcid"],
                "title": title,
                "abstract": abstract[:500] if len(abstract) > 500 else abstract,
                "relevance_score": round(float(scores[idx]), 2),
                "gene_in_title": check_query_in_text(title, query),
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Search the Drosophila literature corpus")
    parser.add_argument("query", help="Search query (gene name, keyword, or phrase)")
    parser.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")
    args = parser.parse_args()

    results = search(args.query, args.limit)
    json.dump(results, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
