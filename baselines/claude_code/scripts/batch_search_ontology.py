#!/usr/bin/env python3
"""Batch ontology search — resolve multiple queries in a single invocation.

Usage:
    # JSON array via stdin
    echo '[{"ontology":"go","query":"wing development","aspect":"P","limit":5}]' | python3 scripts/batch_search_ontology.py

    # JSON array from file
    python3 scripts/batch_search_ontology.py --input queries.json

Each query object supports:
    ontology (required): "go", "anatomy", or "stage"
    query    (required): search text
    aspect   (optional): "P", "F", or "C" (GO only)
    limit    (optional): max results per query (default 5)

Output: JSON object with a "results" array to stdout.
"""

import argparse
import json
import sys
from pathlib import Path

# Import search functions from sibling module
from search_ontology import search_ontology, get_children


def run_batch(queries: list[dict]) -> dict:
    results = []
    for q in queries:
        ontology = q.get("ontology")
        query = q.get("query", "")

        if not ontology or not query:
            results.append({"query": query, "ontology": ontology, "error": "missing 'ontology' or 'query'"})
            continue

        if ontology == "children":
            matches = get_children(query, limit=q.get("limit", 20))
            results.append({"query": query, "ontology": "children", "matches": matches})
        elif ontology in ("go", "anatomy", "stage"):
            matches = search_ontology(
                ontology,
                query,
                aspect=q.get("aspect"),
                limit=q.get("limit", 5),
            )
            results.append({"query": query, "ontology": ontology, "matches": matches})
        else:
            results.append({"query": query, "ontology": ontology, "error": f"unknown ontology '{ontology}'"})

    return {"results": results}


def main():
    parser = argparse.ArgumentParser(description="Batch ontology search")
    parser.add_argument("--input", type=Path, help="JSON file with query array (default: read stdin)")
    args = parser.parse_args()

    if args.input:
        with open(args.input) as f:
            queries = json.load(f)
    else:
        queries = json.load(sys.stdin)

    if not isinstance(queries, list):
        print(json.dumps({"error": "Input must be a JSON array of query objects"}))
        sys.exit(1)

    output = run_batch(queries)
    json.dump(output, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
