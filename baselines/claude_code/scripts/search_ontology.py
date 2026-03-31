#!/usr/bin/env python3
"""Search GO, FBbt (anatomy), and FBdv (developmental stage) ontologies.

Usage:
    python3 scripts/search_ontology.py go "wing development" [--aspect P] [--limit 10]
    python3 scripts/search_ontology.py anatomy "wing disc" [--limit 10]
    python3 scripts/search_ontology.py stage "embryonic" [--limit 10]
    python3 scripts/search_ontology.py children GO:0003700 [--limit 20]

Output: JSON array to stdout with term_id, name, definition, parents, children_count.

Requires: whoosh
Data:     /data/ontology_indices/ (pre-built Whoosh indices)
          /data/ontologies/ (OBO files, for hierarchy lookup)
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from whoosh import index
from whoosh.qparser import MultifieldParser

INDEX_DIR = Path("/data/ontology_indices")
ONTOLOGY_DIR = Path("/data/ontologies")


# --- OBO parsing (for hierarchy) ---

@dataclass
class OntologyTerm:
    term_id: str
    name: str
    namespace: str | None = None
    definition: str | None = None
    synonyms: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)


def parse_obo_file(filepath: Path) -> list[OntologyTerm]:
    terms = []
    current = None

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                if current and current.get("id"):
                    terms.append(OntologyTerm(
                        term_id=current["id"],
                        name=current.get("name", ""),
                        namespace=current.get("namespace"),
                        definition=current.get("def"),
                        synonyms=current.get("synonyms", []),
                        parents=current.get("parents", []),
                    ))
                current = {"synonyms": [], "parents": []}
            elif line.startswith("[") and line.endswith("]"):
                if current and current.get("id"):
                    terms.append(OntologyTerm(
                        term_id=current["id"],
                        name=current.get("name", ""),
                        namespace=current.get("namespace"),
                        definition=current.get("def"),
                        synonyms=current.get("synonyms", []),
                        parents=current.get("parents", []),
                    ))
                current = None
            elif current is not None:
                if line.startswith("id:"):
                    current["id"] = line[3:].strip()
                elif line.startswith("name:"):
                    current["name"] = line[5:].strip()
                elif line.startswith("namespace:"):
                    current["namespace"] = line[10:].strip()
                elif line.startswith("def:"):
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        current["def"] = match.group(1)
                elif line.startswith("synonym:"):
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        current["synonyms"].append(match.group(1))
                elif line.startswith("is_a:"):
                    parent_id = line[5:].strip().split(" ")[0]
                    current["parents"].append(parent_id)

        if current and current.get("id"):
            terms.append(OntologyTerm(
                term_id=current["id"],
                name=current.get("name", ""),
                namespace=current.get("namespace"),
                definition=current.get("def"),
                synonyms=current.get("synonyms", []),
                parents=current.get("parents", []),
            ))

    return terms


class OntologyHierarchy:
    def __init__(self, terms: list[OntologyTerm]):
        self.term_names: dict[str, str] = {}
        self.children: dict[str, list[str]] = {}
        self.parents: dict[str, list[str]] = {}
        for term in terms:
            self.term_names[term.term_id] = term.name
            self.parents[term.term_id] = term.parents
            for parent_id in term.parents:
                if parent_id not in self.children:
                    self.children[parent_id] = []
                self.children[parent_id].append(term.term_id)


# --- Hierarchy cache ---

_hierarchies: dict[str, OntologyHierarchy] = {}

OBO_FILES = {
    "go": "go-basic.obo",
    "anatomy": "fly_anatomy.obo",
    "stage": "fly_development.obo",
}

INDEX_NAMES = {
    "go": "go_index",
    "anatomy": "fbbt_index",
    "stage": "fbdv_index",
}


def get_hierarchy(ontology: str) -> OntologyHierarchy:
    if ontology not in _hierarchies:
        obo_path = ONTOLOGY_DIR / OBO_FILES[ontology]
        if not obo_path.exists():
            print(f"Error: OBO file not found at {obo_path}", file=sys.stderr)
            sys.exit(1)
        terms = parse_obo_file(obo_path)
        _hierarchies[ontology] = OntologyHierarchy(terms)
    return _hierarchies[ontology]


# --- Search ---

def search_ontology(ontology: str, query: str, aspect: str | None = None, limit: int = 10) -> list[dict]:
    index_name = INDEX_NAMES[ontology]
    index_path = INDEX_DIR / index_name

    if not index_path.exists():
        print(f"Error: Index not found at {index_path}", file=sys.stderr)
        sys.exit(1)

    ix = index.open_dir(str(index_path))
    hierarchy = get_hierarchy(ontology)

    with ix.searcher() as searcher:
        parser = MultifieldParser(["name", "definition", "synonyms"], schema=ix.schema)
        parsed_query = parser.parse(query)
        results = searcher.search(parsed_query, limit=limit * 2)

        matches = []
        for hit in results:
            if aspect:
                namespace = hit.get("namespace", "")
                aspect_map = {
                    "P": "biological_process",
                    "F": "molecular_function",
                    "C": "cellular_component",
                }
                expected_ns = aspect_map.get(aspect, aspect)
                if expected_ns not in namespace:
                    continue

            term_id = hit["term_id"]
            result = {
                "term_id": term_id,
                "name": hit["name"],
            }

            if hit.get("namespace"):
                result["namespace"] = hit["namespace"]

            if hit.get("definition"):
                result["definition"] = hit["definition"]

            synonyms_str = hit.get("synonyms_list", "")
            if synonyms_str:
                synonyms = [s for s in synonyms_str.split("|") if s][:5]
                if synonyms:
                    result["synonyms"] = synonyms

            parent_ids = hierarchy.parents.get(term_id, [])
            if parent_ids:
                result["parents"] = [
                    f"{pid} ({hierarchy.term_names.get(pid, '')})"
                    for pid in parent_ids[:3]
                ]
            children_count = len(hierarchy.children.get(term_id, []))
            if children_count > 0:
                result["children_count"] = children_count

            matches.append(result)
            if len(matches) >= limit:
                break

    return matches


def get_children(term_id: str, limit: int = 20) -> dict:
    if term_id.startswith("GO:"):
        ontology = "go"
    elif term_id.startswith("FBbt:"):
        ontology = "anatomy"
    elif term_id.startswith("FBdv:"):
        ontology = "stage"
    else:
        return {"error": "Unknown ontology prefix. Expected GO:, FBbt:, or FBdv:"}

    hierarchy = get_hierarchy(ontology)
    term_name = hierarchy.term_names.get(term_id)
    if term_name is None:
        return {"term_id": term_id, "error": "Term not found in ontology"}

    child_ids = hierarchy.children.get(term_id, [])
    children = []
    for child_id in child_ids[:limit]:
        child_name = hierarchy.term_names.get(child_id, "")
        child_children_count = len(hierarchy.children.get(child_id, []))
        entry = {"term_id": child_id, "name": child_name}
        if child_children_count > 0:
            entry["children_count"] = child_children_count
        children.append(entry)

    return {
        "term_id": term_id,
        "term_name": term_name,
        "children_count": len(child_ids),
        "children": children,
        "truncated": len(child_ids) > limit,
    }


def main():
    parser = argparse.ArgumentParser(description="Search ontologies or get term children")
    parser.add_argument(
        "command",
        choices=["go", "anatomy", "stage", "children"],
        help="Ontology to search (go/anatomy/stage) or 'children' to get child terms",
    )
    parser.add_argument("query", help="Search query or term ID (for children)")
    parser.add_argument("--aspect", choices=["P", "F", "C"], help="GO aspect filter")
    parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    args = parser.parse_args()

    if args.command == "children":
        result = get_children(args.query, limit=args.limit)
        json.dump(result, sys.stdout, indent=2)
    else:
        results = search_ontology(args.command, args.query, aspect=args.aspect, limit=args.limit)
        json.dump(results, sys.stdout, indent=2)

    print()


if __name__ == "__main__":
    main()
