"""Core ontology search functionality for GO, FBbt (anatomy), and FBdv (developmental stage)."""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, KEYWORD, TEXT, Schema
from whoosh.qparser import MultifieldParser

# Directory for ontology indices
INDEX_DIR = Path(__file__).parent.parent.parent / "ontology_indices"
ONTOLOGY_DIR = Path(__file__).parent.parent.parent / "ontologies"
DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Lazy-loaded hidden terms
_hidden_go_terms: set[str] | None = None


def load_hidden_go_terms() -> set[str]:
    """Load hidden GO terms from JSON file.

    Returns empty set if HIDE_GO_TERMS env var is not set or file doesn't exist.
    """
    global _hidden_go_terms
    if _hidden_go_terms is not None:
        return _hidden_go_terms

    # Check environment variable
    if not os.environ.get("HIDE_GO_TERMS"):
        _hidden_go_terms = set()
        return _hidden_go_terms

    hidden_terms_path = DATA_DIR / "hidden_go_terms.json"
    if not hidden_terms_path.exists():
        _hidden_go_terms = set()
        return _hidden_go_terms

    with open(hidden_terms_path) as f:
        data = json.load(f)
        _hidden_go_terms = set(data.get("hidden_terms", []))

    return _hidden_go_terms


def is_hidden_go_term(term_id: str) -> bool:
    """Check if a GO term is hidden.

    Args:
        term_id: GO term ID (e.g., "GO:0001234")

    Returns:
        True if term is hidden, False otherwise
    """
    return term_id in load_hidden_go_terms()


@dataclass
class OntologyTerm:
    """Represents an ontology term."""

    term_id: str
    name: str
    namespace: str | None = None
    definition: str | None = None
    synonyms: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)  # is_a relationships


def parse_obo_file(filepath: Path) -> list[OntologyTerm]:
    """Parse an OBO file and extract terms.

    Args:
        filepath: Path to the OBO file

    Returns:
        List of OntologyTerm objects
    """
    terms: list[OntologyTerm] = []
    current_term: dict[str, Any] | None = None

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "[Term]":
                if current_term and current_term.get("id"):
                    terms.append(
                        OntologyTerm(
                            term_id=current_term["id"],
                            name=current_term.get("name", ""),
                            namespace=current_term.get("namespace"),
                            definition=current_term.get("def"),
                            synonyms=current_term.get("synonyms", []),
                            parents=current_term.get("parents", []),
                        )
                    )
                current_term = {"synonyms": [], "parents": []}

            elif line.startswith("[") and line.endswith("]"):
                # End of terms section (e.g., [Typedef])
                if current_term and current_term.get("id"):
                    terms.append(
                        OntologyTerm(
                            term_id=current_term["id"],
                            name=current_term.get("name", ""),
                            namespace=current_term.get("namespace"),
                            definition=current_term.get("def"),
                            synonyms=current_term.get("synonyms", []),
                            parents=current_term.get("parents", []),
                        )
                    )
                current_term = None

            elif current_term is not None:
                if line.startswith("id:"):
                    current_term["id"] = line[3:].strip()
                elif line.startswith("name:"):
                    current_term["name"] = line[5:].strip()
                elif line.startswith("namespace:"):
                    current_term["namespace"] = line[10:].strip()
                elif line.startswith("def:"):
                    # Extract definition text from quoted string
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        current_term["def"] = match.group(1)
                elif line.startswith("synonym:"):
                    # Extract synonym text from quoted string
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        current_term["synonyms"].append(match.group(1))
                elif line.startswith("is_a:"):
                    # Extract parent ID (format: "is_a: GO:0000001 ! name")
                    parent_id = line[5:].strip().split(" ")[0]
                    current_term["parents"].append(parent_id)

        # Handle last term if file doesn't end with a new stanza
        if current_term and current_term.get("id"):
            terms.append(
                OntologyTerm(
                    term_id=current_term["id"],
                    name=current_term.get("name", ""),
                    namespace=current_term.get("namespace"),
                    definition=current_term.get("def"),
                    synonyms=current_term.get("synonyms", []),
                    parents=current_term.get("parents", []),
                )
            )

    return terms


def create_ontology_schema() -> Schema:
    """Create Whoosh schema for ontology indexing."""
    analyzer = StemmingAnalyzer()
    return Schema(
        term_id=ID(stored=True, unique=True),
        name=TEXT(stored=True, analyzer=analyzer),
        namespace=KEYWORD(stored=True),
        definition=TEXT(stored=True, analyzer=analyzer),
        synonyms=TEXT(stored=True, analyzer=analyzer),
        synonyms_list=KEYWORD(stored=True),  # Store original synonyms for display
        parents=KEYWORD(stored=True),  # Store parent IDs
    )


def build_index(terms: list[OntologyTerm], index_name: str) -> index.Index:
    """Build or open a Whoosh index for ontology terms.

    Args:
        terms: List of OntologyTerm objects
        index_name: Name for the index directory

    Returns:
        Whoosh Index object
    """
    index_path = INDEX_DIR / index_name
    index_path.mkdir(parents=True, exist_ok=True)

    schema = create_ontology_schema()

    # Always rebuild the index if terms are provided
    ix = index.create_in(str(index_path), schema)

    writer = ix.writer()
    for term in terms:
        writer.add_document(
            term_id=term.term_id,
            name=term.name or "",
            namespace=term.namespace or "",
            definition=term.definition or "",
            synonyms=" ".join(term.synonyms) if term.synonyms else "",
            synonyms_list="|".join(term.synonyms) if term.synonyms else "",
            parents="|".join(term.parents) if term.parents else "",
        )
    writer.commit()

    return ix


def load_or_build_index(obo_filename: str, index_name: str) -> index.Index:
    """Load existing index or build from OBO file.

    Args:
        obo_filename: Name of the OBO file in ontologies directory
        index_name: Name for the index

    Returns:
        Whoosh Index object
    """
    index_path = INDEX_DIR / index_name

    # Use existing index if available
    if index_path.exists() and index.exists_in(str(index_path)):
        return index.open_dir(str(index_path))

    # Build new index
    obo_path = ONTOLOGY_DIR / obo_filename
    if not obo_path.exists():
        raise FileNotFoundError(
            f"Ontology file not found: {obo_path}. "
            f"Please download it first (see ontologies/README.md)."
        )

    terms = parse_obo_file(obo_path)
    return build_index(terms, index_name)


class OntologyHierarchy:
    """Tracks parent/child relationships and term names for an ontology."""

    def __init__(self, terms: list[OntologyTerm]):
        self.term_names: dict[str, str] = {}  # term_id -> name
        self.children: dict[str, list[str]] = {}  # parent_id -> [child_ids]
        self.parents: dict[str, list[str]] = {}  # term_id -> [parent_ids]

        for term in terms:
            self.term_names[term.term_id] = term.name
            self.parents[term.term_id] = term.parents

            for parent_id in term.parents:
                if parent_id not in self.children:
                    self.children[parent_id] = []
                self.children[parent_id].append(term.term_id)

    def get_children_count(self, term_id: str) -> int:
        """Get number of direct children for a term."""
        return len(self.children.get(term_id, []))

    def get_parent_info(self, term_id: str) -> list[dict[str, str]]:
        """Get parent term IDs with their names."""
        parent_ids = self.parents.get(term_id, [])
        return [{"id": pid, "name": self.term_names.get(pid, "")} for pid in parent_ids]


# Lazy-loaded indices and hierarchies
_go_index: index.Index | None = None
_fbbt_index: index.Index | None = None
_fbdv_index: index.Index | None = None
_go_hierarchy: OntologyHierarchy | None = None
_fbbt_hierarchy: OntologyHierarchy | None = None
_fbdv_hierarchy: OntologyHierarchy | None = None


def get_go_index() -> index.Index:
    """Get or create the GO ontology index."""
    global _go_index
    if _go_index is None:
        _go_index = load_or_build_index("go-basic.obo", "go_index")
    return _go_index


def get_fbbt_index() -> index.Index:
    """Get or create the FBbt (anatomy) ontology index."""
    global _fbbt_index
    if _fbbt_index is None:
        _fbbt_index = load_or_build_index("fly_anatomy.obo", "fbbt_index")
    return _fbbt_index


def get_fbdv_index() -> index.Index:
    """Get or create the FBdv (developmental stage) ontology index."""
    global _fbdv_index
    if _fbdv_index is None:
        _fbdv_index = load_or_build_index("fly_development.obo", "fbdv_index")
    return _fbdv_index


def _load_hierarchy(obo_filename: str) -> OntologyHierarchy:
    """Load ontology hierarchy from OBO file."""
    obo_path = ONTOLOGY_DIR / obo_filename
    if not obo_path.exists():
        raise FileNotFoundError(f"Ontology file not found: {obo_path}")
    terms = parse_obo_file(obo_path)
    return OntologyHierarchy(terms)


def get_go_hierarchy() -> OntologyHierarchy:
    """Get or create the GO ontology hierarchy."""
    global _go_hierarchy
    if _go_hierarchy is None:
        _go_hierarchy = _load_hierarchy("go-basic.obo")
    return _go_hierarchy


def get_fbbt_hierarchy() -> OntologyHierarchy:
    """Get or create the FBbt (anatomy) ontology hierarchy."""
    global _fbbt_hierarchy
    if _fbbt_hierarchy is None:
        _fbbt_hierarchy = _load_hierarchy("fly_anatomy.obo")
    return _fbbt_hierarchy


def get_fbdv_hierarchy() -> OntologyHierarchy:
    """Get or create the FBdv (developmental stage) ontology hierarchy."""
    global _fbdv_hierarchy
    if _fbdv_hierarchy is None:
        _fbdv_hierarchy = _load_hierarchy("fly_development.obo")
    return _fbdv_hierarchy


def search_ontology(
    ix: index.Index,
    query: str,
    hierarchy: OntologyHierarchy | None = None,
    aspect: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search an ontology index.

    Args:
        ix: Whoosh index
        query: Search query
        hierarchy: Optional hierarchy for parent/children info
        aspect: Optional namespace filter (e.g., "P", "F", "C" for GO)
        limit: Maximum results to return

    Returns:
        List of matching terms as dictionaries with enriched metadata
    """
    with ix.searcher() as searcher:
        parser = MultifieldParser(
            ["name", "definition", "synonyms"],
            schema=ix.schema,  # type: ignore[attr-defined]
        )
        parsed_query = parser.parse(query)

        results = searcher.search(parsed_query, limit=limit * 2)  # Get extra for filtering

        matches = []
        for hit in results:
            # Apply namespace filter if provided
            if aspect:
                namespace = hit.get("namespace", "")
                # Map GO namespaces to aspect codes
                aspect_map = {
                    "P": "biological_process",
                    "F": "molecular_function",
                    "C": "cellular_component",
                }
                expected_ns = aspect_map.get(aspect, aspect)
                if expected_ns not in namespace:
                    continue

            term_id = hit["term_id"]
            result: dict[str, Any] = {
                "term_id": term_id,
                "name": hit["name"],
                "namespace": hit.get("namespace"),
                "definition": hit.get("definition"),
            }

            # Add synonyms (first 5 to avoid clutter)
            synonyms_str = hit.get("synonyms_list", "")
            if synonyms_str:
                synonyms = [s for s in synonyms_str.split("|") if s][:5]
                if synonyms:
                    result["synonyms"] = synonyms

            # Add hierarchy info if available
            if hierarchy:
                parents = hierarchy.get_parent_info(term_id)
                if parents:
                    # Format as "ID (name)" for readability
                    result["parents"] = [
                        f"{p['id']} ({p['name']})" if p["name"] else p["id"]
                        for p in parents[:2]  # Limit to 2 parents
                    ]
                children_count = hierarchy.get_children_count(term_id)
                if children_count > 0:
                    result["children_count"] = children_count

            matches.append(result)

            if len(matches) >= limit:
                break

        return matches


def search_go_core(query: str, aspect: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    """Search Gene Ontology for matching terms.

    This is the framework-agnostic core function.

    Args:
        query: Search term (e.g., "DNA binding", "transcription factor")
        aspect: Optional filter - "P" (Biological Process), "F" (Molecular Function),
                or "C" (Cellular Component)
        limit: Maximum number of results (default 10)

    Returns:
        List of matching GO terms with go_id, name, namespace, definition,
        synonyms, parents, and children_count

    Note:
        If HIDE_GO_TERMS environment variable is set, hidden terms will be
        filtered from results (for specificity gap benchmark).
    """
    ix = get_go_index()
    hierarchy = get_go_hierarchy()
    hidden_terms = load_hidden_go_terms()

    # Request more results if we need to filter hidden terms
    search_limit = limit * 2 if hidden_terms else limit
    results = search_ontology(ix, query, hierarchy=hierarchy, aspect=aspect, limit=search_limit)

    # Rename term_id to go_id for consistency
    for r in results:
        r["go_id"] = r.pop("term_id")

    # Filter out hidden terms if enabled
    if hidden_terms:
        results = [r for r in results if r["go_id"] not in hidden_terms]

    return results[:limit]


def search_anatomy_core(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search FBbt ontology for Drosophila anatomy terms.

    This is the framework-agnostic core function.

    Args:
        query: Search term (e.g., "wing disc", "neuron", "muscle")
        limit: Maximum number of results (default 10)

    Returns:
        List of matching anatomy terms with fbbt_id, name, definition,
        synonyms, parents, and children_count
    """
    ix = get_fbbt_index()
    hierarchy = get_fbbt_hierarchy()
    results = search_ontology(ix, query, hierarchy=hierarchy, limit=limit)
    # Rename term_id to fbbt_id for consistency
    for r in results:
        r["fbbt_id"] = r.pop("term_id")
        r.pop("namespace", None)  # Not relevant for display
    return results


def search_stage_core(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search FBdv ontology for Drosophila developmental stage terms.

    This is the framework-agnostic core function.

    Args:
        query: Search term (e.g., "embryo", "larval stage", "pupal")
        limit: Maximum number of results (default 10)

    Returns:
        List of matching stage terms with fbdv_id, name, definition,
        synonyms, parents, and children_count
    """
    ix = get_fbdv_index()
    hierarchy = get_fbdv_hierarchy()
    results = search_ontology(ix, query, hierarchy=hierarchy, limit=limit)
    # Rename term_id to fbdv_id for consistency
    for r in results:
        r["fbdv_id"] = r.pop("term_id")
        r.pop("namespace", None)  # Not relevant for display
    return results


def get_term_children_core(term_id: str, limit: int = 20) -> dict[str, Any]:
    """Get direct children of any ontology term.

    Works with GO, FBbt (anatomy), and FBdv (stage) terms.
    Detects ontology from term ID prefix.

    Args:
        term_id: Term ID (GO:XXXXXXX, FBbt:XXXXXXXX, or FBdv:XXXXXXXX)
        limit: Maximum number of children to return (default 20)

    Returns:
        Dictionary with:
        - term_id: The queried term ID
        - term_name: Name of the queried term
        - children: List of child terms, each with id, name, children_count
        - error: Error message if term not found or invalid prefix

    Note:
        If HIDE_GO_TERMS environment variable is set, hidden GO terms will be
        filtered from children (for specificity gap benchmark).
    """
    # Detect ontology from prefix
    if term_id.startswith("GO:"):
        hierarchy = get_go_hierarchy()
        id_key = "go_id"
        hidden_terms = load_hidden_go_terms()
    elif term_id.startswith("FBbt:"):
        hierarchy = get_fbbt_hierarchy()
        id_key = "fbbt_id"
        hidden_terms = set()  # No hidden terms for anatomy
    elif term_id.startswith("FBdv:"):
        hierarchy = get_fbdv_hierarchy()
        id_key = "fbdv_id"
        hidden_terms = set()  # No hidden terms for stages
    else:
        return {
            "term_id": term_id,
            "error": "Unknown ontology prefix. Expected GO:, FBbt:, or FBdv:",
        }

    # Check if term exists (and is not hidden)
    if term_id in hidden_terms:
        return {"term_id": term_id, "error": "Term not found in ontology"}

    term_name = hierarchy.term_names.get(term_id)
    if term_name is None:
        return {"term_id": term_id, "error": "Term not found in ontology"}

    # Get children, filtering out hidden terms
    all_child_ids = hierarchy.children.get(term_id, [])
    if hidden_terms:
        child_ids = [c for c in all_child_ids if c not in hidden_terms]
    else:
        child_ids = all_child_ids

    children = []
    for child_id in child_ids[:limit]:
        child_name = hierarchy.term_names.get(child_id, "")
        # Count only non-hidden children
        if hidden_terms:
            child_children = hierarchy.children.get(child_id, [])
            child_children_count = len([c for c in child_children if c not in hidden_terms])
        else:
            child_children_count = hierarchy.get_children_count(child_id)
        child_entry: dict[str, Any] = {
            id_key: child_id,
            "name": child_name,
        }
        if child_children_count > 0:
            child_entry["children_count"] = child_children_count
        children.append(child_entry)

    return {
        "term_id": term_id,
        "term_name": term_name,
        "children_count": len(child_ids),
        "children": children,
        "truncated": len(child_ids) > limit,
    }
