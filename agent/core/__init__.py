"""Framework-agnostic core functions for gene annotation tools.

This module provides the core logic that can be wrapped by different frameworks
(LangChain, MCP servers, etc.) without any framework-specific dependencies.
"""

from .corpus import CorpusIndex, get_corpus_index, search_corpus_core
from .metadata import capture_metadata, format_version_string, get_git_info
from .ontology import (
    OntologyTerm,
    get_fbbt_index,
    get_fbdv_index,
    get_go_index,
    is_hidden_go_term,
    load_hidden_go_terms,
    search_anatomy_core,
    search_go_core,
    search_stage_core,
)
from .papers import PaperCache, get_paper_cache, get_paper_text_core
from .prompt_templates import (
    ANNOTATION_AVOID_LIST,
    EVIDENCE_PATTERNS,
    GO_QUALIFIERS,
    ID_FORMATS,
    OUTPUT_SCHEMA_JSON,
    format_qualifier_docs,
)
from .validation import ValidationError, submit_annotations_core, validate_output

__all__ = [
    # Metadata
    "capture_metadata",
    "format_version_string",
    "get_git_info",
    # Corpus
    "CorpusIndex",
    "get_corpus_index",
    "search_corpus_core",
    # Papers
    "PaperCache",
    "get_paper_cache",
    "get_paper_text_core",
    # Ontology
    "OntologyTerm",
    "get_go_index",
    "get_fbbt_index",
    "get_fbdv_index",
    "is_hidden_go_term",
    "load_hidden_go_terms",
    "search_go_core",
    "search_anatomy_core",
    "search_stage_core",
    # Validation
    "ValidationError",
    "validate_output",
    "submit_annotations_core",
    # Prompt templates
    "GO_QUALIFIERS",
    "OUTPUT_SCHEMA_JSON",
    "EVIDENCE_PATTERNS",
    "ANNOTATION_AVOID_LIST",
    "ID_FORMATS",
    "format_qualifier_docs",
]
