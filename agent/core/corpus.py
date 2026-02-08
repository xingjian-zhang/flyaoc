"""Core corpus search functionality using BM25 over the HuggingFace literature corpus."""

import json
import pickle
from pathlib import Path
from typing import Any

from datasets import load_dataset
from rank_bm25 import BM25Okapi

# Cache directory for preprocessed data
CACHE_DIR = Path(__file__).parent.parent.parent / "corpus_cache"


class CorpusIndex:
    """BM25 index over the Drosophila literature corpus."""

    def __init__(self):
        self.corpus: list[dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None
        self.tokenized_docs: list[list[str]] | None = None
        self._loaded = False

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()

    def _get_searchable_text(self, paper: dict) -> str:
        """Extract searchable text from a paper."""
        parts = [paper.get("title", ""), paper.get("abstract", "")]

        # Include section text if available
        sections = paper.get("sections", {})
        if sections:
            for section_name in ["INTRO", "RESULTS", "DISCUSS", "CONCL"]:
                section_content = sections.get(section_name)
                if section_content:
                    if isinstance(section_content, list):
                        parts.extend(section_content)
                    else:
                        parts.append(section_content)

        return " ".join(filter(None, parts))

    def load(self, force_rebuild: bool = False):
        """Load or build the corpus index.

        Args:
            force_rebuild: If True, rebuild the index even if cached
        """
        if self._loaded and not force_rebuild:
            return

        cache_path = CACHE_DIR / "bm25_index.pkl"
        corpus_path = CACHE_DIR / "corpus_metadata.json"

        # Try to load from cache
        if not force_rebuild and cache_path.exists() and corpus_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    self.bm25 = pickle.load(f)
                with open(corpus_path) as f:
                    self.corpus = json.load(f)
                self._loaded = True
                print(f"Loaded cached index with {len(self.corpus)} papers")
                return
            except Exception as e:
                print(f"Cache load failed: {e}, rebuilding...")

        # Load from HuggingFace
        print("Loading corpus from HuggingFace...")
        dataset = load_dataset("jimmyzxj/drosophila-literature-corpus", split="train")

        # Extract metadata and build index
        self.corpus = []
        texts_for_index: list[str] = []

        for row in dataset:
            paper: dict[str, Any] = dict(row)  # type: ignore[arg-type]
            # Store metadata
            self.corpus.append(
                {
                    "pmcid": paper["pmcid"],
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                }
            )

            # Build searchable text
            searchable_text = self._get_searchable_text(paper)
            texts_for_index.append(searchable_text)

        # Tokenize and build BM25 index
        print("Building BM25 index...")
        self.tokenized_docs = [self._tokenize(text) for text in texts_for_index]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        # Cache the results
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(corpus_path, "w") as f:
            json.dump(self.corpus, f)

        self._loaded = True
        print(f"Built index with {len(self.corpus)} papers")

    def _check_query_in_text(self, text: str, query: str) -> bool:
        """Check if query (as phrase or individual words) appears in text."""
        text_lower = text.lower()
        query_lower = query.lower()
        # Check for exact phrase match first
        if query_lower in text_lower:
            return True
        # Check for all query words present
        query_words = query_lower.split()
        return all(word in text_lower for word in query_words)

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search the corpus for papers matching a query.

        Args:
            query: Search query (gene name, keyword, or phrase)
            limit: Maximum results to return

        Returns:
            List of papers with enriched metadata:
            - pmcid, title, abstract, relevance_score (basic)
            - gene_in_title: bool - whether query appears in title
        """
        if not self._loaded:
            self.load()

        if self.bm25 is None:
            return []

        # Get BM25 scores
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include papers with positive scores
                paper = self.corpus[idx]
                title = paper["title"]
                abstract = paper["abstract"]

                result: dict[str, Any] = {
                    "pmcid": paper["pmcid"],
                    "title": title,
                    "abstract": abstract[:500] if len(abstract) > 500 else abstract,
                    "relevance_score": round(float(scores[idx]), 2),
                    "gene_in_title": self._check_query_in_text(title, query),
                }
                results.append(result)

        return results


# Global index instance (lazy loaded)
_corpus_index: CorpusIndex | None = None


def get_corpus_index() -> CorpusIndex:
    """Get or create the corpus index."""
    global _corpus_index
    if _corpus_index is None:
        _corpus_index = CorpusIndex()
        _corpus_index.load()
    return _corpus_index


def search_corpus_core(query: str, limit: int = 20) -> list[dict]:
    """Search the Drosophila literature corpus for papers matching a query.

    This is the framework-agnostic core function.

    Args:
        query: Search query - can be a gene symbol, keyword, or phrase
        limit: Maximum number of results to return (default 20)

    Returns:
        List of matching papers with:
        - pmcid: PubMed Central ID
        - title: Paper title
        - abstract: Paper abstract (truncated to 500 chars)
        - relevance_score: BM25 relevance score
        - gene_in_title: True if query appears in title (high relevance signal)
    """
    index = get_corpus_index()
    return index.search(query, limit=limit)
