"""Core paper reading functionality for fetching full text from the corpus."""

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

# Cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / "corpus_cache"


class PaperCache:
    """Cache for paper full texts from the HuggingFace corpus."""

    def __init__(self):
        self.papers: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def load(self):
        """Load papers from HuggingFace dataset or cache."""
        if self._loaded:
            return

        cache_path = CACHE_DIR / "papers_full.json"

        # Try to load from cache
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    self.papers = json.load(f)
                self._loaded = True
                print(f"Loaded {len(self.papers)} papers from cache")
                return
            except Exception as e:
                print(f"Cache load failed: {e}, loading from HuggingFace...")

        # Load from HuggingFace
        print("Loading full corpus from HuggingFace...")
        dataset = load_dataset("jimmyzxj/drosophila-literature-corpus", split="train")

        for row in dataset:
            paper: dict[str, Any] = dict(row)  # type: ignore[arg-type]
            pmcid = str(paper["pmcid"])
            self.papers[pmcid] = {
                "pmcid": pmcid,
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "sections": paper.get("sections", {}),
            }

        # Cache for faster subsequent loads
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(self.papers, f)

        self._loaded = True
        print(f"Loaded and cached {len(self.papers)} papers")

    def get(self, pmcid: str) -> dict[str, Any] | None:
        """Get a paper by PMCID."""
        if not self._loaded:
            self.load()
        return self.papers.get(pmcid)

    def exists(self, pmcid: str) -> bool:
        """Check if a paper exists in the corpus."""
        if not self._loaded:
            self.load()
        return pmcid in self.papers


# Global cache instance (lazy loaded)
_paper_cache: PaperCache | None = None


def get_paper_cache() -> PaperCache:
    """Get or create the paper cache."""
    global _paper_cache
    if _paper_cache is None:
        _paper_cache = PaperCache()
        _paper_cache.load()
    return _paper_cache


def get_paper_text_core(pmcid: str, sections: list[str] | None = None) -> dict[str, Any]:
    """Retrieve the full text of a specific paper from the corpus.

    This is the framework-agnostic core function.

    Args:
        pmcid: PubMed Central ID (e.g., "PMC1234567")
        sections: Optional list of sections to retrieve. If not specified,
                  returns all sections.

    Returns:
        Dictionary with paper content (pmcid, title, abstract, sections)
        or error dict if paper not found
    """
    cache = get_paper_cache()
    paper = cache.get(pmcid)

    if paper is None:
        return {"error": f"Paper {pmcid} not found in corpus"}

    result: dict[str, Any] = {
        "pmcid": paper["pmcid"],
        "title": paper["title"],
    }

    # Handle section filtering
    if sections is None:
        result["abstract"] = paper["abstract"]
        result["sections"] = paper["sections"]
    else:
        # Normalize section names
        sections_upper = [s.upper() for s in sections]

        if "ABSTRACT" in sections_upper:
            result["abstract"] = paper["abstract"]
        else:
            result["abstract"] = None

        # Filter sections
        filtered_sections = {}
        paper_sections = paper.get("sections", {})
        for sec_name in sections_upper:
            if sec_name != "ABSTRACT" and sec_name in paper_sections:
                filtered_sections[sec_name] = paper_sections[sec_name]

        result["sections"] = filtered_sections

    return result
