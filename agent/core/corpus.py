"""Core corpus search functionality using BM25 over the HuggingFace literature corpus."""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from rank_bm25 import BM25Okapi

from __future__ import annotations
from typing import Dict, Any, Optional

# Cache directory for preprocessed data
CACHE_DIR = Path(__file__).parent.parent.parent / "corpus_cache"
CHUNKS_DIR = Path(__file__).parent / "drosophila_bm25_chunks"
MEDCPT_DIR = Path(__file__).parent / "drosophila_medcpt"


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

def search_corpus_core(
    query: str,
    limit: int = 20,
    method: str = "bm25_then_medcpt",
    *,
    # bm25_then_medcpt knobs
    bm25_first_n: int = 15,
    medcpt_after_n: int = 5,
    # how many results to retrieve from each search system before fusion
    wrrf_k_each: int = 20,
    # how many final documents are returned after fusion
    wrrf_total: int = 20,
    # smoothing constant in the RRF formula
    wrrf_k: int = 60,
    # weight applied to BM25 contributions
    wrrf_w_lex: float = 1.0,
    # weight applied to MedCPT contributions
    wrrf_w_sem: float = 1.0,
) -> list[dict]:
    """
    Search the Drosophila literature corpus for papers matching a query.

    Supported methods:
      - "wrrf":
          Retrieve from BM25 and MedCPT, then fuse by weighted reciprocal rank fusion.
      - "bm25_then_medcpt":
          Return exactly:
            * positions 1-15 from BM25
            * positions 16-20 from MedCPT
          skipping duplicate PMCIDs already selected from BM25.

    Returns list of dicts with at least:
      - pmcid, title, abstract, relevance_score, gene_in_title
    """
    method = method.lower().strip()

    def _pmcid(r: dict) -> str:
        return str(r.get("pmcid", "")).strip()

    def bm25_then_medcpt_results(
        bm25_list: list[dict],
        med_list: list[dict],
        *,
        bm25_n: int,
        med_n: int,
    ) -> list[dict]:
        """
        Take top bm25_n BM25 results as ranks 1..bm25_n, then append the next
        med_n unique MedCPT results as ranks bm25_n+1 .. bm25_n+med_n.
        """
        out: list[dict] = []
        seen: set[str] = set()

        # Positions 1..bm25_n from BM25
        for r in bm25_list:
            pm = _pmcid(r)
            if pm and pm not in seen:
                out.append(r)
                seen.add(pm)
            if len(out) >= bm25_n:
                break

        # Positions bm25_n+1 .. bm25_n+med_n from MedCPT
        sem_added = 0
        for r in med_list:
            pm = _pmcid(r)
            if pm and pm not in seen:
                out.append(r)
                seen.add(pm)
                sem_added += 1
            if sem_added >= med_n:
                break

        return out[: bm25_n + med_n]

    def weighted_rrf_fuse(
        lex: list[dict],
        sem: list[dict],
        *,
        total: int,
        k: int,
        w_lex: float,
        w_sem: float,
    ) -> list[dict]:
        """
        Weighted Reciprocal Rank Fusion:
            score(d) = sum_s w_s / (k + rank_s(d))
        where ranks are 1-based within each ranked list.
        """
        fused: Dict[str, Dict[str, Any]] = {}

        def add_list(lst: list[dict], weight: float, tag: str) -> None:
            for i, r in enumerate(lst, start=1):
                pm = _pmcid(r)
                if not pm:
                    continue

                contrib = float(weight) / float(k + i)

                if pm not in fused:
                    fused[pm] = dict(r)
                    fused[pm]["rrf_score"] = 0.0
                    fused[pm]["rrf_sources"] = {}

                fused[pm]["rrf_score"] = float(fused[pm]["rrf_score"]) + contrib
                fused[pm]["rrf_sources"][tag] = i

        add_list(lex, w_lex, "bm25_rank")
        add_list(sem, w_sem, "medcpt_rank")

        items = list(fused.values())
        items.sort(key=lambda x: float(x.get("rrf_score", 0.0)), reverse=True)
        return items[:total]

    # --- Base retrievers ---
    index = get_corpus_index()

    if method == "wrrf":
        bm25_res = index.search(query, limit=wrrf_k_each)
        med_res = search_corpus_medcpt_core(
            query,
            retrieve_k=max(50, wrrf_k_each * 5),
            top_n=wrrf_k_each,
        )
        return weighted_rrf_fuse(
            bm25_res,
            med_res,
            total=min(limit, wrrf_total),
            k=wrrf_k,
            w_lex=wrrf_w_lex,
            w_sem=wrrf_w_sem,
        )

    if method in ("bm25_then_medcpt", "bm25_medcpt_split", "15_bm25_5_medcpt"):
        total_needed = bm25_first_n + medcpt_after_n

        bm25_res = index.search(query, limit=max(bm25_first_n, total_needed))
        med_res = search_corpus_medcpt_core(
            query,
            retrieve_k=max(50, medcpt_after_n * 10),
            top_n=max(20, medcpt_after_n * 5),
        )

        results = bm25_then_medcpt_results(
            bm25_res,
            med_res,
            bm25_n=bm25_first_n,
            med_n=medcpt_after_n,
        )
        return results[: min(limit, total_needed)]

    raise ValueError(
        f"unknown corpus search method '{method}'. "
        f"Supported methods: 'wrrf', 'bm25_then_medcpt'"
    )


def search_corpus_medcpt_core(
    query: str,
    retrieve_k: int = 50,
    top_n: int = 20,
) -> list[dict]:
    """
    Semantic search using MedCPT embeddings + FAISS retrieval + cross-encoder rerank.

    Args:
        query: query string
        retrieve_k: initial FAISS retrieval size
        top_n: final top-N to return after reranking
    """
    from . import rerank_medcpt as rr

    index_dir = str(MEDCPT_DIR)
    meta_path = os.path.join(index_dir, "metadata.parquet")
    if not os.path.exists(meta_path):
        return []

    meta = pd.read_parquet(meta_path)

    if "text" not in meta.columns:
        title_col = meta["title"] if "title" in meta.columns else pd.Series("", index=meta.index)
        abstract_col = meta["abstract"] if "abstract" in meta.columns else pd.Series("", index=meta.index)
        meta["text"] = (title_col.fillna("") + "\n\n" + abstract_col.fillna("")).str.strip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) encode query
    try:
        q_vec = rr.encode_query(
            query,
            model_name="ncbi/MedCPT-Query-Encoder",
            device=device,
            fp16=False,
            normalize=True,
        )
    except Exception:
        return []

    # 2) retrieve with FAISS or brute-force fallback
    index_path = os.path.join(index_dir, "faiss.index")
    if os.path.exists(index_path):
        index = rr.load_faiss_index(index_path)
        emb_scores, emb_ids = rr.retrieve_faiss(index, q_vec, retrieve_k)
    else:
        memmap_path = os.path.join(index_dir, "embeddings.memmap")
        if not os.path.exists(memmap_path):
            return []

        n = len(meta)
        dim = None
        try:
            import json as _json
            with open(os.path.join(index_dir, "checkpoint.json"), "r") as f:
                ck = _json.load(f)
            dim = int(ck.get("dim")) if ck.get("dim") else None
        except Exception:
            dim = None

        if dim is None:
            return []

        emb_scores, emb_ids = rr.retrieve_bruteforce_memmap(
            memmap_path,
            n=n,
            dim=dim,
            q_vec=q_vec,
            k=retrieve_k,
        )

    keep = emb_ids >= 0
    emb_ids = emb_ids[keep]
    emb_scores = emb_scores[keep]

    if len(emb_ids) == 0:
        return []

    cand = meta.iloc[emb_ids].copy().reset_index(drop=True)
    cand["emb_score"] = emb_scores
    texts = cand["text"].astype(str).tolist()

    # 3) rerank with cross-encoder
    try:
        rr_scores = rr.rerank_cross_encoder(
            query=query,
            texts=texts,
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=device,
            batch_size=16,
            fp16=False,
        )
    except Exception:
        rr_scores = np.array(cand["emb_score"].astype(float))

    cand["rerank_score"] = rr_scores
    cand = cand.sort_values("rerank_score", ascending=False).head(top_n)

    out: list[dict] = []
    for _, row in cand.iterrows():
        pmcid = str(row.get("pmcid", "")).strip()
        title = row.get("title", "")
        abstract = row.get("abstract", "")
        score = float(row.get("rerank_score", row.get("emb_score", 0.0)))

        out.append(
            {
                "pmcid": pmcid,
                "title": title,
                "abstract": (abstract[:500] + "…")
                if isinstance(abstract, str) and len(abstract) > 500
                else abstract,
                "relevance_score": round(score, 4),
                "gene_in_title": query.lower() in str(title).lower(),
            }
        )

    return out