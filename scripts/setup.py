#!/usr/bin/env python3
"""Setup script for the agent benchmark.

This script:
1. Downloads ontology files
2. Pre-builds ontology indices
3. Downloads and caches the literature corpus

Usage:
    python scripts/setup.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 60)
    print("Drosophila Agent Benchmark Setup")
    print("=" * 60)

    # Step 1: Download ontologies
    print("\n[1/3] Downloading ontology files...")
    from scripts.download_ontologies import main as download_ontologies

    download_ontologies()

    # Step 2: Build ontology indices
    print("\n[2/3] Building ontology indices...")
    from agent.tools.ontology_search import (
        get_fbbt_index,
        get_fbdv_index,
        get_go_index,
    )

    print("  Building GO index...")
    go_ix = get_go_index()
    print(f"  GO index: {go_ix.doc_count()} terms")

    print("  Building FBbt (anatomy) index...")
    fbbt_ix = get_fbbt_index()
    print(f"  FBbt index: {fbbt_ix.doc_count()} terms")

    print("  Building FBdv (stage) index...")
    fbdv_ix = get_fbdv_index()
    print(f"  FBdv index: {fbdv_ix.doc_count()} terms")

    # Step 3: Load and cache corpus
    print("\n[3/3] Loading and caching literature corpus...")
    print("  This may take a few minutes on first run...")
    from agent.tools.corpus_search import get_corpus_index

    corpus_ix = get_corpus_index()
    print(f"  Corpus: {len(corpus_ix.corpus)} papers indexed")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nYou can now run the agents:")
    print("  python -m agent.run_langgraph --gene-id FBgn0000014 --gene-symbol abd-A")
    print("  python -m agent.run_mcp --gene-id FBgn0000014 --gene-symbol abd-A")
    print("\nOr run on benchmark genes:")
    print("  python -m agent.run_langgraph --benchmark data/genes_top100.csv --limit 5")


if __name__ == "__main__":
    main()
