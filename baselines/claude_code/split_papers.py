#!/usr/bin/env python3
"""One-time preprocessing: split papers_full.json into individual PMCID files.

Usage:
    python3 split_papers.py [--input PATH] [--output-dir PATH]

Reads corpus_cache/papers_full.json (661MB, ~17k papers) and writes individual
files to papers_split/{PMCID}.json for volume-mounting into Docker containers.
"""

import argparse
import json
import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = BENCHMARK_ROOT / "corpus_cache" / "papers_full.json"
DEFAULT_OUTPUT = BENCHMARK_ROOT / "baselines" / "claude_code" / "papers_split"


def split_papers(input_path: Path, output_dir: Path) -> int:
    """Split papers_full.json into individual files.

    Returns the number of papers written.
    """
    print(f"Reading {input_path} ...")
    with open(input_path) as f:
        papers = json.load(f)

    print(f"Loaded {len(papers)} papers. Writing to {output_dir}/ ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for pmcid, paper in papers.items():
        out_file = output_dir / f"{pmcid}.json"
        with open(out_file, "w") as f:
            json.dump(paper, f)
        count += 1
        if count % 2000 == 0:
            print(f"  {count}/{len(papers)} ...")

    print(f"Done. Wrote {count} files to {output_dir}/")
    return count


def main():
    parser = argparse.ArgumentParser(description="Split papers_full.json into individual files")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to papers_full.json (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for split files (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    split_papers(args.input, args.output_dir)


if __name__ == "__main__":
    main()
