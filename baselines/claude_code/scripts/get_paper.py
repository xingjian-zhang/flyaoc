#!/usr/bin/env python3
"""Fetch a paper by PMCID from pre-split individual files.

Usage:
    python3 scripts/get_paper.py PMC1234567
    python3 scripts/get_paper.py PMC1234567 --sections INTRO RESULTS
    python3 scripts/get_paper.py PMC1234567 --summary

Output: JSON to stdout with paper content.

Paper budget enforcement:
    Set MAX_PAPERS env var to limit how many papers can be read.
    A counter file at /tmp/papers_read.count tracks reads across invocations.
    --summary calls do NOT count toward the budget.

No external dependencies.
Data: /data/papers/{PMCID}.json (pre-split individual files)
"""

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

PAPERS_DIR = Path("/data/papers")
COUNTER_FILE = Path("/tmp/papers_read.count")
LOCK_FILE = Path("/tmp/papers_read.lock")


def _acquire_budget() -> tuple[int, int] | None:
    """Atomically check budget and increment counter.

    Returns (new_count, max_papers) if budget allows, or None if exhausted.
    Uses file locking to handle concurrent subagent access.
    """
    max_papers = int(os.environ.get("MAX_PAPERS", "0"))
    if max_papers <= 0:
        return (0, 0)  # No limit — signal with max_papers=0

    with open(LOCK_FILE, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            current = int(COUNTER_FILE.read_text().strip()) if COUNTER_FILE.exists() else 0
        except (ValueError, OSError):
            current = 0

        if current >= max_papers:
            return None  # Budget exhausted

        new_count = current + 1
        COUNTER_FILE.write_text(str(new_count))
        return (new_count, max_papers)


def get_paper(pmcid: str, sections: list[str] | None = None, summary: bool = False) -> dict:
    paper_path = PAPERS_DIR / f"{pmcid}.json"

    if not paper_path.exists():
        return {"error": f"Paper {pmcid} not found in corpus"}

    # Summary calls don't count toward budget
    if not summary:
        result_budget = _acquire_budget()
        if result_budget is None:
            max_papers = int(os.environ.get("MAX_PAPERS", "0"))
            return {
                "error": (
                    f"Paper budget exhausted ({max_papers}/{max_papers} papers read). "
                    f"You must now aggregate annotations from the papers already read and submit."
                )
            }
        count, max_papers = result_budget
        remaining = (max_papers - count) if max_papers > 0 else None

    with open(paper_path) as f:
        paper = json.load(f)

    if summary:
        # Return just title, abstract, and section names
        section_names = list(paper.get("sections", {}).keys())
        section_lengths = {}
        for name, content in paper.get("sections", {}).items():
            if isinstance(content, list):
                section_lengths[name] = sum(len(p) for p in content)
            elif isinstance(content, str):
                section_lengths[name] = len(content)
        return {
            "pmcid": paper["pmcid"],
            "title": paper["title"],
            "abstract": paper.get("abstract", ""),
            "sections_available": section_names,
            "section_lengths_chars": section_lengths,
        }

    result = {}

    # Budget info first — must survive output truncation
    if not summary and remaining is not None:
        result["_budget"] = {
            "papers_read": count,
            "papers_remaining": remaining,
            "max_papers": max_papers,
        }

    result["pmcid"] = paper["pmcid"]
    result["title"] = paper["title"]

    if sections is None:
        result["abstract"] = paper.get("abstract", "")
        result["sections"] = paper.get("sections", {})
    else:
        sections_upper = [s.upper() for s in sections]
        if "ABSTRACT" in sections_upper:
            result["abstract"] = paper.get("abstract", "")
        filtered = {}
        paper_sections = paper.get("sections", {})
        for sec_name in sections_upper:
            if sec_name != "ABSTRACT" and sec_name in paper_sections:
                filtered[sec_name] = paper_sections[sec_name]
        result["abstract"] = result.get("abstract")
        result["sections"] = filtered

    return result


def main():
    parser = argparse.ArgumentParser(description="Get paper content by PMCID")
    parser.add_argument("pmcid", help="PubMed Central ID (e.g., PMC1234567)")
    parser.add_argument(
        "--sections", nargs="+",
        help="Specific sections to retrieve (e.g., INTRO RESULTS DISCUSS)",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Return only metadata (title, abstract, section names)",
    )
    args = parser.parse_args()

    result = get_paper(args.pmcid, sections=args.sections, summary=args.summary)
    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
