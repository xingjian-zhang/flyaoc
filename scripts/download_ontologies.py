#!/usr/bin/env python3
"""Download required ontology files for the agent.

This script downloads:
- go-basic.obo: Gene Ontology (basic version)
- fly_anatomy.obo: FlyBase anatomy ontology (FBbt)
- fly_development.obo: FlyBase developmental stage ontology (FBdv)

Usage:
    python scripts/download_ontologies.py
"""

import urllib.request
from pathlib import Path

# URLs for ontology files
ONTOLOGY_URLS = {
    "go-basic.obo": "http://purl.obolibrary.org/obo/go/go-basic.obo",
    "fly_anatomy.obo": "http://purl.obolibrary.org/obo/fbbt/fbbt-simple.obo",
    "fly_development.obo": "http://purl.obolibrary.org/obo/fbdv/fbdv-simple.obo",
}

# Target directory
ONTOLOGY_DIR = Path(__file__).parent.parent / "ontologies"


def download_file(url: str, target_path: Path) -> None:
    """Download a file from URL to target path."""
    print(f"Downloading {target_path.name}...")
    print(f"  URL: {url}")

    try:
        urllib.request.urlretrieve(url, target_path)
        size_mb = target_path.stat().st_size / (1024 * 1024)
        print(f"  Downloaded: {size_mb:.2f} MB")
    except Exception as e:
        print(f"  Error: {e}")
        raise


def main():
    """Download all required ontology files."""
    ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading ontology files...\n")

    for filename, url in ONTOLOGY_URLS.items():
        target_path = ONTOLOGY_DIR / filename

        if target_path.exists():
            print(f"Skipping {filename} (already exists)")
            continue

        download_file(url, target_path)
        print()

    print("Done! Ontology files are in:", ONTOLOGY_DIR)


if __name__ == "__main__":
    main()
