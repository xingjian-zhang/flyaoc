"""Recompute table-ready metrics from normalized frozen predictions."""

from __future__ import annotations

from pathlib import Path

from flyaoc.reporting.tables import evaluate_manifest, write_csv, write_markdown


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "artifacts/predictions/manifest.json"
    rows = evaluate_manifest(manifest)
    write_csv(rows, root / "artifacts/tables/main_results.csv")
    write_markdown(rows, root / "artifacts/tables/main_results.md")
    print(f"Wrote {len(rows)} table row(s) to artifacts/tables/main_results.*")


if __name__ == "__main__":
    main()
