"""Review-artifact data locations for the original FlyAOC harness."""

from __future__ import annotations

import os
from pathlib import Path

from flyaoc.data import DatasetConfig, download_dataset_file


def repo_root() -> Path:
    """Return the review artifact repository root."""
    return Path(__file__).resolve().parents[3]


def cache_dir() -> Path:
    """Return the local ignored cache directory used by rerunnable baselines."""
    configured = os.environ.get("FLYAOC_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return repo_root() / ".flyaoc_cache"


def hf_cache_dir() -> Path:
    path = cache_dir() / "hf"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ontology_file_path(obo_filename: str) -> Path:
    """Resolve an ontology filename from the released HF dataset."""
    cfg = DatasetConfig()
    mapping = {
        "go-basic.obo": cfg.go_obo_file,
        "fly_anatomy.obo": cfg.anatomy_obo_file,
        "fly_development.obo": cfg.development_obo_file,
    }
    try:
        dataset_path = mapping[obo_filename]
    except KeyError as exc:
        raise FileNotFoundError(f"Unknown FlyAOC ontology file: {obo_filename}") from exc
    return download_dataset_file(dataset_path, cfg, cache_dir=hf_cache_dir())
