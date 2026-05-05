"""Hugging Face data access for FlyAOC."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    """Location of the anonymous HF dataset and canonical files."""

    dataset_id: str = "anonymous-042/flyaoc"
    benchmark_file: str = "benchmark.jsonl"
    corpus_file: str = "corpus.jsonl"
    go_obo_file: str = "ontologies/go-basic.obo"
    anatomy_obo_file: str = "ontologies/fly_anatomy.obo"
    development_obo_file: str = "ontologies/fly_development.obo"


def load_benchmark_records(
    config: DatasetConfig | None = None,
    *,
    limit: int | None = None,
    streaming: bool = False,
) -> list[dict[str, Any]]:
    """Load benchmark records from the anonymous HF dataset."""

    cfg = config or DatasetConfig()
    if streaming:
        raise ValueError("Streaming benchmark records is not supported for local JSONL loading.")

    benchmark_path = download_dataset_file(cfg.benchmark_file, cfg)
    records: list[dict[str, Any]] = []
    with benchmark_path.open() as handle:
        for line in handle:
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def download_dataset_file(
    path: str,
    config: DatasetConfig | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download a single file from the anonymous HF dataset and return its local path."""

    cfg = config or DatasetConfig()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised by environment setup
        raise RuntimeError("Install dependencies with `uv sync` before downloading HF files.") from exc

    kwargs = {
        "repo_id": cfg.dataset_id,
        "repo_type": "dataset",
        "filename": path,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
    }
    try:
        downloaded = hf_hub_download(**kwargs, local_files_only=True)
    except Exception:
        downloaded = hf_hub_download(**kwargs)
    return Path(downloaded)
