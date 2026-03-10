#!/usr/bin/env python3
"""
Compute MedCPT (Article Encoder) embeddings for a corpus of paper titles + abstracts
with:
- tqdm progress bar
- checkpointed, resumable batching (saves progress every N batches)
- outputs:
    - embeddings.memmap (float32; [N, dim])  <-- main durable store during encoding
    - embeddings.npy    (optional final export)
    - metadata.parquet
    - (optional) faiss.index (over L2-normalized vectors if --normalize)

Usage:
  python learn_medcpt_embeddings.py \
    --outdir ./drosophila_medcpt \
    --batch_size 32 \
    --max_length 512 \
    --normalize \
    --build_faiss \
    --save_every 25 \
    --resume \
    --export_npy
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("Missing dependency: pandas. Install with `pip install pandas pyarrow`.") from e

try:
    from tqdm import tqdm
except ImportError as e:
    raise SystemExit("Missing dependency: tqdm. Install with `pip install tqdm`.") from e


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [B, T, 1]
    summed = (last_hidden * mask).sum(dim=1)                  # [B, H]
    denom = mask.sum(dim=1).clamp(min=1e-6)                   # [B, 1]
    return summed / denom


def build_faiss_index(vectors: np.ndarray, outpath: str) -> None:
    try:
        import faiss  # type: ignore
    except ImportError as e:
        raise SystemExit("Missing dependency: faiss. Install with `pip install faiss-cpu` (or faiss-gpu).") from e

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype(np.float32))
    faiss.write_index(index, outpath)


def load_checkpoint(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="jimmyzxj/drosophila-literature-corpus")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="ncbi/MedCPT-Article-Encoder")
    ap.add_argument("--outdir", default="./medcpt_embeddings")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--fp16", action="store_true", help="Use fp16 autocast on CUDA.")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize embeddings (recommended for cosine/IP search).")

    ap.add_argument("--save_every", type=int, default=25, help="Flush checkpoint every N batches.")
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint if present.")
    ap.add_argument("--export_npy", action="store_true", help="Export embeddings.npy at the end (copies memmap).")
    ap.add_argument("--build_faiss", action="store_true", help="Also write a FAISS IndexFlatIP index.")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load corpus
    ds = load_dataset(args.dataset, split=args.split)

    pmcid = ds["pmcid"]
    title = ds["title"]
    abstract = ds["abstract"]

    texts: List[str] = []
    for t, a in zip(title, abstract):
        t = (t or "").strip()
        a = (a or "").strip()
        texts.append(f"{t}\n\n{a}".strip() if (t and a) else (t or a or ""))

    # Save metadata once
    meta_path = os.path.join(args.outdir, "metadata.parquet")
    if not os.path.exists(meta_path):
        meta = pd.DataFrame({"pmcid": pmcid, "title": title, "abstract": abstract, "text": texts})
        meta.to_parquet(meta_path, index=False)

    # Load MedCPT Article Encoder
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    # Determine embedding dim from model config (preferred) or a tiny forward pass
    dim = getattr(getattr(model, "config", None), "hidden_size", None)
    if dim is None:
        sample = tokenizer(["test"], return_tensors="pt").to(device)
        out = model(**sample)
        dim = out.last_hidden_state.shape[-1]

    n = len(texts)

    # Files
    ckpt_path = os.path.join(args.outdir, "checkpoint.json")
    memmap_path = os.path.join(args.outdir, "embeddings.memmap")

    # Resume logic
    start_idx = 0
    ckpt = load_checkpoint(ckpt_path) if args.resume else None
    if ckpt is not None:
        # Basic compatibility checks
        if ckpt.get("dataset") != args.dataset or ckpt.get("split") != args.split:
            raise SystemExit(f"Checkpoint dataset mismatch: {ckpt.get('dataset')}[{ckpt.get('split')}] vs {args.dataset}[{args.split}]")
        if ckpt.get("model") != args.model:
            raise SystemExit(f"Checkpoint model mismatch: {ckpt.get('model')} vs {args.model}")
        if ckpt.get("n") != n or ckpt.get("dim") != dim:
            raise SystemExit(f"Checkpoint shape mismatch: n/dim {ckpt.get('n')}/{ckpt.get('dim')} vs {n}/{dim}")
        if bool(ckpt.get("normalize")) != bool(args.normalize):
            raise SystemExit("Checkpoint normalize flag mismatch (resume requires same --normalize setting).")

        start_idx = int(ckpt.get("next_idx", 0))
        if start_idx < 0 or start_idx > n:
            raise SystemExit(f"Invalid checkpoint next_idx={start_idx}")
    else:
        # If not resuming but files exist, avoid accidental overwrite
        if os.path.exists(memmap_path) and not args.resume:
            raise SystemExit(f"{memmap_path} exists. Use --resume to continue, or delete the outdir file(s).")

    # Create/load memmap
    # mode='r+' requires file exists; mode='w+' creates/overwrites.
    mm_mode = "r+" if (os.path.exists(memmap_path) and start_idx > 0) else "w+"
    emb_mm = np.memmap(memmap_path, dtype="float32", mode=mm_mode, shape=(n, dim))

    # Autocast
    use_fp16 = bool(args.fp16 and device.type == "cuda")
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_fp16 else torch.no_grad()

    # Encode in batches with progress bar
    total_batches = (n + args.batch_size - 1) // args.batch_size
    start_batch = start_idx // args.batch_size

    pbar = tqdm(total=total_batches, initial=start_batch, desc="Encoding batches", unit="batch")

    next_idx = start_idx
    batches_since_save = 0

    while next_idx < n:
        b_start = next_idx
        b_end = min(n, b_start + args.batch_size)
        batch_texts = texts[b_start:b_end]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)

        with autocast_ctx:
            out = model(**enc)
            vec = mean_pool(out.last_hidden_state, enc["attention_mask"])  # [B, H]

        if args.normalize:
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)

        vec_np = vec.detach().float().cpu().numpy().astype(np.float32)
        emb_mm[b_start:b_end, :] = vec_np

        next_idx = b_end
        pbar.update(1)
        batches_since_save += 1

        if batches_since_save >= args.save_every or next_idx == n:
            emb_mm.flush()
            save_checkpoint(
                ckpt_path,
                {
                    "dataset": args.dataset,
                    "split": args.split,
                    "model": args.model,
                    "n": n,
                    "dim": dim,
                    "batch_size": args.batch_size,
                    "max_length": args.max_length,
                    "normalize": bool(args.normalize),
                    "next_idx": next_idx,
                },
            )
            batches_since_save = 0

    pbar.close()

    # Optional export to .npy (copies from memmap)
    if args.export_npy:
        emb_npy_path = os.path.join(args.outdir, "embeddings.npy")
        np.save(emb_npy_path, np.asarray(emb_mm, dtype=np.float32))

    # Optional FAISS
    if args.build_faiss:
        # For normalized vectors, IP == cosine similarity
        index_path = os.path.join(args.outdir, "faiss.index")
        build_faiss_index(np.asarray(emb_mm, dtype=np.float32), index_path)

    if args.export_npy:
        print(f"Wrote:   {os.path.join(args.outdir, 'embeddings.npy')}")
    if args.build_faiss:
        print(f"Wrote:   {os.path.join(args.outdir, 'faiss.index')}")


if __name__ == "__main__":
    main()