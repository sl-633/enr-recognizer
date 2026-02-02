#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bi-encoder prediction: export top-K retrieval results to JSONL (no eval).

Input queries JSONL (one line per query):
  {"query_text": "..."}                         # minimal
  {"doc_id": "...", "query_text": "..."}        # optional id
  {"query_id": "...", "query_text": "..."}      # optional id

Concept catalog must include concept id + name (json/jsonl):
  - JSON dict: { "HP_0000001": {"name": "..."} , ... }
  - JSON list: [ {"id": "...", "name": "..."}, ... ]
  - JSONL: each line {"id": "...", "name": "..."}

Output predictions JSONL (one line per query):
{
  "query_text": "...",
  "items": [
    {"rank": 0, "score": 0.12, "term_id": "HP_0000739", "term_text": "Anxiety"},
    ...
  ]
}
If input contains doc_id/query_id, it will be copied to the output.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

LOGGER = logging.getLogger(__name__)


# -------------------------
# IO helpers
# -------------------------


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# Data model
# -------------------------


@dataclass(frozen=True)
class Concept:
    cid: str
    name: str


# -------------------------
# Encoder
# -------------------------


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
    summed = (last_hidden_state * mask).sum(dim=1)  # [B, H]
    denom = mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]
    return summed / denom


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    texts: Sequence[str],
    max_len: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Return L2-normalized embeddings on CPU float32."""
    vecs: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        chunk = list(texts[i : i + batch_size])
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        out = model(**enc).last_hidden_state
        pooled = mean_pooling(out, enc["attention_mask"])
        pooled = F.normalize(pooled, dim=-1)
        vecs.append(pooled.detach().cpu())
    return torch.cat(vecs, dim=0).float() if vecs else torch.empty(0)


# -------------------------
# Catalog / queries
# -------------------------


def load_concept_catalog(path: Path) -> List[Concept]:
    """Load concept catalog with concept_id + name."""
    concepts: List[Concept] = []
    seen: set[str] = set()

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = obj.get("id") or obj.get("concept_id") or obj.get("term_id")
                name = obj.get("name") or obj.get("term_text") or obj.get("label")
                if cid is None or name is None:
                    continue
                cid = str(cid)
                name = str(name)
                if not cid or not name or cid in seen:
                    continue
                seen.add(cid)
                concepts.append(Concept(cid=cid, name=name))
        return concepts

    obj = load_json(path)
    if isinstance(obj, dict):
        for k, v in obj.items():
            cid = str(k)
            name = None
            if isinstance(v, dict):
                name = v.get("name") or v.get("term_text") or v.get("label")
            if name is None:
                continue
            name = str(name)
            if not cid or not name or cid in seen:
                continue
            seen.add(cid)
            concepts.append(Concept(cid=cid, name=name))
        return concepts

    if isinstance(obj, list):
        for it in obj:
            if not isinstance(it, dict):
                continue
            cid = it.get("id") or it.get("concept_id") or it.get("term_id")
            name = it.get("name") or it.get("term_text") or it.get("label")
            if cid is None or name is None:
                continue
            cid = str(cid)
            name = str(name)
            if not cid or not name or cid in seen:
                continue
            seen.add(cid)
            concepts.append(Concept(cid=cid, name=name))
        return concepts

    raise ValueError(f"Unsupported concept catalog format: {path}")


def load_queries(path: Path) -> List[dict]:
    """Read query records. Each record must contain non-empty 'query_text'."""
    rows = read_jsonl(path)
    unique_q = set()
    out: List[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        q = r.get("query_text", "")
        if not isinstance(q, str) or not q.strip():
            continue
        if q in unique_q:
            continue
        unique_q.add(q)
        out.append(r)
    return out


# -------------------------
# Retrieval (FAISS optional)
# -------------------------


def _try_build_faiss_index(vectors: np.ndarray, use_gpu: bool, gpu_id: int):
    try:
        import faiss  # type: ignore
    except Exception:
        return None, None

    dim = vectors.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(vectors)

    if not use_gpu:
        return faiss, cpu_index

    if not hasattr(faiss, "StandardGpuResources"):
        return faiss, cpu_index

    try:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
        return faiss, gpu_index
    except Exception:
        return faiss, cpu_index


def retrieve_topk(
    term_vecs: torch.Tensor,   # [Nt, D] CPU
    query_vecs: torch.Tensor,  # [Nq, D] CPU
    topk: int,
    use_faiss: bool,
    use_faiss_gpu: bool,
    faiss_gpu_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scores, indices) with shape [Nq, topk]."""
    term_np = term_vecs.numpy().astype(np.float32, copy=False)
    query_np = query_vecs.numpy().astype(np.float32, copy=False)

    if use_faiss:
        _, index = _try_build_faiss_index(term_np, use_gpu=use_faiss_gpu, gpu_id=faiss_gpu_id)
        if index is not None:
            scores, indices = index.search(query_np, int(topk))
            return scores, indices

    sims = torch.from_numpy(query_np) @ torch.from_numpy(term_np).t()  # [Nq, Nt]
    vals, idxs = torch.topk(sims, k=min(int(topk), sims.size(1)), dim=1, largest=True, sorted=True)
    return vals.numpy(), idxs.numpy()


# -------------------------
# Export
# -------------------------


def export_predictions(
    query_rows: List[dict],
    scores: np.ndarray,
    indices: np.ndarray,
    concepts: List[Concept],
    out_path: Path,
) -> None:
    def gen():
        for r, sc_row, idx_row in zip(query_rows, scores, indices):
            q = str(r.get("query_text", ""))

            items: List[dict] = []
            for rank, (s, j) in enumerate(zip(sc_row.tolist(), idx_row.tolist())):
                j = int(j)
                if j < 0 or j >= len(concepts):
                    continue
                c = concepts[j]
                items.append(
                    {
                        "rank": int(rank),   # 0-based
                        "score": float(s),
                        "term_id": c.cid,
                        "term_text": c.name,
                    }
                )

            out: dict = {"query_text": q, "items": items}

            # copy optional identifiers if present
            for key in ("doc_id", "query_id", "id"):
                if key in r and key not in out:
                    out[key] = r[key]

            yield out

    write_jsonl(out_path, gen())
    LOGGER.info("[Write] %s", out_path)


# -------------------------
# CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bi-encoder prediction: export top-K retrieval JSONL (no eval).")
    p.add_argument("--model_dir", type=Path, required=True, help="Bi-encoder checkpoint dir (HF format).")

    p.add_argument("--queries_jsonl", type=Path, required=True, help="JSONL: one query per line, requires query_text.")
    p.add_argument(
        "--concept_catalog_json",
        type=Path,
        required=True,
        help="Concept catalog containing concept_id + name (json/jsonl).",
    )

    p.add_argument("--out_jsonl", type=Path, required=True, help="Output prediction JSONL.")
    p.add_argument("--topk", type=int, default=100)

    p.add_argument("--q_max_len", type=int, default=512)
    p.add_argument("--t_max_len", type=int, default=64)
    p.add_argument("--encode_bs", type=int, default=64)

    # retrieval backend
    p.add_argument("--use_faiss", action="store_true", help="Use FAISS if available (recommended for large catalogs).")
    p.add_argument("--faiss_gpu", action="store_true", help="Use FAISS GPU index if available.")
    p.add_argument("--gpu", type=int, default=0)

    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    query_rows = load_queries(args.queries_jsonl)
    if not query_rows:
        raise ValueError("No valid queries found in --queries_jsonl (needs non-empty query_text).")

    concepts = load_concept_catalog(args.concept_catalog_json)
    if not concepts:
        raise ValueError("Empty concept catalog. Check --concept_catalog_json.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModel.from_pretrained(args.model_dir).to(device)
    model.eval()

    LOGGER.info("[Load] queries=%d, concepts=%d", len(query_rows), len(concepts))

    term_texts = [c.name for c in concepts]
    query_texts = [str(r["query_text"]) for r in query_rows]

    LOGGER.info("[Encode] concepts...")
    term_vecs = encode_texts(model, tokenizer, term_texts, args.t_max_len, device, batch_size=args.encode_bs)

    LOGGER.info("[Encode] queries...")
    q_vecs = encode_texts(model, tokenizer, query_texts, args.q_max_len, device, batch_size=args.encode_bs)

    LOGGER.info("[Retrieve] topk=%d (faiss=%s, faiss_gpu=%s)", args.topk, args.use_faiss, args.faiss_gpu)
    scores, indices = retrieve_topk(
        term_vecs=term_vecs,
        query_vecs=q_vecs,
        topk=args.topk,
        use_faiss=args.use_faiss,
        use_faiss_gpu=(args.faiss_gpu and torch.cuda.is_available()),
        faiss_gpu_id=args.gpu,
    )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    export_predictions(query_rows, scores, indices, concepts, args.out_jsonl)
    LOGGER.info("[Done]")


if __name__ == "__main__":
    main()
