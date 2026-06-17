#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-encoder full-ontology prediction: stream-score all concepts and export top-K JSONL.

This script is the full-catalog counterpart of a reranker-style cross-encoder predictor.
It does NOT materialize query x concept pairs into an intermediate JSONL file.
Instead, it loads the ontology/concept catalog once, scores every concept for each query in
mini-batches, keeps only top-K results per query, and writes prediction JSONL.

Input queries JSONL (one line per query):
  {"query_text": "..."}                         # minimal
  {"doc_id": "...", "query_text": "..."}        # optional id
  {"query_id": "...", "query_text": "..."}      # optional id

Concept catalog must include concept id + name (json/jsonl):
  - JSON dict: { "HP_0000001": {"name": "..."}, ... }
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
If input contains doc_id/query_id/id, it will be copied to the output.

Notes:
- Cross-encoder scores are raw logits.
- For binary/one-logit sequence classification heads, logits.squeeze(-1) is used.
- For two-label classification heads, the positive-class logit at index 1 is used by default.
- This is computationally expensive: #queries x #concepts forward passes through the cross-encoder.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

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
# Catalog / queries
# -------------------------


def load_concept_catalog(path: Path) -> List[Concept]:
    """Load concept catalog with concept_id + name.

    Supported forms intentionally mirror the bi-encoder full-ontology script:
      - .jsonl rows: id/concept_id/term_id + name/term_text/label
      - .json dict: {id: {name: ...}}
      - .json list: [{id: ..., name: ...}]
    """
    concepts: List[Concept] = []
    seen: set[str] = set()

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                cid = obj.get("id") or obj.get("concept_id") or obj.get("term_id")
                name = obj.get("name") or obj.get("term_text") or obj.get("label")
                if cid is None or name is None:
                    continue
                cid = str(cid).strip()
                name = str(name).strip()
                if not cid or not name or cid in seen:
                    continue
                seen.add(cid)
                concepts.append(Concept(cid=cid, name=name))
        return concepts

    obj = load_json(path)

    if isinstance(obj, dict):
        for k, v in obj.items():
            cid = str(k).strip()
            name = None
            if isinstance(v, dict):
                name = v.get("name") or v.get("term_text") or v.get("label")
            elif isinstance(v, str):
                # permissive fallback: {"ID": "name"}
                name = v
            if name is None:
                continue
            name = str(name).strip()
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
            cid = str(cid).strip()
            name = str(name).strip()
            if not cid or not name or cid in seen:
                continue
            seen.add(cid)
            concepts.append(Concept(cid=cid, name=name))
        return concepts

    raise ValueError(f"Unsupported concept catalog format: {path}")


def load_queries(path: Path) -> List[dict]:
    """Read query records. Each record must contain non-empty 'query_text'.

    Duplicate query_text values are skipped, matching the existing bi-/cross-encoder scripts.
    """
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
# Scoring
# -------------------------


def chunked(seq: Sequence[Any], n: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def logits_to_scores(logits: torch.Tensor, positive_label_index: int) -> torch.Tensor:
    """Convert sequence-classification logits to one scalar score per pair.

    Cases:
      [B] or [B,1] -> raw scalar logit
      [B,2+]       -> logit of positive_label_index, default 1
    """
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.size(-1) == 1:
        return logits.squeeze(-1)
    if logits.ndim == 2:
        if positive_label_index < 0 or positive_label_index >= logits.size(-1):
            raise ValueError(
                f"--positive_label_index={positive_label_index} out of range for logits shape {tuple(logits.shape)}"
            )
        return logits[:, positive_label_index]
    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")


@torch.no_grad()
def score_query_against_all_concepts(
    tokenizer,
    model,
    query_text: str,
    concept_texts: Sequence[str],
    max_len: int,
    batch_size: int,
    device: torch.device,
    positive_label_index: int,
    query_first: bool,
    use_amp: bool,
) -> np.ndarray:
    """Return raw cross-encoder scores aligned to concept_texts."""
    scores: List[np.ndarray] = []
    model.eval()

    amp_enabled = bool(use_amp and device.type == "cuda")

    for chunk in chunked(concept_texts, batch_size):
        chunk_list = list(chunk)
        if query_first:
            text_a = [query_text] * len(chunk_list)
            text_b = chunk_list
        else:
            # mirrors the original cross_encoder_predict.py: tokenizer(candidates, [query] * n)
            text_a = chunk_list
            text_b = [query_text] * len(chunk_list)

        enc = tokenizer(
            text_a,
            text_b,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(**enc).logits
            sc = logits_to_scores(logits, positive_label_index=positive_label_index)

        scores.append(sc.detach().float().cpu().numpy())

    return np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=np.float32)


def topk_from_scores(scores: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted top-k scores and indices from a 1D score array."""
    n = scores.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    k = min(max(1, int(topk)), n)

    # argpartition is faster than full sort for large catalogs; final argsort restores ranking.
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return scores[idx], idx


# -------------------------
# Export
# -------------------------


def export_full_predictions(
    query_rows: List[dict],
    concepts: List[Concept],
    tokenizer,
    model,
    out_path: Path,
    topk: int,
    max_len: int,
    batch_size: int,
    device: torch.device,
    positive_label_index: int,
    query_first: bool,
    use_amp: bool,
) -> None:
    concept_texts = [c.name for c in concepts]

    def gen():
        for r in tqdm(query_rows, desc="Predict full ontology"):
            q = str(r.get("query_text", ""))

            scores = score_query_against_all_concepts(
                tokenizer=tokenizer,
                model=model,
                query_text=q,
                concept_texts=concept_texts,
                max_len=max_len,
                batch_size=batch_size,
                device=device,
                positive_label_index=positive_label_index,
                query_first=query_first,
                use_amp=use_amp,
            )
            top_scores, top_indices = topk_from_scores(scores, topk=topk)

            items: List[dict] = []
            for rank, (s, j) in enumerate(zip(top_scores.tolist(), top_indices.tolist())):
                c = concepts[int(j)]
                items.append(
                    {
                        "rank": int(rank),
                        "score": float(s),
                        "term_id": c.cid,
                        "term_text": c.name,
                    }
                )

            out: dict = {"query_text": q, "items": items}

            # copy optional identifiers if present, same convention as your other scripts
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
    p = argparse.ArgumentParser(
        description="Cross-encoder full-ontology prediction: stream-score all concepts and export top-K JSONL."
    )
    p.add_argument("--model_dir", type=Path, required=True, help="Cross-encoder checkpoint dir (HF format).")

    p.add_argument("--queries_jsonl", type=Path, required=True, help="JSONL: one query per line, requires query_text.")
    p.add_argument(
        "--concept_catalog_json",
        type=Path,
        required=True,
        help="Concept catalog containing concept_id + name (json/jsonl).",
    )

    p.add_argument("--out_jsonl", type=Path, required=True, help="Output prediction JSONL.")
    p.add_argument("--topk", type=int, default=100, help="Number of top concepts to keep per query.")

    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument(
        "--positive_label_index",
        type=int,
        default=1,
        help="For multi-label logits [B,C], use this class logit as score. Ignored for [B] or [B,1] logits.",
    )
    p.add_argument(
        "--query_first",
        action="store_true",
        help="Use tokenizer(query, concept). Default mirrors existing script: tokenizer(concept, query).",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Use CUDA autocast fp16 for faster inference. Keep off if you want exact fp32 behavior.",
    )

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

    LOGGER.info("[Load] queries=%d, concepts=%d", len(query_rows), len(concepts))
    LOGGER.info("[Device] %s", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    export_full_predictions(
        query_rows=query_rows,
        concepts=concepts,
        tokenizer=tokenizer,
        model=model,
        out_path=args.out_jsonl,
        topk=args.topk,
        max_len=args.max_len,
        batch_size=args.batch_size,
        device=device,
        positive_label_index=args.positive_label_index,
        query_first=args.query_first,
        use_amp=args.fp16,
    )
    LOGGER.info("[Done]")


if __name__ == "__main__":
    main()
