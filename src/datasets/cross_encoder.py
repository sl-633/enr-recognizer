#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare cross-encoder training data (jsonl) for MedMentions-style MM-{go,hpo}.

This script consolidates the old one-off helpers into a clean, modular CLI:

1) Build gold-derived hard negatives (BM25) -> train jsonl
2) Extract false positives (FP) from model prediction outputs -> fp json
3) Merge FP maps -> fp json
4) Build FP-only train jsonl
5) Build FP + gold(BM25) mixed train jsonl (with pool-size grid)
6) (Optional) Distribution-matched thresholding to map scored preds to MA-COIR beam10 FP counts
7) (Optional) Stats: avg hard negatives per passage

------------------------------------------------------------
Expected data layout (same as your rewritten pipeline)

Gold:
  {data_root}/mm-{dataset}/ori/all_wo_mention.json
  {data_root}/mm-{dataset}/ori/split_list.json

BM25 hard negatives (JSONL; concept-level):
  Each line:
  {
    "doc_id": "GO_0000018",                      # concept id
    "positives": ["GO_0000018"],                 # ignored
    "candidates": [{"concept_id":..., "name":..., "score":...}, ...]
  }

Prediction outputs (doc-level JSON dict):
  pred_json[doc_id] = {
    "gold": [{"id": "...", "name": "..."}, ...],
    "pred": [{"id": "...", "name": "...", "score": 0.12?}, ...]
  }

Folders & naming (your latest convention):
  macoir-output:   {split}_beam10.json
  bi-encoder-output / xrt-output: {split}_thresholded.json, {split}_top100.json

------------------------------------------------------------
Examples

# 1) Extract FP from MA-COIR beam10 outputs
python prepare_cross_encoder_data.py extract-fp \
  --dataset go \
  --pred-dir logs/macoir-output/mm-go \
  --variant beam10

# 2) Extract FP from bi-encoder thresholded outputs
python prepare_cross_encoder_data.py extract-fp \
  --dataset go \
  --pred-dir logs/bi-encoder-output/mm-go \
  --variant thresholded

# 3) Merge FP from multiple sources
python prepare_cross_encoder_data.py merge-fp \
  --inputs data/mm-go/mld/fp/fp_macoir-output_beam10.json data/mm-go/mld/fp/fp_bi-encoder-output_thresholded.json \
  --out data/mm-go/mld/fp/fp_macoir_be.json

# 4) Build FP-only cross-encoder training jsonl
python prepare_cross_encoder_data.py build-fp-only \
  --dataset go \
  --fp-json data/mm-go/mld/fp/fp_macoir_be.json \
  --out data/mm-go/mld/for_cross_encoder_training/train_macoir_be_fp.jsonl

# 5) Build gold(BM25)-only with pool sizes
python prepare_cross_encoder_data.py build-gold-bm25 \
  --dataset go \
  --bm25-jsonl data/mm-go/mld/hard_negative_all_concept_bm25.jsonl \
  --pool-sizes 5 10 20 50 \
  --out-dir data/mm-go/mld/for_cross_encoder_training \
  --tag gold_bm25

# 6) Build FP + gold(BM25) mix with pool-size grid
python prepare_cross_encoder_data.py build-mix \
  --dataset go \
  --fp-json data/mm-go/mld/fp/fp_macoir_be.json \
  --bm25-jsonl data/mm-go/mld/hard_negative_all_concept_bm25.jsonl \
  --pool-sizes 5 10 20 50 \
  --out-dir data/mm-go/mld/for_cross_encoder_training \
  --tag macoir_be

# 7) (Optional) map scored top100 predictions to match macoir beam10 FP distribution
python prepare_cross_encoder_data.py map-to-macoir-by-fpcount \
  --dataset go \
  --macoir-fp-json data/mm-go/mld/fp/fp_macoir-output_beam10.json \
  --scored-pred-json logs/bi-encoder-output/mm-go/train_top100.json \
  --out-fp-json data/mm-go/mld/fp/fp_be_top100_map_to_macoir_beam10.json

"""

from __future__ import annotations

import argparse
import json
import math
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -------------------------
# IO helpers
# -------------------------


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_jsonl(path: Path) -> List[dict]:
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
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def norm_name(s: Any) -> str:
    """Robust key for name->id lookup."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = " ".join(s.strip().split())
    return s.lower()


def load_concept_catalog_id2name(path: Path) -> Dict[str, str]:
    """Load concept catalog as id -> name.

    Supported:
      - JSON dict: {id: {"name":...}, ...} OR {id: {"term_text":...}}
      - JSON list: [{"id":..., "name":...}, ...]
      - JSONL: each line {"id":..., "name":...}
    """
    id2name: Dict[str, str] = {}

    if path.suffix.lower() == ".jsonl":
        for r in load_jsonl(path):
            if not isinstance(r, dict):
                continue
            cid = r.get("id") or r.get("concept_id") or r.get("term_id")
            name = r.get("name") or r.get("term_text") or r.get("label")
            if cid is None or name is None:
                continue
            cid = str(cid).strip()
            name = str(name).strip()
            if cid and name and cid not in id2name:
                id2name[cid] = name
        return id2name

    obj = load_json(path)
    if isinstance(obj, dict):
        # could be {id: {"name":...}}
        for k, v in obj.items():
            cid = str(k).strip()
            if not cid or cid in id2name:
                continue
            name = None
            if isinstance(v, dict):
                name = v.get("name") or v.get("term_text") or v.get("label")
            if name is None:
                continue
            name = str(name).strip()
            if name:
                id2name[cid] = name
        return id2name

    if isinstance(obj, list):
        for r in obj:
            if not isinstance(r, dict):
                continue
            cid = r.get("id") or r.get("concept_id") or r.get("term_id")
            name = r.get("name") or r.get("term_text") or r.get("label")
            if cid is None or name is None:
                continue
            cid = str(cid).strip()
            name = str(name).strip()
            if cid and name and cid not in id2name:
                id2name[cid] = name
        return id2name

    raise ValueError(f"Unsupported concept catalog format: {path}")


def build_name2id(id2name: Dict[str, str]) -> Dict[str, str]:
    """Build normalized name -> id.
    If multiple ids share the same normalized name, choose the lexicographically smallest id (stable).
    """
    tmp: Dict[str, List[str]] = {}
    for cid, nm in id2name.items():
        key = norm_name(nm)
        if not key:
            continue
        tmp.setdefault(key, []).append(cid)

    name2id: Dict[str, str] = {}
    for k, ids in tmp.items():
        ids_sorted = sorted(set(ids))
        name2id[k] = ids_sorted[0]
    return name2id


# -------------------------
# Core loaders
# -------------------------


def load_mm_gold(data_root: Path, dataset: str) -> Tuple[dict, dict]:
    all_path = data_root / f"mm-{dataset}" / "ori" / "all_wo_mention.json"
    split_path = data_root / f"mm-{dataset}" / "ori" / "split_list.json"
    all_data = load_json(all_path)
    split_list = load_json(split_path)
    if not isinstance(all_data, dict) or not isinstance(split_list, dict):
        raise ValueError("all_wo_mention.json and split_list.json must both be JSON dicts")
    return all_data, split_list


def load_bm25_concept_candidates(path: Path) -> Dict[str, List[str]]:
    """Load BM25 candidates as concept_id -> [candidate_name,...], sorted by score desc.

    Input JSONL line format:
    {
      "doc_id": "<concept_id>",
      "candidates": [{"concept_id":..., "name":..., "score":...}, ...]
    }
    """
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"--bm25-jsonl must be .jsonl: {path}")

    out: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            cid = obj.get("doc_id", "")
            if not isinstance(cid, str) or not cid:
                continue

            candidates = obj.get("candidates", [])
            if not isinstance(candidates, list):
                candidates = []

            pairs: List[Tuple[str, float]] = []
            for c in candidates:
                if not isinstance(c, dict):
                    continue
                name = c.get("name", "")
                if not isinstance(name, str):
                    name = str(name)
                score = c.get("score", 0.0)
                try:
                    score_f = float(score)
                except Exception:
                    score_f = 0.0
                if name:
                    pairs.append((name, score_f))

            pairs.sort(key=lambda x: x[1], reverse=True)
            out[cid] = [name for name, _ in pairs]

    return out


# -------------------------
# Extract FP from predictions
# -------------------------


@dataclass(frozen=True)
class ConceptPair:
    cid: str
    name: str


def _as_pair(obj: Any) -> Optional[ConceptPair]:
    if isinstance(obj, dict):
        cid = obj.get("id", "")
        name = obj.get("name", "")
        cid = str(cid) if cid is not None else ""
        name = str(name) if name is not None else ""
        if cid:
            return ConceptPair(cid=cid, name=name)
        return None
    if isinstance(obj, (list, tuple)) and len(obj) >= 1:
        cid = str(obj[0]) if obj[0] is not None else ""
        name = str(obj[1]) if len(obj) > 1 and obj[1] is not None else ""
        if cid:
            return ConceptPair(cid=cid, name=name)
    return None


def extract_fp_map_from_pred_json(pred_json: Dict[str, Any]) -> Dict[str, List[List[str]]]:
    """Return fp_map[doc_id] = [[id, name], ...] where fp = pred - gold by concept id."""
    fp_map: Dict[str, List[List[str]]] = {}

    for doc_id, rec in pred_json.items():
        if not isinstance(rec, dict):
            continue

        gold_raw = rec.get("gold", [])
        pred_raw = rec.get("pred", [])
        if not isinstance(gold_raw, list):
            gold_raw = []
        if not isinstance(pred_raw, list):
            pred_raw = []

        gold_ids = {p.cid for p in (_as_pair(x) for x in gold_raw) if p is not None and p.cid}

        seen: set[str] = set()
        fps: List[List[str]] = []
        for it in pred_raw:
            p = _as_pair(it)
            if p is None or not p.cid:
                continue
            if p.cid in gold_ids:
                continue
            key = f"{p.cid}\t{p.name}"
            if key in seen:
                continue
            seen.add(key)
            fps.append([p.cid, p.name])

        fp_map[str(doc_id)] = fps

    return fp_map


def default_pred_file(pred_dir: Path, split: str, variant: str) -> Path:
    if variant not in {"beam10", "thresholded", "top100"}:
        raise ValueError(f"Unknown --variant: {variant}")
    return pred_dir / f"{split}_{variant}.json"


# -------------------------
# Build cross-encoder training JSONL
# -------------------------


def _dedup_shuffle(names: List[str], seed: int) -> List[str]:
    uniq = list(dict.fromkeys([n for n in names if isinstance(n, str) and n.strip()]))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    return uniq


def build_train_gold_bm25(
        all_data: dict,
        split_list: dict,
        dataset: str,
        bm25_map: Dict[str, List[str]],
        pool_size: int,
        out_path: Path,
        seed: int,
) -> Tuple[int, float]:
    """Write train JSONL with negatives from BM25 concept candidates."""
    train_ids = set(split_list.get("train", []))
    hard_counts: List[int] = []

    def iter_records() -> Iterable[dict]:
        for doc_id, doc in all_data.items():
            if doc_id not in train_ids:
                continue
            passage = doc.get("passage", "")
            concepts = doc.get("concepts", [])
            if not isinstance(passage, str) or not passage:
                continue
            if not isinstance(concepts, list) or not concepts:
                continue

            pos_names = [c.get("name", "") for c in concepts if isinstance(c, dict) and c.get("name")]
            pos_set = set(pos_names)

            for c in concepts:
                if not isinstance(c, dict):
                    continue
                pos_id = str(c.get("id", ""))
                pos_name = c.get("name", "")
                if not isinstance(pos_name, str) or not pos_name:
                    continue

                cand_names = bm25_map.get(pos_id, [])
                # exclude itself + other positives in same passage
                negs = [n for n in cand_names if n and n != pos_name and n not in pos_set][: max(0, int(pool_size))]
                negs = _dedup_shuffle(negs, seed=seed + hash((doc_id, pos_id)) % 10_000_000)

                hard_counts.append(len(negs))
                yield {
                    "query_text": passage,
                    "pos_term_text": pos_name,
                    "hard_neg_term_texts": negs,
                }

    write_jsonl(out_path, iter_records())
    avg = float(sum(hard_counts) / max(1, len(hard_counts)))
    return len(hard_counts), avg


def build_train_fp_only(
        all_data: dict,
        split_list: dict,
        fp_map: Dict[str, List[List[str]]],
        out_path: Path,
        seed: int,
) -> Tuple[int, float]:
    """Write train JSONL where negatives are FPs for that doc/passage."""
    train_ids = set(split_list.get("train", []))
    hard_counts: List[int] = []

    def iter_records() -> Iterable[dict]:
        for doc_id, doc in all_data.items():
            if doc_id not in train_ids:
                continue
            passage = doc.get("passage", "")
            concepts = doc.get("concepts", [])
            if not isinstance(passage, str) or not passage:
                continue
            if not isinstance(concepts, list) or not concepts:
                continue

            pos_names = [c.get("name", "") for c in concepts if isinstance(c, dict) and c.get("name")]
            pos_set = set(pos_names)

            fps = fp_map.get(str(doc_id), [])
            fp_names = [p[1] for p in fps if isinstance(p, list) and len(p) >= 2 and isinstance(p[1], str)]
            fp_names = [n for n in fp_names if n and n not in pos_set]
            fp_names = list(dict.fromkeys(fp_names))

            for c in concepts:
                if not isinstance(c, dict):
                    continue
                pos_name = c.get("name", "")
                if not isinstance(pos_name, str) or not pos_name:
                    continue

                negs = _dedup_shuffle(fp_names, seed=seed + hash((doc_id, pos_name)) % 10_000_000)
                hard_counts.append(len(negs))
                yield {
                    "query_text": passage,
                    "pos_term_text": pos_name,
                    "hard_neg_term_texts": negs,
                }

    write_jsonl(out_path, iter_records())
    avg = float(sum(hard_counts) / max(1, len(hard_counts)))
    return len(hard_counts), avg


def build_train_mix_fp_plus_bm25(
        all_data: dict,
        split_list: dict,
        fp_map: Dict[str, List[List[str]]],
        bm25_map: Dict[str, List[str]],
        pool_size: int,
        out_path: Path,
        seed: int,
) -> Tuple[int, float]:
    """Write train JSONL where negatives = FP(doc) + BM25(concept, top pool_size)."""
    train_ids = set(split_list.get("train", []))
    hard_counts: List[int] = []

    def iter_records() -> Iterable[dict]:
        for doc_id, doc in all_data.items():
            if doc_id not in train_ids:
                continue
            passage = doc.get("passage", "")
            concepts = doc.get("concepts", [])
            if not isinstance(passage, str) or not passage:
                continue
            if not isinstance(concepts, list) or not concepts:
                continue

            pos_names = [c.get("name", "") for c in concepts if isinstance(c, dict) and c.get("name")]
            pos_set = set(pos_names)

            fps = fp_map.get(str(doc_id), [])
            fp_names = [p[1] for p in fps if isinstance(p, list) and len(p) >= 2 and isinstance(p[1], str)]
            fp_names = [n for n in fp_names if n and n not in pos_set]
            fp_names = list(dict.fromkeys(fp_names))

            for c in concepts:
                if not isinstance(c, dict):
                    continue
                pos_id = str(c.get("id", ""))
                pos_name = c.get("name", "")
                if not isinstance(pos_name, str) or not pos_name:
                    continue

                cand_names = bm25_map.get(pos_id, [])
                bm25_negs = [n for n in cand_names if n and n != pos_name and n not in pos_set][
                            : max(0, int(pool_size))]

                negs = fp_names + bm25_negs
                negs = _dedup_shuffle(negs, seed=seed + hash((doc_id, pos_id)) % 10_000_000)

                hard_counts.append(len(negs))
                yield {
                    "query_text": passage,
                    "pos_term_text": pos_name,
                    "hard_neg_term_texts": negs,
                }

    write_jsonl(out_path, iter_records())
    avg = float(sum(hard_counts) / max(1, len(hard_counts)))
    return len(hard_counts), avg


# -------------------------
# Merge FP maps
# -------------------------


def merge_fp_maps(fp_maps: List[Dict[str, List[List[str]]]]) -> Dict[str, List[List[str]]]:
    merged: Dict[str, List[List[str]]] = {}
    for m in fp_maps:
        for doc_id, items in m.items():
            if doc_id not in merged:
                merged[doc_id] = []
            merged[doc_id].extend(items)

    # dedup per doc by (id,name)
    for doc_id, items in merged.items():
        seen: set[str] = set()
        uniq: List[List[str]] = []
        for it in items:
            if not isinstance(it, list) or len(it) < 2:
                continue
            cid, name = str(it[0]), str(it[1])
            key = f"{cid}\t{name}"
            if key in seen:
                continue
            seen.add(key)
            uniq.append([cid, name])
        merged[doc_id] = uniq

    return merged


# -------------------------
# Distribution-matched mapping (optional)
# -------------------------


def _extract_scored_pred_pool(pred_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return doc_id -> {"gold_ids": set[str], "pred": List[Tuple[id,name,score]]} for scored preds."""
    out: Dict[str, Dict[str, Any]] = {}
    for doc_id, rec in pred_json.items():
        if not isinstance(rec, dict):
            continue
        gold_raw = rec.get("gold", [])
        pred_raw = rec.get("pred", [])
        if not isinstance(gold_raw, list):
            gold_raw = []
        if not isinstance(pred_raw, list):
            pred_raw = []

        gold_ids = {p.cid for p in (_as_pair(x) for x in gold_raw) if p is not None and p.cid}

        scored: List[Tuple[str, str, float]] = []
        for it in pred_raw:
            if not isinstance(it, dict):
                continue
            cid = it.get("id", "")
            name = it.get("name", "")
            score = it.get("score", None)
            cid = str(cid) if cid is not None else ""
            name = str(name) if name is not None else ""
            if not cid:
                continue
            if score is None:
                continue
            try:
                score_f = float(score)
            except Exception:
                continue
            scored.append((cid, name, score_f))

        out[str(doc_id)] = {"gold_ids": gold_ids, "pred": scored}
    return out


def map_scored_preds_to_macoir_fpcount(
        macoir_fp: Dict[str, List[List[str]]],
        scored_pred_json: Dict[str, Any],
        out_fp_json: Path,
        tol_per_doc: float = 1 / 80,
        max_thr_candidates: int = 4096,
) -> None:
    """Distribution-matched mapping: choose threshold on scored preds so FP count ~= MA-COIR beam10 FP count.

    Key fix: SAFE ALIGNMENT by doc_id intersection.
      - If macoir_fp contains train/dev/test but scored_pred_json contains train only,
        we compute target + search threshold ONLY on intersection(doc_ids).
      - Output fp json is also only for those intersection doc_ids (safe + correct).

    macoir_fp:
      doc_id -> [[id, name], ...]   # only used for per-doc FP count target
    scored_pred_json:
      doc-level prediction json with score in pred items (only scored model)
    """

    scored = _extract_scored_pred_pool(scored_pred_json)  # doc_id -> {"gold_ids":..., "pred":[(cid,name,score),...]}

    macoir_keys = set(str(k) for k in macoir_fp.keys())
    scored_keys = set(str(k) for k in scored.keys())
    doc_ids = sorted(macoir_keys & scored_keys)

    print(
        f"[map-to-macoir] macoir_docs={len(macoir_keys)} scored_docs={len(scored_keys)} "
        f"intersection={len(doc_ids)} dropped_macoir={len(macoir_keys) - len(doc_ids)} "
        f"dropped_scored={len(scored_keys) - len(doc_ids)}"
    )

    if not doc_ids:
        raise ValueError("Empty doc_id intersection between macoir_fp and scored_pred_json. Cannot do safe mapping.")

    # Target FP total ONLY on intersection
    target_fp_total = sum(len(macoir_fp[d]) for d in doc_ids if d in macoir_fp)
    if target_fp_total == 0:
        # If target is 0, set threshold to +inf so we always output empty FP lists.
        out_map = {d: [] for d in doc_ids}
        write_json(out_fp_json, out_map)
        print(f"[map-to-macoir] target_fp_total=0 => wrote empty fp map for {len(doc_ids)} docs: {out_fp_json}")
        return

    # Collect candidate thresholds from scored preds ONLY on intersection,
    # using FP-candidate scores (i.e., excluding gold ids) makes the search tighter/cleaner.
    all_scores: List[float] = []
    for d in doc_ids:
        dd = scored.get(d)
        if not dd:
            continue
        gold_ids = set(dd.get("gold_ids") or [])
        for (cid, _, s) in dd.get("pred", []):
            try:
                sf = float(s)
            except Exception:
                continue
            if cid and cid not in gold_ids:
                all_scores.append(sf)

    if not all_scores:
        raise ValueError("No scored FP-candidate scores found on doc_id intersection. Cannot map by threshold.")

    uniq = sorted(set(all_scores))
    if len(uniq) > max_thr_candidates:
        # quantile downsample (monotonic property still holds well)
        import numpy as np

        arr = np.asarray(uniq, dtype="float32")
        qs = np.linspace(0.0, 1.0, max_thr_candidates)
        uniq = sorted(set(float(x) for x in np.quantile(arr, qs)))

    # allowable absolute error (keep your original semantics but in ints)
    n_docs = len(doc_ids)
    tol = max(1, int(round(n_docs * float(tol_per_doc))))

    def count_fp(thr: float) -> int:
        total = 0
        for d in doc_ids:
            dd = scored.get(d)
            if not dd:
                continue
            gold_ids = set(dd.get("gold_ids") or [])
            for cid, _, s in dd.get("pred", []):
                try:
                    sf = float(s)
                except Exception:
                    continue
                if not cid or cid in gold_ids:
                    continue
                if sf >= thr:
                    total += 1
        return total

    # Binary search on threshold over uniq scores (monotonic: higher thr => fewer preds)
    lo, hi = 0, len(uniq) - 1
    best_thr = uniq[hi]
    best_diff = float("inf")
    best_fp = None

    while lo <= hi:
        mid = (lo + hi) // 2
        thr = float(uniq[mid])
        fp_cnt = count_fp(thr)
        diff = abs(fp_cnt - target_fp_total)

        if diff < best_diff or (math.isclose(diff, best_diff) and thr > best_thr):
            best_diff = diff
            best_thr = thr
            best_fp = fp_cnt

        if diff <= tol:
            break

        # too many preds => raise threshold (need fewer fp)
        if fp_cnt > target_fp_total:
            lo = mid + 1
        else:
            hi = mid - 1

    # Build mapped FP json ONLY for intersection keys
    out_map: Dict[str, List[List[str]]] = {}
    for d in doc_ids:
        dd = scored.get(d)
        if not dd:
            out_map[d] = []
            continue

        gold_ids = set(dd.get("gold_ids") or [])
        fps: List[List[str]] = []
        seen: set[Tuple[str, str]] = set()

        for cid, name, s in dd.get("pred", []):
            try:
                sf = float(s)
            except Exception:
                continue
            if sf < best_thr:
                continue
            if not cid or cid in gold_ids:
                continue

            nm = name if isinstance(name, str) else str(name)
            key = (str(cid), nm)
            if key in seen:
                continue
            seen.add(key)
            fps.append([str(cid), nm])

        out_map[d] = fps

    write_json(out_fp_json, out_map)

    mapped_fp_total = sum(len(v) for v in out_map.values())
    print(
        f"[map-to-macoir] target_fp_total={target_fp_total} mapped_fp_total={mapped_fp_total} "
        f"gap={abs(mapped_fp_total - target_fp_total)} tol={tol} best_thr={best_thr:.6f} "
        f"candidates={len(uniq)} docs={len(doc_ids)}"
    )
    if best_fp is not None and best_fp != mapped_fp_total:
        # this should usually match; if not, it indicates duplicates removal changed counts slightly
        print(f"[map-to-macoir] note: counted_fp_at_thr={best_fp} but dedupbed_mapped_fp_total={mapped_fp_total}")

    print(f"[map-to-macoir] wrote: {out_fp_json}")


# -------------------------
# Stats
# -------------------------


def avg_hard_per_passage(train_jsonl: Path) -> float:
    """Compute avg unique hard negatives per passage from a train jsonl."""
    rows = load_jsonl(train_jsonl)
    d2hard: Dict[str, List[str]] = {}
    for r in rows:
        q = r.get("query_text", "")
        h = r.get("hard_neg_term_texts", [])
        if not isinstance(q, str) or not q:
            continue
        if not isinstance(h, list):
            h = []
        d2hard.setdefault(q, []).extend([x for x in h if isinstance(x, str) and x.strip()])

    vals: List[int] = []
    for q, hs in d2hard.items():
        vals.append(len(set(hs)))
    return float(sum(vals) / max(1, len(vals)))


# -------------------------
# Attach IDs to prediction JSONL
# -------------------------

def cmd_attach_ids_preds(args: argparse.Namespace) -> None:
    """Add term_id for each prediction item in topK prediction JSONL."""
    rows = load_jsonl(Path(args.in_pred_jsonl))
    id2name = load_concept_catalog_id2name(Path(args.concept_catalog_json))
    name2id = build_name2id(id2name)

    out_rows: List[dict] = []
    total_items = 0
    filled = 0
    missing = 0

    for r in rows:
        if not isinstance(r, dict):
            continue
        items = r.get("items", [])
        if not isinstance(items, list):
            items = []

        new_items: List[dict] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            total_items += 1
            term_text = it.get("term_text", "")
            term_id = it.get("term_id", None)

            # only fill if missing/empty
            if term_id is None or str(term_id).strip() == "":
                cid = name2id.get(norm_name(term_text), "")
                if cid:
                    it2 = dict(it)
                    it2["term_id"] = cid
                    new_items.append(it2)
                    filled += 1
                else:
                    it2 = dict(it)
                    it2["term_id"] = ""  # explicit empty
                    new_items.append(it2)
                    missing += 1
            else:
                new_items.append(dict(it))

        r2 = dict(r)
        r2["items"] = new_items
        out_rows.append(r2)

    write_jsonl(Path(args.out_pred_jsonl), out_rows)
    print(
        f"[attach-ids-preds] wrote: {args.out_pred_jsonl}\n"
        f"  items: total={total_items} filled={filled} still_missing={missing} "
        f"miss_rate={missing / max(1, total_items):.4f}"
    )


# -------------------------
# CLI commands
# -------------------------


def cmd_extract_fp(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    out_dir = data_root / f"mm-{args.dataset}" / "mld" / "fp"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = Path(args.pred_dir)
    tag = args.tag or f"{pred_dir.name}_{args.variant}"
    out_path = out_dir / f"fp_{tag}.json"

    merged: Dict[str, List[List[str]]] = {}
    for split in ("train", "dev", "test"):
        pred_path = Path(args.pred_file) if args.pred_file else default_pred_file(pred_dir, split, args.variant)
        if not pred_path.exists():
            if args.allow_missing_splits:
                continue
            raise SystemExit(f"Prediction file not found: {pred_path}")

        pred_json = load_json(pred_path)
        if not isinstance(pred_json, dict):
            raise SystemExit(f"Prediction JSON must be dict: {pred_path}")

        fp_map = extract_fp_map_from_pred_json(pred_json)
        merged.update(fp_map)

        docs = len(fp_map)
        fp_total = sum(len(v) for v in fp_map.values())
        print(f"[{split}] docs={docs} fp_total={fp_total} avg_fp={fp_total / max(1, docs):.2f} ({pred_path.name})")

    write_json(out_path, merged)
    print(f"[extract-fp] wrote: {out_path}")


def cmd_merge_fp(args: argparse.Namespace) -> None:
    maps: List[Dict[str, List[List[str]]]] = []
    for p in args.inputs:
        obj = load_json(Path(p))
        if not isinstance(obj, dict):
            raise SystemExit(f"FP json must be dict: {p}")
        maps.append(obj)
    merged = merge_fp_maps(maps)
    write_json(Path(args.out), merged)
    docs = len(merged)
    fp_total = sum(len(v) for v in merged.values())
    print(f"[merge-fp] docs={docs} fp_total={fp_total} avg_fp={fp_total / max(1, docs):.2f}")
    print(f"[merge-fp] wrote: {args.out}")


def cmd_build_gold_bm25(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    all_data, split_list = load_mm_gold(data_root, args.dataset)
    bm25_map = load_bm25_concept_candidates(Path(args.bm25_jsonl))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pool in args.pool_sizes:
        out_path = out_dir / f"train_{args.tag}_{pool}.jsonl"
        n, avg = build_train_gold_bm25(
            all_data=all_data,
            split_list=split_list,
            dataset=args.dataset,
            bm25_map=bm25_map,
            pool_size=int(pool),
            out_path=out_path,
            seed=args.seed,
        )
        print(f"[gold-bm25] pool={pool} records={n} avg_hard={avg:.2f} -> {out_path}")


def cmd_build_fp_only(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    all_data, split_list = load_mm_gold(data_root, args.dataset)
    fp_map = load_json(Path(args.fp_json))
    if not isinstance(fp_map, dict):
        raise SystemExit("--fp-json must be a dict")
    out_path = Path(args.out)
    n, avg = build_train_fp_only(all_data, split_list, fp_map, out_path=out_path, seed=args.seed)
    print(f"[fp-only] records={n} avg_hard={avg:.2f} -> {out_path}")


def cmd_build_mix(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    all_data, split_list = load_mm_gold(data_root, args.dataset)

    fp_map = load_json(Path(args.fp_json))
    if not isinstance(fp_map, dict):
        raise SystemExit("--fp-json must be a dict")
    bm25_map = load_bm25_concept_candidates(Path(args.bm25_jsonl))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pool in args.pool_sizes:
        out_path = out_dir / f"train_{args.tag}_gold_bm25_{pool}.jsonl"
        n, avg = build_train_mix_fp_plus_bm25(
            all_data=all_data,
            split_list=split_list,
            fp_map=fp_map,
            bm25_map=bm25_map,
            pool_size=int(pool),
            out_path=out_path,
            seed=args.seed,
        )
        print(f"[mix] pool={pool} records={n} avg_hard={avg:.2f} -> {out_path}")


def cmd_stats_hard(args: argparse.Namespace) -> None:
    v = avg_hard_per_passage(Path(args.train_jsonl))
    print(f"[stats] avg_unique_hard_per_passage={v:.4f}  ({args.train_jsonl})")


def cmd_map_to_macoir(args: argparse.Namespace) -> None:
    macoir_fp = load_json(Path(args.macoir_fp_json))
    if not isinstance(macoir_fp, dict):
        raise SystemExit("--macoir-fp-json must be dict")

    scored_pred = load_json(Path(args.scored_pred_json))
    if not isinstance(scored_pred, dict):
        raise SystemExit("--scored-pred-json must be dict")

    out_fp = Path(args.out_fp_json)
    map_scored_preds_to_macoir_fpcount(
        macoir_fp=macoir_fp,
        scored_pred_json=scored_pred,
        out_fp_json=out_fp,
        tol_per_doc=float(args.tol_per_doc),
    )


# -------------------------
# Parser
# -------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare cross-encoder training data / FP utilities.")
    p.add_argument("--dataset", required=True, choices=["go", "hpo"])
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--seed", type=int, default=42)

    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("extract-fp", help="Extract FP map from prediction outputs (doc-level JSON).")
    s.add_argument("--pred-dir", required=True, help="Folder containing {split}_{variant}.json")
    s.add_argument("--variant", required=True, choices=["beam10", "thresholded", "top100"])
    s.add_argument("--pred-file", default="", help="Optional explicit file path (overrides default split naming)")
    s.add_argument("--tag", default="", help="Output tag used in fp_<tag>.json; default: <pred_dir>_<variant>")
    s.add_argument("--allow-missing-splits", action="store_true")
    s.set_defaults(func=cmd_extract_fp)

    s = sub.add_parser("merge-fp", help="Merge multiple FP maps (fp_*.json) into one (dedup).")
    s.add_argument("--inputs", nargs="+", required=True, help="List of fp json files to merge")
    s.add_argument("--out", required=True, help="Output fp json")
    s.set_defaults(func=cmd_merge_fp)

    s = sub.add_parser("build-gold-bm25", help="Build gold-derived BM25 negatives training jsonl(s).")
    s.add_argument("--bm25-jsonl", required=True, help="BM25 hard negative jsonl (concept-level)")
    s.add_argument("--pool-sizes", nargs="+", type=int, required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--tag", default="gold_bm25")
    s.set_defaults(func=cmd_build_gold_bm25)

    s = sub.add_parser("build-fp-only", help="Build FP-only training jsonl.")
    s.add_argument("--fp-json", required=True, help="FP map json (doc_id -> [[id,name],...])")
    s.add_argument("--out", required=True, help="Output train jsonl path")
    s.set_defaults(func=cmd_build_fp_only)

    s = sub.add_parser("build-mix", help="Build training jsonl(s) with FP + gold(BM25) mix.")
    s.add_argument("--fp-json", required=True)
    s.add_argument("--bm25-jsonl", required=True)
    s.add_argument("--pool-sizes", nargs="+", type=int, required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--tag", required=True, help="Name prefix, e.g. macoir_be or macoir_xrt_be")
    s.set_defaults(func=cmd_build_mix)

    s = sub.add_parser("stats-hard", help="Avg unique hard negatives per passage for a train jsonl.")
    s.add_argument("--train-jsonl", required=True)
    s.set_defaults(func=cmd_stats_hard)

    s = sub.add_parser(
        "map-to-macoir-by-fpcount",
        help="Distribution-matched thresholding: map scored preds to match MA-COIR beam10 FP count.",
    )
    s.add_argument("--macoir-fp-json", required=True, help="FP json from MA-COIR beam10 (used for counts)")
    s.add_argument("--scored-pred-json", required=True, help="Doc-level prediction json where pred items include score")
    s.add_argument("--out-fp-json", required=True, help="Output FP json after thresholding to match counts")
    s.add_argument("--tol-per-doc", type=float, default=1 / 80, help="Tolerance as avg FP error per doc")
    s.set_defaults(func=cmd_map_to_macoir)

    s = sub.add_parser(
        "attach-ids-preds",
        help="Add missing item.term_id to prediction JSONL using concept catalog (postprocess after CE predict).",
    )
    s.add_argument("--in-pred-jsonl", required=True, help="Input CE prediction JSONL (query_text + items[])")
    s.add_argument("--out-pred-jsonl", required=True, help="Output prediction JSONL with term_id filled")
    s.add_argument("--concept-catalog-json", required=True, help="Ontology catalog (json/jsonl) with id+name")
    s.set_defaults(func=cmd_attach_ids_preds)

    return p


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    args.func(args)


if __name__ == "__main__":
    main()
