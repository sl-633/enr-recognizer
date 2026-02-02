#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-encoder (sequence classification) prediction + ranker-style eval (dev threshold -> test apply).

This script mirrors the bi-encoder workflow:
  1) predict: score rerank-style candidates and export top-K predictions JSONL (no eval)
  2) eval:    take dev/test prediction JSONL, tune best threshold on dev (set-micro F1), apply on test,
              and export per-split doc_id->{"gold":[...],"pred":[...]} JSONs.

Input rerank JSONL (one line per query):
{
  "query_text": str,
  "candidate_term_texts": [str, ...],
  "gold_term_texts": [str, ...],
  // optional (recommended):
  "candidate_term_ids": [str/int, ...],
  "gold_term_ids": [str/int, ...],
  // optional query identifiers (copied through):
  "doc_id": str/int, "query_id": str/int, "id": str/int
}

Prediction JSONL output (one line per query):
{
  "query_text": "...",
  "items": [
    {"rank": 0, "score": 0.12, "term_id": "...", "term_text": "..."},
    ...
  ],
  // optional identifiers copied through
}

Eval exports (JSON):
- dev_topK.json / test_topK.json              (topK selection, no threshold)
- dev_thresholded.json / test_thresholded.json (threshold selection using dev best thr)
- metrics_thresholded.json                    (dev tuned thr + dev/test set-micro P/R/F1)

Notes:
- Cross-encoder scores are raw logits (float). Threshold tuning uses these scores.
- Set-micro P/R/F1 is computed on concept IDs when ids are available; otherwise normalized text.
- Safe alignment: by doc_id if present, else by query_text mapping from gold-json + split-list-json.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


# -------------------------
# IO helpers
# -------------------------


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# Utils
# -------------------------


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_text(s: Any) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = " ".join(s.strip().split())
    return s.lower()


def chunked(seq: Sequence[Any], n: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# -------------------------
# Data parsing (rerank-style input)
# -------------------------


@dataclass(frozen=True)
class RerankExample:
    query_text: str
    candidate_texts: List[str]
    gold_texts: List[str]
    candidate_ids: Optional[List[str]] = None
    gold_ids: Optional[List[str]] = None
    meta: Optional[dict] = None  # doc_id/query_id passthrough


def _as_str_list(x: Any) -> List[str]:
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for it in x:
        if it is None:
            continue
        s = str(it).strip()
        if s:
            out.append(s)
    return out


def load_rerank_jsonl(path: Path, eval_topk: int) -> List[RerankExample]:
    rows = read_jsonl(path)
    out: List[RerankExample] = []
    seen_q = set()

    for r in rows:
        if not isinstance(r, dict):
            continue
        q = r.get("query_text", "")
        if not isinstance(q, str) or not q.strip():
            continue

        # de-dup by query_text (consistent with your other scripts)
        if q in seen_q:
            continue
        seen_q.add(q)

        c_texts = _as_str_list(r.get("candidate_term_texts", []))
        g_texts = _as_str_list(r.get("gold_term_texts", []))

        c_ids = r.get("candidate_term_ids", None)
        g_ids = r.get("gold_term_ids", None)
        c_ids_list = _as_str_list(c_ids) if c_ids is not None else None
        g_ids_list = _as_str_list(g_ids) if g_ids is not None else None

        if eval_topk and eval_topk > 0:
            c_texts = c_texts[:eval_topk]
            if c_ids_list is not None:
                c_ids_list = c_ids_list[:eval_topk]

        if not c_texts:
            continue

        meta = {}
        for k in ("doc_id", "query_id", "id"):
            if k in r:
                meta[k] = r[k]

        out.append(
            RerankExample(
                query_text=q,
                candidate_texts=c_texts,
                gold_texts=g_texts,
                candidate_ids=c_ids_list,
                gold_ids=g_ids_list,
                meta=meta if meta else None,
            )
        )
    return out


# -------------------------
# Prediction (cross-encoder scoring)
# -------------------------


@torch.no_grad()
def score_query_candidates(
    tokenizer,
    model,
    query_text: str,
    cand_texts: List[str],
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return scores (raw logits) aligned to cand_texts."""
    scores: List[np.ndarray] = []
    model.eval()

    for chunk in chunked(cand_texts, batch_size):
        enc = tokenizer(
            list(chunk),
            [query_text] * len(chunk),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        logits = model(**enc).logits.squeeze(-1)
        scores.append(logits.detach().cpu().numpy())

    return np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=np.float32)


def export_topk_predictions(
    examples: List[RerankExample],
    tokenizer,
    model,
    out_jsonl: Path,
    pred_topk: int,
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> None:
    def gen():
        for ex in tqdm(examples, desc="Predict"):
            scores = score_query_candidates(
                tokenizer, model, ex.query_text, ex.candidate_texts, max_len, batch_size, device
            )
            n = min(len(ex.candidate_texts), len(scores))
            scores = scores[:n]
            c_texts = ex.candidate_texts[:n]
            c_ids = ex.candidate_ids[:n] if ex.candidate_ids is not None else None

            order = np.argsort(-scores)
            k = min(int(pred_topk), n) if pred_topk and pred_topk > 0 else n

            items: List[dict] = []
            for rank, idx in enumerate(order[:k].tolist()):
                rec = {
                    "rank": int(rank),  # 0-based
                    "score": float(scores[idx]),
                    "term_text": c_texts[idx],
                    "term_id": (c_ids[idx] if c_ids is not None else None),
                }
                items.append(rec)

            out = {"query_text": ex.query_text, "items": items}
            if ex.meta:
                out.update(ex.meta)
            yield out

    write_jsonl(out_jsonl, gen())
    LOGGER.info("[Write] %s", out_jsonl)


# -------------------------
# Eval (ranker_top100_eval style)
# -------------------------


@dataclass(frozen=True)
class PredItem:
    term_id: str
    score: float
    term_text: str


@dataclass(frozen=True)
class PredExample:
    query_text: str
    items: List[PredItem]
    meta: Optional[dict] = None


@dataclass(frozen=True)
class GoldConcept:
    cid: str
    name: str


@dataclass(frozen=True)
class GoldExample:
    doc_id: str
    query_text: str
    concepts: List[GoldConcept]


def parse_predictions_jsonl(path: Path) -> List[PredExample]:
    rows = read_jsonl(path)
    out: List[PredExample] = []
    seen_q = set()

    for r in rows:
        if not isinstance(r, dict):
            continue
        q = r.get("query_text", "")
        if not isinstance(q, str) or not q:
            continue
        if q in seen_q:
            continue
        seen_q.add(q)

        items_raw = r.get("items", [])
        if not isinstance(items_raw, list):
            items_raw = []

        items: List[PredItem] = []
        for it in items_raw:
            if not isinstance(it, dict):
                continue
            cid = it.get("term_id", "")
            if cid is None:
                cid = ""
            cid = str(cid)

            name = it.get("term_text", "")
            if not isinstance(name, str):
                name = str(name)

            sc = it.get("score", 0.0)
            try:
                sc_f = float(sc)
            except Exception:
                sc_f = 0.0

            items.append(PredItem(term_id=cid, term_text=name, score=sc_f))

        meta = {}
        for k in ("doc_id", "query_id", "id"):
            if k in r:
                meta[k] = r[k]
        out.append(PredExample(query_text=q, items=items, meta=meta if meta else None))
    return out


def build_gold_examples(all_wo_mention: dict, split_list: dict, split: str) -> Dict[str, GoldExample]:
    doc_ids = split_list.get(split, [])
    if not isinstance(doc_ids, list):
        doc_ids = []

    out: Dict[str, GoldExample] = {}
    for doc_id in doc_ids:
        doc = all_wo_mention.get(str(doc_id))
        if not isinstance(doc, dict):
            continue
        passage = doc.get("passage", "")
        concepts = doc.get("concepts", [])
        if not isinstance(passage, str) or not passage:
            continue
        if not isinstance(concepts, list):
            concepts = []

        seen: set[str] = set()
        gold: List[GoldConcept] = []
        for c in concepts:
            if not isinstance(c, dict):
                continue
            cid = c.get("id", "")
            if cid is None:
                continue
            cid = str(cid)
            if not cid or cid in seen:
                continue
            seen.add(cid)
            name = c.get("name", "")
            if not isinstance(name, str):
                name = str(name)
            gold.append(GoldConcept(cid=cid, name=name))

        out[str(doc_id)] = GoldExample(doc_id=str(doc_id), query_text=passage, concepts=gold)
    return out


def index_gold_by_query(gold_by_doc: Dict[str, GoldExample]) -> Dict[str, str]:
    return {g.query_text: doc_id for doc_id, g in gold_by_doc.items()}


def set_micro_prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def select_items(ex: PredExample, mode: str, topk: int, threshold: float) -> List[PredItem]:
    pool = ex.items[: max(0, int(topk))]
    if mode == "topk":
        return pool
    if mode == "thresholded":
        return [it for it in pool if it.score >= float(threshold)]
    raise ValueError(f"Unknown mode: {mode}")


def _doc_id_for_pred(
    ex: PredExample,
    query_to_doc: Dict[str, str],
) -> Optional[str]:
    # prefer doc_id in prediction if present
    if ex.meta and "doc_id" in ex.meta:
        did = ex.meta["doc_id"]
        if did is not None:
            return str(did)
    return query_to_doc.get(ex.query_text)


def compute_counts(
    preds: List[PredExample],
    gold_by_doc: Dict[str, GoldExample],
    query_to_doc: Dict[str, str],
    mode: str,
    topk: int,
    threshold: float,
) -> Tuple[int, int, int]:
    tp = fp = fn = 0

    for ex in preds:
        doc_id = _doc_id_for_pred(ex, query_to_doc)
        if doc_id is None:
            continue
        gold = gold_by_doc.get(doc_id)
        if gold is None:
            continue

        gold_ids = {c.cid for c in gold.concepts if c.cid}
        gold_texts = {norm_text(c.name) for c in gold.concepts if c.name}

        picked = select_items(ex, mode, topk, threshold)
        # id-first if at least one id is non-empty
        has_ids = any(it.term_id for it in picked) and bool(gold_ids)

        if has_ids:
            pred_ids = {it.term_id for it in picked if it.term_id}
            tp += len(pred_ids & gold_ids)
            fp += len(pred_ids - gold_ids)
            fn += len(gold_ids - pred_ids)
        else:
            pred_texts = {norm_text(it.term_text) for it in picked if it.term_text}
            tp += len(pred_texts & gold_texts)
            fp += len(pred_texts - gold_texts)
            fn += len(gold_texts - pred_texts)

    return tp, fp, fn


def tune_threshold_on_dev(
    dev_preds: List[PredExample],
    dev_gold_by_doc: Dict[str, GoldExample],
    dev_query_to_doc: Dict[str, str],
    topk: int,
    max_candidates: int,
) -> Tuple[float, Dict[str, float]]:
    scores: List[float] = []
    for ex in dev_preds:
        for it in ex.items[: max(0, int(topk))]:
            scores.append(float(it.score))

    if not scores:
        return 1.0, {"P": 0.0, "R": 0.0, "F1": 0.0}

    arr = np.asarray(scores, dtype=np.float32)
    uniq = np.unique(arr)

    # IMPORTANT: match old behavior (quantile on raw scores, not uniq)
    if uniq.size > max_candidates:
        cand = np.quantile(arr, np.linspace(0.0, 1.0, max_candidates))
        cand = np.unique(cand.astype(np.float32))
    else:
        cand = uniq

    best_thr = None
    best_p = best_r = 0.0
    best_f1 = -1.0

    for thr in cand:
        tp, fp, fn = compute_counts(
            dev_preds, dev_gold_by_doc, dev_query_to_doc,
            mode="thresholded", topk=topk, threshold=float(thr)
        )
        p, r, f1 = set_micro_prf1(tp, fp, fn)

        # IMPORTANT: tie-break matches old script -> choose higher threshold when F1 ties
        if (f1 > best_f1) or (math.isclose(f1, best_f1) and (best_thr is None or float(thr) > best_thr)):
            best_thr = float(thr)
            best_p, best_r, best_f1 = p, r, f1

    return (best_thr if best_thr is not None else 1.0), {"P": best_p, "R": best_r, "F1": best_f1}


def build_split_dump(
    preds: List[PredExample],
    gold_by_doc: Dict[str, GoldExample],
    query_to_doc: Dict[str, str],
    mode: str,
    topk: int,
    threshold: float,
    with_score: bool,
) -> Tuple[Dict[str, dict], Dict[str, List[dict]]]:
    per_doc: Dict[str, dict] = {}
    pred_only: Dict[str, List[dict]] = {}

    for ex in preds:
        doc_id = _doc_id_for_pred(ex, query_to_doc)
        if doc_id is None:
            continue
        gold = gold_by_doc.get(doc_id)
        if gold is None:
            continue

        picked = select_items(ex, mode, topk, threshold)

        pred_recs: List[dict] = []
        for it in picked:
            # always export id+name if present; id can be ""
            rec = {"id": it.term_id, "name": it.term_text}
            if with_score:
                rec["score"] = float(it.score)
            pred_recs.append(rec)

        gold_recs: List[dict] = [{"id": c.cid, "name": c.name} for c in gold.concepts]
        per_doc[doc_id] = {"gold": gold_recs, "pred": pred_recs}
        pred_only[doc_id] = pred_recs

    return per_doc, pred_only


# -------------------------
# CLI
# -------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cross-encoder predict + ranker-style eval.")
    sub = p.add_subparsers(dest="command", required=True)

    # ---- predict ----
    pr = sub.add_parser("predict", help="Score rerank JSONL and export top-K prediction JSONL.")
    pr.add_argument("--model_dir", type=Path, required=True)
    pr.add_argument("--in_jsonl", type=Path, required=True, help="Rerank-style input JSONL (dev/test/train).")
    pr.add_argument("--out_jsonl", type=Path, required=True, help="Top-K prediction JSONL.")
    pr.add_argument("--pred_topk", type=int, default=100)
    pr.add_argument("--max_len", type=int, default=256)
    pr.add_argument("--batch_size", type=int, default=64)
    pr.add_argument("--eval_topk", type=int, default=300, help="Truncate candidate list before scoring.")
    pr.add_argument("--gpu", type=int, default=0)
    pr.add_argument("--seed", type=int, default=42)

    # ---- eval (ranker_top100_eval style) ----
    ev = sub.add_parser("eval", help="Tune threshold on dev predictions and evaluate test predictions.")
    ev.add_argument("--dev_pred_jsonl", type=Path, required=True)
    ev.add_argument("--test_pred_jsonl", type=Path, required=True)
    ev.add_argument("--train_pred_jsonl", type=Path, default=None)

    ev.add_argument("--gold_json", type=Path, required=True, help="all_wo_mention.json (id+name in concepts).")
    ev.add_argument("--split_list_json", type=Path, required=True, help="split_list.json")

    ev.add_argument("--topk", type=int, default=100, help="Only use top-K items for selection.")
    ev.add_argument("--max_thr_candidates", type=int, default=4096)
    ev.add_argument("--with_score", action="store_true", help="Include score in output pred items.")
    ev.add_argument("--out_dir", type=Path, required=True)

    return p


def cmd_predict(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    examples = load_rerank_jsonl(args.in_jsonl, eval_topk=args.eval_topk)
    if not examples:
        raise ValueError("No valid examples in --in_jsonl (needs query_text + non-empty candidate list).")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    export_topk_predictions(
        examples=examples,
        tokenizer=tokenizer,
        model=model,
        out_jsonl=args.out_jsonl,
        pred_topk=args.pred_topk,
        max_len=args.max_len,
        batch_size=args.batch_size,
        device=device,
    )


def cmd_eval(args: argparse.Namespace) -> None:
    all_wo_mention = load_json(args.gold_json)
    split_list = load_json(args.split_list_json)
    if not isinstance(all_wo_mention, dict) or not isinstance(split_list, dict):
        raise ValueError("--gold_json and --split_list_json must be JSON dicts")

    dev_gold_by_doc = build_gold_examples(all_wo_mention, split_list, "dev")
    test_gold_by_doc = build_gold_examples(all_wo_mention, split_list, "test")
    train_gold_by_doc = build_gold_examples(all_wo_mention, split_list, "train") if args.train_pred_jsonl else {}

    dev_query_to_doc = index_gold_by_query(dev_gold_by_doc)
    test_query_to_doc = index_gold_by_query(test_gold_by_doc)
    train_query_to_doc = index_gold_by_query(train_gold_by_doc) if args.train_pred_jsonl else {}

    dev_preds = parse_predictions_jsonl(args.dev_pred_jsonl)
    test_preds = parse_predictions_jsonl(args.test_pred_jsonl)
    train_preds = parse_predictions_jsonl(args.train_pred_jsonl) if args.train_pred_jsonl else []

    # tune threshold on dev (thresholded mode)
    best_thr, best_dev = tune_threshold_on_dev(
        dev_preds,
        dev_gold_by_doc,
        dev_query_to_doc,
        topk=args.topk,
        max_candidates=args.max_thr_candidates,
    )
    LOGGER.info("[DEV] best_thr=%.6f best=%s", best_thr, best_dev)

    # dumps: topk + thresholded
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_topk_dump, dev_topk_pred = build_split_dump(
        dev_preds, dev_gold_by_doc, dev_query_to_doc, mode="topk", topk=args.topk, threshold=best_thr, with_score=args.with_score
    )
    test_topk_dump, test_topk_pred = build_split_dump(
        test_preds, test_gold_by_doc, test_query_to_doc, mode="topk", topk=args.topk, threshold=best_thr, with_score=args.with_score
    )

    dev_th_dump, dev_th_pred = build_split_dump(
        dev_preds, dev_gold_by_doc, dev_query_to_doc, mode="thresholded", topk=args.topk, threshold=best_thr, with_score=args.with_score
    )
    test_th_dump, test_th_pred = build_split_dump(
        test_preds, test_gold_by_doc, test_query_to_doc, mode="thresholded", topk=args.topk, threshold=best_thr, with_score=args.with_score
    )

    train_th_dump: Dict[str, dict] = {}
    train_th_pred: Dict[str, List[dict]] = {}
    if args.train_pred_jsonl:
        train_th_dump, train_th_pred = build_split_dump(
            train_preds,
            train_gold_by_doc,
            train_query_to_doc,
            mode="thresholded",
            topk=args.topk,
            threshold=best_thr,
            with_score=args.with_score,
        )

    # metrics (set-micro on IDs if available else text)
    dev_tp, dev_fp, dev_fn = compute_counts(
        dev_preds, dev_gold_by_doc, dev_query_to_doc, mode="thresholded", topk=args.topk, threshold=best_thr
    )
    test_tp, test_fp, test_fn = compute_counts(
        test_preds, test_gold_by_doc, test_query_to_doc, mode="thresholded", topk=args.topk, threshold=best_thr
    )
    dev_p, dev_r, dev_f1 = set_micro_prf1(dev_tp, dev_fp, dev_fn)
    test_p, test_r, test_f1 = set_micro_prf1(test_tp, test_fp, test_fn)

    # write outputs
    write_json(out_dir / f"dev_top{args.topk}.json", dev_topk_dump)
    write_json(out_dir / f"test_top{args.topk}.json", test_topk_dump)

    write_json(out_dir / "dev_thresholded.json", dev_th_dump)
    write_json(out_dir / "test_thresholded.json", test_th_dump)
    if args.train_pred_jsonl:
        write_json(out_dir / "train_thresholded.json", train_th_dump)

    pred_all: Dict[str, List[dict]] = {}
    pred_all.update(dev_th_pred)
    pred_all.update(test_th_pred)
    if args.train_pred_jsonl:
        pred_all.update(train_th_pred)
    write_json(out_dir / "predictions_all_thresholded.json", pred_all)

    metrics: Dict[str, Any] = {
        "topk": args.topk,
        "threshold_tuned_on_dev": best_thr,
        "dev_best": best_dev,
        "set_micro_metrics_thresholded": {
            "dev": {"TP": dev_tp, "FP": dev_fp, "FN": dev_fn, "P": dev_p, "R": dev_r, "F1": dev_f1},
            "test": {"TP": test_tp, "FP": test_fp, "FN": test_fn, "P": test_p, "R": test_r, "F1": test_f1},
        },
        "outputs": {
            "dev_topk": str(out_dir / f"dev_top{args.topk}.json"),
            "test_topk": str(out_dir / f"test_top{args.topk}.json"),
            "dev_thresholded": str(out_dir / "dev_thresholded.json"),
            "test_thresholded": str(out_dir / "test_thresholded.json"),
            "train_thresholded": str(out_dir / "train_thresholded.json") if args.train_pred_jsonl else None,
            "all_thresholded": str(out_dir / "predictions_all_thresholded.json"),
        },
    }
    write_json(out_dir / "metrics_thresholded.json", metrics)

    print(f"dev\t{dev_tp}\t{dev_fp}\t{dev_fn}\t{dev_p*100:.4f}\t{dev_r*100:.4f}\t{dev_f1*100:.4f}")
    print(f"test\t{test_tp}\t{test_fp}\t{test_fn}\t{test_p*100:.4f}\t{test_r*100:.4f}\t{test_f1*100:.4f}")
    print(f"[DEV thr={best_thr:.6f}] test set-micro P/R/F1 = {test_p:.4f}/{test_r:.4f}/{test_f1:.4f}")
    LOGGER.info("[Write] %s", out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args()

    if args.command == "predict":
        cmd_predict(args)
        return
    if args.command == "eval":
        cmd_eval(args)
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
