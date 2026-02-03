#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Postprocess ranker top-K predictions: evaluate and export in two modes (top-100 / thresholded).

Input prediction JSONL (one line per query):
{
  "query_text": "...",
  "items": [
    {"rank": 0, "score": 0.12, "term_id": "HP_0000739", "term_text": "Anxiety"},
    ...
  ]
}

Gold is required:
- `--gold-json`: all_wo_mention.json (doc_id -> {"passage":..., "concepts":[{"id":..., "name":...}, ...]})
- `--split-list-json`: split_list.json ({"train":[...], "dev":[...], "test":[...]})

Modes:
- topk: keep top-K candidates (default K=100)
- thresholded: tune threshold on dev by set-micro F1, apply to all splits

Outputs:
- Per split: JSON (doc_id -> {"gold":[...], "pred":[...]})
- Aggregated: predictions_all_<tag>.json (doc_id -> pred list)
- metrics_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


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
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                if json.loads(line) in records:
                    continue
                records.append(json.loads(line))
    return records


# -------------------------
# Data model
# -------------------------


@dataclass(frozen=True)
class PredItem:
    term_id: str
    score: float
    term_text: str = ""


@dataclass(frozen=True)
class PredExample:
    query_text: str
    items: List[PredItem]


@dataclass(frozen=True)
class GoldConcept:
    cid: str
    name: str


@dataclass(frozen=True)
class GoldExample:
    doc_id: str
    query_text: str
    concepts: List[GoldConcept]


def parse_predictions(rows: List[dict]) -> List[PredExample]:
    preds: List[PredExample] = []
    unique_q = set()
    for r in rows:
        q = r.get("query_text", "")
        if not isinstance(q, str) or not q:
            continue
        if q in unique_q:
            continue
        unique_q.add(q)

        items_raw = r.get("items", [])
        if not isinstance(items_raw, list):
            items_raw = []

        items: List[PredItem] = []
        for it in items_raw:
            if not isinstance(it, dict):
                continue

            cid = it.get("term_id", "")
            if not isinstance(cid, str):
                cid = str(cid)

            score = it.get("score", 0.0)
            try:
                score_f = float(score)
            except Exception:
                score_f = 0.0

            name = it.get("term_text", "")
            if not isinstance(name, str):
                name = str(name)

            items.append(PredItem(term_id=cid, score=score_f, term_text=name))

        preds.append(PredExample(query_text=q, items=items))
    return preds


def build_gold_examples(
        all_wo_mention: dict,
        split_list: dict,
        split: str,
) -> Dict[str, GoldExample]:
    """Return doc_id -> GoldExample for a split, using concept {id,name} from gold-json."""
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
        gold_concepts: List[GoldConcept] = []
        for c in concepts:
            if not isinstance(c, dict) or "id" not in c:
                continue
            cid = str(c.get("id", ""))
            if not cid or cid in seen:
                continue
            seen.add(cid)
            name = c.get("name", "")
            if not isinstance(name, str):
                name = str(name)
            gold_concepts.append(GoldConcept(cid=cid, name=name))

        out[str(doc_id)] = GoldExample(doc_id=str(doc_id), query_text=passage, concepts=gold_concepts)

    return out


def index_gold_by_query(gold_by_doc: Dict[str, GoldExample]) -> Dict[str, str]:
    """query_text -> doc_id (assumes unique passages)."""
    idx: Dict[str, str] = {}
    for doc_id, g in gold_by_doc.items():
        idx[g.query_text] = doc_id
    return idx


# -------------------------
# Core logic
# -------------------------


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
        doc_id = query_to_doc.get(ex.query_text)
        if doc_id is None:
            continue
        gold = gold_by_doc.get(doc_id)
        if gold is None:
            continue

        pred_ids = {it.term_id for it in select_items(ex, mode, topk, threshold) if it.term_id}
        gold_ids = {c.cid for c in gold.concepts if c.cid}

        tp += len(pred_ids & gold_ids)
        fp += len(pred_ids - gold_ids)
        fn += len(gold_ids - pred_ids)

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

    # match old behavior: quantile on raw scores (keeps frequency)
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

        # tie-break: prefer higher threshold when F1 ties
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
        doc_id = query_to_doc.get(ex.query_text)
        if doc_id is None:
            continue
        gold = gold_by_doc.get(doc_id)
        if gold is None:
            continue

        pred_items = select_items(ex, mode, topk, threshold)
        pred_recs: List[dict] = []
        for it in pred_items:
            if not it.term_id:
                continue
            rec: dict = {"id": it.term_id, "name": it.term_text}
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


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["topk", "thresholded"], default="topk")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--with-score", action="store_true")
    parser.add_argument("--max-thr-candidates", type=int, default=4096)

    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, default=None)

    parser.add_argument("--gold-json", type=Path, required=True, help="all_wo_mention.json (concepts include id+name)")
    parser.add_argument("--split-list-json", type=Path, required=True, help="split_list.json")

    parser.add_argument("--out-dir", type=Path, required=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Postprocess ranker top-K predictions (generic).")
    add_shared_args(parser)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args()

    all_wo_mention = load_json(args.gold_json)
    split_list = load_json(args.split_list_json)
    if not isinstance(all_wo_mention, dict) or not isinstance(split_list, dict):
        raise ValueError("gold-json and split-list-json must be JSON dicts")

    dev_gold_by_doc = build_gold_examples(all_wo_mention, split_list, "dev")
    test_gold_by_doc = build_gold_examples(all_wo_mention, split_list, "test")
    train_gold_by_doc = build_gold_examples(all_wo_mention, split_list, "train") if args.train_jsonl else {}

    dev_query_to_doc = index_gold_by_query(dev_gold_by_doc)
    test_query_to_doc = index_gold_by_query(test_gold_by_doc)
    train_query_to_doc = index_gold_by_query(train_gold_by_doc) if args.train_jsonl else {}

    dev_preds = parse_predictions(load_jsonl(args.dev_jsonl))
    test_preds = parse_predictions(load_jsonl(args.test_jsonl))
    train_preds = parse_predictions(load_jsonl(args.train_jsonl)) if args.train_jsonl else []

    threshold = 1.0
    thr_stats: Optional[Dict[str, float]] = None
    if args.mode == "thresholded":
        threshold, thr_stats = tune_threshold_on_dev(
            dev_preds,
            dev_gold_by_doc,
            dev_query_to_doc,
            topk=args.topk,
            max_candidates=args.max_thr_candidates,
        )
        LOGGER.info("[DEV] tuned threshold=%.6f stats=%s", threshold, thr_stats)

    tag = f"top{args.topk}" if args.mode == "topk" else "thresholded"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_dump, dev_pred_only = build_split_dump(
        dev_preds, dev_gold_by_doc, dev_query_to_doc, args.mode, args.topk, threshold, with_score=args.with_score
    )
    test_dump, test_pred_only = build_split_dump(
        test_preds, test_gold_by_doc, test_query_to_doc, args.mode, args.topk, threshold, with_score=args.with_score
    )
    train_dump: Dict[str, dict] = {}
    train_pred_only: Dict[str, List[dict]] = {}
    if args.train_jsonl:
        train_dump, train_pred_only = build_split_dump(
            train_preds,
            train_gold_by_doc,
            train_query_to_doc,
            args.mode,
            args.topk,
            threshold,
            with_score=args.with_score,
        )

    dev_tp, dev_fp, dev_fn = compute_counts(
        dev_preds, dev_gold_by_doc, dev_query_to_doc, args.mode, args.topk, threshold
    )
    test_tp, test_fp, test_fn = compute_counts(
        test_preds, test_gold_by_doc, test_query_to_doc, args.mode, args.topk, threshold
    )
    dev_p, dev_r, dev_f1 = set_micro_prf1(dev_tp, dev_fp, dev_fn)
    test_p, test_r, test_f1 = set_micro_prf1(test_tp, test_fp, test_fn)

    print(f"dev\t{dev_tp}\t{dev_fp}\t{dev_fn}\t{dev_p * 100:.4f}\t{dev_r * 100:.4f}\t{dev_f1 * 100:.4f}")
    print(f"test\t{test_tp}\t{test_fp}\t{test_fn}\t{test_p * 100:.4f}\t{test_r * 100:.4f}\t{test_f1 * 100:.4f}")

    metrics: Dict[str, Any] = {
        "mode": args.mode,
        "topk": args.topk,
        "with_score": args.with_score,
        "threshold_tuned_on_dev": threshold if args.mode == "thresholded" else None,
        "dev_threshold_stats": thr_stats,
        "set_micro_metrics": {
            "dev": {"TP": dev_tp, "FP": dev_fp, "FN": dev_fn, "P": dev_p, "R": dev_r, "F1": dev_f1},
            "test": {"TP": test_tp, "FP": test_fp, "FN": test_fn, "P": test_p, "R": test_r, "F1": test_f1},
        },
        "outputs": {
            "dev": str(out_dir / f"dev_{tag}.json"),
            "test": str(out_dir / f"test_{tag}.json"),
            "train": str(out_dir / f"train_{tag}.json") if args.train_jsonl else None,
            "all": str(out_dir / f"predictions_all_{tag}.json"),
        },
    }

    write_json(out_dir / f"dev_{tag}.json", dev_dump)
    write_json(out_dir / f"test_{tag}.json", test_dump)
    if args.train_jsonl:
        write_json(out_dir / f"train_{tag}.json", train_dump)

    pred_all: Dict[str, List[dict]] = {}
    pred_all.update(dev_pred_only)
    pred_all.update(test_pred_only)
    if args.train_jsonl:
        pred_all.update(train_pred_only)

    write_json(out_dir / f"predictions_all_{tag}.json", pred_all)
    write_json(out_dir / f"metrics_{tag}.json", metrics)


if __name__ == "__main__":
    main()
