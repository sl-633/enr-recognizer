#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare bi-encoder inputs and convert top-K predictions to cross-encoder inputs.

Commands
--------
1) preprocess
   Build bi-encoder training inputs with BM25 hard negatives.

2) build-cross-rerank
   Convert bi-encoder top-K prediction JSONL (dev/test) into cross-encoder rerank-dev JSONL:
   {
     "query_text": str,
     "candidate_term_texts": [str, ...],
     "gold_term_texts": [str, ...]
   }

3) build-cross-proxy
   Convert bi-encoder top-K prediction JSONL into cross-encoder proxy-dev JSONL:
   { "query_text": str, "pos_term_text": str, "hard_neg_term_texts": [str, ...] }
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

LOGGER = logging.getLogger(__name__)


# -------------------------
# IO helpers
# -------------------------


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


# -------------------------
# Bi-encoder preprocess (BM25 hard negatives)  [unchanged]
# -------------------------


def load_hard_negative_map(path: Path) -> Dict[str, List[List[Any]]]:
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"--hard-negative-path must be a .jsonl file: {path}")

    hard_map: Dict[str, List[List[Any]]] = {}
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
                score = c.get("score", 0.0)
                if not isinstance(name, str):
                    name = str(name)
                try:
                    score_f = float(score)
                except Exception:
                    score_f = 0.0
                if name:
                    pairs.append((name, score_f))

            pairs.sort(key=lambda x: x[1], reverse=True)
            hard_map[cid] = [[i, name] for i, (name, _) in enumerate(pairs)]

    return hard_map


def iter_biencoder_records(
    all_data: Dict[str, Any],
    split_list: Dict[str, Any],
    hard_map: Dict[str, Any],
    split: str,
    k_hard_neg: int,
) -> Iterable[dict]:
    split_ids = split_list.get(split, [])
    if not isinstance(split_ids, list):
        split_ids = []
    split_id_set = set(str(x) for x in split_ids)

    for doc_id, doc in all_data.items():
        if str(doc_id) not in split_id_set:
            continue
        if not isinstance(doc, dict):
            continue

        passage = doc.get("passage", "")
        concepts = doc.get("concepts", [])
        if not isinstance(passage, str) or not passage:
            continue
        if not isinstance(concepts, list) or not concepts:
            continue

        query_text = passage
        pos_names = [str(c.get("name", "")) for c in concepts if isinstance(c, dict)]

        for c in concepts:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("id", ""))
            pos_name = str(c.get("name", ""))
            if not cid or not pos_name:
                continue

            cand = hard_map.get(cid, [])
            if not isinstance(cand, list):
                cand = []

            hard_negs: List[str] = []
            for pair in cand:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                neg_name = pair[1]
                if not isinstance(neg_name, str):
                    neg_name = str(neg_name)

                if neg_name == pos_name:
                    continue
                if neg_name in pos_names:
                    continue

                hard_negs.append(neg_name)
                if len(hard_negs) >= int(k_hard_neg):
                    break

            yield {
                "query_text": query_text,
                "pos_term_text": pos_name,
                "hard_neg_term_texts": hard_negs,
            }


def build_biencoder_inputs(
    data_name: str,
    data_root: Path,
    hard_negative_path: Path,
    k_hard_neg: int,
) -> None:
    dataset_root = data_root / f"mm-{data_name}"
    all_data = load_json(dataset_root / "ori" / "all_wo_mention.json")
    split_list = load_json(dataset_root / "ori" / "split_list.json")
    hard_map = load_hard_negative_map(hard_negative_path)

    if not isinstance(all_data, dict) or not isinstance(split_list, dict):
        raise ValueError("all_wo_mention.json and split_list.json must be JSON dicts")

    for split in ["train", "dev", "test"]:
        out_path = dataset_root / "mld" / "bi-encoder-input" / f"{split}_biencoder.jsonl"
        records = iter_biencoder_records(all_data, split_list, hard_map, split, k_hard_neg=k_hard_neg)
        write_jsonl(out_path, records)
        LOGGER.info("Finished transformation, output: %s", out_path)


# -------------------------
# New: prediction -> cross-encoder inputs
# -------------------------


def build_passage_to_doc_id(
    all_data: Dict[str, Any],
    split_list: Dict[str, Any],
    split: str,
) -> Dict[str, str]:
    split_ids = split_list.get(split, [])
    if not isinstance(split_ids, list):
        split_ids = []
    split_id_set = set(str(x) for x in split_ids)

    idx: Dict[str, str] = {}
    for doc_id, doc in all_data.items():
        if str(doc_id) not in split_id_set:
            continue
        if not isinstance(doc, dict):
            continue
        passage = doc.get("passage", "")
        if isinstance(passage, str) and passage:
            idx[passage] = str(doc_id)
    return idx


def extract_gold_term_texts(doc: Dict[str, Any]) -> List[str]:
    concepts = doc.get("concepts", [])
    if not isinstance(concepts, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for c in concepts:
        if not isinstance(c, dict):
            continue
        name = c.get("name", "")
        if not isinstance(name, str):
            name = str(name)
        name = name.strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def extract_candidate_term_texts(pred_row: Dict[str, Any], keep_topk: int) -> List[str]:
    items = pred_row.get("items", [])
    if not isinstance(items, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for it in items[: max(0, int(keep_topk))]:
        if not isinstance(it, dict):
            continue
        t = it.get("term_text", "")
        if not isinstance(t, str):
            t = str(t)
        t = t.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def iter_cross_rerank_records(
    pred_rows: List[Dict[str, Any]],
    all_data: Dict[str, Any],
    split_list: Dict[str, Any],
    split: str,
    keep_topk: int,
    ensure_gold_in_candidates: bool,
) -> Iterable[dict]:
    passage_to_doc = build_passage_to_doc_id(all_data, split_list, split)

    total = 0
    missing = 0

    for r in pred_rows:
        q = r.get("query_text", "")
        if not isinstance(q, str) or not q:
            continue

        total += 1
        doc_id = passage_to_doc.get(q)
        if doc_id is None:
            missing += 1
            continue

        doc = all_data.get(doc_id, {})
        if not isinstance(doc, dict):
            continue

        gold_terms = extract_gold_term_texts(doc)
        if not gold_terms:
            continue

        cands = extract_candidate_term_texts(r, keep_topk=keep_topk)

        if ensure_gold_in_candidates:
            # inject missing golds at the front (stable & guarantees evaluability)
            gold_set = set(gold_terms)
            cand_set = set(cands)
            missing_golds = [g for g in gold_terms if g not in cand_set]
            cands = missing_golds + cands
            # keep order + dedup again
            dedup: List[str] = []
            seen: set[str] = set()
            for t in cands:
                if t not in seen:
                    seen.add(t)
                    dedup.append(t)
            cands = dedup[: max(0, int(keep_topk))]  # keep final length bounded

        yield {
            "query_text": q,
            "candidate_term_texts": cands,
            "gold_term_texts": gold_terms,
        }

    if total > 0:
        LOGGER.info(
            "Prediction alignment: total_queries=%d matched=%d missing=%d (%.2f%% missing)",
            total,
            total - missing,
            missing,
            100.0 * missing / total,
        )


def build_cross_rerank_from_predictions(
    data_name: str,
    data_root: Path,
    split: str,
    pred_jsonl: Path,
    out_jsonl: Path,
    keep_topk: int,
    ensure_gold_in_candidates: bool,
) -> None:
    dataset_root = data_root / f"mm-{data_name}"
    all_data = load_json(dataset_root / "ori" / "all_wo_mention.json")
    split_list = load_json(dataset_root / "ori" / "split_list.json")
    if not isinstance(all_data, dict) or not isinstance(split_list, dict):
        raise ValueError("all_wo_mention.json and split_list.json must be JSON dicts")

    pred_rows = load_jsonl(pred_jsonl)
    records = iter_cross_rerank_records(
        pred_rows=pred_rows,
        all_data=all_data,
        split_list=split_list,
        split=split,
        keep_topk=keep_topk,
        ensure_gold_in_candidates=ensure_gold_in_candidates,
    )
    write_jsonl(out_jsonl, records)
    LOGGER.info("Wrote cross-encoder rerank inputs: %s", out_jsonl)


def iter_cross_proxy_records_from_predictions(
    pred_rows: List[Dict[str, Any]],
    all_data: Dict[str, Any],
    split_list: Dict[str, Any],
    split: str,
    keep_topk: int,
    k_hard_neg: int,
) -> Iterable[dict]:
    """If you still need proxy-dev style: per gold => one record."""
    passage_to_doc = build_passage_to_doc_id(all_data, split_list, split)

    for r in pred_rows:
        q = r.get("query_text", "")
        if not isinstance(q, str) or not q:
            continue
        doc_id = passage_to_doc.get(q)
        if doc_id is None:
            continue

        doc = all_data.get(doc_id, {})
        if not isinstance(doc, dict):
            continue

        gold_terms = extract_gold_term_texts(doc)
        if not gold_terms:
            continue

        cand_terms = extract_candidate_term_texts(r, keep_topk=keep_topk)
        gold_set = set(gold_terms)
        hard_pool = [t for t in cand_terms if t not in gold_set][: max(0, int(k_hard_neg))]

        for g in gold_terms:
            yield {"query_text": q, "pos_term_text": g, "hard_neg_term_texts": hard_pool}


def build_cross_proxy_from_predictions(
    data_name: str,
    data_root: Path,
    split: str,
    pred_jsonl: Path,
    out_jsonl: Path,
    keep_topk: int,
    k_hard_neg: int,
) -> None:
    dataset_root = data_root / f"mm-{data_name}"
    all_data = load_json(dataset_root / "ori" / "all_wo_mention.json")
    split_list = load_json(dataset_root / "ori" / "split_list.json")
    if not isinstance(all_data, dict) or not isinstance(split_list, dict):
        raise ValueError("all_wo_mention.json and split_list.json must be JSON dicts")

    pred_rows = load_jsonl(pred_jsonl)
    records = iter_cross_proxy_records_from_predictions(
        pred_rows=pred_rows,
        all_data=all_data,
        split_list=split_list,
        split=split,
        keep_topk=keep_topk,
        k_hard_neg=k_hard_neg,
    )
    write_jsonl(out_jsonl, records)
    LOGGER.info("Wrote cross-encoder proxy inputs: %s", out_jsonl)


# -------------------------
# CLI
# -------------------------


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--data-name", type=str, required=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare bi-encoder inputs and cross-encoder inputs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preprocess", help="Build bi-encoder JSONL training inputs.")
    add_shared_args(p)
    p.add_argument("--hard-negative-path", type=Path, required=True)
    p.add_argument("--k-hard-neg", type=int, default=20)

    r = sub.add_parser("build-cross-rerank", help="Build cross-encoder rerank-dev JSONL from top-K predictions.")
    add_shared_args(r)
    r.add_argument("--split", choices=["train", "dev", "test"], required=True)
    r.add_argument("--pred-jsonl", type=Path, required=True)
    r.add_argument("--out-jsonl", type=Path, required=True)
    r.add_argument("--keep-topk", type=int, default=100)
    r.add_argument(
        "--ensure-gold-in-candidates",
        action="store_true",
        help="If set, inject missing gold term_texts into candidates (keeps final size <= keep-topk).",
    )

    x = sub.add_parser("build-cross-proxy", help="Build cross-encoder proxy JSONL from top-K predictions.")
    add_shared_args(x)
    x.add_argument("--split", choices=["train", "dev", "test"], required=True)
    x.add_argument("--pred-jsonl", type=Path, required=True)
    x.add_argument("--out-jsonl", type=Path, required=True)
    x.add_argument("--keep-topk", type=int, default=100)
    x.add_argument("--k-hard-neg", type=int, default=100)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args()

    if args.command == "preprocess":
        build_biencoder_inputs(args.data_name, args.data_root, args.hard_negative_path, args.k_hard_neg)
        return

    if args.command == "build-cross-rerank":
        build_cross_rerank_from_predictions(
            data_name=args.data_name,
            data_root=args.data_root,
            split=args.split,
            pred_jsonl=args.pred_jsonl,
            out_jsonl=args.out_jsonl,
            keep_topk=args.keep_topk,
            ensure_gold_in_candidates=bool(args.ensure_gold_in_candidates),
        )
        return

    if args.command == "build-cross-proxy":
        build_cross_proxy_from_predictions(
            data_name=args.data_name,
            data_root=args.data_root,
            split=args.split,
            pred_jsonl=args.pred_jsonl,
            out_jsonl=args.out_jsonl,
            keep_topk=args.keep_topk,
            k_hard_neg=args.k_hard_neg,
        )
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
