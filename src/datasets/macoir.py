#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MA-COIR data-related pipeline.

Commands
--------
1) preprocess
   Build MA-COIR training JSONL from gold data + ssi_id_sid.json.

2) postprocess
   Convert MA-COIR prediction JSONL (beam10 / tuned-top1) into set-style predictions:
   - Outputs JSON (doc_id -> {"gold":[{"id","name"}], "pred":[{"id","name"}]})
   - Also outputs predictions_all.json (doc_id -> pred list)

3) eval
   Compute micro TP/FP/FN/P/R/F1 on dev/test (and optionally train).

Key point
---------
MA-COIR outputs do NOT have scores, so there is no ranking/thresholding here.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)
OBO_PREFIX = "http://purl.obolibrary.org/obo/"


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


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# -------------------------
# Preprocess (build MA-COIR inputs)
# -------------------------


def normalize_sid_map(raw_map: Dict[str, List[int]]) -> Dict[str, List[int]]:
    normalized: Dict[str, List[int]] = {}
    for key, value in raw_map.items():
        if isinstance(key, str) and key.startswith(OBO_PREFIX):
            key = key.replace(OBO_PREFIX, "")
        try:
            normalized[str(key)] = [int(i) for i in value]
        except Exception:
            continue
    return normalized


def build_macoir_inputs(data_name: str, data_root: Path, sid_map_path: Path) -> None:
    all_data = load_json(data_root / f"mm-{data_name}" / "ori" / "all_wo_mention.json")
    split_list = load_json(data_root / f"mm-{data_name}" / "ori" / "split_list.json")
    sid_map = normalize_sid_map(load_json(sid_map_path))

    if not isinstance(all_data, dict) or not isinstance(split_list, dict):
        raise ValueError("all_wo_mention.json and split_list.json must be JSON dicts")

    for split in ["train", "dev", "test"]:
        out_path = data_root / f"mm-{data_name}" / "mld" / "macoir-input" / f"{split}_ssi_macoir.jsonl"
        records: List[dict] = []

        for doc_id, doc in all_data.items():
            if doc_id not in split_list.get(split, []):
                continue
            if not isinstance(doc, dict):
                continue

            query_text = doc.get("passage", "")
            concepts = doc.get("concepts", [])
            if not isinstance(query_text, str) or not query_text.strip():
                continue
            if not isinstance(concepts, list):
                continue

            gold_terms: Dict[str, str] = {}
            for c in concepts:
                if not isinstance(c, dict):
                    continue
                cid = str(c.get("id", ""))
                name = c.get("name", "")
                if cid in sid_map:
                    sid = "-".join(str(i) for i in sid_map[cid])
                    gold_terms[sid] = str(name) if isinstance(name, str) else str(name)

            records.append({"query_text": query_text, "pos_terms": gold_terms})

        write_jsonl(out_path, records)
        print(f"[preprocess] wrote: {out_path}")


# -------------------------
# Postprocess: SID -> term_id (+name), align with gold by doc_id
# -------------------------


@dataclass(frozen=True)
class GoldConcept:
    cid: str
    name: str


@dataclass(frozen=True)
class GoldExample:
    doc_id: str
    query_text: str
    concepts: List[GoldConcept]


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
        if not isinstance(passage, str) or not passage.strip():
            continue
        if not isinstance(concepts, list):
            continue

        seen: set[str] = set()
        gold_concepts: List[GoldConcept] = []
        for c in concepts:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("id", ""))
            if not cid or cid in seen:
                continue
            seen.add(cid)
            name = c.get("name", "")
            name = name if isinstance(name, str) else str(name)
            gold_concepts.append(GoldConcept(cid=cid, name=name))

        out[str(doc_id)] = GoldExample(doc_id=str(doc_id), query_text=passage.strip(), concepts=gold_concepts)

    return out


def index_gold_by_query(gold_by_doc: Dict[str, GoldExample]) -> Dict[str, str]:
    """query_text -> doc_id (assumes passages are unique in a split)."""
    return {g.query_text: doc_id for doc_id, g in gold_by_doc.items()}


def build_sid_to_term_id(ssi_id_sid: dict) -> Dict[str, str]:
    """ssi_id_sid.json: {term_id: [ints]} -> {sid_str: term_id}."""
    out: Dict[str, str] = {}
    if not isinstance(ssi_id_sid, dict):
        return out
    for term_id, sid_list in ssi_id_sid.items():
        if not isinstance(sid_list, list) or not sid_list:
            continue
        try:
            sid = "-".join(str(int(x)) for x in sid_list)
        except Exception:
            continue
        tid = str(term_id)
        if sid and sid not in out:
            out[sid] = tid
    return out


def build_term_id_to_name(concept_meta: Optional[dict | list]) -> Dict[str, str]:
    """Optional: meta file to map term_id -> name."""
    if concept_meta is None:
        return {}
    out: Dict[str, str] = {}

    if isinstance(concept_meta, dict):
        for tid, v in concept_meta.items():
            if not isinstance(v, dict):
                continue
            name = v.get("name") or v.get("label") or v.get("term_text")
            if isinstance(name, str) and name.strip():
                out[str(tid)] = name.strip()
        return out

    if isinstance(concept_meta, list):
        for v in concept_meta:
            if not isinstance(v, dict):
                continue
            tid = v.get("id") or v.get("term_id") or v.get("concept_id")
            name = v.get("name") or v.get("label") or v.get("term_text")
            if tid is None or name is None:
                continue
            tid_s = str(tid)
            name_s = str(name)
            if tid_s and name_s and tid_s not in out:
                out[tid_s] = name_s
        return out

    return out


def _split_sids(raw: str) -> List[str]:
    raw = raw.strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(";")]
    return [p for p in parts if p]


def extract_sids_from_row(row: dict, mode: str, topk: int) -> List[str]:
    """
    mode:
      - tuned1: row["pred_ids"] is a list of strings (often len=1)
      - beam10: row["pred_ids_top10"] is a list of strings; each string may contain ';'
    """
    if mode == "tuned1":
        pred_list = row.get("pred_ids", [])
    elif mode == "beam10":
        pred_list = row.get("pred_ids_top10", [])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not isinstance(pred_list, list):
        pred_list = []

    pred_list = pred_list[: max(0, int(topk))]
    sids: List[str] = []
    for x in pred_list:
        sids.extend(_split_sids(str(x)))
    return sids


def convert_preds_for_split(
        pred_rows: List[dict],
        gold_by_doc: Dict[str, GoldExample],
        query_to_doc: Dict[str, str],
        sid2tid: Dict[str, str],
        tid2name: Dict[str, str],
        mode: str,
        topk: int,
) -> Tuple[Dict[str, dict], Dict[str, List[dict]], Dict[str, int]]:
    """
    Returns:
      per_doc: doc_id -> {"gold":[{id,name}], "pred":[{id,name}]}
      pred_only: doc_id -> pred list
      stats: counters
    """
    per_doc: Dict[str, dict] = {}
    pred_only: Dict[str, List[dict]] = {}
    stats = {"rows_in": len(pred_rows), "rows_used": 0, "no_doc_match": 0, "unknown_sid": 0}

    for r in pred_rows:
        if not isinstance(r, dict):
            continue
        q = r.get("query_text", "")
        if not isinstance(q, str) or not q.strip():
            continue

        doc_id = query_to_doc.get(q.strip())
        if doc_id is None:
            stats["no_doc_match"] += 1
            continue
        gold = gold_by_doc.get(doc_id)
        if gold is None:
            stats["no_doc_match"] += 1
            continue

        sids = extract_sids_from_row(r, mode=mode, topk=topk)

        pred_ids: List[str] = []
        for sid in sids:
            tid = sid2tid.get(sid)
            if tid is None:
                stats["unknown_sid"] += 1
                continue
            pred_ids.append(tid)

        # uniq preserve order
        seen: set[str] = set()
        uniq_pred: List[str] = []
        for tid in pred_ids:
            if tid not in seen:
                seen.add(tid)
                uniq_pred.append(tid)

        pred_recs = [{"id": tid, "name": tid2name.get(tid, "")} for tid in uniq_pred]
        gold_recs = [{"id": c.cid, "name": c.name} for c in gold.concepts]

        per_doc[doc_id] = {"gold": gold_recs, "pred": pred_recs}
        pred_only[doc_id] = pred_recs
        stats["rows_used"] += 1

    return per_doc, pred_only, stats


# -------------------------
# Eval: micro P/R/F1 on sets
# -------------------------


def micro_counts(per_doc: Dict[str, dict]) -> Tuple[int, int, int]:
    tp = fp = fn = 0
    for _, v in per_doc.items():
        gold = v.get("gold", [])
        pred = v.get("pred", [])
        g = {str(x.get("id", "")) for x in gold if isinstance(x, dict) and x.get("id")}
        p = {str(x.get("id", "")) for x in pred if isinstance(x, dict) and x.get("id")}
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    return tp, fp, fn


def micro_prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


# -------------------------
# CLI
# -------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MA-COIR pipeline (preprocess + postprocess + eval).")
    sub = parser.add_subparsers(dest="command", required=True)

    # preprocess
    prep = sub.add_parser("preprocess", help="Build MA-COIR JSONL inputs from gold data + ssi_id_sid map.")
    prep.add_argument("--data-name", required=True, choices=["go", "hpo"])
    prep.add_argument("--data-root", type=Path, default=Path("data"))
    prep.add_argument("--sid-map", type=Path, required=True, help="ssi_id_sid.json (term_id -> sid list)")

    # postprocess (beam10 / tuned1)
    post = sub.add_parser("postprocess", help="Convert MA-COIR prediction JSONL to set-style outputs (no ranking).")
    post.add_argument("--mode", choices=["beam10", "tuned1"], required=True)
    post.add_argument("--pred-jsonl", type=Path, required=True)
    post.add_argument("--topk", type=int, default=10, help="max beams/strings to read per example")
    post.add_argument("--ssi-id-sid-json", type=Path, required=True)
    post.add_argument("--concept-meta-json", type=Path, default=None, help="optional term_id->name mapping")
    post.add_argument("--gold-json", type=Path, required=True, help="all_wo_mention.json")
    post.add_argument("--split-list-json", type=Path, required=True, help="split_list.json")
    post.add_argument("--split", choices=["train", "dev", "test"], required=True)
    post.add_argument("--out-dir", type=Path, required=True)
    post.add_argument("--out-name", type=str, default=None, help="default: <split>_<mode>.json")

    # eval (on already-postprocessed split jsons)
    ev = sub.add_parser("eval", help="Compute micro P/R/F1 for postprocessed outputs.")
    ev.add_argument("--dev-json", type=Path, required=True)
    ev.add_argument("--test-json", type=Path, required=True)
    ev.add_argument("--train-json", type=Path, default=None)
    ev.add_argument("--out-dir", type=Path, required=True)
    ev.add_argument("--tag", type=str, default="macoir")

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args()

    if args.command == "preprocess":
        build_macoir_inputs(args.data_name, args.data_root, args.sid_map)
        return

    if args.command == "postprocess":
        ssi = load_json(args.ssi_id_sid_json)
        sid2tid = build_sid_to_term_id(ssi)
        tid2name = build_term_id_to_name(load_json(args.concept_meta_json)) if args.concept_meta_json else {}

        all_wo_mention = load_json(args.gold_json)
        split_list = load_json(args.split_list_json)
        if not isinstance(all_wo_mention, dict) or not isinstance(split_list, dict):
            raise SystemExit("gold-json and split-list-json must be JSON dicts")

        gold_by_doc = build_gold_examples(all_wo_mention, split_list, args.split)
        query_to_doc = index_gold_by_query(gold_by_doc)

        pred_rows = load_jsonl(args.pred_jsonl)
        per_doc, pred_only, st = convert_preds_for_split(
            pred_rows,
            gold_by_doc,
            query_to_doc,
            sid2tid,
            tid2name,
            mode=args.mode,
            topk=args.topk,
        )

        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = out_dir
        pred_dir.mkdir(parents=True, exist_ok=True)

        out_name = args.out_name or f"{args.split}_{args.mode}.json"
        out_path = pred_dir / out_name
        write_json(out_path, per_doc)

        all_path = pred_dir / f"predictions_all_{args.split}_{args.mode}.json"
        write_json(all_path, pred_only)

        LOGGER.info("[postprocess] split=%s mode=%s in=%d used=%d no_doc=%d unknown_sid=%d -> %s",
                    args.split, args.mode, st["rows_in"], st["rows_used"], st["no_doc_match"], st["unknown_sid"],
                    out_path)
        return

    if args.command == "eval":
        dev = load_json(args.dev_json)
        test = load_json(args.test_json)
        train = load_json(args.train_json) if args.train_json else None

        if not isinstance(dev, dict) or not isinstance(test, dict) or (
                train is not None and not isinstance(train, dict)):
            raise SystemExit("eval inputs must be JSON dicts: doc_id -> {gold,pred}")

        dev_tp, dev_fp, dev_fn = micro_counts(dev)
        test_tp, test_fp, test_fn = micro_counts(test)
        dev_p, dev_r, dev_f1 = micro_prf1(dev_tp, dev_fp, dev_fn)
        test_p, test_r, test_f1 = micro_prf1(test_tp, test_fp, test_fn)

        print(f"dev\t{dev_tp}\t{dev_fp}\t{dev_fn}\t{dev_p * 100:.4f}\t{dev_r * 100:.4f}\t{dev_f1 * 100:.4f}")
        print(f"test\t{test_tp}\t{test_fp}\t{test_fn}\t{test_p * 100:.4f}\t{test_r * 100:.4f}\t{test_f1 * 100:.4f}")

        out = {
            "tag": args.tag,
            "dev": {"TP": dev_tp, "FP": dev_fp, "FN": dev_fn, "P": dev_p, "R": dev_r, "F1": dev_f1},
            "test": {"TP": test_tp, "FP": test_fp, "FN": test_fn, "P": test_p, "R": test_r, "F1": test_f1},
        }

        if train is not None:
            tr_tp, tr_fp, tr_fn = micro_counts(train)
            tr_p, tr_r, tr_f1 = micro_prf1(tr_tp, tr_fp, tr_fn)
            print(f"train\t{tr_tp}\t{tr_fp}\t{tr_fn}\t{tr_p * 100:.4f}\t{tr_r * 100:.4f}\t{tr_f1 * 100:.4f}")
            out["train"] = {"TP": tr_tp, "FP": tr_fp, "FN": tr_fn, "P": tr_p, "R": tr_r, "F1": tr_f1}

        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.out_dir / f"metrics_{args.tag}.json", out)
        return


if __name__ == "__main__":
    main()
