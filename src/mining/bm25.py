#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BM25 concept mining utilities.

Subcommands:
- query: mine candidates for a single query string or file.
- eval : evaluate candidate generation with recall@k, hit@k, MRR and MRR@10.
- batch: mine candidates for a corpus and write JSONL.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from rank_bm25 import BM25Okapi

WORD_RE = re.compile(r"[A-Za-z0-9\u00C0-\u024F\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF_]+", flags=re.UNICODE)

DEFAULT_STOPWORDS: set[str] = set()


def normalize(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return " ".join(text.lower().split())


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text)


def default_preprocess(text: str, stopwords: Optional[set[str]] = None) -> List[str]:
    sw = stopwords or set()
    return [tok for tok in tokenize(normalize(text)) if tok not in sw]


def load_stopwords(path: Optional[Path]) -> Optional[set[str]]:
    if not path:
        return None
    stopwords = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            tok = line.strip()
            if tok:
                stopwords.add(tok.lower())
    return stopwords


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class ConceptRecord:
    concept_id: str
    name: str
    description: str
    synonyms: List[str]

    @classmethod
    def from_obj(cls, obj: Dict[str, Union[str, List[str]]]) -> "ConceptRecord":
        return cls(
            concept_id=str(obj.get("id", "")),
            name=str(obj.get("name", "")),
            description=str(obj.get("description", "")),
            synonyms=[s for s in (obj.get("synonyms") or []) if isinstance(s, str)],
        )

    def tokens(self, fields: Sequence[str], stopwords: set[str]) -> List[str]:
        chunks: List[str] = []
        for field in fields:
            val = getattr(self, field, "")
            if isinstance(val, list):
                chunks.extend([v for v in val if isinstance(v, str)])
            else:
                chunks.append(val)
        return default_preprocess(" ".join(chunks), stopwords)


class BM25ConceptIndex:
    """Build a BM25 index over concept textual fields."""

    def __init__(
        self,
        records: List[ConceptRecord],
        fields: Sequence[str] = ("name", "description", "synonyms"),
        stopwords: Optional[set[str]] = None,
    ) -> None:
        self.fields = list(fields)
        self.stopwords = stopwords or set()
        self.records = records
        self.doc_tokens = [record.tokens(self.fields, self.stopwords) for record in records]
        self.bm25 = BM25Okapi(self.doc_tokens)

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        fields: Sequence[str] = ("name", "description", "synonyms"),
        stopwords: Optional[set[str]] = None,
        max_records: Optional[int] = None,
    ) -> "BM25ConceptIndex":
        records: List[ConceptRecord] = []
        for idx, obj in enumerate(iter_jsonl(path)):
            if max_records is not None and idx >= max_records:
                break
            if "id" not in obj:
                continue
            records.append(ConceptRecord.from_obj(obj))
        return cls(records, fields=fields, stopwords=stopwords)

    def search(
        self,
        query_text: str,
        topk: int = 50,
        exclude_ids: Optional[Iterable[str]] = None,
    ) -> List[Tuple[ConceptRecord, float]]:
        tokens = default_preprocess(query_text, self.stopwords)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        exclude = set(exclude_ids) if exclude_ids else set()
        results: List[Tuple[ConceptRecord, float]] = []
        for doc_id, score in ranked:
            record = self.records[doc_id]
            if record.concept_id in exclude:
                continue
            if score <= 0:
                break
            results.append((record, float(score)))
            if len(results) >= topk:
                break
        return results


def query_single(
    idx: BM25ConceptIndex,
    query_text: str,
    topk: int,
    exclude_ids: Optional[Iterable[str]] = None,
) -> List[dict]:
    results = idx.search(query_text, topk=topk, exclude_ids=exclude_ids)
    return [
        {
            "concept_id": record.concept_id,
            "name": record.name,
            "score": score,
            "description": record.description,
            "synonyms": record.synonyms,
        }
        for record, score in results
    ]


def evaluate(
    idx: BM25ConceptIndex,
    docs_gold_path: Path,
    topk: int,
    ks: Sequence[int],
    preds_out: Optional[Path] = None,
) -> Dict[str, Union[int, Dict[str, float], float]]:
    ks = sorted({k for k in ks if 0 < k <= topk})
    if not ks:
        raise ValueError("No valid --k values <= --topk")

    num_docs = 0
    sum_recall_at_k = {k: 0.0 for k in ks}
    sum_hit_at_k = {k: 0.0 for k in ks}
    sum_mrr = 0.0
    sum_mrr_at_10 = 0.0

    preds_handle = preds_out.open("w", encoding="utf-8") if preds_out else None

    for obj in iter_jsonl(docs_gold_path):
        doc_id = obj.get("doc_id")
        text = obj.get("text", "")
        positives = set(obj.get("positive_concept_ids", []) or [])
        if not text or not positives:
            continue

        num_docs += 1
        results = idx.search(text, topk=topk)
        ranked_ids = [record.concept_id for record, _ in results]

        for k in ks:
            top_ids = set(ranked_ids[:k])
            hits = len(positives & top_ids)
            sum_recall_at_k[k] += hits / max(1, len(positives))
            sum_hit_at_k[k] += 1.0 if hits > 0 else 0.0

        mrr_val = 0.0
        for rank, concept_id in enumerate(ranked_ids, start=1):
            if concept_id in positives:
                mrr_val = 1.0 / rank
                break
        sum_mrr += mrr_val

        mrr10 = 0.0
        for rank, concept_id in enumerate(ranked_ids[:10], start=1):
            if concept_id in positives:
                mrr10 = 1.0 / rank
                break
        sum_mrr_at_10 += mrr10

        if preds_handle:
            payload = {
                "doc_id": doc_id,
                "positives": sorted(list(positives)),
                "candidates": [
                    {"concept_id": record.concept_id, "name": record.name, "score": score}
                    for record, score in results
                ],
            }
            preds_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    if preds_handle:
        preds_handle.close()

    if num_docs == 0:
        raise ValueError("No valid labeled docs found in --docs_gold")

    return {
        "num_docs": num_docs,
        "topk": topk,
        "hit@k": {str(k): sum_hit_at_k[k] / num_docs for k in ks},
        "recall@k": {str(k): sum_recall_at_k[k] / num_docs for k in ks},
        "MRR": sum_mrr / num_docs,
        "MRR@10": sum_mrr_at_10 / num_docs,
    }


def batch_mine(idx: BM25ConceptIndex, docs_path: Path, topk: int, out_path: Path) -> None:
    out_records = []
    for obj in iter_jsonl(docs_path):
        doc_id = obj.get("doc_id")
        text = obj.get("text", "")
        if not doc_id or not text:
            continue
        results = idx.search(text, topk=topk)
        out_records.append(
            {
                "doc_id": doc_id,
                "candidates": [
                    {"concept_id": record.concept_id, "name": record.name, "score": score}
                    for record, score in results
                ],
            }
        )
    write_jsonl(out_path, out_records)


def build_index(args: argparse.Namespace) -> BM25ConceptIndex:
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    stopwords = load_stopwords(args.stopwords) or DEFAULT_STOPWORDS
    return BM25ConceptIndex.from_jsonl(
        path=args.entities,
        fields=fields,
        stopwords=stopwords,
        max_records=args.max_records,
    )


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--entities", type=Path, required=True, help="Path to entities JSONL.")
    parser.add_argument("--fields", type=str, default="name,description,synonyms", help="Comma-separated fields.")
    parser.add_argument("--stopwords", type=Path, default=None, help="Stopword file path (optional).")
    parser.add_argument("--max_records", type=int, default=None, help="Limit number of entities (debug).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BM25 concept mining utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    query = sub.add_parser("query", help="Mine candidates for a single query string.")
    add_shared_args(query)
    doc_group = query.add_mutually_exclusive_group(required=True)
    doc_group.add_argument("--doc", type=str, help="Raw document text to query.")
    doc_group.add_argument("--doc_file", type=Path, help="Path to a text file containing the document.")
    query.add_argument("--topk", type=int, default=50, help="Number of candidates to return.")
    query.add_argument("--exclude_ids", type=Path, default=None, help="File with IDs to exclude (one per line).")
    query.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    eval_parser = sub.add_parser("eval", help="Evaluate recall@k, hit@k, MRR and MRR@10.")
    add_shared_args(eval_parser)
    eval_parser.add_argument("--docs_gold", type=Path, required=True, help="Gold JSONL with doc_id/text/positives.")
    eval_parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 50], help="List of K values.")
    eval_parser.add_argument("--topk", type=int, default=100, help="Candidate pool size.")
    eval_parser.add_argument("--metrics_out", type=Path, default=None, help="Write metrics JSON to this path.")
    eval_parser.add_argument("--preds_out", type=Path, default=None, help="Write per-doc predictions JSONL.")
    eval_parser.add_argument("--pretty", action="store_true", help="Pretty JSON output.")

    batch = sub.add_parser("batch", help="Batch mine candidates for a corpus.")
    add_shared_args(batch)
    batch.add_argument("--docs", type=Path, required=True, help="Docs JSONL with doc_id and text.")
    batch.add_argument("--topk", type=int, default=50, help="Top-k candidates to output.")
    batch.add_argument("--out", type=Path, required=True, help="Output JSONL path.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    idx = build_index(args)

    if args.command == "query":
        doc_text = args.doc if args.doc is not None else read_text(args.doc_file)
        exclude_ids = None
        if args.exclude_ids:
            exclude_ids = [line.strip() for line in args.exclude_ids.read_text(encoding="utf-8").splitlines() if line]
        payload = query_single(idx, doc_text, topk=args.topk, exclude_ids=exclude_ids)
        if args.pretty:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False))
        return

    if args.command == "eval":
        metrics = evaluate(
            idx=idx,
            docs_gold_path=args.docs_gold,
            topk=args.topk,
            ks=args.k,
            preds_out=args.preds_out,
        )
        output = json.dumps(metrics, ensure_ascii=False, indent=2 if args.pretty else None)
        print(output)
        if args.metrics_out:
            args.metrics_out.write_text(output, encoding="utf-8")
        return

    if args.command == "batch":
        batch_mine(idx=idx, docs_path=args.docs, topk=args.topk, out_path=args.out)
        print(json.dumps({"status": "ok", "written": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
