#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MA-COIR decoding utilities: eval (dev-tune beam) or predict (beam-k dump only)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import BartConfig, BartModel, BartTokenizerFast

from src.evaluation.macoir import compute_micro_prf1_from_ids, parse_sids_from_token_ids
from src.models.macoir import (
    MACOIR,
    build_restricted_vocab,
    enable_bart_reorder_cache_for_generation,
    make_prefix_allowed_fn,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_valid_sid_set(path: Path) -> set:
    data = json.loads(path.read_text(encoding="utf-8"))

    def to_sid(value: Sequence[int]) -> Optional[str]:
        try:
            return "-".join(str(int(x)) for x in value)
        except Exception:
            return None

    valid = set()
    for value in data.values():
        if isinstance(value, list) and all(isinstance(x, int) for x in value):
            sid = to_sid(value)
            if sid:
                valid.add(sid)
        elif isinstance(value, list) and any(isinstance(x, list) for x in value):
            for sub in value:
                if isinstance(sub, list):
                    sid = to_sid(sub)
                    if sid:
                        valid.add(sid)
    return valid


def normalize_search_id(val: Any) -> Optional[str]:
    if isinstance(val, (list, tuple)) and len(val) > 0:
        try:
            return "-".join(str(int(x)) for x in val)
        except Exception:
            pass
    if isinstance(val, str):
        if val.replace("-", "").isdigit():
            return val
        digits = [ch for ch in val if ch.isdigit() or ch == "-"]
        if digits:
            s = "".join(digits).strip("-")
            if s.replace("-", "").isdigit():
                return s
    return None


def pos_terms_to_target_string(pos_terms: Dict[str, Any]) -> str:
    ids: List[str] = []
    for key, value in (pos_terms or {}).items():
        sid = normalize_search_id(key) or normalize_search_id(value)
        if sid:
            ids.append(sid)
    ids = sorted(set(ids))
    return ";".join(ids) + ";" if ids else ""


def tokenize_inputs(
    tokenizer,
    rows: List[Dict[str, Any]],
    max_src_len: int,
    max_tgt_len: int,
):
    src_texts = [r["query_text"] for r in rows]
    tgt_texts = [pos_terms_to_target_string(r.get("pos_terms", {})) for r in rows]
    enc_src = tokenizer(src_texts, padding=True, truncation=True, max_length=max_src_len, return_tensors="pt")
    enc_tgt = tokenizer(tgt_texts, padding=True, truncation=True, max_length=max_tgt_len, return_tensors="pt")
    return enc_src["input_ids"], enc_src["attention_mask"].float(), enc_tgt["input_ids"]


@torch.no_grad()
def decode_with_beam(
    model,
    tokenizer,
    src_ids,
    src_mask,
    prefix_fn,
    gen_max_len: int,
    num_beams: int,
    device,
    num_return_sequences: int = 1,
) -> torch.Tensor:
    beams = model.generate(
        input_ids=src_ids.to(device),
        attention_mask=src_mask.to(device),
        max_length=gen_max_len,
        prefix_allowed_tokens_fn=prefix_fn,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        do_sample=False,
        use_cache=True,
    )
    return beams[:, 2:].detach().cpu()


def decode_top1_batched(
    model,
    tokenizer,
    src_ids,
    src_mask,
    prefix_fn,
    gen_max_len: int,
    beam: int,
    device,
    batch_size: int = 8,
) -> List[List[int]]:
    preds: List[List[int]] = []
    for i in range(0, src_ids.size(0), batch_size):
        out = decode_with_beam(
            model,
            tokenizer,
            src_ids[i : i + batch_size],
            src_mask[i : i + batch_size],
            prefix_fn,
            gen_max_len,
            beam,
            device,
            num_return_sequences=1,
        )
        preds += out.tolist()
    return preds


def decode_topk_batched(
    model,
    tokenizer,
    rows: List[Dict[str, Any]],
    src_ids,
    src_mask,
    prefix_fn,
    gen_max_len: int,
    k: int,
    device,
    out_path: Path,
    valid_sid_set: Optional[set] = None,
    batch_size: int = 4,
    include_gold: bool = True,
) -> None:
    """Dump beam-K hypotheses. If include_gold=False, gold_ids will be omitted."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok_ids = {
        "hyphen_id": tokenizer.convert_tokens_to_ids("-"),
        "semi_id": tokenizer.convert_tokens_to_ids(";"),
        "eos_id": tokenizer.eos_token_id,
        "pad_id": tokenizer.pad_token_id,
    }

    with out_path.open("w", encoding="utf-8") as handle:
        for i in range(0, src_ids.size(0), batch_size):
            batch = src_ids[i : i + batch_size]
            mask = src_mask[i : i + batch_size]
            out = decode_with_beam(
                model,
                tokenizer,
                batch,
                mask,
                prefix_fn,
                gen_max_len,
                num_beams=k,
                device=device,
                num_return_sequences=k,
            )
            t_len = out.size(1)
            out = out.view(batch.size(0), k, t_len)

            for j in range(batch.size(0)):
                preds_tok = out[j].tolist()
                pred_sids_k = [
                    parse_sids_from_token_ids([seq], tokenizer, tok_ids, gen_max_len - 1)[0]
                    for seq in preds_tok
                ]
                if valid_sid_set is not None:
                    pred_sids_k = [[s for s in sid_list if s in valid_sid_set] for sid_list in pred_sids_k]
                pred_ids_topk = [";".join(sids) for sids in pred_sids_k]

                record: Dict[str, Any] = {
                    "query_text": rows[i + j].get("query_text", ""),
                    "pred_ids_top10": pred_ids_topk,
                    "num_beams": k,
                }

                if include_gold:
                    gold_sid: List[str] = []
                    pos_terms = rows[i + j].get("pos_terms", {}) or {}
                    for key, val in pos_terms.items():
                        sid = normalize_search_id(key) or normalize_search_id(val)
                        if sid:
                            gold_sid.append(sid)
                    record["gold_ids"] = sorted(set(gold_sid))

                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def dump_top1_jsonl(
    rows: List[Dict[str, Any]],
    preds_top1_ids: List[List[int]],
    tokenizer,
    tok_ids: Dict[str, int],
    gen_max_len: int,
    out_path: Path,
    num_beams_for_this_run: int,
    valid_sid_set: Optional[set] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for ex, pred_seq in zip(rows, preds_top1_ids):
            pred_sid_list = parse_sids_from_token_ids([pred_seq], tokenizer, tok_ids, gen_max_len - 1)[0]
            if valid_sid_set is not None:
                pred_sid_list = [s for s in pred_sid_list if s in valid_sid_set]

            gold_sid: List[str] = []
            pos_terms = ex.get("pos_terms", {}) or {}
            for key, val in pos_terms.items():
                sid = normalize_search_id(key) or normalize_search_id(val)
                if sid:
                    gold_sid.append(sid)
            gold_sid = sorted(set(gold_sid))

            record = {
                "query_text": ex.get("query_text", ""),
                "pred_ids": pred_sid_list,
                "gold_ids": gold_sid,
                "num_beams": num_beams_for_this_run,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_macoir(model_dir: Path, transformer_name: str, device: torch.device):
    tokenizer = BartTokenizerFast.from_pretrained(model_dir)
    if tokenizer.eos_token_id is None:
        tokenizer = BartTokenizerFast.from_pretrained(transformer_name)
    config = BartConfig.from_pretrained(transformer_name)
    backbone = BartModel.from_pretrained(transformer_name)
    enable_bart_reorder_cache_for_generation(backbone)

    model = MACOIR(config, backbone, tokenizer).to(device)
    ckpt = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer


# -------------------------
# CLI
# -------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MA-COIR: eval (tune beam on dev) or predict (beam dump).")
    sub = parser.add_subparsers(dest="command", required=True)

    # shared model args
    def add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--transformer_name", default="facebook/bart-large")
        p.add_argument("--model_dir", type=Path, required=True)
        p.add_argument("--sid_catalog_json", type=Path, required=True)
        p.add_argument("--max_src_len", type=int, default=1024)
        p.add_argument("--max_tgt_len", type=int, default=300)
        p.add_argument("--gen_max_len", type=int, default=300)
        p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--seed", type=int, default=42)

    # eval mode
    ev = sub.add_parser("eval", help="Tune best beam on dev, evaluate on test, and dump predictions.")
    add_model_args(ev)
    ev.add_argument("--dataset", required=True, help="Dataset key (e.g., mm-go or mm-hpo).")
    ev.add_argument("--dev_jsonl", type=Path, required=True)
    ev.add_argument("--test_jsonl", type=Path, required=True)
    ev.add_argument("--beam_grid", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ev.add_argument("--dev_pred_beam10", type=Path, default=None)
    ev.add_argument("--test_pred_beam10", type=Path, default=None)
    ev.add_argument("--dev_pred_beam_tuned_1", type=Path, default=None)
    ev.add_argument("--test_pred_beam_tuned_1", type=Path, default=None)
    ev.add_argument("--metrics_out", type=Path, default=None)

    # predict mode (no gold)
    pr = sub.add_parser("predict", help="Decode a JSONL of queries and dump beam-K hypotheses (no eval, no gold).")
    add_model_args(pr)
    pr.add_argument("--input_jsonl", type=Path, required=True, help="JSONL with at least {query_text: ...}")
    pr.add_argument("--output_jsonl", type=Path, required=True)
    pr.add_argument("--beam", type=int, default=10, help="beam size (also number of returned sequences)")
    pr.add_argument("--batch_size", type=int, default=4, help="decode batch size (reduce if OOM)")

    return parser


# -------------------------
# Entry points
# -------------------------


def run_eval(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    base_dir = Path("logs") / "macoir" / args.dataset
    dev_pred_beam10 = args.dev_pred_beam10 or (base_dir / "model_outputs" / "dev-beam-10-predictions.jsonl")
    test_pred_beam10 = args.test_pred_beam10 or (base_dir / "model_outputs" / "test-beam-10-predictions.jsonl")
    dev_pred_beam_tuned = args.dev_pred_beam_tuned_1 or (
        base_dir / "model_outputs" / "dev-beam-tuned-top-1-predictions.jsonl"
    )
    test_pred_beam_tuned = args.test_pred_beam_tuned_1 or (
        base_dir / "model_outputs" / "test-beam-tuned-top-1-predictions.jsonl"
    )
    metrics_out = args.metrics_out or (base_dir / "metrics.json")

    dev_rows = load_jsonl(args.dev_jsonl)
    test_rows = load_jsonl(args.test_jsonl)
    model, tokenizer = load_macoir(args.model_dir, args.transformer_name, device)

    restricted = build_restricted_vocab(tokenizer)
    prefix_fn = make_prefix_allowed_fn(restricted.allowed_ids)
    tok_ids = {
        "hyphen_id": tokenizer.convert_tokens_to_ids("-"),
        "semi_id": tokenizer.convert_tokens_to_ids(";"),
        "eos_id": tokenizer.eos_token_id,
        "pad_id": tokenizer.pad_token_id,
    }
    valid_sid_set = load_valid_sid_set(args.sid_catalog_json)

    dev_src, dev_mask, dev_gold = tokenize_inputs(tokenizer, dev_rows, args.max_src_len, args.max_tgt_len)
    test_src, test_mask, test_gold = tokenize_inputs(tokenizer, test_rows, args.max_src_len, args.max_tgt_len)
    dev_src, dev_mask, test_src, test_mask = [x.to(device) for x in [dev_src, dev_mask, test_src, test_mask]]

    # tune beam on dev
    best_f1 = -1.0
    best_beam = 1
    best_pr = (0.0, 0.0)
    for beam in args.beam_grid:
        batch_size = 8 if beam <= 5 else 4
        preds = decode_top1_batched(
            model, tokenizer, dev_src, dev_mask, prefix_fn, args.gen_max_len, beam, device, batch_size=batch_size
        )
        p, r, f1 = compute_micro_prf1_from_ids(
            preds, dev_gold.tolist(), tokenizer, tok_ids, args.gen_max_len - 1, valid_sid_set=valid_sid_set
        )
        print(f"beam={beam}: P={p:.4f} R={r:.4f} F1={f1:.4f}")
        if f1 > best_f1 or (math.isclose(f1, best_f1) and beam < best_beam):
            best_f1 = f1
            best_beam = beam
            best_pr = (p, r)

    print(f"[DEV] best beam={best_beam} F1={best_f1:.4f}")

    # dump dev
    decode_topk_batched(
        model,
        tokenizer,
        dev_rows,
        dev_src,
        dev_mask,
        prefix_fn,
        args.gen_max_len,
        10,
        device,
        dev_pred_beam10,
        valid_sid_set=valid_sid_set,
        batch_size=4,
        include_gold=True,
    )
    dev_preds_best = decode_top1_batched(
        model, tokenizer, dev_src, dev_mask, prefix_fn, args.gen_max_len, best_beam, device, batch_size=8
    )
    dump_top1_jsonl(
        dev_rows,
        dev_preds_best,
        tokenizer,
        tok_ids,
        args.gen_max_len,
        dev_pred_beam_tuned,
        num_beams_for_this_run=best_beam,
        valid_sid_set=valid_sid_set,
    )

    # test eval + dump
    test_preds = decode_top1_batched(
        model, tokenizer, test_src, test_mask, prefix_fn, args.gen_max_len, best_beam, device, batch_size=8
    )
    p_t, r_t, f1_t = compute_micro_prf1_from_ids(
        test_preds, test_gold.tolist(), tokenizer, tok_ids, args.gen_max_len - 1, valid_sid_set=valid_sid_set
    )
    print(f"[TEST] beam={best_beam} P={p_t:.4f} R={r_t:.4f} F1={f1_t:.4f}")

    decode_topk_batched(
        model,
        tokenizer,
        test_rows,
        test_src,
        test_mask,
        prefix_fn,
        args.gen_max_len,
        10,
        device,
        test_pred_beam10,
        valid_sid_set=valid_sid_set,
        batch_size=4,
        include_gold=True,
    )
    dump_top1_jsonl(
        test_rows,
        test_preds,
        tokenizer,
        tok_ids,
        args.gen_max_len,
        test_pred_beam_tuned,
        num_beams_for_this_run=best_beam,
        valid_sid_set=valid_sid_set,
    )

    summary = {
        "dev_best": {"beam": best_beam, "P": best_pr[0], "R": best_pr[1], "F1": best_f1},
        "test_with_best_beam": {"beam": best_beam, "P": p_t, "R": r_t, "F1": f1_t},
        "dev_pred_beam10_top10": str(dev_pred_beam10),
        "test_pred_beam10_top10": str(test_pred_beam10),
        "dev_pred_beam_tuned_top1": str(dev_pred_beam_tuned),
        "test_pred_beam_tuned_top1": str(test_pred_beam_tuned),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def run_predict(args: argparse.Namespace) -> None:
    """Decode arbitrary queries; dump beam-K hypotheses only (no gold required)."""
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    rows = load_jsonl(args.input_jsonl)
    # allow "query_text" only, no pos_terms required
    rows = [{"query_text": r.get("query_text", "")} for r in rows if isinstance(r, dict) and r.get("query_text")]

    model, tokenizer = load_macoir(args.model_dir, args.transformer_name, device)
    restricted = build_restricted_vocab(tokenizer)
    prefix_fn = make_prefix_allowed_fn(restricted.allowed_ids)
    valid_sid_set = load_valid_sid_set(args.sid_catalog_json)

    # we don't need targets, but reuse tokenizer path (target len can be small)
    src_ids, src_mask, _ = tokenize_inputs(tokenizer, rows, args.max_src_len, max_tgt_len=8)
    src_ids = src_ids.to(device)
    src_mask = src_mask.to(device)

    decode_topk_batched(
        model,
        tokenizer,
        rows,
        src_ids,
        src_mask,
        prefix_fn,
        args.gen_max_len,
        k=args.beam,
        device=device,
        out_path=args.output_jsonl,
        valid_sid_set=valid_sid_set,
        batch_size=args.batch_size,
        include_gold=False,
    )
    print(f"[predict] wrote: {args.output_jsonl}")


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "eval":
        run_eval(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
