#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train MA-COIR using JSONL inputs."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BartConfig, BartModel, BartTokenizerFast, get_linear_schedule_with_warmup

from src.evaluation.macoir import evaluate
from src.models.macoir import MACOIR, build_restricted_vocab, enable_bart_reorder_cache_for_generation, \
    make_prefix_allowed_fn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_valid_sid_set(path: Path) -> set:
    data = json.loads(path.read_text(encoding="utf-8"))

    def to_sid(value: Iterable[int]) -> Optional[str]:
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


def normalize_search_id(key_or_val: Any) -> Optional[str]:
    if isinstance(key_or_val, (list, tuple)) and len(key_or_val) > 0:
        try:
            ints = [int(x) for x in key_or_val]
            return "-".join(str(i) for i in ints)
        except Exception:
            pass
    if isinstance(key_or_val, str):
        if re.fullmatch(r"\d+(?:-\d+)*", key_or_val):
            return key_or_val
        digits = re.findall(r"\d+|-", key_or_val)
        if digits:
            s = "".join(digits)
            s = re.sub(r"-{2,}", "-", s).strip("-")
            if re.fullmatch(r"\d+(?:-\d+)*", s):
                return s
    return None


def pos_terms_to_target_string(pos_terms: Dict[str, Any]) -> str:
    ids: List[str] = []
    for key, value in pos_terms.items():
        sid = normalize_search_id(key)
        if sid is None:
            sid = normalize_search_id(value)
        if sid:
            ids.append(sid)
    ids = sorted(set(ids))
    return ";".join(ids) + ";" if ids else ""


class MacoirJsonlDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def build_augmented_train_rows(
        train_rows: List[Dict[str, Any]],
        enable_aug: bool,
) -> Tuple[List[Dict[str, Any]], int]:
    if not enable_aug:
        return train_rows, 0

    pairs: Set[Tuple[str, str]] = set()
    for row in train_rows:
        pos_terms = row.get("pos_terms", {}) or {}
        for sid_raw, term_text in pos_terms.items():
            sid = normalize_search_id(sid_raw)
            if sid is None:
                sid = normalize_search_id(term_text)
            if sid is None:
                continue
            if not isinstance(term_text, str) or not term_text.strip():
                continue
            pairs.add((sid, term_text.strip()))

    aug_rows = [{"query_text": tt, "pos_terms": {sid: tt}} for sid, tt in pairs]
    return list(train_rows) + aug_rows, len(aug_rows)


def make_collate_fn(tokenizer: BartTokenizerFast, max_src_len: int, max_tgt_len: int):
    def collate(batch: List[Dict[str, Any]]):
        src_texts = [b["query_text"] for b in batch]
        tgt_texts = [pos_terms_to_target_string(b.get("pos_terms", {})) for b in batch]
        enc_src = tokenizer(src_texts, padding=True, truncation=True, max_length=max_src_len, return_tensors="pt")
        enc_tgt = tokenizer(tgt_texts, padding=True, truncation=True, max_length=max_tgt_len, return_tensors="pt")
        src_ids = enc_src["input_ids"]
        src_mask = enc_src["attention_mask"].float()
        tgt_ids = enc_tgt["input_ids"]
        return (
            src_ids,
            src_mask,
            tgt_ids,
            tgt_ids,
            [b.get("doc_key", "") for b in batch],
            [b.get("query_text", "") for b in batch],
        )

    return collate


def save_best_ckpt(output_dir: Path, model: MACOIR, best_f1: float, tokenizer: BartTokenizerFast) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "best_model.pt"
    state = {"model_state_dict": model.state_dict(), "best_f1": best_f1}
    torch.save(state, path)
    tokenizer.save_pretrained(output_dir)
    print(f"[Save] best model -> {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MA-COIR (JSONL inputs).")
    parser.add_argument("--transformer_name", type=str, default="facebook/bart-large")
    parser.add_argument("--output_dir", type=Path, default=Path("./ckpt-macoir"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--train_jsonl", type=Path, required=True)
    parser.add_argument("--dev_jsonl", type=Path, required=True)
    parser.add_argument(
        "--sid_catalog_json",
        type=Path,
        default=None,
        help="Optional id_sid.json to filter predicted SIDs before metrics.",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--eval_every_steps", type=int, default=0)

    parser.add_argument("--max_src_len", type=int, default=1024)
    parser.add_argument("--max_tgt_len", type=int, default=300)
    parser.add_argument("--gen_max_len", type=int, default=300)
    parser.add_argument("--augment_term_spans", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    tokenizer = BartTokenizerFast.from_pretrained(args.transformer_name)
    config = BartConfig.from_pretrained(args.transformer_name)
    backbone = BartModel.from_pretrained(args.transformer_name)
    enable_bart_reorder_cache_for_generation(backbone)
    model = MACOIR(config=config, model=backbone, tokenizer=tokenizer).to(device)
    model.config.return_dict = True

    restricted = build_restricted_vocab(tokenizer)
    prefix_fn = make_prefix_allowed_fn(restricted.allowed_ids)
    tok_ids = {
        "hyphen_id": tokenizer.convert_tokens_to_ids("-"),
        "semi_id": tokenizer.convert_tokens_to_ids(";"),
        "eos_id": tokenizer.eos_token_id,
        "pad_id": tokenizer.pad_token_id,
    }

    valid_sid_set = None
    if args.sid_catalog_json:
        valid_sid_set = load_valid_sid_set(args.sid_catalog_json)
        print(f"[Ontology] Loaded {len(valid_sid_set)} valid SIDs from {args.sid_catalog_json}")

    train_rows_raw = load_jsonl(args.train_jsonl)
    dev_rows = load_jsonl(args.dev_jsonl)
    train_rows, n_aug = build_augmented_train_rows(train_rows_raw, enable_aug=args.augment_term_spans)
    print(f"[Data] train original={len(train_rows_raw)}, +aug={n_aug} → total={len(train_rows)}; dev={len(dev_rows)}")

    collate = make_collate_fn(tokenizer, args.max_src_len, args.max_tgt_len)
    train_loader = DataLoader(MacoirJsonlDataset(train_rows), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate)
    dev_loader = DataLoader(MacoirJsonlDataset(dev_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.lr)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum_steps)))
    total_updates = steps_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    best_f1 = -1.0
    global_step = 0
    running_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, start=1):
            src_ids, src_mask, tgt_ids, labels, _, _ = batch
            src_ids = src_ids.to(device)
            src_mask = src_mask.to(device)
            labels = labels.to(device)

            out = model(input_ids=src_ids, attention_mask=src_mask, labels=labels[:, 1:])
            loss = out.loss / max(1, args.grad_accum_steps)
            loss.backward()
            running_loss += loss.item()

            if (step % args.grad_accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if args.log_steps and (global_step % args.log_steps == 0):
                    avg_loss = running_loss / args.log_steps
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "gs": global_step})
                    running_loss = 0.0

            if args.eval_every_steps and epoch > 5 and (global_step % args.eval_every_steps == 0):
                v_p, v_r, v_f1 = evaluate(
                    model, dev_loader, device, tokenizer, prefix_fn, tok_ids, args.gen_max_len, valid_sid_set
                )
                if v_f1 > best_f1:
                    best_f1 = v_f1
                    save_best_ckpt(args.output_dir, model, best_f1, tokenizer)

        if not args.eval_every_steps:
            v_p, v_r, v_f1 = evaluate(
                model, dev_loader, device, tokenizer, prefix_fn, tok_ids, args.gen_max_len, valid_sid_set
            )
            if v_f1 > best_f1:
                best_f1 = v_f1
                save_best_ckpt(args.output_dir, model, best_f1, tokenizer)

    print(f"[Done] Training finished. Best dev F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
