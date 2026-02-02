#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train cross-encoder from JSONL inputs (proxy-dev or rerank-dev)."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.evaluation.cross_encoder import eval_proxy, listwise_loss_and_acc1, pairwise_margin_loss_and_winrate, eval_rerank_streaming
from src.models.cross_encoder import build_cross_encoder, init_encoder_from_biencoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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


def save_ckpt(out_dir: Path, model, tokenizer) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[Save] -> {out_dir}")


# -------------------------
# Datasets
# -------------------------


class ProxyJsonlDataset(Dataset):
    """Each line:
    {"query_text": str, "pos_term_text": str, "hard_neg_term_texts": [str, ...]}
    """

    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        return {
            "q": r["query_text"],
            "p": r["pos_term_text"],
            "h": r.get("hard_neg_term_texts", []) or [],
        }


class RerankJsonlDataset(Dataset):
    """Each line:
    {"query_text": str, "candidate_term_texts": [...], "gold_term_texts": [...]}
    """

    def __init__(self, rows: List[Dict[str, Any]], topk: int):
        self.items: List[Dict[str, Any]] = []
        for r in rows:
            q = r.get("query_text", "")
            cands = (r.get("candidate_term_texts", []) or [])[:topk]
            golds = set(r.get("gold_term_texts", []) or [])
            if not cands:
                continue
            self.items.append({"q": q, "cands": cands, "golds": golds})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


# -------------------------
# Collators
# -------------------------


class TrainCollator:
    """Training: sample k_hard negatives randomly each call (reshuffles across epochs)."""

    def __init__(self, tokenizer, k_hard: int, max_len: int):
        self.tok = tokenizer
        self.k_hard = k_hard
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pair_q: List[str] = []
        pair_t: List[str] = []
        group_sizes: List[int] = []

        for ex in batch:
            q = str(ex["q"])
            p = str(ex["p"])
            h = [str(x) for x in (ex.get("h", []) or []) if str(x).strip()]

            if len(h) >= self.k_hard:
                negs = random.sample(h, self.k_hard)
            elif h:
                rep = (self.k_hard + len(h) - 1) // len(h)
                negs = (h * rep)[: self.k_hard]
            else:
                negs = ["[NO_NEG]"] * self.k_hard

            for t in [p] + negs:
                pair_q.append(q)
                pair_t.append(t)
            group_sizes.append(1 + self.k_hard)

        enc = self.tok(pair_t, pair_q, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "group_sizes": group_sizes}


class DevCollator:
    """Proxy-dev: deterministic neg selection (first k_hard; cycle if insufficient) for stable eval."""

    def __init__(self, tokenizer, k_hard: int, max_len: int):
        self.tok = tokenizer
        self.k_hard = k_hard
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pair_q: List[str] = []
        pair_t: List[str] = []
        group_sizes: List[int] = []

        for ex in batch:
            q = str(ex["q"])
            p = str(ex["p"])
            h = [str(x) for x in (ex.get("h", []) or []) if str(x).strip()]

            if len(h) >= self.k_hard:
                negs = h[: self.k_hard]
            elif h:
                rep = (self.k_hard + len(h) - 1) // len(h)
                negs = (h * rep)[: self.k_hard]
            else:
                negs = ["[NO_NEG]"] * self.k_hard

            for t in [p] + negs:
                pair_q.append(q)
                pair_t.append(t)
            group_sizes.append(1 + self.k_hard)

        enc = self.tok(pair_t, pair_q, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "group_sizes": group_sizes}


# -------------------------
# CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train cross-encoder (JSONL inputs).")

    p.add_argument("--model_name", type=str, default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    p.add_argument("--init_from_biencoder", type=str, default=None, help="Optional bi-encoder dir to init backbone.")

    p.add_argument("--output_dir", type=Path, default=Path("./ckpt-cross"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--train_jsonl", type=Path, required=True)
    p.add_argument("--dev_jsonl", type=Path, required=True)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)

    p.add_argument("--batch_size", type=int, default=8, help="Number of queries per batch.")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--k_hard", type=int, default=4)

    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_steps", type=int, default=50)
    p.add_argument("--save_every_steps", type=int, default=0)

    p.add_argument("--loss_type", choices=["listwise", "pairwise"], default="listwise")
    p.add_argument("--margin", type=float, default=1.0)

    # dev eval mode
    p.add_argument("--dev_eval_mode", choices=["proxy", "rerank"], default="proxy")
    p.add_argument("--eval_early_metric", type=str, default=None, help="Override early-stop metric key.")
    # rerank-only knobs
    p.add_argument("--dev_eval_topk", type=int, default=100, help="Candidates per query (rerank).")
    p.add_argument("--eval_ks", type=str, default="5,10,20,100")
    p.add_argument("--eval_micro_bsz", type=int, default=64)
    p.add_argument("--amp_eval", action="store_true")

    return p.parse_args()


# -------------------------
# Main
# -------------------------


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = build_cross_encoder(args.model_name).to(device)
    if args.init_from_biencoder:
        init_encoder_from_biencoder(model, args.init_from_biencoder)

    # data
    train_rows = load_jsonl(args.train_jsonl)
    dev_rows = load_jsonl(args.dev_jsonl)

    train_loader = DataLoader(
        ProxyJsonlDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=TrainCollator(tokenizer, k_hard=args.k_hard, max_len=args.max_len),
    )

    if args.dev_eval_mode == "proxy":
        dev_loader = DataLoader(
            ProxyJsonlDataset(dev_rows),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            collate_fn=DevCollator(tokenizer, k_hard=args.k_hard, max_len=args.max_len),
        )
        dev_rerank_iter = None
    else:
        # In rerank mode, dev_jsonl should have candidate_term_texts/gold_term_texts.
        dev_loader = None
        dev_rerank_iter = RerankJsonlDataset(dev_rows, topk=args.dev_eval_topk)

    # optim
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.lr)

    steps_per_epoch = max(1, math.floor(len(train_loader)))
    total_updates = (steps_per_epoch * args.epochs) // max(1, args.grad_accum_steps)
    total_updates = max(1, total_updates)
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    # early stopping
    best = -1.0
    bad_epochs = 0

    model.train()
    global_step = 0
    running_loss = 0.0

    eval_ks = tuple(int(x) for x in args.eval_ks.split(",") if x.strip())

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            group_sizes = batch["group_sizes"]

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)

            if args.loss_type == "listwise":
                loss, metric_val = listwise_loss_and_acc1(logits, group_sizes, device=device)
                metric_name = "acc@1"
            else:
                loss, metric_val = pairwise_margin_loss_and_winrate(logits, group_sizes, margin=args.margin)
                metric_name = "win_rate"

            loss = loss / max(1, args.grad_accum_steps)
            loss.backward()
            running_loss += float(loss.item())

            if (step % args.grad_accum_steps) == 0:
                nn_utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if args.log_steps and (global_step % args.log_steps == 0):
                    avg_loss = running_loss / args.log_steps
                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", metric_name: f"{metric_val:.3f}", "lr": f"{lr:.2e}"})
                    running_loss = 0.0

                if args.save_every_steps and (global_step % args.save_every_steps == 0):
                    save_ckpt(args.output_dir / f"step-{global_step}", model, tokenizer)

        # -------- end-of-epoch eval --------
        if args.dev_eval_mode == "proxy":
            dev_metrics = eval_proxy(model, dev_loader, device, args.loss_type, args.margin)
            # default early metric
            early_key = args.eval_early_metric or (f"dev/{metric_name}" if f"dev/{metric_name}" in dev_metrics else "dev/acc@1")
            early_val = float(dev_metrics.get(early_key, 0.0))
            print(" | ".join([f"{k}={v:.4f}" for k, v in dev_metrics.items()]))

        else:
            dev_metrics = eval_rerank_streaming(
                model=model,
                tokenizer=tokenizer,
                dev_iter=dev_rerank_iter,
                device=device,
                ks=eval_ks,
                max_len=args.max_len,
                micro_bsz=args.eval_micro_bsz,
                amp_eval=args.amp_eval,
            )
            # default early metric: nDCG@100 if present else nDCG@{maxK} else Acc@{maxK}
            cand_keys = [
                f"dev/nDCG@100",
                f"dev/nDCG@{max(eval_ks)}",
                f"dev/Acc@{max(eval_ks)}",
            ]
            early_key = args.eval_early_metric or next((k for k in cand_keys if k in dev_metrics), list(dev_metrics.keys())[-1])
            early_val = float(dev_metrics.get(early_key, 0.0))
            print(" | ".join([f"{k}={v:.4f}" for k, v in dev_metrics.items()]))

        # save_ckpt(args.output_dir / f"epoch-{epoch}", model, tokenizer)

        improved = early_val > best + 1e-6
        if improved:
            best = early_val
            bad_epochs = 0
            save_ckpt(args.output_dir / "best", model, tokenizer)
            print(f"[Dev] best={best:.4f} ({early_key})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[EarlyStop] No improvement for {args.patience} epoch(s). best={best:.4f} ({early_key})")
                break

    print(f"[Done] Training finished. Best dev={best:.4f}")


if __name__ == "__main__":
    main()
