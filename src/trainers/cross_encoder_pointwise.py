#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a TRUE pointwise cross-encoder from JSONL inputs.

This script is intentionally separate from src.trainers.cross_encoder so that the
original pairwise/listwise trainer remains untouched.

Training input JSONL (proxy format; one line per positive query--concept pair):
{
  "query_text": str,
  "pos_term_text": str,
  "hard_neg_term_texts": [str, ...]
}

For each record, the trainer builds independent binary classification examples:
  (query_text, pos_term_text) -> label 1
  (query_text, hard_neg_term_text) -> label 0

This is TRUE pointwise BCE training:
  BCEWithLogitsLoss(s_CE(p, c), y)

Dev modes:
  --dev_eval_mode proxy
    Expects the same proxy JSONL format. Reports BCE loss, pair-level accuracy,
    positive accuracy, negative accuracy, and simple ranking acc@1 within each
    query group for monitoring.

  --dev_eval_mode rerank
    Expects rerank JSONL with candidate_term_texts/gold_term_texts.
    Reuses eval_rerank_streaming from src.evaluation.cross_encoder, so the output
    is comparable to the original trainer's rerank dev evaluation.

Notes:
- The tokenizer pair order defaults to the original code style: tokenizer(term, query).
- Use --query_first if your model was trained/evaluated with tokenizer(query, term).
"""

from __future__ import annotations

import os

offline_env = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TRITON_DISABLE_AUTO_DOWNLOAD": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_EVALUATE_OFFLINE": "1",
}
for k, v in offline_env.items():
    os.environ[k] = v

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.evaluation.cross_encoder import eval_rerank_streaming
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
        self.rows: List[Dict[str, Any]] = []
        for r in rows:
            q = r.get("query_text", "")
            p = r.get("pos_term_text", "")
            if isinstance(q, str) and q.strip() and isinstance(p, str) and p.strip():
                self.rows.append(r)

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
            if isinstance(q, str) and q.strip() and cands:
                self.items.append({"q": q, "cands": cands, "golds": golds})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


# -------------------------
# Collators
# -------------------------


class PointwiseTrainCollator:
    """Build independent pair-level BCE examples.

    For each query group, sample k_hard negatives and create:
      one positive pair with label 1
      k_hard negative pairs with label 0

    group_sizes are kept only for monitoring/eval convenience.
    """

    def __init__(self, tokenizer, k_hard: int, max_len: int, query_first: bool = False):
        self.tok = tokenizer
        self.k_hard = k_hard
        self.max_len = max_len
        self.query_first = query_first

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pair_q: List[str] = []
        pair_t: List[str] = []
        labels: List[float] = []
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

            # positive
            pair_q.append(q)
            pair_t.append(p)
            labels.append(1.0)

            # negatives
            for t in negs:
                pair_q.append(q)
                pair_t.append(t)
                labels.append(0.0)

            group_sizes.append(1 + self.k_hard)

        if self.query_first:
            enc = self.tok(pair_q, pair_t, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        else:
            # Match the original trainer: tokenizer(pair_t, pair_q)
            enc = self.tok(pair_t, pair_q, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float32),
            "group_sizes": group_sizes,
        }


class PointwiseDevCollator:
    """Deterministic dev negative selection for stable proxy eval."""

    def __init__(self, tokenizer, k_hard: int, max_len: int, query_first: bool = False):
        self.tok = tokenizer
        self.k_hard = k_hard
        self.max_len = max_len
        self.query_first = query_first

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pair_q: List[str] = []
        pair_t: List[str] = []
        labels: List[float] = []
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

            pair_q.append(q)
            pair_t.append(p)
            labels.append(1.0)
            for t in negs:
                pair_q.append(q)
                pair_t.append(t)
                labels.append(0.0)
            group_sizes.append(1 + self.k_hard)

        if self.query_first:
            enc = self.tok(pair_q, pair_t, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        else:
            enc = self.tok(pair_t, pair_q, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float32),
            "group_sizes": group_sizes,
        }


# -------------------------
# Eval helpers
# -------------------------


@torch.no_grad()
def eval_pointwise_proxy(model, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    total_loss = 0.0
    total = 0
    correct = 0
    pos_total = pos_correct = 0
    neg_total = neg_correct = 0
    group_total = group_acc1 = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        group_sizes = batch["group_sizes"]

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
        loss = loss_fn(logits, labels)
        total_loss += float(loss.item())
        total += int(labels.numel())

        pred = (logits >= 0.0).float()
        correct += int((pred == labels).sum().item())

        pos_mask = labels == 1.0
        neg_mask = labels == 0.0
        pos_total += int(pos_mask.sum().item())
        neg_total += int(neg_mask.sum().item())
        pos_correct += int((pred[pos_mask] == labels[pos_mask]).sum().item()) if pos_mask.any() else 0
        neg_correct += int((pred[neg_mask] == labels[neg_mask]).sum().item()) if neg_mask.any() else 0

        # Ranking-style monitoring within each query group: positive is first item.
        offset = 0
        for gs in group_sizes:
            group_logits = logits[offset : offset + gs]
            if group_logits.numel() > 0:
                if int(torch.argmax(group_logits).item()) == 0:
                    group_acc1 += 1
                group_total += 1
            offset += gs

    model.train()
    return {
        "dev/bce_loss": total_loss / max(1, total),
        "dev/pair_acc": correct / max(1, total),
        "dev/pos_acc": pos_correct / max(1, pos_total),
        "dev/neg_acc": neg_correct / max(1, neg_total),
        "dev/group_acc@1": group_acc1 / max(1, group_total),
    }


# -------------------------
# CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TRUE pointwise cross-encoder with BCEWithLogitsLoss.")

    p.add_argument("--model_name", type=str, default="/home/shanliu/llm/cambridgeltl-SapBERT-from-PubMedBERT-fulltext")
    p.add_argument("--init_from_biencoder", type=str, default=None, help="Optional bi-encoder dir to init backbone.")

    p.add_argument("--output_dir", type=Path, default=Path("./ckpt-cross-pointwise"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--train_jsonl", type=Path, required=True)
    p.add_argument("--dev_jsonl", type=Path, required=True)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)

    p.add_argument("--batch_size", type=int, default=8, help="Number of queries per batch before pair expansion.")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--k_hard", type=int, default=4)
    p.add_argument("--dev_topk_neg", type=int, default=20)

    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_steps", type=int, default=50)
    p.add_argument("--save_every_steps", type=int, default=0)

    # True pointwise knobs
    p.add_argument("--pos_weight", type=float, default=None, help="Optional BCE pos_weight. Usually leave unset here because each query has 1 pos + k neg.")
    p.add_argument("--query_first", action="store_true", help="Use tokenizer(query, term) instead of tokenizer(term, query).")

    # dev eval mode
    p.add_argument("--dev_eval_mode", choices=["proxy", "rerank"], default="proxy")
    p.add_argument("--eval_early_metric", type=str, default=None, help="Override early-stop metric key.")

    # rerank-only knobs
    p.add_argument("--dev_eval_topk", type=int, default=100, help="Candidates per query for rerank dev eval.")
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

    train_rows = load_jsonl(args.train_jsonl)
    dev_rows = load_jsonl(args.dev_jsonl)

    train_dataset = ProxyJsonlDataset(train_rows)
    if not train_dataset:
        raise ValueError("No valid training rows found. Expected query_text + pos_term_text.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=PointwiseTrainCollator(
            tokenizer,
            k_hard=args.k_hard,
            max_len=args.max_len,
            query_first=args.query_first,
        ),
    )

    if args.dev_eval_mode == "proxy":
        dev_dataset = ProxyJsonlDataset(dev_rows)
        if not dev_dataset:
            raise ValueError("No valid proxy dev rows found. Expected query_text + pos_term_text.")
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            collate_fn=PointwiseDevCollator(
                tokenizer,
                k_hard=args.dev_topk_neg,
                max_len=args.max_len,
                query_first=args.query_first,
            ),
        )
        dev_rerank_iter = None
    else:
        dev_loader = None
        dev_rerank_iter = RerankJsonlDataset(dev_rows, topk=args.dev_eval_topk)
        if not dev_rerank_iter:
            raise ValueError("No valid rerank dev rows found. Expected query_text + candidate_term_texts.")

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.lr)

    steps_per_epoch = max(1, math.floor(len(train_loader)))
    total_updates = (steps_per_epoch * args.epochs) // max(1, args.grad_accum_steps)
    total_updates = max(1, total_updates)
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    if args.pos_weight is not None:
        pos_weight = torch.tensor(float(args.pos_weight), dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    best = -1.0
    bad_epochs = 0
    global_step = 0
    running_loss = 0.0
    eval_ks = tuple(int(x) for x in args.eval_ks.split(",") if x.strip())

    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            loss = loss_fn(logits, labels)

            with torch.no_grad():
                pred = (logits >= 0.0).float()
                pair_acc = float((pred == labels).float().mean().item())

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
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "pair_acc": f"{pair_acc:.3f}", "lr": f"{lr:.2e}"})
                    running_loss = 0.0

                if args.save_every_steps and (global_step % args.save_every_steps == 0):
                    save_ckpt(args.output_dir / f"step-{global_step}", model, tokenizer)

        # -------- end-of-epoch eval --------
        if args.dev_eval_mode == "proxy":
            assert dev_loader is not None
            dev_metrics = eval_pointwise_proxy(model, dev_loader, device)
            early_key = args.eval_early_metric or "dev/group_acc@1"
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
            cand_keys = [
                "dev/nDCG@100",
                f"dev/nDCG@{max(eval_ks)}",
                f"dev/Acc@{max(eval_ks)}",
            ]
            early_key = args.eval_early_metric or next((k for k in cand_keys if k in dev_metrics), list(dev_metrics.keys())[-1])
            early_val = float(dev_metrics.get(early_key, 0.0))
            print(" | ".join([f"{k}={v:.4f}" for k, v in dev_metrics.items()]))

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
