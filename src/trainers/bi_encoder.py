#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a shared bi-encoder with in-batch negatives (JSONL triples).

Input JSONL (one per query-positive pair):
{
  "query_text": "...",
  "pos_term_text": "...",
  "hard_neg_term_texts": ["...", "...", ...]
}

Training:
- shared encoder for query/term
- per query: 1 positive + K hard negatives (from JSONL)
- in-batch negatives: full softmax over all terms in the batch (B*(1+K))

Evaluation (end of each epoch):
- retrieval accuracy@K over a term catalog built from eval JSONL (pos + hard negs)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.evaluation.bi_encoder import eval_acc1_local_candidates
from src.models.bi_encoder import SharedBiEncoder


# -------------------------
# Utilities
# -------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# -------------------------
# Dataset / Collator
# -------------------------


class BiencoderTriplesDataset(Dataset):
    """Keep only needed fields; ensure each example has >= hard_neg_k negatives."""

    def __init__(self, rows: List[Dict[str, Any]], hard_neg_k: int) -> None:
        self.rows = []
        self.hard_neg_k = int(hard_neg_k)

        for r in rows:
            q = r.get("query_text", "")
            pos = r.get("pos_term_text", "")
            hards = r.get("hard_neg_term_texts", []) or []
            if not isinstance(q, str) or not q.strip():
                continue
            if not isinstance(pos, str) or not pos.strip():
                continue
            if not isinstance(hards, list):
                hards = []

            hards = [str(x) for x in hards if str(x).strip()]
            if len(hards) == 0:
                hards = ["[NO_NEG]"]
            while len(hards) < self.hard_neg_k:
                hards.append(hards[-1])

            self.rows.append(
                {
                    "query_text": q,
                    "pos_term_text": pos,
                    "hard_neg_term_texts": hards[: self.hard_neg_k],
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


@dataclass
class BiencoderCollator:
    tokenizer: Any
    q_max_len: int
    t_max_len: int
    hard_neg_k: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        q_texts = [ex["query_text"] for ex in batch]
        q_enc = self.tokenizer(
            q_texts,
            padding=True,
            truncation=True,
            max_length=self.q_max_len,
            return_tensors="pt",
        )

        term_texts: List[str] = []
        pos_indices: List[int] = []
        for ex in batch:
            start = len(term_texts)
            term_texts.append(ex["pos_term_text"])
            for j in range(self.hard_neg_k):
                term_texts.append(ex["hard_neg_term_texts"][j])
            pos_indices.append(start)

        t_enc = self.tokenizer(
            term_texts,
            padding=True,
            truncation=True,
            max_length=self.t_max_len,
            return_tensors="pt",
        )

        return {
            "q_input_ids": q_enc["input_ids"],
            "q_attention_mask": q_enc["attention_mask"],
            "t_input_ids": t_enc["input_ids"],
            "t_attention_mask": t_enc["attention_mask"],
            "targets": torch.tensor(pos_indices, dtype=torch.long),
        }


# -------------------------
# Checkpointing
# -------------------------


def save_encoder(output_dir: Path, model: SharedBiEncoder, tokenizer: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save underlying HF encoder (+ optional projection head via state_dict).
    model.encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # If you add a projection layer in your SharedBiEncoder, save the full state dict as well.
    torch.save({"model_state_dict": model.state_dict()}, output_dir / "biencoder.pt")
    print(f"[Save] checkpoint -> {output_dir}")


# -------------------------
# CLI
# -------------------------


# -------------------------
# CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a shared bi-encoder (JSONL triples).")

    p.add_argument("--model_name", type=str, default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    p.add_argument("--output_dir", type=Path, default=Path("./ckpt-biencoder"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--train_jsonl", type=Path, required=True)
    p.add_argument("--eval_jsonl", type=Path, default=None, help="If omitted, use train_jsonl.")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16, help="Number of queries per batch.")
    p.add_argument("--hard_neg_k", type=int, default=2)
    p.add_argument("--q_max_len", type=int, default=512)
    p.add_argument("--t_max_len", type=int, default=64)

    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--log_steps", type=int, default=100)

    # model knobs
    p.add_argument("--proj_dim", type=int, default=0)
    p.add_argument("--tau", type=float, default=0.07)

    # eval knobs (local-candidates acc@1)
    p.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="Batch size for end-of-epoch evaluation (acc@1 over pos+negs).",
    )

    return p.parse_args()


# -------------------------
# Main
# -------------------------


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    train_rows = load_jsonl(args.train_jsonl)
    train_ds = BiencoderTriplesDataset(train_rows, hard_neg_k=args.hard_neg_k)
    collate = BiencoderCollator(
        tokenizer=tokenizer,
        q_max_len=args.q_max_len,
        t_max_len=args.t_max_len,
        hard_neg_k=args.hard_neg_k,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
    )

    eval_path = args.eval_jsonl or args.train_jsonl
    eval_rows = load_jsonl(eval_path)

    model = SharedBiEncoder(model_name=args.model_name, proj_dim=args.proj_dim, tau=args.tau).to(device)

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

    best_score = -1.0
    global_step = 0
    running_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, start=1):
            q_ids = batch["q_input_ids"].to(device, non_blocking=True)
            q_mask = batch["q_attention_mask"].to(device, non_blocking=True)
            t_ids = batch["t_input_ids"].to(device, non_blocking=True)
            t_mask = batch["t_attention_mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            logits = model(
                {"input_ids": q_ids, "attention_mask": q_mask},
                {"input_ids": t_ids, "attention_mask": t_mask},
            )  # [B, B*(1+K)]

            loss = F.cross_entropy(logits, targets) / max(1, args.grad_accum_steps)
            loss.backward()
            running_loss += loss.item()

            if (step % args.grad_accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if args.log_steps and (global_step % args.log_steps == 0):
                    avg = running_loss / args.log_steps
                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({"loss": f"{avg:.4f}", "lr": f"{lr:.2e}", "gs": global_step})
                    running_loss = 0.0

        # ---- End of epoch eval ----
        model.eval()
        eval_metrics = eval_acc1_local_candidates(
            model=model,
            tokenizer=tokenizer,
            eval_examples=eval_rows,
            q_max_len=args.q_max_len,
            t_max_len=args.t_max_len,
            device=device,
            batch_size=args.eval_batch_size,
        )
        print({k: f"{v:.4f}" for k, v in eval_metrics.items()})

        score = float(eval_metrics["eval/acc@1"])
        if score > best_score:
            best_score = score
            save_encoder(args.output_dir / "best", model, tokenizer)

    print(f"[Done] Training finished. Best eval acc@1 = {best_score:.4f}")


if __name__ == "__main__":
    main()
