#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation helpers for cross-encoder."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def listwise_loss_and_acc1(
    scores_flat: torch.Tensor,
    group_sizes: Sequence[int],
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """Listwise CE where positive is always at index 0 within each group."""
    g = int(group_sizes[0])
    if not all(int(x) == g for x in group_sizes):
        raise ValueError("All groups must have the same size for listwise loss.")
    scores = scores_flat.view(-1, g)
    targets = torch.zeros(scores.size(0), dtype=torch.long, device=device)
    loss = F.cross_entropy(scores, targets)
    acc1 = (scores.argmax(dim=1) == targets).float().mean().item()
    return loss, acc1


def pairwise_margin_loss_and_winrate(
    scores_flat: torch.Tensor,
    group_sizes: Sequence[int],
    margin: float = 1.0,
) -> Tuple[torch.Tensor, float]:
    """Pairwise margin loss: pos vs all negatives in each group."""
    g = int(group_sizes[0])
    if not all(int(x) == g for x in group_sizes):
        raise ValueError("All groups must have the same size for pairwise loss.")
    scores = scores_flat.view(-1, g)
    s_pos = scores[:, 0].unsqueeze(1)  # [B,1]
    s_neg = scores[:, 1:]              # [B,G-1]
    loss = F.relu(margin - s_pos + s_neg).mean()
    win_rate = (s_pos > s_neg).float().mean().item()
    return loss, win_rate


@torch.no_grad()
def eval_proxy(
    model,
    dev_loader,
    device: torch.device,
    loss_type: str,
    margin: float,
) -> Dict[str, float]:
    """Proxy dev: evaluate on (pos + K hard negs) groups."""
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    steps = 0

    for batch in dev_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)

        if loss_type == "listwise":
            loss, metric = listwise_loss_and_acc1(logits, batch["group_sizes"], device=device)
            metric_name = "acc@1"
        else:
            loss, metric = pairwise_margin_loss_and_winrate(logits, batch["group_sizes"], margin=margin)
            metric_name = "win_rate"

        total_loss += float(loss.item())
        total_metric += float(metric)
        steps += 1

    model.train()
    return {
        "dev/loss": total_loss / max(1, steps),
        f"dev/{metric_name}": total_metric / max(1, steps),
    }


@torch.no_grad()
def eval_rerank_streaming(
    model,
    tokenizer,
    dev_iter: Iterable[dict],
    device: torch.device,
    ks: Sequence[int],
    max_len: int,
    micro_bsz: int = 64,
    amp_eval: bool = False,
) -> Dict[str, float]:
    """Rerank dev: for each query, score all candidates (streamed), then compute ranking metrics.

    Expected each example:
      {
        "q": str,
        "cands": List[str],
        "golds": Set[str]
      }

    Metrics:
      - Acc@K: any gold appears in top-K
      - MRR@K : 1/rank(best_gold) if best_gold_rank<=K else 0
      - nDCG@K: DCG over gold best ranks / IDCG (unique-gold best ranks)
    """
    model.eval()

    def mrr_at_k(best_ranks: List[int], k: int) -> float:
        if not best_ranks:
            return 0.0
        rmin = min(best_ranks)
        return 1.0 / rmin if rmin <= k else 0.0

    def ndcg_at_k(best_ranks: List[int], k: int) -> float:
        if not best_ranks:
            return 0.0
        dcg = sum(1.0 / math.log2(r + 1) for r in best_ranks if r <= k)
        m = min(k, len(best_ranks))
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, m + 1))
        return dcg / idcg if idcg > 0 else 0.0

    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if amp_eval and torch.cuda.is_available()
        else torch.cuda.amp.autocast(enabled=False)
    )

    best_ranks_all: List[List[int]] = []
    acc_hits: Dict[int, List[int]] = {int(k): [] for k in ks}

    for ex in tqdm(dev_iter, desc="Dev Rerank Eval"):
        q = str(ex["q"])
        cands = [str(x) for x in (ex.get("cands", []) or [])]
        golds: Set[str] = set(str(x) for x in (ex.get("golds", set()) or set()))
        if not cands:
            continue

        scores_chunks: List[torch.Tensor] = []
        for i in range(0, len(cands), micro_bsz):
            chunk = cands[i : i + micro_bsz]
            enc = tokenizer(
                chunk,
                [q] * len(chunk),
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device, non_blocking=True)
            attention_mask = enc["attention_mask"].to(device, non_blocking=True)
            with autocast_ctx:
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            scores_chunks.append(logits.detach().float().cpu())
            del enc, input_ids, attention_mask, logits

        scores = torch.cat(scores_chunks, dim=0).numpy()
        order = np.argsort(-scores)  # high -> low
        rank_map = {int(idx): rank for rank, idx in enumerate(order.tolist(), start=1)}

        text2idxs: Dict[str, List[int]] = {}
        for i, t in enumerate(cands):
            text2idxs.setdefault(t, []).append(i)

        best_ranks: List[int] = []
        for g in golds:
            if g in text2idxs:
                best_ranks.append(min(rank_map[i] for i in text2idxs[g]))

        best_ranks_all.append(best_ranks)
        for k in ks:
            acc_hits[int(k)].append(1 if (best_ranks and min(best_ranks) <= int(k)) else 0)

    out: Dict[str, float] = {}
    if not best_ranks_all:
        for k in ks:
            out[f"dev/Acc@{int(k)}"] = 0.0
            out[f"dev/MRR@{int(k)}"] = 0.0
            out[f"dev/nDCG@{int(k)}"] = 0.0
        model.train()
        return out

    for k in ks:
        k = int(k)
        out[f"dev/Acc@{k}"] = float(np.mean(acc_hits[k])) if acc_hits[k] else 0.0
        out[f"dev/MRR@{k}"] = float(np.mean([mrr_at_k(rr, k) for rr in best_ranks_all]))
        out[f"dev/nDCG@{k}"] = float(np.mean([ndcg_at_k(rr, k) for rr in best_ranks_all]))

    model.train()
    return out
