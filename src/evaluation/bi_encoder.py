#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation helpers for bi-encoder (acc@1 on per-example candidates)."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


@torch.no_grad()
def eval_acc1_local_candidates(
    model,
    tokenizer,
    eval_examples: List[Dict[str, Any]],
    q_max_len: int,
    t_max_len: int,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Compute acc@1 by ranking within each example's (pos + hard_negs).

    For each example:
      terms = [pos_term_text] + hard_neg_term_texts[:K]
      score_i = sim(query, term_i)
      hit = argmax(score) == 0
    """
    model.eval()

    n = 0
    hit = 0

    for i in range(0, len(eval_examples), batch_size):
        batch = eval_examples[i : i + batch_size]

        q_texts = [str(ex.get("query_text", "")) for ex in batch]
        term_groups: List[List[str]] = []
        for ex in batch:
            pos = str(ex.get("pos_term_text", ""))
            negs = ex.get("hard_neg_term_texts", []) or []
            negs = [str(x) for x in negs if str(x).strip()]
            term_groups.append([pos] + negs)

        # flatten terms
        flat_terms: List[str] = []
        offsets = [0]
        for g in term_groups:
            flat_terms.extend(g)
            offsets.append(offsets[-1] + len(g))

        q_enc = tokenizer(q_texts, padding=True, truncation=True, max_length=q_max_len, return_tensors="pt")
        t_enc = tokenizer(flat_terms, padding=True, truncation=True, max_length=t_max_len, return_tensors="pt")

        q_vec = model.encode_batch(q_enc["input_ids"].to(device), q_enc["attention_mask"].to(device))  # [B, D]
        t_vec = model.encode_batch(t_enc["input_ids"].to(device), t_enc["attention_mask"].to(device))  # [T, D]

        for bi in range(len(batch)):
            start, end = offsets[bi], offsets[bi + 1]
            scores = (q_vec[bi : bi + 1] @ t_vec[start:end].T).squeeze(0)  # [1+K]
            if scores.numel() == 0:
                continue
            pred = int(torch.argmax(scores).item())
            if pred == 0:
                hit += 1
            n += 1

    return {"eval/acc@1": (hit / n) if n > 0 else 0.0}
