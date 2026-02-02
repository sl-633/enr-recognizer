#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation helpers for MA-COIR."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def ids_to_tokens_str(ids: List[int], tokenizer) -> List[str]:
    toks = tokenizer.convert_ids_to_tokens(ids)
    return [t.replace("Ġ", "").replace("▁", "").strip() for t in toks]


def parse_sids_from_token_ids(
    seqs: List[List[int]],
    tokenizer,
    tok_ids: Dict[str, int],
    max_len: int,
) -> List[List[str]]:
    hy_id = tok_ids["hyphen_id"]
    semi_id = tok_ids["semi_id"]
    eos_id = tok_ids["eos_id"]
    pad_id = tok_ids["pad_id"]

    out: List[List[str]] = []
    for arr in seqs:
        arr = arr[:max_len]
        toks = ids_to_tokens_str(arr, tokenizer)
        sids: List[str] = []
        cur_seg: List[str] = []
        cur_num = ""

        for i, tid in enumerate(arr):
            if tid in (eos_id, pad_id):
                if cur_num:
                    cur_seg.append(cur_num)
                    cur_num = ""
                if cur_seg:
                    sids.append("-".join(cur_seg))
                    cur_seg = []
                break

            if tid == semi_id:
                if cur_num:
                    cur_seg.append(cur_num)
                    cur_num = ""
                if cur_seg:
                    sids.append("-".join(cur_seg))
                    cur_seg = []
                continue

            if tid == hy_id:
                if cur_num:
                    cur_seg.append(cur_num)
                    cur_num = ""
                continue

            tok = toks[i]
            if tok.isdigit():
                cur_num += tok
            else:
                if cur_num:
                    cur_seg.append(cur_num)
                    cur_num = ""

        if cur_num:
            cur_seg.append(cur_num)
        if cur_seg:
            sids.append("-".join(cur_seg))

        seen = set()
        uniq = []
        for sid in sids:
            if sid and sid not in seen:
                seen.add(sid)
                uniq.append(sid)
        out.append(uniq)
    return out


def compute_micro_prf1_from_ids(
    pred_ids: List[List[int]],
    gold_ids: List[List[int]],
    tokenizer,
    tok_ids: Dict[str, int],
    max_len: int,
    valid_sid_set: Optional[set] = None,
) -> Tuple[float, float, float]:
    pred_sids = parse_sids_from_token_ids(pred_ids, tokenizer, tok_ids, max_len)
    gold_sids = parse_sids_from_token_ids(gold_ids, tokenizer, tok_ids, max_len)

    tp = fp = fn = 0
    for pred, gold in zip(pred_sids, gold_sids):
        if valid_sid_set is not None:
            pred = [s for s in pred if s in valid_sid_set]
        p_set = set(pred)
        g_set = set(gold)
        tp += len(p_set & g_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


@torch.no_grad()
def evaluate(
    model,
    dev_loader: DataLoader,
    device: torch.device,
    tokenizer,
    prefix_fn,
    tok_ids: Dict[str, int],
    gen_max_len: int,
    valid_sid_set: Optional[set] = None,
) -> Tuple[float, float, float]:
    model.eval()
    preds_all: List[List[int]] = []
    gold_all: List[List[int]] = []

    for batch in tqdm(dev_loader, desc="Eval"):
        src_ids, src_mask, tgt_ids, labels, _, _ = batch
        src_ids = src_ids.to(device)
        src_mask = src_mask.to(device)

        beams = model.generate(
            input_ids=src_ids,
            attention_mask=src_mask,
            max_length=gen_max_len,
            prefix_allowed_tokens_fn=prefix_fn,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=1,
            do_sample=False,
        )

        pred_tok = beams[:, 2:].detach().cpu().tolist()
        gold_tok = labels[:, 1:].detach().cpu().tolist()
        preds_all.extend(pred_tok)
        gold_all.extend(gold_tok)

    return compute_micro_prf1_from_ids(
        preds_all,
        gold_all,
        tokenizer,
        tok_ids,
        max_len=gen_max_len - 1,
        valid_sid_set=valid_sid_set,
    )
