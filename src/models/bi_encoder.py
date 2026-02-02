#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared bi-encoder model (query/term) with mean pooling + optional projection."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B, L, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                   # [B, H]
    denom = mask.sum(dim=1).clamp(min=1e-9)                          # [B, 1]
    return summed / denom


class SharedBiEncoder(nn.Module):
    """Bi-encoder with a shared HF encoder.

    Returns scaled cosine similarity logits:
      logits = (q_vec @ t_vec.T) / tau
    """

    def __init__(self, model_name: str, proj_dim: int = 0, tau: float = 0.07) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = int(self.encoder.config.hidden_size)

        self.proj: Optional[nn.Linear] = None
        if proj_dim and proj_dim > 0:
            self.proj = nn.Linear(hidden, int(proj_dim))

        self.register_buffer("tau", torch.tensor(float(tau), dtype=torch.float32))

    def encode_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(out.last_hidden_state, attention_mask)
        if self.proj is not None:
            pooled = self.proj(pooled)
        return F.normalize(pooled, dim=-1)

    def forward(self, q_inputs: Dict[str, torch.Tensor], t_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        q_vec = self.encode_batch(q_inputs["input_ids"], q_inputs["attention_mask"])  # [B, D]
        t_vec = self.encode_batch(t_inputs["input_ids"], t_inputs["attention_mask"])  # [N, D]
        return (q_vec @ t_vec.T) / self.tau
