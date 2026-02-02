#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-encoder model helpers (seq classification) + optional init from bi-encoder."""

from __future__ import annotations

from collections import OrderedDict
from typing import Tuple

import torch
from transformers import AutoModel, AutoModelForSequenceClassification


def build_cross_encoder(model_name: str) -> AutoModelForSequenceClassification:
    """Regression-style score (num_labels=1)."""
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)


def _get_backbone_and_prefix(model: AutoModelForSequenceClassification) -> Tuple[torch.nn.Module, str]:
    # Common HF backbones: roberta/bert/distilbert/deberta/electra...
    if hasattr(model, "roberta"):
        return model.roberta, "roberta."
    if hasattr(model, "bert"):
        return model.bert, "bert."
    if hasattr(model, "distilbert"):
        return model.distilbert, "distilbert."
    if hasattr(model, "deberta"):
        return model.deberta, "deberta."
    if hasattr(model, "deberta_v2"):
        return model.deberta_v2, "deberta_v2."
    if hasattr(model, "electra"):
        return model.electra, "electra."
    raise ValueError("Unsupported backbone. Extend _get_backbone_and_prefix for your model type.")


def _strip_prefix(state_dict: "OrderedDict[str, torch.Tensor]", prefix: str) -> "OrderedDict[str, torch.Tensor]":
    out = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
    return out or state_dict


def init_encoder_from_biencoder(
    cross_encoder: AutoModelForSequenceClassification,
    biencoder_dir: str,
) -> None:
    """Initialize cross-encoder backbone from a bi-encoder checkpoint dir.

    The bi-encoder checkpoint should be saved by `AutoModel.save_pretrained(...)`.
    We only load the backbone weights; the classification head remains as-is.
    """
    backbone, backbone_prefix = _get_backbone_and_prefix(cross_encoder)
    bi_encoder = AutoModel.from_pretrained(biencoder_dir)
    bi_sd = bi_encoder.state_dict()
    bi_sd = _strip_prefix(bi_sd, backbone_prefix)
    missing, unexpected = backbone.load_state_dict(bi_sd, strict=False)
    print(
        f"[Init] Loaded encoder from {biencoder_dir}. "
        f"Missing={len(missing)} Unexpected={len(unexpected)}"
    )
