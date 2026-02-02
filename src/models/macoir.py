#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MA-COIR model definition and decoding utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartConfig, BartModel
from transformers.models.bart.modeling_bart import BartPreTrainedModel, Seq2SeqLMOutput, shift_tokens_right


@dataclass
class RestrictedVocab:
    allowed_ids: List[int]
    orig2restricted: Dict[int, int]
    restricted2orig: Dict[int, int]


def build_restricted_vocab(tokenizer) -> RestrictedVocab:
    vocab = tokenizer.get_vocab()
    wanted = {str(d) for d in range(10)} | {"-", ";"}
    allowed = [tid for tok, tid in vocab.items() if tok in wanted]
    for special in [tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id]:
        if special is not None:
            allowed.append(special)
    allowed = sorted(set(allowed))
    orig2restricted = {orig: i for i, orig in enumerate(allowed)}
    restricted2orig = {i: orig for i, orig in enumerate(allowed)}
    return RestrictedVocab(allowed_ids=allowed, orig2restricted=orig2restricted, restricted2orig=restricted2orig)


def make_prefix_allowed_fn(allowed_ids: Iterable[int]):
    allowed_set = set(allowed_ids)

    def fn(batch_id: int, input_ids: torch.LongTensor) -> List[int]:
        return list(allowed_set)

    return fn


class MACOIR(BartPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]

    def __init__(self, config: BartConfig, model: BartModel, tokenizer) -> None:
        super().__init__(config)
        self.model = model
        self.model.eval()
        self.config.decoder_start_token_id = self.model.config.decoder_start_token_id

        self.restricted_vocab = build_restricted_vocab(tokenizer)
        self.num_labels = len(self.restricted_vocab.allowed_ids)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.num_labels)))
        self.lm_head = nn.Linear(config.d_model, self.num_labels, bias=False)
        nn.init.xavier_uniform_(self.lm_head.weight)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Seq2SeqLMOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        logits_restricted = self.lm_head(outputs.last_hidden_state) + self.final_logits_bias.to(
            outputs.last_hidden_state.device
        )
        dec_emb = self.model.decoder.embed_tokens.weight[self.restricted_vocab.allowed_ids]
        score_1 = F.linear(outputs.last_hidden_state, dec_emb)
        lm_logits_restricted = (score_1 + logits_restricted) / 2.0

        vocab_size = self.model.config.vocab_size
        lm_logits_full = outputs.last_hidden_state.new_full(
            (outputs.last_hidden_state.size(0), outputs.last_hidden_state.size(1), vocab_size),
            fill_value=-1e24,
        )
        for new_idx, orig_id in self.restricted_vocab.restricted2orig.items():
            lm_logits_full[:, :, orig_id] = lm_logits_restricted[:, :, new_idx]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(lm_logits_full.reshape(-1, vocab_size), labels.reshape(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits_full,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past_key_values=None,
            attention_mask=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            remove = past_length if decoder_input_ids.shape[1] > past_length else decoder_input_ids.shape[1] - 1
            decoder_input_ids = decoder_input_ids[:, remove:]
        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def _reorder_cache(self, past_key_values, beam_idx: torch.LongTensor):
        if past_key_values is None:
            return None
        reordered_past = []
        for layer_past in past_key_values:
            if layer_past is None:
                reordered_past.append(None)
                continue
            reordered_layer = tuple(
                (p.index_select(0, beam_idx) if p is not None else None)
                for p in layer_past
            )
            reordered_past.append(reordered_layer)
        return tuple(reordered_past)


def enable_bart_reorder_cache_for_generation(bart_model) -> None:
    if hasattr(bart_model, "_reorder_cache") and callable(getattr(bart_model, "_reorder_cache")):
        return

    def _bart_reorder_cache(this, past_key_values, beam_idx: torch.LongTensor):
        if past_key_values is None:
            return None
        reordered = []
        for layer_past in past_key_values:
            if layer_past is None:
                reordered.append(None)
                continue
            reordered_layer = tuple((p.index_select(0, beam_idx) if p is not None else None) for p in layer_past)
            reordered.append(reordered_layer)
        return tuple(reordered)

    bart_model._reorder_cache = _bart_reorder_cache.__get__(bart_model, bart_model.__class__)
