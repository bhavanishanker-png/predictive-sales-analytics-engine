"""Text encoding pipeline for hybrid sales prediction.

This module wraps DistilBERT with learned attention pooling and a compact
projection head so downstream fusion receives fixed-size text representations.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer


class TextEncoder(nn.Module):
    """DistilBERT encoder with learned token attention pooling.

    Forward output:
    - text_repr: [batch, output_dim]
    - attention_weights: [batch, seq_len]
    """

    def __init__(
        self,
        output_dim: int = 128,
        model_name: str = "distilbert-base-uncased",
        freeze_bert: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim

        self.bert = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        hidden_size = int(self.bert.config.hidden_size)
        self.attention_weights = nn.Linear(hidden_size, 1)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def unfreeze_last_n_layers(self, n: int = 2) -> None:
        """Unfreeze final `n` transformer layers for controlled fine-tuning."""
        if n <= 0:
            return

        layers = list(self.bert.transformer.layer)
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def tokenize_batch(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a list of conversations for DistilBERT."""
        return self.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text and return pooled representation and token weights."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, T, H]

        scores = self.attention_weights(hidden_states).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        token_weights = torch.softmax(scores, dim=1)  # [B, T]
        pooled = torch.bmm(token_weights.unsqueeze(1), hidden_states).squeeze(1)  # [B, H]
        text_repr = self.projection(pooled)  # [B, output_dim]
        return text_repr, token_weights
