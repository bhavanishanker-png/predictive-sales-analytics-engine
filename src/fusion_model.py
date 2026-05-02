"""Hybrid fusion models for tabular + text sales prediction."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.text_pipeline import TextEncoder


class GatedFusion(nn.Module):
    """Per-dimension gated fusion between tabular and text representations."""

    def __init__(self, repr_dim: int = 128) -> None:
        super().__init__()
        self.repr_dim = repr_dim
        self.gate_layer = nn.Linear(repr_dim * 2, repr_dim)

    def forward(
        self,
        tab_repr: torch.Tensor,
        text_repr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tab_repr.shape != text_repr.shape:
            raise ValueError(
                "tab_repr and text_repr must have the same shape, "
                f"got {tuple(tab_repr.shape)} and {tuple(text_repr.shape)}."
            )

        combined = torch.cat([tab_repr, text_repr], dim=1)
        gate = torch.sigmoid(self.gate_layer(combined))
        fused = gate * tab_repr + (1.0 - gate) * text_repr
        return fused, gate


class HybridSalesPredictor(nn.Module):
    """End-to-end hybrid predictor with tabular branch, text branch, and gate."""

    def __init__(
        self,
        text_encoder: Optional[TextEncoder] = None,
        repr_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.repr_dim = repr_dim
        self.text_encoder = text_encoder or TextEncoder(output_dim=repr_dim)

        self.tab_projection: Optional[nn.Module] = None
        self.fusion = GatedFusion(repr_dim=repr_dim)
        self.classifier = nn.Sequential(
            nn.Linear(repr_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def set_tab_projection(self, leaf_dim: int) -> None:
        """Attach tabular projection after leaf one-hot encoding."""
        self.tab_projection = nn.Sequential(
            nn.Linear(leaf_dim, self.repr_dim),
            nn.LayerNorm(self.repr_dim),
            nn.ReLU(),
        )

    def encode_tabular(self, tab_features: torch.Tensor) -> torch.Tensor:
        """Project tabular input into shared representation space."""
        if self.tab_projection is None:
            raise RuntimeError("Call set_tab_projection(leaf_dim=...) before forward.")
        return self.tab_projection(tab_features)

    def forward(
        self,
        tab_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return probability, gate weights, and token attention weights."""
        tab_repr = self.encode_tabular(tab_features)
        text_repr, token_attention = self.text_encoder(input_ids, attention_mask)

        fused_repr, gate = self.fusion(tab_repr, text_repr)
        logits = self.classifier(fused_repr)
        probability = torch.sigmoid(logits)
        return probability, gate, token_attention
