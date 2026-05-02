"""Explainability helpers for the hybrid sales predictor."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import shap
import torch


class HybridExplainer:
    """Combine SHAP, gate, and attention signals into one explanation."""

    def __init__(self, xgb_model, feature_names: Sequence[str]) -> None:
        self.xgb_model = xgb_model
        self.feature_names = list(feature_names)
        self.tree_explainer = shap.TreeExplainer(self.xgb_model)

    def top_shap_features(
        self,
        tabular_features: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return top-k SHAP feature contributions for a single sample."""
        if tabular_features.ndim == 1:
            tabular_features = tabular_features.reshape(1, -1)

        shap_values = self.tree_explainer.shap_values(tabular_features)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        sample_vals = np.asarray(shap_values)[0]
        indices = np.argsort(np.abs(sample_vals))[::-1][:top_k]
        return [(self.feature_names[i], float(sample_vals[i])) for i in indices]

    @staticmethod
    def summarize_gate(gate_values: torch.Tensor) -> Dict[str, float]:
        """Convert gate vector to tabular/text reliance percentages."""
        if gate_values.ndim == 2:
            gate_values = gate_values[0]

        tabular_reliance = float(gate_values.mean().item())
        return {
            "tabular_reliance": tabular_reliance,
            "text_reliance": 1.0 - tabular_reliance,
        }

    @staticmethod
    def map_attention_to_turns(
        conversation_text: str,
        token_attention: torch.Tensor,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Create a coarse turn-level attention summary from token weights."""
        lines = [line.strip() for line in conversation_text.splitlines() if line.strip()]
        if not lines:
            return []

        if token_attention.ndim == 2:
            token_attention = token_attention[0]
        weights = token_attention.detach().cpu().numpy()
        weights = weights[: max(len(lines), 1)]

        if len(weights) < len(lines):
            pad = np.zeros(len(lines) - len(weights), dtype=float)
            weights = np.concatenate([weights, pad], axis=0)

        turn_scores = [(line, float(weights[idx])) for idx, line in enumerate(lines)]
        turn_scores.sort(key=lambda x: x[1], reverse=True)
        return turn_scores[:top_k]

    def unified_explanation(
        self,
        probability: float,
        tabular_features: np.ndarray,
        gate_values: torch.Tensor,
        token_attention: torch.Tensor,
        conversation_text: str,
    ) -> Dict[str, object]:
        """Build a single explanation payload for UI and notebook analysis."""
        gate_summary = self.summarize_gate(gate_values)
        shap_top = self.top_shap_features(tabular_features, top_k=5)
        attention_top = self.map_attention_to_turns(conversation_text, token_attention, top_k=3)

        return {
            "prediction": float(probability),
            "predicted_class": "Win" if probability >= 0.5 else "Loss",
            "risk_level": self.risk_level(probability),
            "gate_weights": gate_summary,
            "shap_top_features": shap_top,
            "attention_top_turns": attention_top,
            "recommendation": self.recommendation(probability, shap_top, attention_top),
        }

    @staticmethod
    def risk_level(probability: float) -> str:
        """Classify prediction confidence into risk bands."""
        if 0.40 <= probability <= 0.60:
            return "High"
        if 0.25 <= probability <= 0.75:
            return "Medium"
        return "Low"

    @staticmethod
    def recommendation(
        probability: float,
        shap_top_features: Sequence[Tuple[str, float]],
        attention_top_turns: Sequence[Tuple[str, float]],
    ) -> str:
        """Generate a simple action-oriented recommendation message."""
        if probability < 0.3:
            base = "Low win probability."
        elif probability < 0.5:
            base = "Leaning toward loss but recoverable."
        elif probability < 0.7:
            base = "Moderate win potential."
        else:
            base = "Strong win potential."

        hints: List[str] = []
        top_text = " ".join(t.lower() for t, _ in attention_top_turns)
        if any(k in top_text for k in ["price", "pricing", "cost", "budget", "expensive"]):
            hints.append("Pricing concerns detected - prepare ROI justification.")
        if any(k in top_text for k in ["competitor", "alternative", "compare", "comparison"]):
            hints.append("Competitor evaluation detected - prepare a comparison brief.")

        feature_map = {name: val for name, val in shap_top_features}
        if feature_map.get("customer_engagement", 0.0) < 0:
            hints.append("Engagement signal is weak - schedule a more interactive session.")
        if feature_map.get("sales_effectiveness", 0.0) < 0:
            hints.append("Rep effectiveness is a drag - coach objection handling.")

        return " ".join([base] + hints).strip()
