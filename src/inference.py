"""Inference utilities for loading and serving hybrid model predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import xgboost as xgb

from src.fusion_model import HybridSalesPredictor
from src.text_pipeline import TextEncoder


class SalesPredictor:
    """Load trained artifacts and provide prediction helper methods."""

    def __init__(
        self,
        model_dir: str = "models",
        device: str = "cpu",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)

        self.model: Optional[HybridSalesPredictor] = None
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.feature_config: Dict[str, object] = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model, XGBoost artifacts, and feature config if present."""
        feature_path = self.model_dir / "feature_config.json"
        if feature_path.exists():
            with open(feature_path, "r", encoding="utf-8") as f:
                self.feature_config = json.load(f)

        xgb_path = self.model_dir / "xgboost_model.json"
        if xgb_path.exists():
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(str(xgb_path))

        model_path = self.model_dir / "hybrid_model.pt"
        if model_path.exists():
            loaded = torch.load(model_path, map_location=self.device)
            if isinstance(loaded, HybridSalesPredictor):
                self.model = loaded.to(self.device).eval()

    def is_ready(self) -> bool:
        """Return True when required model artifacts are available."""
        return self.model is not None and self.xgb_model is not None

    def predict(
        self,
        conversation_text: str,
        tabular_features: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """Run model forward pass and return prediction internals."""
        if self.model is None:
            raise RuntimeError("Hybrid model artifact not loaded.")

        # Tokenize text using model's text encoder tokenizer.
        encoded = self.model.text_encoder.tokenize_batch([conversation_text])
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        tab_tensor = torch.tensor(tabular_features, dtype=torch.float32, device=self.device)
        if tab_tensor.ndim == 1:
            tab_tensor = tab_tensor.unsqueeze(0)

        with torch.no_grad():
            probability, gate, attn = self.model(tab_tensor, input_ids, attention_mask)

        return {
            "probability": probability.squeeze(-1).cpu(),
            "gate": gate.cpu(),
            "attention": attn.cpu(),
        }


def build_untrained_predictor(leaf_dim: int, device: str = "cpu") -> HybridSalesPredictor:
    """Utility for app/dev mode when trained artifact is not available yet."""
    text_encoder = TextEncoder(output_dim=128, freeze_bert=True)
    model = HybridSalesPredictor(text_encoder=text_encoder, repr_dim=128)
    model.set_tab_projection(leaf_dim=leaf_dim)
    return model.to(torch.device(device)).eval()
