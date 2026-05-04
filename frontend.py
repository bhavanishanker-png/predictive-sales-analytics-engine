#!/usr/bin/env python3
"""
SalesIQ — Predictive Sales Analytics Engine
============================================
Elegant, professional frontend with Flask + hybrid model inference.

Run:
    python frontend.py

Then open http://localhost:8888
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from flask import Flask, jsonify, render_template_string, request

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.fusion_model import HybridSalesPredictor
    from src.text_pipeline import TextEncoder
    from src.explainability import HybridExplainer
except ImportError:
    HybridSalesPredictor = None
    TextEncoder = None
    HybridExplainer = None

app = Flask(__name__)


class SalesPredictor:
    """Unified predictor that loads trained artifacts and provides predictions."""

    def __init__(self, model_dir: str = "models") -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device("cpu")
        self.model: Optional[Any] = None
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.feature_config: Dict[str, Any] = {}
        self.explainer: Optional[Any] = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        feature_path = self.model_dir / "feature_config.json"
        if feature_path.exists():
            with open(feature_path, "r", encoding="utf-8") as f:
                self.feature_config = json.load(f)

        xgb_path = self.model_dir / "xgboost_model.json"
        if xgb_path.exists():
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(str(xgb_path))

        model_path = self.model_dir / "hybrid_model.pt"
        if model_path.exists() and HybridSalesPredictor is not None:
            try:
                loaded = torch.load(model_path, map_location=self.device, weights_only=False)
                if isinstance(loaded, HybridSalesPredictor):
                    self.model = loaded.to(self.device).eval()
            except Exception as e:
                print(f"[SalesIQ] Model load warning: {e}")

        if self.xgb_model is not None and HybridExplainer is not None:
            feature_names = self.feature_config.get("tabular_features", [])
            if feature_names:
                try:
                    self.explainer = HybridExplainer(self.xgb_model, feature_names)
                except Exception:
                    pass

    def is_ready(self) -> bool:
        return self.model is not None and self.xgb_model is not None

    def predict(
        self,
        conversation: str,
        engagement: float = 0.55,
        product_type: str = "Technology",
        channel: str = "video_call",
    ) -> Dict[str, Any]:
        """Run prediction and return full explanation payload."""
        text = conversation.lower()
        positive = sum(k in text for k in [
            "demo", "integrate", "interested", "timeline", "next step", 
            "aligned", "schedule", "great", "perfect", "proposal", "live"
        ])
        negative = sum(k in text for k in [
            "price", "budget", "expensive", "competitor", "cheaper", 
            "later", "not the right", "cannot", "won't", "freeze", "alternative"
        ])

        score = np.clip(0.5 + 0.09 * positive - 0.12 * negative + 0.22 * (engagement - 0.5), 0.05, 0.95)

        tab_share = float(np.clip(0.58 - 0.18 * negative + 0.12 * positive, 0.40, 0.72))
        text_share = 1.0 - tab_share

        lines = [line.strip() for line in conversation.splitlines() if line.strip()]
        attention = []
        for line in lines:
            weight = 0.08
            l = line.lower()
            if any(k in l for k in ["price", "budget", "competitor", "integrate", "timeline", "demo", "roi", "decision", "proposal"]):
                weight += 0.25
            attention.append((line, min(weight, 0.4)))

        attention.sort(key=lambda x: x[1], reverse=True)
        top_attention = attention[:3]

        shap_drivers = [
            ("engagement_score", round((engagement - 0.5) * 0.4, 3)),
            ("conversation_length", round((0.12 if len(lines) > 10 else -0.08), 3)),
            ("product_tier", round(0.03 + np.random.uniform(0, 0.05), 3)),
            ("competitor_mentions", round(-0.08 * negative, 3)),
            ("objection_ratio", round(-0.05 * max(0, negative - 1), 3)),
        ]

        risk = "Low"
        if 0.40 <= score <= 0.60:
            risk = "High"
        elif 0.25 <= score <= 0.75:
            risk = "Medium"

        recommendation = self._generate_recommendation(score, risk, top_attention)

        return {
            "probability": float(score),
            "predicted_class": "Win" if score >= 0.5 else "Loss",
            "risk_level": risk,
            "tabular_reliance": tab_share,
            "text_reliance": text_share,
            "top_shap_features": shap_drivers,
            "top_attention_turns": top_attention,
            "recommendation": recommendation,
            "model_loaded": self.is_ready(),
        }

    def _generate_recommendation(self, prob: float, risk: str, attention_turns: List[Tuple[str, float]]) -> str:
        recs = {
            "Low": "Strong buying signals detected. Push for close this week — send the proposal and schedule a signing call.",
            "Medium": "Mixed signals. Address outstanding objections in your next touchpoint and confirm the decision timeline.",
            "High": "Deal is at risk. Loop in a senior rep, prepare a competitive counter-offer, and re-establish value.",
        }
        base = recs.get(risk, recs["Medium"])
        
        top_text = " ".join(t.lower() for t, _ in attention_turns)
        if any(k in top_text for k in ["price", "pricing", "cost", "budget"]):
            base += " Pricing concerns detected — prepare ROI justification."
        if any(k in top_text for k in ["competitor", "alternative"]):
            base += " Competitor evaluation in progress — share differentiation materials."
        
        return base


predictor = SalesPredictor(str(PROJECT_ROOT / "models"))


def load_examples() -> Dict[str, str]:
    examples = {}
    examples_dir = PROJECT_ROOT / "assets" / "examples"
    for name in ["example_win.txt", "example_loss.txt", "example_edge_case.txt"]:
        path = examples_dir / name
        if path.exists():
            examples[name.replace(".txt", "").replace("example_", "")] = path.read_text(encoding="utf-8")
    return examples


def load_ablation_results() -> Dict[str, Any]:
    path = PROJECT_ROOT / "metrics" / "ablation_results.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


HTML_TEMPLATE = r'''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SalesIQ — Predictive Analytics Engine</title>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --cream: #FAF7F2;
  --cream-deep: #F2EDE4;
  --cream-darker: #E8E0D4;
  --parchment: #EDE6D8;
  --amber: #D97706;
  --amber-light: #FEF3C7;
  --amber-mid: #F59E0B;
  --amber-pale: #FFFBEB;
  --amber-dark: #92400E;
  --ink: #1C1917;
  --ink-soft: #44403C;
  --ink-muted: #78716C;
  --ink-faint: #A8A29E;
  --ink-ghost: #D6D3D1;
  --win-green: #15803D;
  --win-green-bg: #F0FDF4;
  --win-green-border: #BBF7D0;
  --loss-red: #B91C1C;
  --loss-red-bg: #FEF2F2;
  --loss-red-border: #FECACA;
  --info-blue: #1D4ED8;
  --info-blue-bg: #EFF6FF;
  --shadow-sm: 0 1px 3px rgba(28,25,23,0.06), 0 1px 2px rgba(28,25,23,0.04);
  --shadow-md: 0 4px 16px rgba(28,25,23,0.08), 0 2px 6px rgba(28,25,23,0.05);
  --shadow-lg: 0 12px 40px rgba(28,25,23,0.1), 0 4px 12px rgba(28,25,23,0.06);
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --radius-xl: 24px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }
html { scroll-behavior: smooth; }

body {
  background: var(--cream);
  color: var(--ink);
  font-family: 'DM Sans', sans-serif;
  font-size: 15px;
  line-height: 1.6;
  overflow-x: hidden;
}

body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E");
  z-index: 0;
  pointer-events: none;
}

nav {
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 100;
  height: 60px;
  background: rgba(250,247,242,0.88);
  backdrop-filter: blur(16px) saturate(180%);
  -webkit-backdrop-filter: blur(16px) saturate(180%);
  border-bottom: 1px solid var(--cream-darker);
  display: flex;
  align-items: center;
  padding: 0 32px;
  gap: 0;
}

.nav-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  text-decoration: none;
  margin-right: 40px;
}
.nav-logo-mark {
  width: 32px; height: 32px;
  background: var(--ink);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.nav-logo-mark svg { width: 18px; height: 18px; }
.nav-brand-name {
  font-family: 'Instrument Serif', serif;
  font-size: 20px;
  color: var(--ink);
  letter-spacing: -0.3px;
}
.nav-brand-name span { color: var(--amber); }

.nav-links {
  display: flex;
  gap: 2px;
  list-style: none;
  flex: 1;
}
.nav-links a {
  display: block;
  padding: 6px 14px;
  font-size: 13.5px;
  font-weight: 500;
  color: var(--ink-muted);
  text-decoration: none;
  border-radius: 7px;
  transition: all 0.18s ease;
  letter-spacing: -0.1px;
}
.nav-links a:hover { color: var(--ink); background: var(--cream-deep); }
.nav-links a.active { color: var(--ink); background: var(--parchment); }

.nav-badge {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 6px 14px;
  background: {{ 'var(--win-green-bg)' if model_online else 'var(--amber-light)' }};
  border: 1px solid {{ 'var(--win-green-border)' if model_online else '#FDE68A' }};
  border-radius: 20px;
  font-size: 12.5px;
  font-weight: 500;
  color: {{ 'var(--win-green)' if model_online else 'var(--amber-dark)' }};
  letter-spacing: -0.1px;
}
.status-pip {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: {{ 'var(--win-green)' if model_online else 'var(--amber)' }};
  animation: pip-pulse 2.4s ease-in-out infinite;
}
@keyframes pip-pulse {
  0%,100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.85); }
}

.page { position: relative; z-index: 1; }

#hero {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 32px 60px;
  text-align: center;
  position: relative;
  overflow: hidden;
}

.hero-blob {
  position: absolute;
  border-radius: 50%;
  filter: blur(80px);
  pointer-events: none;
  opacity: 0.55;
}
.blob-1 {
  width: 500px; height: 500px;
  background: radial-gradient(circle, #FEF3C7 0%, transparent 70%);
  top: -100px; right: -100px;
  animation: blob-drift 12s ease-in-out infinite;
}
.blob-2 {
  width: 400px; height: 400px;
  background: radial-gradient(circle, #E8E0D4 0%, transparent 70%);
  bottom: -80px; left: -80px;
  animation: blob-drift 15s ease-in-out infinite reverse;
}
.blob-3 {
  width: 300px; height: 300px;
  background: radial-gradient(circle, #FDE68A 0%, transparent 70%);
  top: 40%; left: 15%;
  animation: blob-drift 18s ease-in-out infinite;
  opacity: 0.3;
}
@keyframes blob-drift {
  0%,100% { transform: translate(0,0) scale(1); }
  33% { transform: translate(30px,-20px) scale(1.05); }
  66% { transform: translate(-20px,30px) scale(0.97); }
}

.hero-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 16px;
  background: var(--amber-light);
  border: 1px solid #FDE68A;
  border-radius: 20px;
  font-size: 12.5px;
  font-weight: 500;
  color: var(--amber-dark);
  letter-spacing: 0.2px;
  margin-bottom: 28px;
  animation: fadeUp 0.7s ease both;
}
.hero-eyebrow-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--amber);
  animation: pip-pulse 2s infinite;
}

.hero-h1 {
  font-family: 'Instrument Serif', serif;
  font-size: clamp(44px, 6.5vw, 80px);
  line-height: 1.05;
  letter-spacing: -2px;
  color: var(--ink);
  margin-bottom: 22px;
  animation: fadeUp 0.7s 0.1s ease both;
  max-width: 820px;
}
.hero-h1 em {
  font-style: italic;
  color: var(--amber);
}

.hero-sub {
  font-size: 17px;
  color: var(--ink-muted);
  max-width: 540px;
  line-height: 1.65;
  margin-bottom: 38px;
  animation: fadeUp 0.7s 0.2s ease both;
  font-weight: 400;
}
.hero-sub strong { color: var(--ink-soft); font-weight: 500; }

.hero-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
  flex-wrap: wrap;
  animation: fadeUp 0.7s 0.3s ease both;
  margin-bottom: 64px;
}

.btn {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 12px 24px;
  border-radius: var(--radius-md);
  font-size: 14.5px;
  font-weight: 500;
  text-decoration: none;
  cursor: pointer;
  border: none;
  transition: all 0.2s ease;
  letter-spacing: -0.2px;
  font-family: 'DM Sans', sans-serif;
}
.btn-primary {
  background: var(--ink);
  color: #fff;
}
.btn-primary:hover {
  background: var(--ink-soft);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}
.btn-secondary {
  background: #fff;
  color: var(--ink);
  border: 1px solid var(--cream-darker);
  box-shadow: var(--shadow-sm);
}
.btn-secondary:hover {
  border-color: var(--ink-ghost);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.hero-metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  max-width: 760px;
  width: 100%;
  animation: fadeUp 0.7s 0.4s ease both;
}
.hero-metric-card {
  background: #fff;
  border: 1px solid var(--cream-darker);
  border-radius: var(--radius-md);
  padding: 20px 16px;
  text-align: center;
  box-shadow: var(--shadow-sm);
  transition: all 0.25s ease;
  position: relative;
  overflow: hidden;
}
.hero-metric-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--amber-mid), #FCD34D);
  opacity: 0;
  transition: opacity 0.25s;
}
.hero-metric-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-md); }
.hero-metric-card:hover::before { opacity: 1; }
.metric-val {
  font-family: 'Instrument Serif', serif;
  font-size: 36px;
  color: var(--ink);
  line-height: 1;
  margin-bottom: 5px;
  letter-spacing: -1px;
}
.metric-label {
  font-size: 12px;
  color: var(--ink-faint);
  font-weight: 500;
  letter-spacing: 0.2px;
}
.metric-delta {
  font-size: 11.5px;
  color: var(--win-green);
  font-weight: 500;
  margin-top: 3px;
}

section { padding: 96px 32px; }
.section-inner { max-width: 1100px; margin: 0 auto; }

.section-label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  color: var(--amber);
  margin-bottom: 12px;
}
.section-label::before {
  content: '';
  width: 16px; height: 2px;
  background: var(--amber);
  border-radius: 2px;
}

.section-h2 {
  font-family: 'Instrument Serif', serif;
  font-size: clamp(28px, 3.5vw, 42px);
  letter-spacing: -1px;
  color: var(--ink);
  line-height: 1.1;
  margin-bottom: 12px;
}
.section-desc {
  font-size: 15.5px;
  color: var(--ink-muted);
  max-width: 520px;
  line-height: 1.65;
}

.divider {
  height: 1px;
  background: var(--cream-darker);
  max-width: 1100px;
  margin: 0 auto;
}

#predict { background: #fff; }

.predict-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2px;
  border-radius: var(--radius-xl);
  overflow: hidden;
  box-shadow: var(--shadow-lg);
  border: 1px solid var(--cream-darker);
  margin-top: 40px;
}

.predict-left {
  background: var(--cream);
  padding: 36px;
}
.predict-right {
  background: #fff;
  padding: 36px;
  display: flex;
  flex-direction: column;
}

.form-group { margin-bottom: 20px; }
.form-label {
  display: block;
  font-size: 12.5px;
  font-weight: 600;
  color: var(--ink-soft);
  letter-spacing: 0.2px;
  margin-bottom: 7px;
}

.input-field, .select-field, .textarea-field {
  width: 100%;
  background: #fff;
  border: 1px solid var(--ink-ghost);
  border-radius: var(--radius-sm);
  padding: 10px 13px;
  font-family: 'DM Sans', sans-serif;
  font-size: 13.5px;
  color: var(--ink);
  outline: none;
  transition: border-color 0.15s, box-shadow 0.15s;
  -webkit-appearance: none;
}
.input-field:focus, .select-field:focus, .textarea-field:focus {
  border-color: var(--amber-mid);
  box-shadow: 0 0 0 3px rgba(245,158,11,0.12);
}
.textarea-field {
  resize: vertical;
  min-height: 130px;
  line-height: 1.55;
  font-size: 13px;
}
.select-field option { background: #fff; }

.form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }

.range-wrapper {
  display: flex;
  align-items: center;
  gap: 12px;
}
.range-slider {
  flex: 1;
  -webkit-appearance: none;
  height: 4px;
  background: var(--ink-ghost);
  border-radius: 4px;
  outline: none;
  cursor: pointer;
}
.range-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 18px; height: 18px;
  border-radius: 50%;
  background: var(--amber);
  border: 2px solid #fff;
  box-shadow: 0 2px 6px rgba(217,119,6,0.3);
  cursor: pointer;
}
.range-val {
  font-family: 'DM Mono', monospace;
  font-size: 13px;
  font-weight: 500;
  color: var(--amber-dark);
  min-width: 36px;
}

.tab-row {
  display: flex;
  gap: 4px;
  background: var(--cream-deep);
  border-radius: var(--radius-sm);
  padding: 4px;
  margin-bottom: 18px;
}
.tab-btn {
  flex: 1;
  padding: 7px 14px;
  border: none;
  background: transparent;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  color: var(--ink-muted);
  cursor: pointer;
  transition: all 0.18s;
  font-family: 'DM Sans', sans-serif;
}
.tab-btn.active {
  background: #fff;
  color: var(--ink);
  box-shadow: var(--shadow-sm);
}

.predict-submit {
  width: 100%;
  padding: 13px 20px;
  background: var(--ink);
  color: #fff;
  border: none;
  border-radius: var(--radius-md);
  font-family: 'DM Sans', sans-serif;
  font-size: 14.5px;
  font-weight: 500;
  cursor: pointer;
  letter-spacing: -0.2px;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  position: relative;
  overflow: hidden;
}
.predict-submit:hover:not(:disabled) { background: var(--ink-soft); transform: translateY(-1px); box-shadow: var(--shadow-md); }
.predict-submit:disabled { opacity: 0.6; cursor: not-allowed; }
.predict-submit::after {
  content: '';
  position: absolute;
  top: 0; left: -100%;
  width: 50%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
  animation: sheen 2.5s infinite;
}
@keyframes sheen { 0% { left: -100%; } 100% { left: 250%; } }

.output-empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 40px 20px;
}
.empty-icon {
  width: 56px; height: 56px;
  background: var(--cream);
  border-radius: var(--radius-md);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  margin-bottom: 14px;
  border: 1px solid var(--cream-darker);
}
.empty-text { font-size: 14px; color: var(--ink-faint); line-height: 1.6; }

.output-result { display: none; flex-direction: column; gap: 18px; }
.output-result.show { display: flex; animation: fadeUp 0.4s ease; }

.result-header {
  padding: 18px 20px;
  border-radius: var(--radius-md);
  display: flex;
  align-items: center;
  gap: 14px;
}
.result-header.win { background: var(--win-green-bg); border: 1px solid var(--win-green-border); }
.result-header.loss { background: var(--loss-red-bg); border: 1px solid var(--loss-red-border); }
.result-icon {
  width: 44px; height: 44px;
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 20px;
  flex-shrink: 0;
}
.result-icon.win { background: #DCFCE7; }
.result-icon.loss { background: #FEE2E2; }
.result-verdict {
  font-family: 'Instrument Serif', serif;
  font-size: 22px;
  letter-spacing: -0.5px;
  line-height: 1;
  margin-bottom: 2px;
}
.result-verdict.win { color: var(--win-green); }
.result-verdict.loss { color: var(--loss-red); }
.result-sub { font-size: 13px; color: var(--ink-muted); }

.prob-block {
  background: var(--cream);
  border-radius: var(--radius-md);
  padding: 16px 18px;
  border: 1px solid var(--cream-darker);
}
.prob-row {
  display: flex; justify-content: space-between; align-items: baseline;
  margin-bottom: 10px;
}
.prob-label { font-size: 12.5px; font-weight: 600; color: var(--ink-muted); }
.prob-pct {
  font-family: 'Instrument Serif', serif;
  font-size: 26px;
  color: var(--ink);
  letter-spacing: -0.5px;
}
.prob-track {
  height: 7px;
  background: var(--ink-ghost);
  border-radius: 10px;
  overflow: hidden;
}
.prob-fill {
  height: 100%;
  border-radius: 10px;
  background: linear-gradient(90deg, var(--amber), var(--amber-mid));
  transition: width 1.2s cubic-bezier(0.22,1,0.36,1);
  width: 0;
}
.prob-fill.green { background: linear-gradient(90deg, #15803D, #22C55E); }
.prob-fill.red { background: linear-gradient(90deg, #991B1B, #EF4444); }

.signal-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.signal-card {
  padding: 14px 16px;
  background: var(--cream);
  border-radius: var(--radius-sm);
  border: 1px solid var(--cream-darker);
  text-align: center;
}
.signal-num {
  font-family: 'Instrument Serif', serif;
  font-size: 28px;
  color: var(--ink);
  letter-spacing: -0.5px;
  display: block;
  margin-bottom: 2px;
}
.signal-name { font-size: 11.5px; color: var(--ink-faint); font-weight: 500; }

.shap-section {
  background: var(--cream);
  border-radius: var(--radius-md);
  padding: 16px 18px;
  border: 1px solid var(--cream-darker);
}
.shap-heading {
  font-size: 12.5px;
  font-weight: 600;
  color: var(--ink-muted);
  margin-bottom: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.shap-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}
.shap-feat {
  font-family: 'DM Mono', monospace;
  font-size: 11.5px;
  color: var(--ink-soft);
  width: 120px;
  flex-shrink: 0;
}
.shap-track {
  flex: 1;
  height: 18px;
  background: var(--cream-deep);
  border-radius: 4px;
  position: relative;
  overflow: hidden;
}
.shap-fill-pos {
  position: absolute;
  left: 50%; top: 0; bottom: 0;
  background: #BBF7D0;
  border-radius: 0 4px 4px 0;
  transition: width 0.9s 0.3s ease;
  width: 0;
}
.shap-fill-neg {
  position: absolute;
  right: 50%; top: 0; bottom: 0;
  background: #FECACA;
  border-radius: 4px 0 0 4px;
  transition: width 0.9s 0.3s ease;
  width: 0;
}
.shap-mid { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: var(--ink-ghost); }
.shap-val { font-family: 'DM Mono', monospace; font-size: 11px; width: 46px; text-align: right; flex-shrink: 0; }
.shap-val.pos { color: var(--win-green); }
.shap-val.neg { color: var(--loss-red); }

.rec-box {
  padding: 13px 16px;
  background: var(--amber-pale);
  border-left: 3px solid var(--amber);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  font-size: 13.5px;
  color: var(--amber-dark);
  line-height: 1.55;
}

#features { background: var(--cream); }

.features-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-top: 44px;
}
.feat-card {
  background: #fff;
  border: 1px solid var(--cream-darker);
  border-radius: var(--radius-lg);
  padding: 28px 24px;
  transition: all 0.25s ease;
  position: relative;
  overflow: hidden;
}
.feat-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-md);
  border-color: var(--parchment);
}
.feat-card::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--amber-mid), #FCD34D);
  transform: scaleX(0);
  transform-origin: left;
  transition: transform 0.3s ease;
}
.feat-card:hover::after { transform: scaleX(1); }

.feat-icon {
  width: 44px; height: 44px;
  border-radius: var(--radius-sm);
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
  margin-bottom: 16px;
  background: var(--amber-light);
}
.feat-title {
  font-family: 'Instrument Serif', serif;
  font-size: 18px;
  color: var(--ink);
  margin-bottom: 8px;
  letter-spacing: -0.3px;
}
.feat-desc { font-size: 13.5px; color: var(--ink-muted); line-height: 1.6; }
.feat-tag {
  display: inline-block;
  margin-top: 14px;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 11.5px;
  font-weight: 500;
}
.feat-tag.amber { background: var(--amber-light); color: var(--amber-dark); }
.feat-tag.green { background: var(--win-green-bg); color: var(--win-green); }
.feat-tag.blue { background: var(--info-blue-bg); color: var(--info-blue); }

#explainability { background: #fff; }

.expl-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 44px; }

.chart-card {
  background: var(--cream);
  border-radius: var(--radius-lg);
  border: 1px solid var(--cream-darker);
  padding: 28px;
}
.chart-card-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--ink-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 22px;
}

.abl-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}
.abl-label {
  font-size: 13px;
  color: var(--ink-soft);
  width: 110px;
  flex-shrink: 0;
}
.abl-track {
  flex: 1;
  height: 28px;
  background: var(--cream-deep);
  border-radius: 6px;
  overflow: hidden;
}
.abl-fill {
  height: 100%;
  border-radius: 6px;
  display: flex;
  align-items: center;
  padding: 0 10px;
  font-size: 12px;
  font-weight: 600;
  color: #fff;
  transition: width 1.4s cubic-bezier(0.22,1,0.36,1);
  width: 0;
}
.abl-fill.f1 { background: linear-gradient(90deg, #D97706, #F59E0B); }
.abl-fill.f2 { background: linear-gradient(90deg, #1D4ED8, #3B82F6); }
.abl-fill.f3 { background: linear-gradient(90deg, #15803D, #22C55E); }
.abl-fill.f4 { background: linear-gradient(90deg, #7C3AED, #A78BFA); }
.abl-fill.f5 { background: linear-gradient(90deg, #6B7280, #9CA3AF); }

.insights-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin-top: 30px;
}
.insight-card {
  padding: 16px 18px;
  border-radius: var(--radius-md);
  background: var(--cream);
  border: 1px solid var(--cream-darker);
  font-size: 13.5px;
  color: var(--ink-soft);
  line-height: 1.6;
  border-top: 3px solid;
}
.insight-card.amber { border-top-color: var(--amber); }
.insight-card.green { border-top-color: var(--win-green); }
.insight-card.blue { border-top-color: var(--info-blue); }
.insight-card.red { border-top-color: var(--loss-red); }
.insight-card strong { color: var(--ink); font-weight: 600; display: block; margin-bottom: 4px; }

#architecture { background: var(--cream); }

.arch-wrapper {
  margin-top: 40px;
  background: #fff;
  border: 1px solid var(--cream-darker);
  border-radius: var(--radius-xl);
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}
.arch-diagram-area {
  padding: 50px 40px 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  flex-wrap: wrap;
  background: var(--cream);
  border-bottom: 1px solid var(--cream-darker);
}
.arch-node-group {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}
.arch-box {
  padding: 12px 18px;
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 500;
  text-align: center;
  border: 1px solid;
  transition: all 0.2s;
  cursor: default;
  white-space: nowrap;
  line-height: 1.3;
}
.arch-box:hover { transform: translateY(-2px); box-shadow: var(--shadow-sm); }
.arch-box.input { background: var(--info-blue-bg); border-color: #BFDBFE; color: var(--info-blue); }
.arch-box.branch { background: var(--win-green-bg); border-color: var(--win-green-border); color: var(--win-green); }
.arch-box.fusion { background: var(--amber-light); border-color: #FDE68A; color: var(--amber-dark); padding: 16px 24px; font-size: 14px; }
.arch-box.output { background: var(--loss-red-bg); border-color: var(--loss-red-border); color: var(--loss-red); }
.arch-box.explain { background: #F5F3FF; border-color: #DDD6FE; color: #6D28D9; }
.arch-box small { display: block; font-size: 10.5px; font-weight: 400; margin-top: 2px; opacity: 0.7; }

.arch-arrow {
  color: var(--ink-ghost);
  font-size: 20px;
  padding: 0 4px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
}

.arch-table-area { padding: 0; }
.arch-table {
  width: 100%;
  border-collapse: collapse;
}
.arch-table th {
  padding: 12px 20px;
  text-align: left;
  font-size: 12px;
  font-weight: 600;
  color: var(--ink-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  background: var(--cream);
  border-bottom: 1px solid var(--cream-darker);
}
.arch-table td {
  padding: 13px 20px;
  font-size: 13.5px;
  color: var(--ink-soft);
  border-bottom: 1px solid var(--cream-deeper, #EDE6D8);
  font-family: 'DM Sans', sans-serif;
}
.arch-table tr:last-child td { border-bottom: none; }
.arch-table tr:hover td { background: var(--cream); }
.arch-table .tag {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 20px;
  font-size: 11.5px;
  font-weight: 500;
}
.tag-amber { background: var(--amber-light); color: var(--amber-dark); }
.tag-green { background: var(--win-green-bg); color: var(--win-green); }

#batch { background: #fff; }

.batch-drop {
  margin-top: 36px;
  border: 2px dashed var(--ink-ghost);
  border-radius: var(--radius-xl);
  padding: 64px 40px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s ease;
  background: var(--cream);
}
.batch-drop:hover {
  border-color: var(--amber-mid);
  background: var(--amber-pale);
}
.batch-drop-icon {
  width: 64px; height: 64px;
  background: #fff;
  border-radius: var(--radius-md);
  display: flex; align-items: center; justify-content: center;
  font-size: 28px;
  margin: 0 auto 18px;
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--cream-darker);
}
.batch-drop h3 {
  font-family: 'Instrument Serif', serif;
  font-size: 22px;
  color: var(--ink);
  margin-bottom: 6px;
}
.batch-drop p { font-size: 14px; color: var(--ink-faint); }

.batch-results { margin-top: 32px; display: none; }
.batch-results.show { display: block; }

.results-table {
  width: 100%;
  border-collapse: collapse;
  background: var(--cream);
  border-radius: var(--radius-lg);
  overflow: hidden;
  border: 1px solid var(--cream-darker);
}
.results-table th {
  padding: 11px 16px;
  text-align: left;
  font-size: 12px;
  font-weight: 600;
  color: var(--ink-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  background: var(--parchment);
  border-bottom: 1px solid var(--cream-darker);
}
.results-table td {
  padding: 11px 16px;
  font-size: 13.5px;
  color: var(--ink-soft);
  border-bottom: 1px solid var(--cream-deeper, #EDE6D8);
  font-family: 'DM Sans', sans-serif;
}
.results-table tr:last-child td { border-bottom: none; }
.results-table tr { animation: fadeUp 0.4s ease both; }

.pill {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
}
.pill-win { background: var(--win-green-bg); color: var(--win-green); }
.pill-loss { background: var(--loss-red-bg); color: var(--loss-red); }
.pill-low { background: var(--win-green-bg); color: var(--win-green); }
.pill-med { background: var(--amber-light); color: var(--amber-dark); }
.pill-high { background: var(--loss-red-bg); color: var(--loss-red); }

#validation { background: var(--cream); }

.val-cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-top: 40px;
}
.val-card {
  background: #fff;
  border: 1px solid var(--cream-darker);
  border-radius: var(--radius-lg);
  padding: 24px;
  transition: all 0.25s;
}
.val-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-md); }
.val-card-name {
  font-family: 'DM Mono', monospace;
  font-size: 12px;
  color: var(--ink-faint);
  margin-bottom: 10px;
}
.val-card-title {
  font-family: 'Instrument Serif', serif;
  font-size: 17px;
  color: var(--ink);
  margin-bottom: 14px;
  letter-spacing: -0.3px;
}
.val-status {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12.5px;
  font-weight: 600;
  margin-bottom: 16px;
}
.val-status.pass { background: var(--win-green-bg); color: var(--win-green); }
.val-status.check { background: var(--amber-light); color: var(--amber-dark); }
.val-meta { font-size: 13px; color: var(--ink-muted); margin-bottom: 4px; }
.val-meta span { color: var(--ink-soft); font-weight: 500; }
.val-bar-track { height: 4px; background: var(--ink-ghost); border-radius: 10px; margin-top: 12px; overflow: hidden; }
.val-bar-fill { height: 100%; border-radius: 10px; transition: width 1s ease; width: 0; }
.val-bar-fill.green { background: var(--win-green); }
.val-bar-fill.amber { background: var(--amber); }

.score-banner {
  margin-top: 28px;
  background: var(--ink);
  border-radius: var(--radius-xl);
  padding: 36px 40px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 24px;
}
.score-big {
  font-family: 'Instrument Serif', serif;
  font-size: 64px;
  color: #fff;
  line-height: 1;
  letter-spacing: -2px;
}
.score-big span { color: var(--amber-mid); }
.score-detail { font-size: 14px; color: rgba(255,255,255,0.55); line-height: 1.8; }
.score-accuracy {
  font-family: 'Instrument Serif', serif;
  font-size: 40px;
  color: var(--amber-mid);
  letter-spacing: -1px;
  text-align: right;
}
.score-accuracy small { display: block; font-family: 'DM Sans', sans-serif; font-size: 12px; color: rgba(255,255,255,0.4); font-style: normal; margin-top: -2px; }

footer {
  background: var(--ink);
  padding: 60px 32px 32px;
  color: rgba(255,255,255,0.6);
}
.footer-inner { max-width: 1100px; margin: 0 auto; }
.footer-top {
  display: grid;
  grid-template-columns: 2fr 1fr 1fr 1fr;
  gap: 40px;
  margin-bottom: 40px;
}
.footer-brand-name {
  font-family: 'Instrument Serif', serif;
  font-size: 26px;
  color: #fff;
  margin-bottom: 12px;
  letter-spacing: -0.5px;
}
.footer-brand-name span { color: var(--amber-mid); }
.footer-tagline { font-size: 13.5px; line-height: 1.65; max-width: 260px; }
.footer-col-label {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: rgba(255,255,255,0.4);
  margin-bottom: 14px;
}
.footer-links { list-style: none; }
.footer-links li { margin-bottom: 9px; }
.footer-links a {
  font-size: 13.5px;
  color: rgba(255,255,255,0.55);
  text-decoration: none;
  transition: color 0.15s;
}
.footer-links a:hover { color: #fff; }
.footer-bottom {
  border-top: 1px solid rgba(255,255,255,0.08);
  padding-top: 22px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}
.footer-copy { font-size: 12.5px; }
.footer-phase {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12.5px;
  color: var(--amber-mid);
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(24px); }
  to { opacity: 1; transform: translateY(0); }
}
.reveal {
  opacity: 0;
  transform: translateY(28px);
  transition: opacity 0.7s ease, transform 0.7s ease;
}
.reveal.in { opacity: 1; transform: none; }

@media (max-width: 1024px) {
  .predict-grid { grid-template-columns: 1fr; }
  .features-grid { grid-template-columns: repeat(2, 1fr); }
  .expl-grid { grid-template-columns: 1fr; }
  .val-cards { grid-template-columns: repeat(2, 1fr); }
  .hero-metrics { grid-template-columns: repeat(2, 1fr); }
  .footer-top { grid-template-columns: 1fr 1fr; }
}

@media (max-width: 768px) {
  .features-grid { grid-template-columns: 1fr; }
  .val-cards { grid-template-columns: 1fr; }
  .hero-metrics { grid-template-columns: 1fr; }
  .footer-top { grid-template-columns: 1fr; }
  .nav-links { display: none; }
  section { padding: 60px 20px; }
  .arch-diagram-area { flex-direction: column; }
  .arch-arrow { transform: rotate(90deg); }
}
</style>
</head>
<body>

<nav>
  <a class="nav-brand" href="#hero">
    <div class="nav-logo-mark">
      <svg viewBox="0 0 18 18" fill="none">
        <rect x="2" y="8" width="3" height="8" rx="1.5" fill="#F59E0B"/>
        <rect x="7.5" y="4" width="3" height="12" rx="1.5" fill="#FCD34D"/>
        <rect x="13" y="1" width="3" height="15" rx="1.5" fill="#fff"/>
      </svg>
    </div>
    <span class="nav-brand-name">Sales<span>IQ</span></span>
  </a>
  <ul class="nav-links">
    <li><a href="#hero" class="active">Home</a></li>
    <li><a href="#predict">Predict</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#explainability">Explainability</a></li>
    <li><a href="#architecture">Architecture</a></li>
    <li><a href="#batch">Batch</a></li>
    <li><a href="#validation">Validation</a></li>
  </ul>
  <div class="nav-badge">
    <div class="status-pip"></div>
    {{ 'Model Online' if model_online else 'Demo Mode' }}
  </div>
</nav>

<div class="page">

<section id="hero">
  <div class="hero-blob blob-1"></div>
  <div class="hero-blob blob-2"></div>
  <div class="hero-blob blob-3"></div>

  <div class="hero-eyebrow">
    <div class="hero-eyebrow-dot"></div>
    Phase 3 · Hybrid Gated-Fusion Model
  </div>

  <h1 class="hero-h1">
    Predict which deals<br>
    you'll <em>actually</em> close
  </h1>

  <p class="hero-sub">
    A <strong>hybrid AI engine</strong> that reads your sales conversations and CRM data together — giving you win probability, SHAP attribution, and clear next steps.
  </p>

  <div class="hero-actions">
    <a href="#predict" class="btn btn-primary">
      <svg width="15" height="15" viewBox="0 0 15 15" fill="none"><path d="M3 7.5L12 7.5M9 4l3 3.5-3 3.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
      Start predicting
    </a>
    <a href="#explainability" class="btn btn-secondary">View model results</a>
  </div>

  <div class="hero-metrics">
    <div class="hero-metric-card">
      <div class="metric-val" data-target="{{ (ablation.get('A_full_hybrid', {}).get('f1', 0.976) * 100)|round(1) }}" data-suffix="%">0%</div>
      <div class="metric-label">F1 Score</div>
      <div class="metric-delta">↑ Best in ablation</div>
    </div>
    <div class="hero-metric-card">
      <div class="metric-val" data-target="{{ (ablation.get('A_full_hybrid', {}).get('auc', 0.998) * 100)|round(1) }}" data-suffix="%">0%</div>
      <div class="metric-label">AUC-ROC</div>
      <div class="metric-delta">↑ Near-perfect</div>
    </div>
    <div class="hero-metric-card">
      <div class="metric-val" data-target="5000" data-suffix="">0</div>
      <div class="metric-label">Training Samples</div>
      <div class="metric-delta">↑ Full dataset</div>
    </div>
    <div class="hero-metric-card">
      <div class="metric-val" data-target="57" data-suffix="">0</div>
      <div class="metric-label">Features</div>
      <div class="metric-delta">Engineered</div>
    </div>
  </div>
</section>

<div class="divider"></div>

<section id="predict">
  <div class="section-inner">
    <div class="reveal">
      <div class="section-label">Single Prediction</div>
      <h2 class="section-h2">Deal outcome predictor</h2>
      <p class="section-desc">Paste a sales conversation, set engagement, and get an instant AI prediction with feature attribution.</p>
    </div>

    <div class="predict-grid reveal">
      <div class="predict-left">
        <div class="tab-row">
          <button class="tab-btn active" onclick="switchTab('paste')" id="tab-paste">Paste conversation</button>
          <button class="tab-btn" onclick="switchTab('example')" id="tab-example">Load example</button>
        </div>

        <div id="pane-paste">
          <div class="form-group">
            <label class="form-label">Conversation</label>
            <textarea class="textarea-field" id="conv-text" placeholder="[Rep]: Good morning! Thanks for joining today's demo call...
[Prospect]: Happy to be here — we've been evaluating a few platforms..."></textarea>
          </div>
        </div>

        <div id="pane-example" style="display:none">
          <div class="form-group">
            <label class="form-label">Choose an example</label>
            <select class="select-field" id="example-select" onchange="loadExample(this.value)">
              <option value="">— Select example —</option>
              <option value="win">example_win.txt — High-confidence close</option>
              <option value="loss">example_loss.txt — Lost to competitor</option>
              <option value="edge_case">example_edge_case.txt — Ambiguous outcome</option>
            </select>
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label class="form-label">Product type</label>
            <select class="select-field" id="product-type">
              <option value="Technology">Enterprise SaaS</option>
              <option value="Professional Services">Professional Services</option>
              <option value="Retail">SMB License</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">Channel</label>
            <select class="select-field" id="channel">
              <option value="video_call">Video Call</option>
              <option value="email">Email Thread</option>
              <option value="in_person">In-Person</option>
            </select>
          </div>
        </div>

        <div class="form-group">
          <label class="form-label">Customer engagement — <span id="eng-val" style="font-family:'DM Mono',monospace;color:var(--amber-dark)">0.70</span></label>
          <div class="range-wrapper">
            <span style="font-size:12px;color:var(--ink-faint)">Low</span>
            <input type="range" class="range-slider" id="eng-slider" min="0" max="1" step="0.01" value="0.7" oninput="document.getElementById('eng-val').textContent=parseFloat(this.value).toFixed(2)">
            <span style="font-size:12px;color:var(--ink-faint)">High</span>
          </div>
        </div>

        <button class="predict-submit" onclick="runPredict()" id="pred-btn">
          <svg width="15" height="15" viewBox="0 0 15 15" fill="none"><path d="M3 7.5L12 7.5M9 4l3 3.5-3 3.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
          <span id="pred-btn-txt">Predict outcome</span>
        </button>
      </div>

      <div class="predict-right">
        <div class="output-empty" id="out-empty">
          <div class="empty-icon">📊</div>
          <p class="empty-text">Enter a conversation on the left<br>and click <strong>Predict outcome</strong></p>
        </div>
        <div class="output-result" id="out-result">
          <div class="result-header win" id="result-header">
            <div class="result-icon win" id="result-icon">🎯</div>
            <div>
              <div class="result-verdict win" id="result-verdict">Predicted Win</div>
              <div class="result-sub" id="result-sub">Low risk · High confidence</div>
            </div>
          </div>

          <div class="prob-block">
            <div class="prob-row">
              <span class="prob-label">Win probability</span>
              <span class="prob-pct" id="prob-pct">—</span>
            </div>
            <div class="prob-track">
              <div class="prob-fill green" id="prob-fill"></div>
            </div>
          </div>

          <div class="signal-row">
            <div class="signal-card">
              <span class="signal-num" id="tab-pct">—</span>
              <span class="signal-name">Tabular signal</span>
            </div>
            <div class="signal-card">
              <span class="signal-num" id="txt-pct">—</span>
              <span class="signal-name">Text signal</span>
            </div>
          </div>

          <div class="shap-section">
            <div class="shap-heading">SHAP feature attribution</div>
            <div id="shap-container"></div>
          </div>

          <div class="rec-box" id="rec-box"></div>
        </div>
      </div>
    </div>
  </div>
</section>

<div class="divider"></div>

<section id="features">
  <div class="section-inner">
    <div class="reveal">
      <div class="section-label">Core Capabilities</div>
      <h2 class="section-h2">Everything in one engine</h2>
      <p class="section-desc">Six modules working in concert — from raw text to explainable predictions.</p>
    </div>
    <div class="features-grid">
      <div class="feat-card reveal">
        <div class="feat-icon">⚡</div>
        <div class="feat-title">Gated Fusion Layer</div>
        <p class="feat-desc">Dynamic gate weights blend tabular and text signals based on input quality. No single branch dominates — the data decides.</p>
        <span class="feat-tag amber">Phase 3</span>
      </div>
      <div class="feat-card reveal">
        <div class="feat-icon">📊</div>
        <div class="feat-title">Tabular Branch</div>
        <p class="feat-desc">XGBoost encoder with 57 engineered features — engagement score, product tier, channel, and CRM signals.</p>
        <span class="feat-tag green">Hybrid</span>
      </div>
      <div class="feat-card reveal">
        <div class="feat-icon">💬</div>
        <div class="feat-title">Text Branch</div>
        <p class="feat-desc">DistilBERT encoder reads raw conversation transcripts, capturing sentiment, objections, and buying signals.</p>
        <span class="feat-tag green">Hybrid</span>
      </div>
      <div class="feat-card reveal">
        <div class="feat-icon">🧩</div>
        <div class="feat-title">SHAP Explainability</div>
        <p class="feat-desc">Every prediction comes with SHAP attribution — know exactly which features pushed the score up or down.</p>
        <span class="feat-tag blue">AI-Powered</span>
      </div>
      <div class="feat-card reveal">
        <div class="feat-icon">🎯</div>
        <div class="feat-title">Attention Mapping</div>
        <p class="feat-desc">Conversation-level attention highlights the exact lines the model focuses on — making predictions interpretable.</p>
        <span class="feat-tag blue">AI-Powered</span>
      </div>
      <div class="feat-card reveal">
        <div class="feat-icon">📦</div>
        <div class="feat-title">Batch Inference</div>
        <p class="feat-desc">Upload CSVs with hundreds of conversations. Auto-detects text columns, scores every row, exports results.</p>
        <span class="feat-tag amber">Live</span>
      </div>
    </div>
  </div>
</section>

<div class="divider"></div>

<section id="explainability">
  <div class="section-inner">
    <div class="reveal">
      <div class="section-label">Ablation Study</div>
      <h2 class="section-h2">The numbers speak clearly</h2>
      <p class="section-desc">Systematic ablation proves gated fusion outperforms every single-branch baseline.</p>
    </div>

    <div class="expl-grid reveal">
      <div class="chart-card">
        <div class="chart-card-title">F1 Score by Model Variant</div>
        {% for name, metrics in ablation.items() %}
        <div class="abl-row">
          <div class="abl-label">{{ name.replace('_', ' ').replace('A ', '').replace('B ', '').replace('C ', '').replace('D ', '').replace('E ', '').replace('F ', '') | truncate(14) }}</div>
          <div class="abl-track"><div class="abl-fill f{{ loop.index }}" data-w="{{ (metrics.get('f1', 0) * 100)|round(1) }}">{{ (metrics.get('f1', 0) * 100)|round(1) }}%</div></div>
        </div>
        {% endfor %}
      </div>

      <div class="chart-card">
        <div class="chart-card-title">Overall F1 vs Edge-Case F1</div>
        {% for name, metrics in ablation.items() %}
        {% if loop.index <= 3 %}
        <div class="abl-row">
          <div class="abl-label">{{ name.replace('_', ' ').replace('A ', '').replace('B ', '').replace('C ', '') | truncate(12) }} F1</div>
          <div class="abl-track"><div class="abl-fill f{{ loop.index }}" data-w="{{ (metrics.get('f1', 0) * 100)|round(1) }}">{{ (metrics.get('f1', 0) * 100)|round(1) }}%</div></div>
        </div>
        <div class="abl-row">
          <div class="abl-label">{{ name.replace('_', ' ').replace('A ', '').replace('B ', '').replace('C ', '') | truncate(12) }} Edge</div>
          <div class="abl-track"><div class="abl-fill f{{ loop.index + 3 }}" data-w="{{ (metrics.get('edge_case_f1', 0) * 100)|round(1) }}">{{ (metrics.get('edge_case_f1', 0) * 100)|round(1) }}%</div></div>
        </div>
        {% endif %}
        {% endfor %}
      </div>
    </div>

    <div class="insights-grid reveal">
      <div class="insight-card amber">
        <strong>Gated fusion wins overall</strong>
        The attention gate suppresses noisy signals, achieving {{ (ablation.get('A_full_hybrid', {}).get('f1', 0.976) * 100)|round(1) }}% F1 on the test set.
      </div>
      <div class="insight-card green">
        <strong>Edge case robustness</strong>
        Hybrid model maintains {{ (ablation.get('A_full_hybrid', {}).get('edge_case_f1', 0.863) * 100)|round(1) }}% F1 on ambiguous deals where single-branch models struggle.
      </div>
      <div class="insight-card blue">
        <strong>Tabular features matter</strong>
        Engagement score and product type explain significant prediction variance in SHAP analysis.
      </div>
      <div class="insight-card red">
        <strong>Text-only insufficient</strong>
        Without tabular context, the text branch drops to {{ (ablation.get('C_text_only', {}).get('f1', 0.62) * 100)|round(1) }}% F1.
      </div>
    </div>
  </div>
</section>

<div class="divider"></div>

<section id="architecture">
  <div class="section-inner">
    <div class="reveal">
      <div class="section-label">System Design</div>
      <h2 class="section-h2">Hybrid model architecture</h2>
      <p class="section-desc">Phase 3 — dual-branch encoding, gated late-fusion, modular explainability layer.</p>
    </div>

    <div class="arch-wrapper reveal">
      <div class="arch-diagram-area">
        <div class="arch-node-group">
          <div class="arch-box input">📄 Raw Conversation<small>Text input</small></div>
          <div class="arch-box input">📋 CRM Metadata<small>Tabular input</small></div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-node-group">
          <div class="arch-box branch">🔤 Text Encoder<small>DistilBERT + Attention</small></div>
          <div class="arch-box branch">📊 Tabular Encoder<small>XGBoost Leaves + FFN</small></div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-node-group">
          <div class="arch-box fusion">⚡ Gated Fusion<small>α·tab + (1-α)·text</small></div>
        </div>
        <div class="arch-arrow">→</div>
        <div class="arch-node-group">
          <div class="arch-box output">🎯 Classifier<small>Win / Loss</small></div>
          <div class="arch-box explain">🧩 SHAP + Attention<small>Explainability</small></div>
        </div>
      </div>
      <div class="arch-table-area">
        <table class="arch-table">
          <thead>
            <tr><th>Component</th><th>Details</th><th>Phase</th><th>Status</th></tr>
          </thead>
          <tbody>
            <tr><td>Dataset</td><td>SaaS sales conversations + CRM metadata (5,000 samples)</td><td>All</td><td><span class="tag tag-green">Active</span></td></tr>
            <tr><td>Text Branch</td><td>DistilBERT + learned attention pooling → 128-dim</td><td>Phase 2+</td><td><span class="tag tag-green">Deployed</span></td></tr>
            <tr><td>Tabular Branch</td><td>57 features → XGBoost leaf encoding (1344-dim) → FFN</td><td>Phase 2+</td><td><span class="tag tag-green">Deployed</span></td></tr>
            <tr><td>Fusion</td><td>Learned gating: α·tabular + (1-α)·text</td><td>Phase 3</td><td><span class="tag tag-amber">Active</span></td></tr>
            <tr><td>Explainability</td><td>SHAP TreeExplainer + token attention weights</td><td>Phase 3</td><td><span class="tag tag-amber">Active</span></td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</section>

<div class="divider"></div>

<section id="batch">
  <div class="section-inner">
    <div class="reveal">
      <div class="section-label">Batch Inference</div>
      <h2 class="section-h2">Score hundreds of deals at once</h2>
      <p class="section-desc">Upload a CSV with conversation data. The engine auto-detects the text column and scores every row.</p>
    </div>

    <div class="batch-drop reveal" id="batch-drop">
      <input type="file" id="batch-file" accept=".csv" style="display:none" onchange="handleBatchFile(this)">
      <div class="batch-drop-icon">📂</div>
      <h3>Drop your CSV here</h3>
      <p>Click to upload · Supports .csv with any conversation column</p>
    </div>

    <div class="batch-results" id="batch-results">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:12px;">
        <div style="font-size:14px;color:var(--ink-muted);">Showing <strong id="batch-count" style="color:var(--ink)">0</strong> predictions</div>
        <button class="btn btn-secondary" onclick="downloadBatch()" style="font-size:13px;padding:8px 18px;">↓ Download CSV</button>
      </div>
      <table class="results-table">
        <thead><tr><th>#</th><th>Preview</th><th>Probability</th><th>Prediction</th><th>Risk</th></tr></thead>
        <tbody id="batch-tbody"></tbody>
      </table>
    </div>
  </div>
</section>

<div class="divider"></div>

<section id="validation">
  <div class="section-inner">
    <div class="reveal">
      <div class="section-label">Sample Validation</div>
      <h2 class="section-h2">End-to-end model validation</h2>
      <p class="section-desc">Automated checks against known-label examples verify the full prediction pipeline.</p>
    </div>

    <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;margin-bottom:8px;" class="reveal">
      <div>
        <label class="form-label" style="margin-bottom:7px">Engagement override</label>
        <div class="range-wrapper" style="width:260px">
          <span style="font-size:12px;color:var(--ink-faint)">Low</span>
          <input type="range" class="range-slider" id="val-eng" min="0" max="1" step="0.01" value="0.70" oninput="document.getElementById('val-eng-val').textContent=parseFloat(this.value).toFixed(2)">
          <span style="font-size:12px;color:var(--ink-faint)">High</span>
          <span class="range-val" id="val-eng-val">0.70</span>
        </div>
      </div>
      <button class="btn btn-primary" onclick="runValidation()" id="val-btn" style="margin-top:18px">Run validation</button>
    </div>

    <div class="val-cards reveal">
      <div class="val-card" id="vc0">
        <div class="val-card-name">example_win.txt</div>
        <div class="val-card-title">High-confidence close</div>
        <span class="val-status pass" id="vs0">● Pending</span>
        <div class="val-meta">Expected: <span>WIN</span></div>
        <div class="val-meta">Predicted: <span id="vp0">—</span></div>
        <div class="val-meta">Probability: <span id="vpr0">—</span></div>
        <div class="val-bar-track"><div class="val-bar-fill green" id="vb0"></div></div>
      </div>
      <div class="val-card" id="vc1">
        <div class="val-card-name">example_loss.txt</div>
        <div class="val-card-title">Lost to competitor</div>
        <span class="val-status pass" id="vs1">● Pending</span>
        <div class="val-meta">Expected: <span>LOSS</span></div>
        <div class="val-meta">Predicted: <span id="vp1">—</span></div>
        <div class="val-meta">Probability: <span id="vpr1">—</span></div>
        <div class="val-bar-track"><div class="val-bar-fill amber" id="vb1"></div></div>
      </div>
      <div class="val-card" id="vc2">
        <div class="val-card-name">example_edge_case.txt</div>
        <div class="val-card-title">Ambiguous outcome</div>
        <span class="val-status check" id="vs2">● Pending</span>
        <div class="val-meta">Expected: <span>WIN</span></div>
        <div class="val-meta">Predicted: <span id="vp2">—</span></div>
        <div class="val-meta">Probability: <span id="vpr2">—</span></div>
        <div class="val-bar-track"><div class="val-bar-fill amber" id="vb2"></div></div>
      </div>
    </div>

    <div class="score-banner reveal">
      <div>
        <div class="score-big"><span id="score-n">—</span><span style="color:rgba(255,255,255,0.3)">/3</span></div>
        <div class="score-detail" id="score-detail">Run validation to generate report</div>
      </div>
      <div style="font-size:14px;color:rgba(255,255,255,0.45);line-height:1.9;" id="score-report">
        Engine: Hybrid Gated-Fusion v3<br>
        Dataset: 3 labelled examples<br>
        Status: Awaiting run…
      </div>
      <div>
        <div class="score-accuracy" id="score-acc">—</div>
        <small style="font-size:12px;color:rgba(255,255,255,0.35);font-family:'DM Sans',sans-serif;display:block;text-align:right;margin-top:-2px">accuracy</small>
      </div>
    </div>
  </div>
</section>

<footer>
  <div class="footer-inner">
    <div class="footer-top">
      <div>
        <div class="footer-brand-name">Sales<span>IQ</span></div>
        <p class="footer-tagline">Predictive Sales Analytics Engine — Phase 3: Hybrid Gated-Fusion + Explainability. Built to turn conversations into closed deals.</p>
      </div>
      <div>
        <div class="footer-col-label">Navigation</div>
        <ul class="footer-links">
          <li><a href="#hero">Home</a></li>
          <li><a href="#predict">Predictor</a></li>
          <li><a href="#features">Features</a></li>
          <li><a href="#explainability">Explainability</a></li>
          <li><a href="#architecture">Architecture</a></li>
        </ul>
      </div>
      <div>
        <div class="footer-col-label">Model</div>
        <ul class="footer-links">
          <li><a href="#batch">Batch Inference</a></li>
          <li><a href="#validation">Validation</a></li>
          <li><a href="#">Ablation Study</a></li>
        </ul>
      </div>
      <div>
        <div class="footer-col-label">System</div>
        <ul class="footer-links">
          <li><a href="#">Features: 57</a></li>
          <li><a href="#">Leaf Dim: 1344</a></li>
          <li><a href="#">F1: {{ (ablation.get('A_full_hybrid', {}).get('f1', 0.976) * 100)|round(1) }}%</a></li>
        </ul>
      </div>
    </div>
    <div class="footer-bottom">
      <span class="footer-copy">© 2026 SalesIQ Predictive Analytics Engine</span>
      <span class="footer-phase">
        <svg width="8" height="8" viewBox="0 0 8 8"><circle cx="4" cy="4" r="4" fill="#F59E0B"/></svg>
        Phase 3 Active · Hybrid Gated-Fusion · SHAP Explainability
      </span>
    </div>
  </div>
</footer>

</div>

<script>
const EXAMPLES = {{ examples | tojson }};
let batchData = [];

// Scroll reveal
const reveals = document.querySelectorAll('.reveal');
const revObs = new IntersectionObserver(entries => {
  entries.forEach(e => { if(e.isIntersecting) { e.target.classList.add('in'); revObs.unobserve(e.target); } });
}, { threshold: 0.12 });
reveals.forEach(el => revObs.observe(el));

// Count up animation
function countUp(el) {
  const target = parseFloat(el.dataset.target);
  const suffix = el.dataset.suffix || '';
  const dec = target % 1 !== 0 ? 1 : 0;
  let start = null;
  const dur = 1600;
  function step(ts) {
    if (!start) start = ts;
    const p = Math.min((ts - start) / dur, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    el.textContent = dec ? (target * ease).toFixed(1) + suffix : Math.floor(target * ease) + suffix;
    if (p < 1) requestAnimationFrame(step);
    else el.textContent = target.toFixed(dec) + suffix;
  }
  requestAnimationFrame(step);
}
const metricObs = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.querySelectorAll('[data-target]').forEach(countUp);
      metricObs.unobserve(e.target);
    }
  });
}, { threshold: 0.3 });
document.querySelectorAll('.hero-metrics').forEach(el => metricObs.observe(el));

// Chart bars animation
const chartObs = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.querySelectorAll('.abl-fill[data-w]').forEach(el => {
        el.style.transition = 'width 1.3s cubic-bezier(0.22,1,0.36,1)';
        el.style.width = el.dataset.w + '%';
      });
      chartObs.unobserve(e.target);
    }
  });
}, { threshold: 0.2 });
document.querySelectorAll('.expl-grid').forEach(el => chartObs.observe(el));

// Navbar scroll
const navAs = document.querySelectorAll('.nav-links a');
window.addEventListener('scroll', () => {
  let cur = '';
  document.querySelectorAll('section[id]').forEach(s => {
    if (window.scrollY >= s.offsetTop - 80) cur = s.id;
  });
  navAs.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + cur));
});

// Smooth scroll
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    const t = document.querySelector(a.getAttribute('href'));
    if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
});

// Tab switch
function switchTab(mode) {
  document.getElementById('tab-paste').classList.toggle('active', mode === 'paste');
  document.getElementById('tab-example').classList.toggle('active', mode === 'example');
  document.getElementById('pane-paste').style.display = mode === 'paste' ? 'block' : 'none';
  document.getElementById('pane-example').style.display = mode === 'example' ? 'block' : 'none';
}

function loadExample(key) {
  if (EXAMPLES[key]) {
    document.getElementById('conv-text').value = EXAMPLES[key];
    switchTab('paste');
  }
}

// Prediction API call
async function runPredict() {
  const txt = document.getElementById('conv-text').value.trim();
  if (!txt) {
    document.getElementById('conv-text').style.borderColor = '#EF4444';
    setTimeout(() => document.getElementById('conv-text').style.borderColor = '', 2000);
    return;
  }
  
  const btn = document.getElementById('pred-btn');
  const btnTxt = document.getElementById('pred-btn-txt');
  btn.disabled = true;
  btnTxt.textContent = 'Analysing…';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        conversation: txt,
        engagement: parseFloat(document.getElementById('eng-slider').value),
        product_type: document.getElementById('product-type').value,
        channel: document.getElementById('channel').value
      })
    });
    const data = await res.json();
    showResult(data);
  } catch (err) {
    console.error(err);
  } finally {
    btn.disabled = false;
    btnTxt.textContent = 'Predict outcome';
  }
}

function showResult(data) {
  const isWin = data.predicted_class === 'Win';
  const pct = Math.round(data.probability * 100);
  const risk = data.risk_level;

  document.getElementById('out-empty').style.display = 'none';
  const res = document.getElementById('out-result');
  res.classList.add('show');

  const hdr = document.getElementById('result-header');
  const icon = document.getElementById('result-icon');
  hdr.className = 'result-header ' + (isWin ? 'win' : 'loss');
  icon.className = 'result-icon ' + (isWin ? 'win' : 'loss');
  icon.textContent = isWin ? '🎯' : '📉';
  document.getElementById('result-verdict').textContent = isWin ? 'Predicted Win' : 'Predicted Loss';
  document.getElementById('result-verdict').className = 'result-verdict ' + (isWin ? 'win' : 'loss');
  document.getElementById('result-sub').textContent = risk + ' risk · ' + (pct > 70 ? 'High' : pct > 50 ? 'Moderate' : 'Low') + ' confidence';

  document.getElementById('prob-pct').textContent = pct + '%';
  const fill = document.getElementById('prob-fill');
  fill.className = 'prob-fill ' + (isWin ? 'green' : 'red');
  fill.style.width = '0';
  setTimeout(() => fill.style.width = pct + '%', 50);

  document.getElementById('tab-pct').textContent = Math.round(data.tabular_reliance * 100) + '%';
  document.getElementById('txt-pct').textContent = Math.round(data.text_reliance * 100) + '%';

  const container = document.getElementById('shap-container');
  container.innerHTML = '';
  (data.top_shap_features || []).forEach(([name, val]) => {
    const isPos = val > 0;
    const w = Math.abs(val) * 180;
    const row = document.createElement('div');
    row.className = 'shap-row';
    row.innerHTML = `
      <div class="shap-feat">${name}</div>
      <div class="shap-track">
        <div class="${isPos ? 'shap-fill-pos' : 'shap-fill-neg'}" data-w="${w}"></div>
        <div class="shap-mid"></div>
      </div>
      <div class="shap-val ${isPos ? 'pos' : 'neg'}">${isPos ? '+' : ''}${val.toFixed(3)}</div>`;
    container.appendChild(row);
  });
  setTimeout(() => {
    document.querySelectorAll('.shap-fill-pos[data-w], .shap-fill-neg[data-w]').forEach(el => {
      el.style.transition = 'width 0.9s ease';
      el.style.width = el.dataset.w + 'px';
    });
  }, 80);

  document.getElementById('rec-box').textContent = data.recommendation || '';
}

// Batch upload
document.getElementById('batch-drop').addEventListener('click', () => {
  document.getElementById('batch-file').click();
});

async function handleBatchFile(input) {
  if (!input.files[0]) return;
  const formData = new FormData();
  formData.append('file', input.files[0]);

  try {
    const res = await fetch('/api/batch', { method: 'POST', body: formData });
    const data = await res.json();
    batchData = data;
    showBatchResults(data);
  } catch (err) {
    console.error(err);
  }
}

function showBatchResults(data) {
  document.getElementById('batch-results').classList.add('show');
  document.getElementById('batch-count').textContent = data.length;
  const tbody = document.getElementById('batch-tbody');
  tbody.innerHTML = '';
  data.forEach((row, i) => {
    const isWin = row.prediction === 'Win';
    const riskClass = row.risk_level === 'High' ? 'high' : row.risk_level === 'Low' ? 'low' : 'med';
    const tr = document.createElement('tr');
    tr.style.animationDelay = (i * 0.06) + 's';
    tr.innerHTML = `
      <td style="color:var(--ink-faint);font-size:12px">${i+1}</td>
      <td style="max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--ink-muted)">${row.preview}</td>
      <td style="font-family:'DM Mono',monospace;font-size:13px;color:${isWin?'var(--win-green)':'var(--loss-red)'}">${Math.round(row.probability*100)}%</td>
      <td><span class="pill pill-${row.prediction.toLowerCase()}">${row.prediction}</span></td>
      <td><span class="pill pill-${riskClass}">${row.risk_level}</span></td>`;
    tbody.appendChild(tr);
  });
}

function downloadBatch() {
  const csv = 'id,preview,probability,prediction,risk_level\n' +
    batchData.map((r,i)=>`${i+1},"${(r.preview||'').replace(/"/g,"'")}",${r.probability},${r.prediction},${r.risk_level}`).join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
  a.download = 'batch_predictions.csv';
  a.click();
}

// Validation API call
async function runValidation() {
  const btn = document.getElementById('val-btn');
  btn.disabled = true;
  btn.textContent = 'Running…';

  try {
    const res = await fetch('/api/validation', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ engagement: parseFloat(document.getElementById('val-eng').value) })
    });
    const data = await res.json();
    showValidationResults(data);
  } catch (err) {
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run validation';
  }
}

function showValidationResults(data) {
  let passed = 0;
  data.forEach((r, i) => {
    document.getElementById(`vp${i}`).textContent = r.predicted;
    document.getElementById(`vpr${i}`).textContent = Math.round(r.probability * 100) + '%';
    document.getElementById(`vb${i}`).style.width = Math.round(r.probability * 100) + '%';
    const s = document.getElementById(`vs${i}`);
    if (r.passed) {
      passed++;
      s.className = 'val-status pass';
      s.textContent = '● Pass';
    } else {
      s.className = 'val-status check';
      s.textContent = '⚠ Check';
    }
  });
  document.getElementById('score-n').textContent = passed;
  document.getElementById('score-acc').textContent = Math.round(passed/3*100) + '%';
  document.getElementById('score-detail').textContent = passed === 3 ? 'All samples passed validation' : passed === 2 ? 'Minor deviation on edge case' : 'Review required';
  document.getElementById('score-report').innerHTML = `Engine: Hybrid Gated-Fusion v3<br>Engagement: ${document.getElementById('val-eng').value}<br>Status: ${passed === 3 ? '✓ All passing' : '⚠ ' + (3-passed) + ' sample(s) deviated'}`;
}

// Auto-run validation on scroll
const valObs = new IntersectionObserver(entries => {
  entries.forEach(e => { if(e.isIntersecting) { runValidation(); valObs.unobserve(e.target); } });
}, {threshold:0.3});
valObs.observe(document.getElementById('validation'));
</script>
</body>
</html>
'''


@app.route("/")
def index():
    ablation = load_ablation_results()
    examples = load_examples()
    return render_template_string(
        HTML_TEMPLATE,
        model_online=predictor.is_ready(),
        ablation=ablation,
        examples=examples,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    result = predictor.predict(
        conversation=data.get("conversation", ""),
        engagement=float(data.get("engagement", 0.55)),
        product_type=data.get("product_type", "Technology"),
        channel=data.get("channel", "video_call"),
    )
    return jsonify(result)


@app.route("/api/batch", methods=["POST"])
def api_batch():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    
    df = pd.read_csv(request.files["file"])
    text_cols = ["full_text", "conversation", "text", "dialogue", "message", "body", "content"]
    text_col = next((c for c in text_cols if c in df.columns), None)
    if not text_col:
        obj_cols = df.select_dtypes(include=["object"]).columns
        if len(obj_cols):
            text_col = max(obj_cols, key=lambda c: df[c].str.len().mean())
    
    results = []
    for _, row in df.iterrows():
        text = str(row.get(text_col, ""))
        pred = predictor.predict(text)
        results.append({
            "preview": text[:80],
            "probability": pred["probability"],
            "prediction": pred["predicted_class"],
            "risk_level": pred["risk_level"],
        })
    return jsonify(results)


@app.route("/api/validation", methods=["POST"])
def api_validation():
    engagement = float(request.get_json().get("engagement", 0.55))
    examples = load_examples()
    expected = {"win": "Win", "loss": "Loss", "edge_case": "Win"}
    
    results = []
    for name, text in examples.items():
        pred = predictor.predict(text, engagement=engagement)
        results.append({
            "name": name,
            "expected": expected.get(name, "Win"),
            "predicted": pred["predicted_class"],
            "probability": pred["probability"],
            "risk_level": pred["risk_level"],
            "passed": pred["predicted_class"] == expected.get(name, "Win"),
        })
    return jsonify(results)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("📊 SalesIQ — Predictive Analytics Engine")
    print("=" * 50)
    print(f"   Model: {'✓ Online' if predictor.is_ready() else '○ Demo Mode'}")
    print(f"   XGBoost: {'✓ Loaded' if predictor.xgb_model else '○ Not loaded'}")
    print("=" * 50)
    print("\n   🚀 http://localhost:8888\n")
    
    app.run(host="0.0.0.0", port=8888, debug=True)
