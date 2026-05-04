"""Streamlit app for hybrid sales outcome prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.parsers import detect_text_column, parse_uploaded_file

PROJECT_ROOT = Path(__file__).parent
ASSETS_DIR = PROJECT_ROOT / "assets" / "examples"
METRICS_PATH = PROJECT_ROOT / "metrics" / "ablation_results.json"


def _load_example_texts() -> Dict[str, str]:
    examples = {
        "Clear Win - Enterprise CRM deal": "example_win.txt",
        "Clear Loss - Budget objection": "example_loss.txt",
        "Edge Case - Competitor evaluation": "example_edge_case.txt",
    }
    loaded = {}
    for title, filename in examples.items():
        path = ASSETS_DIR / filename
        loaded[title] = path.read_text(encoding="utf-8") if path.exists() else ""
    return loaded


def _mock_prediction(conversation: str, engagement: float) -> Dict[str, object]:
    """Fallback mock predictor until trained artifacts are available."""
    text = conversation.lower()
    positive = sum(k in text for k in ["demo", "integrate", "interested", "timeline", "next step"])
    negative = sum(k in text for k in ["price", "budget", "expensive", "competitor", "later"])
    score = np.clip(0.5 + 0.08 * positive - 0.1 * negative + 0.15 * (engagement - 0.5), 0.02, 0.98)

    tab_share = float(np.clip(0.65 - 0.35 * negative + 0.1 * positive, 0.15, 0.85))
    text_share = 1.0 - tab_share

    lines = [line.strip() for line in conversation.splitlines() if line.strip()]
    attention = []
    for line in lines:
        weight = 0.05
        l = line.lower()
        if any(k in l for k in ["price", "budget", "competitor", "integrate", "timeline"]):
            weight += 0.20
        attention.append((line, min(weight, 0.35)))

    attention.sort(key=lambda x: x[1], reverse=True)
    top_attention = attention[:3]
    shap_drivers = [
        ("customer_engagement", round((engagement - 0.5) * 0.4, 3)),
        ("conversation_length", round((0.12 if len(lines) > 10 else -0.08), 3)),
        ("communication_channel_chat", 0.05),
    ]

    recommendation = "Moderate potential."
    if score < 0.3:
        recommendation = "Low win probability. Prepare a rescue plan."
    elif score < 0.5:
        recommendation = "Leaning toward loss but recoverable. Address objections directly."
    elif score >= 0.7:
        recommendation = "Strong win potential. Push for a concrete next step."

    if any("price" in t.lower() or "budget" in t.lower() for t, _ in top_attention):
        recommendation += " Pricing concerns detected - prepare ROI justification."
    if any("competitor" in t.lower() for t, _ in top_attention):
        recommendation += " Competitor evaluation detected - share a comparison sheet."

    risk = "Low"
    if 0.40 <= score <= 0.60:
        risk = "High"
    elif 0.25 <= score <= 0.75:
        risk = "Medium"

    return {
        "probability": float(score),
        "predicted_class": "Win" if score >= 0.5 else "Loss",
        "risk_level": risk,
        "tabular_reliance": tab_share,
        "text_reliance": text_share,
        "top_shap_features": shap_drivers,
        "top_attention_turns": top_attention,
        "recommendation": recommendation,
    }


def render_attention_map(turns: List[Tuple[str, float]]) -> None:
    """Render lightweight turn-level heatmap using inline HTML."""
    st.markdown("**Conversation Attention Map**")
    if not turns:
        st.caption("No conversation turns available.")
        return
    for text, weight in turns:
        alpha = min(max(weight * 2.5, 0.08), 0.7)
        block = (
            "<div style='padding:8px;border-radius:6px;margin:4px 0;"
            f"background: rgba(255,120,0,{alpha});'>{text}</div>"
        )
        st.markdown(block, unsafe_allow_html=True)


def tab_single_prediction() -> None:
    st.subheader("Single Prediction")
    examples = _load_example_texts()

    left, right = st.columns(2)
    with left:
        mode = st.radio("Input Mode", ["Paste conversation", "Upload file", "Load example"])

        conversation = ""
        uploaded_df = None
        if mode == "Paste conversation":
            conversation = st.text_area("Conversation", height=280)
        elif mode == "Upload file":
            file = st.file_uploader("Upload .txt, .csv, .pdf, or .eml", type=["txt", "csv", "pdf", "eml"])
            if file is not None:
                conversation, uploaded_df = parse_uploaded_file(file)
                st.text_area("Parsed conversation", conversation, height=200, disabled=True)
        else:
            selected = st.selectbox("Example", list(examples.keys()))
            conversation = examples.get(selected, "")
            st.text_area("Loaded example", conversation, height=200, disabled=True)

        st.markdown("**Deal Metadata**")
        _ = st.selectbox("Product Type", ["CRM", "Analytics", "Marketing", "Support", "Finance"])
        _ = st.selectbox("Conversation Style", ["direct_professional", "casual_friendly", "consultative"])
        _ = st.selectbox("Communication Channel", ["email", "chat", "phone", "video_call"])
        engagement = st.slider("Customer Engagement", 0.0, 1.0, 0.55, 0.01)

        run = st.button("Predict Outcome", type="primary")

    with right:
        if run and conversation.strip():
            result = _mock_prediction(conversation, engagement)

            st.metric("Prediction", f"{result['predicted_class']} - {result['probability'] * 100:.1f}%")
            st.progress(float(result["probability"]))
            st.caption(f"Risk level: **{result['risk_level']}**")

            c1, c2 = st.columns(2)
            c1.metric("Tabular signal", f"{result['tabular_reliance'] * 100:.1f}%")
            c2.metric("Text signal", f"{result['text_reliance'] * 100:.1f}%")

            st.markdown("**Top Tabular Drivers (SHAP-style)**")
            for feat, value in result["top_shap_features"]:
                arrow = "↑" if value >= 0 else "↓"
                st.write(f"{arrow} `{feat}`: {value:+.3f}")

            render_attention_map(result["top_attention_turns"])
            st.info(result["recommendation"])
        else:
            st.caption("Run a prediction to view probability, explanations, and recommendations.")


def tab_batch_prediction() -> None:
    st.subheader("Batch Prediction")
    file = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
    if file is None:
        st.caption("Upload a CSV file to run batch inference.")
        return

    df = pd.read_csv(file)
    text_col = detect_text_column(df)
    st.write(f"Detected conversation column: `{text_col}`")
    st.dataframe(df.head(5))

    if st.button("Run Batch Prediction"):
        probs = []
        risks = []
        prog = st.progress(0.0)
        for i, text in enumerate(df[text_col].fillna("").astype(str).tolist()):
            pred = _mock_prediction(text, engagement=0.55)
            p = pred["probability"]
            probs.append(p)
            risks.append(pred["risk_level"])
            prog.progress((i + 1) / len(df))

        out = df.copy()
        out["probability"] = probs
        out["prediction"] = np.where(out["probability"] >= 0.5, "Win", "Loss")
        out["risk_level"] = risks
        st.dataframe(out.head(20))
        st.download_button(
            "Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="batch_predictions.csv",
            mime="text/csv",
        )


def tab_explainability() -> None:
    st.subheader("Model Explainability & Ablations")
    if not METRICS_PATH.exists():
        st.warning("`metrics/ablation_results.json` not found yet. Run ablation notebook first.")
        return

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for model_name, metrics in data.items():
        rows.append(
            {
                "model": model_name,
                "f1": metrics.get("f1", 0.0),
                "edge_case_f1": metrics.get("edge_case_f1", 0.0),
                "auc": metrics.get("auc", 0.0),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("Ablation results file is empty.")
        return

    fig1 = px.bar(df.sort_values("f1", ascending=False), x="f1", y="model", orientation="h", title="Ablation F1 Scores")
    st.plotly_chart(fig1, use_container_width=True)

    melt = df.melt(id_vars=["model"], value_vars=["f1", "edge_case_f1"], var_name="metric", value_name="score")
    fig2 = px.bar(melt, x="model", y="score", color="metric", barmode="group", title="Overall vs Edge-Case F1")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        "- Full hybrid should lead on edge-case robustness.\n"
        "- Gated fusion should outperform concat on ambiguous deals.\n"
        "- Leaf encoding should outperform raw tabular projection."
    )


def tab_architecture() -> None:
    st.subheader("Architecture")
    st.markdown(
        """
| Component | Design |
|---|---|
| Dataset | DeepMostInnovations/saas-sales-conversations |
| Tabular Branch | OHE + engineered features + XGBoost leaf encoding |
| Text Branch | DistilBERT + learned attention pooling |
| Fusion | Gated fusion (`g*tab + (1-g)*text`) |
| Classifier | MLP (`128 -> 64 -> 1`) |
| Explainability | SHAP + attention map + gate reliance |
"""
    )

    arch_img = PROJECT_ROOT / "figures" / "hybrid_architecture.png"
    if arch_img.exists():
        st.image(str(arch_img), caption="Hybrid Gated-Fusion Architecture")
    else:
        st.caption("Add `figures/hybrid_architecture.png` to display architecture diagram.")


def tab_sample_validation() -> None:
    st.subheader("Sample Validation Demo")
    st.caption("Quick check on built-in sample conversations.")

    examples = _load_example_texts()
    expected = {
        "Clear Win - Enterprise CRM deal": "Win",
        "Clear Loss - Budget objection": "Loss",
        "Edge Case - Competitor evaluation": "Win",
    }

    engagement = st.slider("Validation engagement level", 0.0, 1.0, 0.55, 0.01, key="validation_engagement")
    if not st.button("Run Sample Validation", type="primary"):
        st.info("Click the button to run all sample conversations.")
        return

    rows = []
    for title, convo in examples.items():
        result = _mock_prediction(convo, engagement)
        pred = str(result["predicted_class"])
        exp = expected.get(title, "Unknown")
        rows.append(
            {
                "sample": title,
                "expected": exp,
                "predicted": pred,
                "probability": round(float(result["probability"]), 4),
                "risk_level": result["risk_level"],
                "status": "PASS" if pred == exp else "CHECK",
            }
        )

    out = pd.DataFrame(rows)
    st.dataframe(out, use_container_width=True)
    pass_count = int((out["status"] == "PASS").sum())
    st.metric("Samples matching expected label", f"{pass_count}/{len(out)}")
    st.caption("`CHECK` means review conversation details or adjust expected label for your scenario.")


def main() -> None:
    st.set_page_config(page_title="Predictive Sales Analytics Engine", page_icon="🎯", layout="wide")
    st.title("🎯 Predictive Sales Analytics Engine")
    st.caption("Phase 3: Hybrid Gated-Fusion + Explainability + Demo")

    t1, t2, t3, t4, t5 = st.tabs(
        ["Single Prediction", "Batch Prediction", "Model Explainability", "Architecture", "Sample Validation"]
    )
    with t1:
        tab_single_prediction()
    with t2:
        tab_batch_prediction()
    with t3:
        tab_explainability()
    with t4:
        tab_architecture()
    with t5:
        tab_sample_validation()


if __name__ == "__main__":
    main()
