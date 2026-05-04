# Predictive Sales Analytics Engine

Hybrid ML/DL system for predicting SaaS deal outcomes from sales conversations.  
The project combines tabular features + text modeling, compares ablations, and provides explainability artifacts (SHAP, attention, gating behavior).

Dataset source: [`DeepMostInnovations/saas-sales-conversations`](https://huggingface.co/datasets/DeepMostInnovations/saas-sales-conversations)

## Project Highlights

- **Multimodal pipeline**: tabular branch + text branch + fusion logic
- **Ablation-first evaluation**: full hybrid vs reduced variants
- **Explainability**: SHAP summaries, per-sample feature effects, attention/gate analysis
- **Interactive app**: Streamlit UI for single/batch prediction demos
- **Reproducible workflow**: notebooks for staged experimentation + scripts for orchestration

## Tech Stack

- Python, NumPy, Pandas, scikit-learn
- PyTorch, Transformers
- XGBoost, SHAP
- Jupyter notebooks
- Streamlit + Plotly

## Setup

### 1) Create environment

```bash
cd predictive-sales-analytics-engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install jupyter ipykernel
```

### 2) Select kernel in Jupyter/VS Code

Use the project interpreter:

```bash
.venv/bin/python
```

### 3) (Optional) Hugging Face token

Without token, you may see an HF warning and lower rate limits:

```bash
export HF_TOKEN=your_token_here
```

## Recommended Execution Flow

Run from repo root with the project kernel active.

### Core notebook path (current Phase-3 focus)

1. `notebooks/06_hybrid_fusion.ipynb`  
   Train/build hybrid components and save artifacts.
2. `notebooks/07_ablation_studies.ipynb`  
   Compare architectural variants and write ablation metrics.
3. `notebooks/08_explainability.ipynb`  
   Generate SHAP and interpretability outputs.

### Earlier phase notebooks (baseline + DL progression)

- `notebooks/02_eda_preparation.ipynb`
- `notebooks/03_feature_engineering.ipynb`
- `notebooks/04_model_training.ipynb`
- `notebooks/05_model_evaluation.ipynb`
- `notebooks/06_dl_data_preprocessing.ipynb`
- `notebooks/07_dl_model_training.ipynb`
- `notebooks/08_dl_evaluation.ipynb`
- `notebooks/09_dl_interactive_prediction.ipynb`

## Run Scripts and App

### Pipeline script

```bash
python scripts/run_pipeline.py --device cpu --sample-size 40000
```

Useful flags:

- `--skip-ablation`
- `--batch-size`
- `--epochs-frozen`
- `--epochs-finetune`

### Streamlit app

```bash
streamlit run app.py
```

App tabs include single prediction, batch prediction, explainability/ablations, architecture view, and sample validation.

## Repository Structure

```text
.
├── app.py
├── scripts/
│   ├── run_pipeline.py
│   ├── run_notebook.py
│   └── retrain_lstm_with_glove.py
├── src/
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── deep_learning.py
│   ├── fusion_model.py
│   ├── explainability.py
│   ├── inference.py
│   ├── parsers.py
│   └── text_pipeline.py
├── notebooks/
├── models/
├── metrics/
├── figures/
├── docs/
└── data/
```

## Artifacts and Versioning

This repo generates large artifacts (`models/`, tokenizer files, figures, and some metrics).  
`.gitignore` is configured to avoid accidentally committing heavy generated files.

If a file was already tracked before adding ignore rules, remove it from tracking once:

```bash
git rm --cached <path>
```

## Troubleshooting

- **Kernel crashes in notebook**:
  - Confirm kernel path is `.venv/bin/python`
  - Reduce sample sizes for heavy cells
  - Keep optional heavy loads disabled unless needed (for example hybrid/gate analysis)
- **HF warning about unauthenticated requests**:
  - Warning only; set `HF_TOKEN` to improve limits/speed
- **XGBoost instability on some environments**:
  - Use the safe tree fallback path in notebook when needed

## Documentation

- `data/README.md`
- `notebooks/README.md`
- `docs/README.md`
- `docs/PROJECT_ANALYSIS.md`
- `docs/THEORETICAL_RIGOR.md`
