# Predictive Sales Analytics Engine

End-to-end machine learning workflow to predict **SaaS sales conversation outcomes** from structured fields, categorical attributes, and text (or precomputed embeddings). Data is loaded from Hugging Face (`DeepMostInnovations/saas-sales-conversations`). The project implements a dual-track approach comparing **Classical ML models** with **Deep Learning architectures** (BiLSTM with GloVe embeddings, DistilBERT). Reusable logic lives under `src/`, experiments and reporting under `notebooks/`.

## Quick start

1. **Python**: 3.10+ recommended (3.8+ supported per `setup.py`).

2. **Environment**

   ```bash
   cd predictive-sales-analytics-engine
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install jupyter matplotlib seaborn joblib nbformat nbclient ipykernel torch transformers
   ```

   Core training also uses `numpy` (pulled in by scikit-learn). For headless notebook execution: `pip install nbclient`.

3. **Data**

   Download or stream the dataset via Hugging Face (see [data/README.md](data/README.md)). Large artifacts are gitignored.

4. **Run the pipeline (order)**

   Open Jupyter from the repo root or `notebooks/`, then run in sequence:

   | Step | Notebook | Purpose |
   |------|----------|---------|
   | 1 | [02_eda_preparation.ipynb](notebooks/02_eda_preparation.ipynb) | Load data,  quality checks,  cleaning,  EDA |
   | 2 | [03_feature_engineering.ipynb](notebooks/03_feature_engineering.ipynb) | Features, optional PCA, saves `results/*.npy` |
   | 3 | [04_model_training.ipynb](notebooks/04_model_training.ipynb) | Logistic regression + Random Forest, `GridSearchCV` |
   | 4 | [05_model_evaluation.ipynb](notebooks/05_model_evaluation.ipynb) | ROC, confusion matrices, metrics JSON/CSV |
   | 5 | [06_dl_data_preprocessing.ipynb](notebooks/06_dl_data_preprocessing.ipynb) | Text tokenization, word-level & WordPiece sequencing |
   | 6 | [07_dl_model_training.ipynb](notebooks/07_dl_model_training.ipynb) | BiLSTM + Attention (Random & GloVe), DistilBERT fine-tuning |
   | 7 | [08_dl_evaluation.ipynb](notebooks/08_dl_evaluation.ipynb) | Deep learning evaluation and Classical ML comparison |

## Repository layout

```
├── README.md                 # This file
├── requirements.txt          # Minimal runtime deps for src + HF load
├── setup.py                  # Optional editable install metadata
├── src/
│   ├── data_preparation.py   # HF load, cleaning, missing values
│   ├── feature_engineering.py # Numeric ratios, OHE, TF-IDF / embeddings
│   └── deep_learning.py      # PyTorch BiLSTM and DistilBERT architectures
├── notebooks/                # EDA → features → train → evaluate
├── figures/                  # Saved plots (e.g. ROC, feature logs)
├── metrics/                  # JSON/CSV metrics committed for reporting
├── docs/
│   ├── README.md             # Documentation index
│   ├── PROJECT_ANALYSIS.md   # Deep dive evaluation and senior review notes
│   ├── Literature Review.pdf # Submitted literature (PDF)
│   └── THEORETICAL_RIGOR.md  # Theory tied to this project
├── data/                     # raw/ and processed/ (see data/README.md)
└── results/                  # Generated arrays, checkpoints, tokenizers
```

## Results directory (`results/`)

Feature engineering writes arrays such as `X_pca.npy` and `y.npy` for downstream notebooks. Deep learning preprocessing caches tokenized tensors (`dl_train_ids.pt`, `dl_bert_train.pt`) and custom vocabulary (`dl_tokenizer.json`). Training saves `model_*_best.pt` models. Large binaries are listed in `.gitignore`.

## Documentation

- [data/README.md](data/README.md) — data layout and Hugging Face usage  
- [notebooks/README.md](notebooks/README.md) — notebook order and conventions  
- [docs/README.md](docs/README.md) — literature + theory + pointers for the written report  
- [docs/PROJECT_ANALYSIS.md](docs/PROJECT_ANALYSIS.md) — full project analysis, metrics breakdown & interview prep
- [docs/THEORETICAL_RIGOR.md](docs/THEORETICAL_RIGOR.md) — assumptions, models, metrics

## Models and metrics (summary)

- **Classical Baseline**: Logistic regression on engineered features (interpretable linear boundary).
- **Classical Comparator**: Random Forest with **GridSearchCV** (cross-validated).
- **Deep Learning Baseline**: BiLSTM + Self-Attention with random embedding initialization.
- **Deep Learning Advanced**: BiLSTM + Self-Attention using **GloVe pretrained embeddings**, and **DistilBERT** fine-tuning on raw conversation text.
- **Evaluation**: Accuracy, precision, recall, **F1**, **ROC-AUC**, average precision; confusion matrices and ROC plots under `figures/`.

## Contributing / course submission

Replace placeholder author fields in `setup.py` if you publish the repo. Keep `metrics/` and `figures/` updated when you re-run evaluation so the repository matches the report.
