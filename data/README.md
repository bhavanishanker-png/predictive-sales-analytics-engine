# Data directory

## Layout

```
data/
├── raw/                    # Optional local cache of Hugging Face files
│   └── saas-sales-conversations/
└── processed/              # Train/test splits or cleaned CSVs (if you export them)
    ├── train_data.csv      # (optional; gitignored when large)
    ├── test_data.csv
    └── validation_data.csv
```

## Source dataset

- **Hugging Face**: `DeepMostInnovations/saas-sales-conversations`  
- **Loading in code**: `src/data_preparation.py` (`load_dataset` via `datasets`).  
- **Notebooks**: Primary exploration in `notebooks/02_eda_preparation.ipynb`; feature pipeline in `notebooks/03_feature_engineering.ipynb`.

The published snapshot includes many **embedding\_*** numeric columns (dimensionality reduction already applied upstream) plus conversation metadata and labels. Treat embeddings as **fixed features** unless you recompute them yourself with a documented model.

## Usage guidelines

- Do **not** edit files under `raw/`; treat them as immutable inputs.  
- Write derived tables to `processed/` or to `results/` (arrays) as implemented in the notebooks.  
- Large CSVs and pickles under `data/raw/` and `data/processed/` are **gitignored**; reviewers clone the repo and regenerate data per root `README.md`.

## Reproducibility

- Set `HF_TOKEN` if Hugging Face rate-limits anonymous downloads.  
- After pulling data, run notebooks **02 → 03 → 04 → 05** so `metrics/` and `figures/` match your report.
