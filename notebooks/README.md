# Notebooks

## Run order

Execute in this order from the repository (run Jupyter with working directory `notebooks/` or adjust paths consistently):

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | `02_eda_preparation.ipynb` | Load HF data via `src.data_preparation`, structure, missing values, duplicates, cleaning, EDA plots and narrative cells |
| 2 | `03_feature_engineering.ipynb` | Build feature matrix (`SalesFeatureEngineer`), optional PCA, save `results/X_pca.npy`, `results/y.npy`, figures under `figures/` |
| 3 | `04_model_training.ipynb` | Train/test split, logistic regression baseline, Random Forest + `GridSearchCV` (F1), save comparisons to `metrics/` |
| 4 | `05_model_evaluation.ipynb` | ROC, confusion matrices, feature importance, consolidated metrics |

**Note:** There is no `01_data_loading.ipynb` or `06_deployment_demo.ipynb` in this repository; numbering matches the analysis stages above.

## Conventions

- First cells should ensure `sys.path` includes the project root so `import src` works.  
- Save plots to `../figures/` and tabular metrics to `../metrics/` when paths are relative to `notebooks/`.  
- Prefer extracting reusable logic into `src/` rather than duplicating large functions across notebooks.

## Automation

From repo root (after `pip install nbclient`):

```bash
python scripts/run_notebook.py notebooks/02_eda_preparation.ipynb
```

Use a generous `--timeout` for training notebooks if Random Forest search is slow.
