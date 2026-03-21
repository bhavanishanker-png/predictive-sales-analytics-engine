# Documentation

## What is in this folder

| File | Description |
|------|-------------|
| [Literature Review.pdf](Literature%20Review.pdf) | Course literature review (PDF). Keep citations and narrative aligned with the notebooks and `README.md` at repo root. |
| [THEORETICAL_RIGOR.md](THEORETICAL_RIGOR.md) | Theory for the report and viva: problem setup, features, logistic regression vs. Random Forest, CV/F1, metrics, limitations. |

## Suggested report outline (LaTeX)

Map your write-up to the rubric:

1. **Introduction** — business problem, contribution, paper gap (from literature PDF).  
2. **Data** — source (Hugging Face), schema, EDA findings (from `02_eda_preparation.ipynb`).  
3. **Methods** — cleaning (`src/data_preparation.py`), features (`src/feature_engineering.py`), models and tuning (`04_model_training.ipynb`); cite **THEORETICAL_RIGOR.md** for formal justification.  
4. **Results** — tables/plots from `metrics/` and `figures/`.  
5. **Discussion** — limitations (i.i.d., leakage, drift), ethical use of predictions.  

## Optional additions (not required to exist in-repo)

If you extend the project, you may add:

- `SETUP.md` — environment pinning (`pip freeze`), GPU notes, HF token.  
- `DATA_DICTIONARY.md` — column definitions from the dataset card.  
- `RESULTS.md` — narrative interpretation of saved metrics.

## Best practices

- Update the literature PDF and this folder when you change the modeling story.  
- Cross-link figure filenames (`figures/evaluation_roc_curves.png`, etc.) in the LaTeX report for traceability.
