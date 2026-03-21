# Data Directory

## Structure

```
data/
├── raw/              # Original datasets from HuggingFace
│   └── saas-sales-conversations/
└── processed/        # Cleaned and preprocessed data
    ├── train_data.csv
    ├── test_data.csv
    └── validation_data.csv
```

## Usage

- **raw/**: Download HuggingFace datasets here (loading logic lives in `notebooks/02_eda_preparation.ipynb` and `notebooks/03_feature_engineering.ipynb`)
- **processed/**: Cleaned data automatically saved here after preprocessing

## Important

- ⚠️ Never modify raw data directly
- ⚠️ Always create processed versions for modeling
- Add `data/raw/` and `data/processed/*.csv` to `.gitignore` (large files)
