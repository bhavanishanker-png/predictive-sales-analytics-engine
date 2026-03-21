# Source Code Directory

## Module Structure

```
src/
├── __init__.py                 # Package initialization
├── data_preparation.py         # Data loading and cleaning
├── feature_engineering.py      # Feature creation pipeline
├── model_training.py           # Model development
├── model_evaluation.py         # Metrics and validation
├── prediction.py               # Inference interface
└── utils.py                    # Helper functions
```

## Module Descriptions

### data_preparation.py

- Load datasets from HuggingFace
- Data cleaning and validation
- Train/test split creation
- Output: `data/processed/`

### feature_engineering.py

- Numerical feature transformations
- Text embedding and TF-IDF
- Feature interaction creation
- Scaling and normalization
- Output: Feature matrices (sparse)

### model_training.py

- Baseline model (Logistic Regression)
- Advanced models (Random Forest, XGBoost)
- Fusion model architecture
- Hyperparameter tuning
- Output: `results/models/`

### model_evaluation.py

- Accuracy, precision, recall, F1-score
- Confusion matrices
- ROC/PR curves
- Cross-validation
- Feature importance analysis
- Output: `metrics/` and `figures/`

### prediction.py

- Load trained models
- Make predictions on new data
- Confidence scores
- Feature attribution (SHAP)

### utils.py

- Common utility functions
- Config loading
- Logger setup
- Path helpers

## Usage

```python
from src import data_preparation, feature_engineering, model_training

# Load data
df = data_preparation.load_and_clean_data()

# Engineer features
X, y, feature_names = feature_engineering.create_features(df)

# Train model
model = model_training.train_fusion_model(X, y)
```

## Best Practices

- Keep modules focused and single-responsibility
- Add docstrings to all functions
- Use type hints for clarity
- Test functions independently
- Document assumptions and edge cases
