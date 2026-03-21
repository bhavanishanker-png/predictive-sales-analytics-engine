# Predictive Sales Analytics Engine

A Python-based machine learning pipeline for predicting sales outcomes from SaaS sales conversations. This project processes conversation data, performs exploratory analysis, and engineers features for predictive modeling.

## Overview

The Predictive Sales Analytics Engine automates the pipeline for:

- **Data Preparation**: Loading and cleaning SaaS sales conversation datasets from Hugging Face
- **Exploratory Data Analysis**: Analyzing conversation patterns and outcomes
- **Feature Engineering**: Converting raw conversation data into ML-ready feature matrices

## Project Structure

```
├── data_preparation.py       # Data loading and cleaning pipeline
├── feature_engineering.py    # Feature engineering for ML models
├── eda_preparation.ipynb     # Exploratory data analysis notebook
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Key Modules

### `data_preparation.py`

Handles data loading and basic cleaning for the SaaS sales conversations dataset:

- Loads data from Hugging Face `DeepMostInnovations/saas-sales-conversations`
- Converts between Hugging Face Dataset and pandas DataFrames
- Manages multiple data splits (train, test, validation)

### `feature_engineering.py`

Transforms cleaned data into ML-ready features:

- Supports two text processing strategies: TF-IDF vectorization or precomputed embeddings
- Extracts numerical features (engagement, effectiveness, conversation length)
- Encodes categorical variables with one-hot encoding
- Outputs scipy sparse matrices for efficient model training

### `eda_preparation.ipynb`

Interactive Jupyter notebook for exploratory data analysis:

- Visualizes conversation patterns
- Analyzes outcome distributions
- Identifies feature relationships

## Requirements

- Python 3.8+
- pandas >= 2.2.0
- scikit-learn >= 1.5.0
- scipy >= 1.11.0
- datasets >= 2.19.0

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd predictive-sales-analytics-engine
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Load Data

```python
from data_preparation import load_hf_dataset, dataset_to_dataframe

# Load dataset from Hugging Face
dataset = load_hf_dataset("DeepMostInnovations/saas-sales-conversations")
df = dataset_to_dataframe(dataset)
```

### Engineer Features

```python
from feature_engineering import SalesFeatureEngineer, FeatureEngineeringConfig

config = FeatureEngineeringConfig(
    target_col="outcome",
    text_strategy="tfidf",
    tfidf_max_features=3000
)

engineer = SalesFeatureEngineer(config)
X, y = engineer.build_features(df)
```

## License

[Add your license here]

## Contact

[Add your contact information here]
