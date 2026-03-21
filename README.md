# Predictive Sales Analytics Engine

## 🎯 Overview

A machine learning system for predicting SaaS sales deal outcomes using **multimodal learning** that combines:

- 📝 **Conversation Text Data** (3072-dimensional embeddings)
- 📊 **CRM Metrics** (customer engagement, sales effectiveness, conversation length)

**Goal**: Help sales teams predict deal success probability with 75-82% accuracy using advanced fusion models.

---

## 📊 Project Highlights

| Metric                  | Value                           |
| ----------------------- | ------------------------------- |
| **Dataset Size**        | 100,000 conversations           |
| **Data Balance**        | 50.07% Won / 49.93% Lost        |
| **Features**            | 3,187 (embeddings + engineered) |
| **Target Accuracy**     | 75-82%                          |
| **Primary ML Approach** | Multimodal Fusion Model         |

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
cd /Users/bhavanishanker/predictive-sales-analytics-engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package locally
pip install -e .
```

### 2. Explore Data

```bash
jupyter notebook notebooks/02_eda_preparation.ipynb
```

### 3. Train Models

```bash
python -m src.model_training
```

### 4. Evaluate Results

```bash
python -m src.model_evaluation
```

---

## 📁 Project Structure

```
predictive-sales-analytics-engine/
│
├── 📂 data/                  # Datasets (raw and processed)
├── 📂 notebooks/             # Jupyter notebooks (01-06 sequence)
├── 📂 src/                   # Production Python code
├── 📂 results/               # Models, predictions, logs
├── 📂 metrics/               # Performance metrics (JSON/CSV)
├── 📂 figures/               # Visualizations and charts
├── 📂 docs/                  # Documentation
├── 📂 presentations/         # PPT and video scripts
│
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── setup.py                 # Package installation
├── PROJECT_STRUCTURE.md     # Detailed structure guide
└── MIGRATION_GUIDE.md       # File organization instructions
```

**→ See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed organization**

---

## 📖 Key Documentation

| Document                                           | Purpose                                            |
| -------------------------------------------------- | -------------------------------------------------- |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)       | Complete directory structure and file organization |
| [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)           | How to move and organize your files                |
| [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) | Feature descriptions and data schema               |
| [docs/MODELS.md](docs/MODELS.md)                   | Model architectures and training details           |
| [docs/RESULTS.md](docs/RESULTS.md)                 | Detailed results and performance metrics           |

---

## 🔄 Pipeline Workflow

```
1. Load Data
   ↓
2. Exploratory Data Analysis (EDA)
   └─→ Class distribution, sentiment analysis, feature correlations
   ↓
3. Feature Engineering
   └─→ 3,187 features from embeddings + numerical engineering
   ↓
4. Model Training
   └─→ Baseline (LR) → Advanced (RF) → Fusion Model
   ↓
5. Evaluation
   └─→ Metrics, confusion matrices, feature importance
   ↓
6. Visualization & Results
   └─→ figures/ + metrics/ + results/
```

---

## 📊 Data Overview

### Dataset

- **Source**: SaaS Sales Conversations (HuggingFace)
- **Samples**: 100,000 conversations
- **Balance**: Perfectly balanced (50/50 Won/Lost)

### Key Features

| Feature                 | Type      | Range    | Role                |
| ----------------------- | --------- | -------- | ------------------- |
| **customer_engagement** | Numerical | 0-100    | Engagement signal   |
| **sales_effectiveness** | Numerical | 0-100    | Sales quality       |
| **conversation_length** | Numerical | Minutes  | Duration            |
| **embeddings**          | Vector    | 3072-dim | Text representation |
| **outcome**             | Binary    | 0/1      | Target (Lost/Won)   |

---

## 🎯 Results Summary (Current)

### EDA Findings

- ✅ Perfectly balanced classes (no imbalance)
- ✅ Sentiment: Won avg 0.0059 vs Lost avg 0.0050
- ✅ No highly skewed features
- ✅ 7.42% outliers in engagement (manageable)
- ✅ Text patterns differ between Won/Lost conversations

### Engineering Results

- ✅ 3,187 features created
- ✅ Includes: embeddings + numerical + interaction terms
- ✅ Sparse matrix format for efficiency
- ✅ StandardScaler normalization applied

### Model Performance (Expected)

| Model         | Accuracy   | F1-Score    |
| ------------- | ---------- | ----------- |
| Baseline (LR) | 50-55%     | 0.50-55     |
| Random Forest | 65-75%     | 0.65-75     |
| **Fusion**    | **75-82%** | **0.75-82** |

_Will be updated with actual metrics after training_

---

## 📊 Metrics & Figures

### Available Metrics

- `metrics/performance_metrics.json` - Accuracy, F1, precision, recall
- `metrics/feature_importance.json` - Top features by importance
- `metrics/confusion_matrices/` - Classification matrices

### Available Figures

- `figures/eda/` - Data exploration visualizations
- `figures/model_results/` - Performance comparison charts
- `figures/feature_importance/` - Feature analysis plots

---

## 🔧 Key Files

### Source Code

- **src/data_preparation.py** - Load and clean data
- **src/feature_engineering.py** - Create 3,187 features
- **src/model_training.py** - Baseline, advanced, fusion models
- **src/model_evaluation.py** - Metrics and validation
- **src/prediction.py** - Inference interface

### Notebooks (Sequential)

1. `notebooks/01_data_loading.ipynb` - Data import
2. `notebooks/02_eda_preparation.ipynb` - Exploratory analysis ⭐
3. `notebooks/03_feature_engineering.ipynb` - Feature creation
4. `notebooks/04_model_training.ipynb` - Model development
5. `notebooks/05_model_evaluation.ipynb` - Performance analysis
6. `notebooks/06_deployment_demo.ipynb` - Live demo

---

## 💾 Savepoints

Key outputs saved to organized directories:

| Output      | Location               | Format     |
| ----------- | ---------------------- | ---------- |
| Models      | `results/models/`      | .pkl       |
| Predictions | `results/predictions/` | .csv       |
| Metrics     | `metrics/`             | JSON/CSV   |
| Figures     | `figures/`             | PNG/PDF    |
| Logs        | `results/logs/`        | .txt/.json |

---

## 🎤 Presentation

### Available Resources

- 📊 **PPT Prompt** → `presentations/PPT_GENERATOR_PROMPT.md`
- 🎯 **Rubric Guide** → `PRESENTATION_RUBRIC_GUIDE.md`
- 📝 **Video Script** → `presentations/video_script.md` (create as needed)

### Key Stats for Presentation

- **Dataset**: 100,000 conversations, perfectly balanced
- **Features**: 3,072-dim embeddings + 115 engineered features
- **Model**: Multimodal fusion combining text + CRM data
- **Performance**: 75-82% accuracy expected on test set

---

## 📋 Usage Examples

### Load and use a trained model

```python
import pickle
from pathlib import Path

# Load model
model_path = Path('results/models/fusion_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Access metrics

```python
import json

with open('metrics/performance_metrics.json') as f:
    metrics = json.load(f)

print(f"Fusion Model F1-Score: {metrics['fusion_model']['f1_score']}")
```

### Load processed data

```python
import pandas as pd

train_data = pd.read_csv('data/processed/train_data.csv')
test_data = pd.read_csv('data/processed/test_data.csv')
```

---

## 🔍 For New Team Members

1. **Start here**: Read [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
2. **Setup**: Follow [docs/SETUP.md](docs/SETUP.md)
3. **Understand data**: Review [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md)
4. **Explore**: Run `notebooks/02_eda_preparation.ipynb`
5. **Understand models**: Read [docs/MODELS.md](docs/MODELS.md)

---

## 📞 Questions or Issues?

See relevant documentation:

- 📚 **Structure questions** → [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- 🔧 **Setup issues** → [docs/SETUP.md](docs/SETUP.md)
- 📊 **Data questions** → [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md)
- 🤖 **Model questions** → [docs/MODELS.md](docs/MODELS.md)
- 📈 **Results questions** → [docs/RESULTS.md](docs/RESULTS.md)

---

## ✅ Checklist for Success

- ✅ Clean directory structure (done)
- ✅ Separated notebooks, code, data, results
- ✅ Comprehensive documentation
- ✅ Production-ready code organization
- ⏳ Complete model training (in progress)
- ⏳ Generate all metrics and figures
- ⏳ Create 10-minute presentation video
- ⏳ Final presentation to stakeholders

---

**Status**: 🔄 In Development | **Next**: Model training and evaluation

_Last Updated: March 21, 2026_

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
