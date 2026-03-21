# Metrics Directory

## Structure

```
metrics/
├── confusion_matrices/          # Classification confusion matrices
│   ├── baseline_cm.json
│   ├── rf_cm.json
│   └── fusion_cm.json
├── performance_metrics.json     # All key metrics
├── feature_importance.json      # Feature importance scores
├── cross_validation_scores.csv  # CV results
└── model_comparison.json        # Side-by-side comparison
```

## File Contents

### performance_metrics.json

```json
{
  "baseline_model": {
    "accuracy": 0.785,
    "precision": 0.80,
    "recall": 0.76,
    "f1_score": 0.76,
    "auc_roc": 0.82,
    "auc_pr": 0.80
  },
  "random_forest": {...},
  "fusion_model": {...}
}
```

### confusion_matrices/

```json
{
  "true_negatives": 10000,
  "false_positives": 2000,
  "false_negatives": 1500,
  "true_positives": 11500
}
```

### feature_importance.json

```json
{
  "customer_engagement": 0.28,
  "conversation_sentiment": 0.22,
  "sales_effectiveness": 0.19,
  "engagement_ratio": 0.16,
  "effectiveness_ratio": 0.15
}
```

### cross_validation_scores.csv

Columns: fold, train_accuracy, val_accuracy, train_f1, val_f1, training_time

### model_comparison.json

Side-by-side metrics for all models for easy comparison

## Usage

```python
import json
import pandas as pd

# Load metrics
with open('metrics/performance_metrics.json') as f:
    metrics = json.load(f)

# Load CV scores
cv_df = pd.read_csv('metrics/cross_validation_scores.csv')

# Access specific metric
f1_score = metrics['fusion_model']['f1_score']
```

## Best Practices

- ✅ Export all metrics in structured format (JSON/CSV)
- ✅ Include timestamps for reproducibility
- ✅ Document metric definitions and calculation methods
- ✅ Compare across all models consistently
