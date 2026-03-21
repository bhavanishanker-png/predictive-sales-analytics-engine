# Results Directory

## Structure

```
results/
├── models/              # Trained model files
│   ├── baseline_model.pkl
│   ├── random_forest_model.pkl
│   └── fusion_model.pkl
├── predictions/         # Model outputs on datasets
│   ├── train_predictions.csv
│   ├── test_predictions.csv
│   └── validation_predictions.csv
└── logs/               # Training and execution logs
    ├── training_log.txt
    └── model_history.json
```

## File Formats

### Model Files (.pkl)

- Serialized scikit-learn models
- Load with: `pickle.load(open('model.pkl', 'rb'))`

### Prediction Files (.csv)

Columns:

- `index`: Original row index
- `true_label`: Actual outcome
- `predicted_label`: Model prediction
- `prediction_probability`: Confidence score
- (Optional) Feature attributions

### Logs (.txt, .json)

- Training parameters
- Execution time per epoch
- Training/validation loss curves
- Model architecture info

## Naming Convention

- `{model_name}_{dataset_type}_{timestamp}`
- Example: `fusion_model_test_2026-03-21_v1.pkl`

## Best Practices

- ✅ Always version model files
- ✅ Save predictions with metadata
- ✅ Log all training runs
- ✅ Keep originals for reproducibility
