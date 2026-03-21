# Notebooks Directory

## Sequential Workflow

Execute notebooks in numerical order for proper pipeline flow:

1. **01_data_loading.ipynb** - Import and explore raw dataset
2. **02_eda_preparation.ipynb** - Exploratory data analysis and insights
3. **03_feature_engineering.ipynb** - Create and transform features
4. **04_model_training.ipynb** - Train baseline, advanced, and fusion models
5. **05_model_evaluation.ipynb** - Evaluate performance and generate metrics
6. **06_deployment_demo.ipynb** - Live prediction demo and API testing

## Purpose

- Exploratory analysis and visualization
- Model experimentation and tuning
- Results documentation and reporting
- NOT for production code (extract shared logic to a package if you need reuse outside Jupyter)

## Best Practices

- Each notebook should be self-contained and reproducible
- Install required packages in first cell
- Document findings and key insights
- Save outputs (figures, metrics) to appropriate directories
- Run notebooks from the `notebooks/` directory so relative paths (e.g. to `figures/`, `metrics/`) resolve correctly
