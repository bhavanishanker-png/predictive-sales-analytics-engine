# Figures Directory

## Structure

```
figures/
├── eda/                     # Exploratory Data Analysis plots
├── model_results/           # Model performance visualizations
└── feature_importance/      # Feature analysis plots
```

## EDA Figures

- `class_distribution.png` - Won vs Lost counts
- `sentiment_histogram.png` - Sentiment score distribution
- `engagement_distribution.png` - Customer engagement scores
- `conversation_length_boxplot.png` - Conversation duration analysis
- `correlation_heatmap.png` - Feature correlations
- `top_words_comparison.png` - Token frequency Won vs Lost

## Model Results

- `accuracy_comparison.png` - Bar chart of accuracy across models
- `f1_score_comparison.png` - F1-score progression
- `roc_curves.png` - ROC curves for all models
- `confusion_matrices.png` - Heatmaps for all models
- `pr_curves.png` - Precision-Recall curves

## Feature Importance

- `feature_importance_bar.png` - Top features ranked by importance
- `shap_values_plot.png` - SHAP value contributions
- `embedding_space_tsne.png` - t-SNE visualization of embeddings

## Export Settings

- **Format**: PNG (or PDF for presentations)
- **Resolution**: 300 DPI for print quality
- **Size**: 8x6 inches (or 16:9 aspect ratio)
- **Colors**: Use colorblind-friendly palettes
- **Fonts**: Sans-serif, minimum 12pt for readability

## Generation

All figures should be generated programmatically:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
# ... plotting code ...
plt.savefig('figures/eda/my_plot.png', dpi=300, bbox_inches='tight')
plt.close()
```

## Usage in Presentations

- Reference figures by relative path
- Embed into slides or reports
- Include captions and legends
- Source and date each visualization
