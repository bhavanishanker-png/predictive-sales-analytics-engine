# Theoretical Rigor — Predictive Sales Analytics Engine

This note ties the implemented pipeline to standard statistical learning theory so you can reuse it in a LaTeX report, slides, or viva. It is aligned with the code in `src/feature_engineering.py` and notebooks `04_model_training` / `05_model_evaluation`.

## 1. Problem formulation

- **Task**: Binary classification of sales **outcome** \(y \in \{0,1\}\) from a high-dimensional feature vector \(x\) built from numeric scores, low-cardinality categoricals, and text (TF-IDF bag-of-n-grams or fixed-dimensional **embeddings** supplied with the dataset).
- **Goal**: Learn a hypothesis \(h: \mathcal{X} \to [0,1]\) that estimates \(\mathbb{P}(y=1 \mid x)\) (or a decision boundary) from a finite sample \(\{(x_i, y_i)\}_{i=1}^n\).

**Independence note (i.i.d. assumption)**: Standard supervised guarantees assume training pairs are drawn i.i.d. from a joint distribution. Conversations from the **same company or product** may be correlated; the rubric rewards acknowledging this as a possible violation and treating held-out evaluation as an operational estimate rather than a formal i.i.d. guarantee.

## 2. Feature construction (bias vs. variance)

- **Numeric scaling (`StandardScaler`)**: Linear models and regularized logistic regression are sensitive to feature scales; tree models are scale-invariant on numeric inputs, but a single pipeline block keeps the design consistent.
- **Ratios and interactions** (e.g. engagement per length, effectiveness per length, engagement \(\times\) effectiveness): These encode **domain structure**—intensity normalized by conversation size and nonlinear coupling—reducing the need for the model to infer those relationships from raw columns alone (lower **approximation error** / bias for smooth models).
- **One-hot encoding with cardinality caps**: High-cardinality categoricals would explode dimensionality and variance; restricting levels limits sparsity and overfitting risk (explicit **bias–variance** tradeoff in feature design).
- **Text as TF-IDF**: Bag-of-n-grams map text to a sparse high-dimensional space; TF-IDF down-weights ubiquitous terms. This is a classical **VSM** representation; it assumes conditional independence of terms given the class at scoring time (naive Bayes motivation), which is not strictly true but often works for linear separators.
- **Precomputed embeddings**: When present, they act as a deterministic nonlinear map from text to \(\mathbb{R}^d\), typically trained on large corpora (transfer learning). **Leakage check**: If embeddings were computed using information from the test set at dataset construction time, validation metrics would be optimistic; for grading, state clearly whether embeddings are conversation-local only.

## 3. Dimensionality reduction (PCA, when used)

If notebook 03 applies PCA to dense numeric / embedding blocks (or to a combined matrix), the motivation is:

- **Variance preservation**: Principal components align with directions of maximal empirical variance.
- **Regularization effect**: Projecting onto the top-\(k\) components restricts the hypothesis class, often reducing variance at the cost of a controlled increase in bias—consistent with the bias–variance decomposition of expected prediction error.

## 4. Logistic regression (baseline)

- **Model**: \(\mathbb{P}(y=1 \mid x) = \sigma(w^\top x + b)\) with \(\sigma\) the sigmoid.
- **Loss**: Average **log-loss** (binary cross-entropy), convex in \((w,b)\) for linear-in-features models → **global optimum** for the penalized objective (e.g. L2) under standard conditions.
- **Interpretation**: Weights indicate direction of association in **log-odds** space after the chosen feature map; useful for explaining which engineered signals align with wins.

## 5. Random Forest

- **Ensemble**: An average (or vote) over **bootstrap**-sampled trees with random feature subsets at splits.
- **Mechanism**: Decorrelating trees reduces variance relative to a single deep tree while retaining low bias on structured data.
- **Why here**: Mixed numeric + sparse high-dimensional text or embeddings; tree ensembles handle **nonlinearity** and **interactions** without manual specification. Compared to SVMs on huge sparse spaces, tree methods with subsampling can be more pragmatic at scale (rubric: justify model vs. data properties).

## 6. Hyperparameter search and cross-validation

- **`GridSearchCV` with F1 scoring**: Optimizes a **harmonic mean** of precision and recall, appropriate when **false positives and false negatives** both matter (e.g. mis-ranking a lost deal vs. chasing a dead lead).
- **Role of CV**: Estimates generalization error by rotating validation folds; reduces reliance on a single lucky train/test split (though it does not remove distributional shift between training data and future conversations).

## 7. Evaluation metrics (why not accuracy alone?)

- With **balanced** classes, accuracy can still hide poor minority-class performance; **F1** focuses on the positive class’s precision/recall tradeoff.
- **ROC-AUC** summarizes ranking quality across thresholds; **average precision (AP)** is informative under **class skew** (if present in other splits or production).

## 8. Limitations (honest rubric-level discussion)

- **Linearity of logistic regression** in feature space: Powerful only if features make the problem approximately linearly separable.
- **Computational cost**: Large sparse text features vs. PCA-compressed pipelines trade interpretability, training time, and memory.
- **Temporal drift**: Sales language and products evolve; static models need monitoring and refresh.

Use this section to answer viva questions on **assumptions**, **why F1 for tuning**, **bias–variance**, and **why forests vs. linear baselines** for your specific feature mix.
