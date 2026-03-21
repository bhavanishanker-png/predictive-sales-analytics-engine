"""Reusable feature engineering pipeline for sales prediction.

This module converts a cleaned DataFrame into:
- Feature matrix X (scipy sparse matrix)
- Target vector y (pandas Series)

It supports two text strategies:
1) TF-IDF vectorization
2) Precomputed embeddings when available
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FeatureEngineeringConfig:
    """Configuration for building the feature matrix."""

    target_col: str = "outcome"
    text_col: str = "conversation_text"

    # Base numerical columns expected from the cleaned dataset.
    engagement_col: str = "customer_engagement"
    effectiveness_col: str = "sales_effectiveness"
    length_col: str = "conversation_length"

    # Text strategy options: "auto", "tfidf", "embeddings".
    text_strategy: str = "auto"
    embedding_prefix: str = "embedding_"

    # TF-IDF settings.
    tfidf_max_features: int = 3000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_min_df: int = 3

    # Keep one-hot encoding practical by skipping very high-cardinality columns.
    max_categorical_levels: int = 50
    max_categorical_ratio: float = 0.10


class SalesFeatureEngineer:
    """Builds engineered features and transforms them into a model-ready matrix."""

    def __init__(self, config: Optional[FeatureEngineeringConfig] = None) -> None:
        self.config = config or FeatureEngineeringConfig()

        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.embedding_cols_: List[str] = []
        self.text_mode_: str = ""

        self.scaler_: Optional[StandardScaler] = None
        self.encoder_: Optional[OneHotEncoder] = None
        self.tfidf_: Optional[TfidfVectorizer] = None

        self.feature_names_: List[str] = []
        self._fitted: bool = False

    def fit_transform(self, df: pd.DataFrame) -> Tuple[csr_matrix, pd.Series]:
        """Fit the feature pipeline and return transformed X, y."""
        prepared_df, y, text_series = self._prepare_base_dataframe(df)

        self._set_column_groups(prepared_df)
        self._set_text_mode(prepared_df)

        X_numeric = self._fit_transform_numeric(prepared_df)
        X_categorical = self._fit_transform_categorical(prepared_df)
        X_text = self._fit_transform_text(prepared_df, text_series)

        X = hstack([X_numeric, X_categorical, X_text], format="csr")
        self._build_feature_names()

        self._fitted = True
        return X, y

    def transform(self, df: pd.DataFrame) -> Tuple[csr_matrix, pd.Series]:
        """Transform new data with already-fitted transformers."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform.")

        prepared_df, y, text_series = self._prepare_base_dataframe(df)

        X_numeric = self._transform_numeric(prepared_df)
        X_categorical = self._transform_categorical(prepared_df)
        X_text = self._transform_text(prepared_df, text_series)

        X = hstack([X_numeric, X_categorical, X_text], format="csr")
        return X, y

    def get_feature_names(self) -> List[str]:
        """Return ordered feature names for matrix columns."""
        if not self._fitted:
            raise RuntimeError("Feature names are available only after fit_transform.")
        return self.feature_names_

    def get_text_mode(self) -> str:
        """Return effective text feature mode used by the pipeline."""
        if not self._fitted:
            raise RuntimeError("Text mode is available only after fit_transform.")
        return self.text_mode_

    def _prepare_base_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare base columns and engineered features before encoding/vectorization."""
        if self.config.target_col not in df.columns:
            raise ValueError(f"Target column '{self.config.target_col}' is missing.")

        # Resolve text column robustly if preferred name is unavailable.
        text_col = self.config.text_col if self.config.text_col in df.columns else self._infer_text_column(df)
        if not text_col:
            raise ValueError("No suitable text column found for text feature engineering.")

        # Shallow copy avoids expensive block consolidation on very wide tables.
        working_df = df.copy(deep=False)

        # Create conversation length if missing.
        if self.config.length_col not in working_df.columns:
            working_df[self.config.length_col] = (
                working_df[text_col].fillna("").astype(str).str.split().str.len()
            )

        required_numeric = [
            self.config.engagement_col,
            self.config.effectiveness_col,
            self.config.length_col,
        ]
        missing_numeric = [col for col in required_numeric if col not in working_df.columns]
        if missing_numeric:
            raise ValueError(f"Missing numeric columns required for engineering: {missing_numeric}")

        numeric_converted = {
            col: pd.to_numeric(working_df[col], errors="coerce")
            for col in required_numeric + [self.config.target_col]
        }
        for col, series in numeric_converted.items():
            working_df[col] = series

        # Keep rows where target and core numeric columns are valid.
        working_df = working_df.dropna(subset=required_numeric + [self.config.target_col]).copy()
        working_df[self.config.target_col] = working_df[self.config.target_col].astype(int)

        # Core engineered numeric features.
        eps = 1e-6
        working_df["engagement_per_length"] = (
            working_df[self.config.engagement_col] / (working_df[self.config.length_col] + eps)
        )
        working_df["effectiveness_per_length"] = (
            working_df[self.config.effectiveness_col] / (working_df[self.config.length_col] + eps)
        )
        working_df["engagement_x_effectiveness"] = (
            working_df[self.config.engagement_col] * working_df[self.config.effectiveness_col]
        )

        # Optional categorical columns (excluding text and target).
        excluded = {
            text_col,
            self.config.target_col,
        }
        candidate_cats = [
            col
            for col in working_df.select_dtypes(include=["object", "category", "string"]).columns
            if col not in excluded
        ]

        # Encode only low/moderate-cardinality columns to avoid giant sparse matrices.
        cat_cols: List[str] = []
        for col in candidate_cats:
            nunique = int(working_df[col].nunique(dropna=True))
            ratio = nunique / max(len(working_df), 1)
            if nunique <= self.config.max_categorical_levels and ratio <= self.config.max_categorical_ratio:
                cat_cols.append(col)

        self.categorical_cols_ = cat_cols

        text_series = working_df[text_col].fillna("").astype(str)
        y = working_df[self.config.target_col]

        return working_df, y, text_series

    def _set_column_groups(self, prepared_df: pd.DataFrame) -> None:
        """Define numeric and embedding feature groups."""
        self.embedding_cols_ = [
            col for col in prepared_df.columns if col.startswith(self.config.embedding_prefix)
        ]

        base_numeric = [
            self.config.engagement_col,
            self.config.effectiveness_col,
            self.config.length_col,
            "engagement_per_length",
            "effectiveness_per_length",
            "engagement_x_effectiveness",
        ]
        self.numeric_cols_ = [col for col in base_numeric if col in prepared_df.columns]

    def _set_text_mode(self, prepared_df: pd.DataFrame) -> None:
        """Select effective text strategy based on config and data availability."""
        strategy = self.config.text_strategy.lower()
        if strategy not in {"auto", "tfidf", "embeddings"}:
            raise ValueError("text_strategy must be one of: auto, tfidf, embeddings")

        if strategy == "auto":
            self.text_mode_ = "embeddings" if self.embedding_cols_ else "tfidf"
        elif strategy == "embeddings":
            if not self.embedding_cols_:
                raise ValueError("Embeddings strategy requested, but no embedding columns were found.")
            self.text_mode_ = "embeddings"
        else:
            self.text_mode_ = "tfidf"

    def _fit_transform_numeric(self, prepared_df: pd.DataFrame) -> csr_matrix:
        self.scaler_ = StandardScaler()
        numeric_values = prepared_df[self.numeric_cols_].to_numpy(dtype=float)
        numeric_scaled = self.scaler_.fit_transform(numeric_values)
        return csr_matrix(numeric_scaled)

    def _transform_numeric(self, prepared_df: pd.DataFrame) -> csr_matrix:
        if self.scaler_ is None:
            raise RuntimeError("Numeric scaler is not fitted.")
        numeric_values = prepared_df[self.numeric_cols_].to_numpy(dtype=float)
        numeric_scaled = self.scaler_.transform(numeric_values)
        return csr_matrix(numeric_scaled)

    def _fit_transform_categorical(self, prepared_df: pd.DataFrame) -> csr_matrix:
        if not self.categorical_cols_:
            self.encoder_ = None
            return csr_matrix((len(prepared_df), 0))

        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        return self.encoder_.fit_transform(prepared_df[self.categorical_cols_])

    def _transform_categorical(self, prepared_df: pd.DataFrame) -> csr_matrix:
        if not self.categorical_cols_:
            return csr_matrix((len(prepared_df), 0))
        if self.encoder_ is None:
            raise RuntimeError("Categorical encoder is not fitted.")
        return self.encoder_.transform(prepared_df[self.categorical_cols_])

    def _fit_transform_text(self, prepared_df: pd.DataFrame, text_series: pd.Series) -> csr_matrix:
        if self.text_mode_ == "embeddings":
            return csr_matrix(prepared_df[self.embedding_cols_].to_numpy(dtype=float))

        self.tfidf_ = TfidfVectorizer(
            max_features=self.config.tfidf_max_features,
            ngram_range=self.config.tfidf_ngram_range,
            min_df=self.config.tfidf_min_df,
            lowercase=True,
            strip_accents="unicode",
        )
        return self.tfidf_.fit_transform(text_series)

    def _transform_text(self, prepared_df: pd.DataFrame, text_series: pd.Series) -> csr_matrix:
        if self.text_mode_ == "embeddings":
            return csr_matrix(prepared_df[self.embedding_cols_].to_numpy(dtype=float))

        if self.tfidf_ is None:
            raise RuntimeError("TF-IDF vectorizer is not fitted.")
        return self.tfidf_.transform(text_series)

    def _build_feature_names(self) -> None:
        """Compose ordered feature names for all matrix blocks."""
        feature_names: List[str] = []

        feature_names.extend(self.numeric_cols_)

        if self.encoder_ is not None and self.categorical_cols_:
            feature_names.extend(self.encoder_.get_feature_names_out(self.categorical_cols_).tolist())

        if self.text_mode_ == "embeddings":
            feature_names.extend(self.embedding_cols_)
        else:
            if self.tfidf_ is None:
                raise RuntimeError("TF-IDF vectorizer missing during feature name generation.")
            feature_names.extend([f"tfidf__{t}" for t in self.tfidf_.get_feature_names_out()])

        self.feature_names_ = feature_names

    @staticmethod
    def _infer_text_column(df: pd.DataFrame) -> str:
        """Infer a natural language text column while avoiding ID-like columns."""
        object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if not object_cols:
            return ""

        preferred_tokens = ["conversation_text", "transcript", "utterance", "message", "chat", "text"]

        scored: List[Tuple[str, float]] = []
        for col in object_cols:
            col_l = col.lower()
            id_penalty = -5 if any(tok in col_l for tok in ["_id", "id", "uuid", "key"]) else 0
            name_bonus = sum(tok in col_l for tok in preferred_tokens)

            sample = df[col].dropna().astype(str).head(1000)
            if sample.empty:
                content_bonus = -1.0
            else:
                has_space_ratio = sample.str.contains(r"\s", regex=True).mean()
                avg_words = sample.str.split().str.len().mean()
                content_bonus = (2.0 if has_space_ratio >= 0.4 else 0.0) + (2.0 if avg_words >= 4 else 0.0)

            total_score = float(name_bonus + id_penalty + content_bonus)
            scored.append((col, total_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_col, best_score = scored[0]
        return best_col if best_score >= 2 else ""


def get_feature_rationale_markdown(text_mode: str) -> str:
    """Human-readable explanation of why each feature block is created."""
    text_choice = "precomputed embeddings" if text_mode == "embeddings" else "TF-IDF"

    return (
        "### Why These Features Are Created\n"
        "- **engagement_per_length**: Captures engagement density, not just raw engagement.\n"
        "- **effectiveness_per_length**: Measures sales effectiveness normalized by conversation size.\n"
        "- **engagement_x_effectiveness**: Models interaction where strong engagement and effectiveness together can amplify conversion probability.\n"
        "- **Categorical one-hot features**: Preserve segment-level information (for example, channel or region) without imposing fake ordinal relationships.\n"
        f"- **Text features ({text_choice})**: Convert conversation language into machine-usable signals for intent, objections, urgency, and buying cues.\n"
        "- **Standard scaling on numeric block**: Brings numeric features to comparable scale for stable optimization and fair regularization impact."
    )
