"""Data loading and basic cleaning pipeline for SaaS sales conversations.

This module prepares the dataset for exploratory data analysis (EDA) only.
No feature engineering or modeling is performed here.
"""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


DATASET_NAME = "DeepMostInnovations/saas-sales-conversations"


def load_hf_dataset(dataset_name: str, split: Optional[str] = None) -> Union[Dataset, DatasetDict]:
    """Load a dataset from Hugging Face.

    Args:
        dataset_name: The Hugging Face dataset identifier.
        split: Optional split name (for example: "train"). If omitted,
            the full DatasetDict is loaded.

    Returns:
        A Dataset or DatasetDict depending on the provided split.
    """
    if split:
        return load_dataset(dataset_name, split=split)
    return load_dataset(dataset_name)


def dataset_to_dataframe(data: Union[Dataset, DatasetDict]) -> pd.DataFrame:
    """Convert Hugging Face Dataset/DatasetDict into a pandas DataFrame.

    If multiple splits are present, they are concatenated and a `split` column
    is added to preserve provenance.
    """
    if isinstance(data, Dataset):
        return data.to_pandas()

    if isinstance(data, DatasetDict):
        frames = []
        for split_name, split_ds in data.items():
            split_df = split_ds.to_pandas()
            split_df["split"] = split_name
            frames.append(split_df)

        return pd.concat(frames, ignore_index=True)

    raise TypeError("Input must be a Hugging Face Dataset or DatasetDict.")


def display_basic_info(df: pd.DataFrame) -> None:
    """Print dataset shape, column names, and dtypes."""
    print("\n=== BASIC DATASET INFO ===")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)


def check_missing_and_duplicates(df: pd.DataFrame) -> None:
    """Print missing-value counts and duplicate-row counts."""
    print("\n=== DATA QUALITY CHECKS ===")

    missing_counts = df.isna().sum().sort_values(ascending=False)
    print("\nMissing values per column:")
    print(missing_counts)

    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicate_count}")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic cleaning for EDA readiness.

    Cleaning strategy:
    1. Drop fully empty rows.
    2. Remove duplicate rows.
    3. Fill missing values:
       - Numeric columns -> median
       - Non-numeric columns -> "Unknown"
    """
    cleaned_df = df.copy()

    # Remove rows that are entirely empty.
    cleaned_df = cleaned_df.dropna(how="all")

    # Keep first occurrence of each duplicated row.
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

    # Fill missing values by data type to make data EDA-friendly.
    numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns
    categorical_cols = cleaned_df.select_dtypes(exclude=["number"]).columns

    if len(numeric_cols) > 0:
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].apply(
            lambda col: col.fillna(col.median())
        )

    if len(categorical_cols) > 0:
        cleaned_df[categorical_cols] = cleaned_df[categorical_cols].fillna("Unknown")

    return cleaned_df


def run_data_preparation(dataset_name: str = DATASET_NAME, split: Optional[str] = None) -> pd.DataFrame:
    """Run full data preparation flow and return cleaned DataFrame."""
    hf_data = load_hf_dataset(dataset_name=dataset_name, split=split)
    raw_df = dataset_to_dataframe(hf_data)

    display_basic_info(raw_df)
    check_missing_and_duplicates(raw_df)

    cleaned_df = clean_dataframe(raw_df)

    print("\n=== AFTER CLEANING ===")
    print(f"Shape: {cleaned_df.shape}")
    print(f"Duplicate rows: {cleaned_df.duplicated().sum()}")
    print("Missing values (sum across all columns):", int(cleaned_df.isna().sum().sum()))

    return cleaned_df


if __name__ == "__main__":
    # By default, load all splits. Pass split="train" if you want one split.
    dataframe = run_data_preparation()

    print("\nPreview of cleaned data:")
    print(dataframe.head())
