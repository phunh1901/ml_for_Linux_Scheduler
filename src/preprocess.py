"""
preprocess.py - Data Cleaning & Feature Engineering Pipeline

Provides utilities to load raw process telemetry data, clean it,
engineer new features, normalise numeric columns, and run the
full preprocessing pipeline end-to-end.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import (
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    ENGINEERED_FEATURE,
    get_all_feature_columns,
)


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load a raw CSV data file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded data.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    print(f"[LOAD] Loaded {len(df)} rows from '{path}'.")
    return df


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame by handling missing values and outliers.

    Steps performed:
    1. Drop rows with any NaN values.
    2. Clip ``cpu_utilization`` to the valid range [0, 100].
    3. Remove rows where ``memory_utilization`` exceeds 100.
    4. Drop duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (or partially processed) DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    initial_len = len(df)

    # Drop missing values
    df = df.dropna()
    after_na = len(df)

    # Clip CPU utilization to [0, 100]
    df["cpu_utilization"] = df["cpu_utilization"].clip(lower=0, upper=100)

    # Remove rows with unrealistic memory utilization
    df = df[df["memory_utilization"] <= 100].copy()
    after_mem = len(df)

    # Remove duplicates
    df = df.drop_duplicates()
    final_len = len(df)

    print(f"[CLEAN] Rows: {initial_len} -> {final_len}  "
          f"(dropped {initial_len - final_len}: "
          f"NaN={initial_len - after_na}, "
          f"mem>100={after_na - after_mem}, "
          f"duplicates={after_mem - final_len})")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw columns.

    Currently adds:
    - ``combined_load_score``:
        priority_level * 20.0 + cpu_utilization * 1.2 + memory_utilization * 0.5

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the raw feature columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with the new engineered feature appended.
    """
    df = df.copy()
    df[ENGINEERED_FEATURE] = (
        df["priority_level"] * 20.0
        + df["cpu_utilization"] * 1.2
        + df["memory_utilization"] * 0.5
    )

    print(f"[ENGINEER] Added '{ENGINEERED_FEATURE}' column. "
          f"Range: [{df[ENGINEERED_FEATURE].min():.2f}, "
          f"{df[ENGINEERED_FEATURE].max():.2f}]")

    return df


# ---------------------------------------------------------------------------
# 4. Normalisation
# ---------------------------------------------------------------------------

def normalize_features(
    df: pd.DataFrame,
    feature_cols: list,
) -> tuple:
    """
    Normalise numeric feature columns using StandardScaler (zero mean, unit variance).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the feature columns to normalise.
    feature_cols : list of str
        Column names to normalise.

    Returns
    -------
    tuple of (pd.DataFrame, StandardScaler)
        - DataFrame with the specified columns replaced by their scaled values.
        - The fitted StandardScaler instance (for later inverse-transform or
          applying the same transform to new data).
    """
    df = df.copy()
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    print(f"[NORMALIZE] Scaled {len(feature_cols)} feature column(s) "
          f"using StandardScaler.")

    return df, scaler


# ---------------------------------------------------------------------------
# 5. Full Pipeline
# ---------------------------------------------------------------------------

def preprocess_pipeline(
    raw_path: str,
    processed_path: str,
) -> pd.DataFrame:
    """
    Execute the complete preprocessing pipeline.

    Pipeline steps:
    1. Load raw CSV data.
    2. Clean data (NaN removal, outlier handling, deduplication).
    3. Engineer new features.
    4. Save the processed DataFrame to ``processed_path``.

    Parameters
    ----------
    raw_path : str
        Path to the raw CSV file.
    processed_path : str
        Destination path for the cleaned & engineered CSV.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame (before normalisation, so original scale is
        preserved in the saved file).
    """
    print("=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1 — Load
    df = load_raw_data(raw_path)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape  : {df.shape}")

    # Step 2 — Clean
    df = clean_data(df)

    # Step 3 — Feature engineering
    df = engineer_features(df)

    # Save processed data
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"\n[SAVE] Processed data saved to '{processed_path}' "
          f"({len(df)} rows, {len(df.columns)} columns).")
    print("=" * 60)

    return df
