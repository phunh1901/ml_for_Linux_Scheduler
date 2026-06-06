"""
train.py - Model Training Module for ML-Based Linux Scheduler

This module provides functions to instantiate, train, save, and load
six machine learning models used for process scheduling prediction:
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
  - K-Means (unsupervised clustering)

It also includes a data preparation utility that splits a labeled
DataFrame into stratified train/test sets using the project's
standard feature columns and target column.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from src import TARGET_COLUMN, get_all_feature_columns


# ---------------------------------------------------------------------------
# Model catalogue helpers
# ---------------------------------------------------------------------------

def get_all_models() -> dict:
    """
    Return a dictionary of all 6 model instances (untrained).

    Returns:
        dict: Mapping of model_name (str) -> scikit-learn estimator instance.
    """
    return {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial',
            class_weight='balanced'
        ),
        'K-Means': KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        ),
    }


def get_supervised_models() -> dict:
    """
    Return only the 5 supervised models (excludes K-Means).

    Returns:
        dict: Mapping of model_name (str) -> scikit-learn estimator instance.
    """
    all_models = get_all_models()
    all_models.pop('K-Means', None)
    return all_models


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_model(model, X_train, y_train):
    """
    Fit a single model on the provided training data.

    Args:
        model: A scikit-learn estimator (classifier or clusterer).
        X_train (array-like): Feature matrix for training.
        y_train (array-like): Target labels for training.

    Returns:
        The fitted model instance.
    """
    print(f"  -> Training {type(model).__name__} on {X_train.shape[0]} samples "
          f"with {X_train.shape[1]} features ...")
    model.fit(X_train, y_train)
    print(f"  -> Training complete for {type(model).__name__}.")
    return model


def train_all_models(X_train, y_train, include_kmeans: bool = False) -> dict:
    """
    Train all models and return them as a dictionary.

    Supervised models are trained with (X_train, y_train).
    If *include_kmeans* is True, K-Means is trained on X_train only
    (unsupervised – no target labels).

    Args:
        X_train (array-like): Feature matrix for training.
        y_train (array-like): Target labels for training.
        include_kmeans (bool): Whether to include and train K-Means.

    Returns:
        dict: Mapping of model_name (str) -> trained estimator.
    """
    if include_kmeans:
        models = get_all_models()
    else:
        models = get_supervised_models()

    trained_models = {}

    for name, model in models.items():
        print(f"\n[Training] {name} ...")

        if name == 'K-Means':
            # K-Means is unsupervised – fit on features only
            print(f"  -> Fitting K-Means on {X_train.shape[0]} samples "
                  f"(unsupervised, no target labels) ...")
            model.fit(X_train)
            print(f"  -> K-Means training complete.")
        else:
            model = train_model(model, X_train, y_train)

        trained_models[name] = model

    print(f"\n[Training] All {len(trained_models)} model(s) trained successfully.")
    return trained_models


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_model(model, path: str) -> None:
    """
    Serialize and save a trained model to disk using joblib.

    Creates the parent directory if it does not exist.

    Args:
        model: A fitted scikit-learn estimator.
        path (str): File path to save the model (e.g. 'models/rf.joblib').
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[Save] Model saved to: {path}")


def load_model(path: str):
    """
    Load a previously saved model from disk using joblib.

    Args:
        path (str): File path of the serialized model.

    Returns:
        The deserialized scikit-learn estimator.
    """
    model = joblib.load(path)
    print(f"[Load] Model loaded from: {path}")
    return model


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(df: pd.DataFrame, test_size: float = 0.30, random_state: int = 42):
    """
    Split a DataFrame into stratified train and test sets.

    Feature columns are obtained from ``src.get_all_feature_columns()``
    and the target column from ``src.TARGET_COLUMN``.

    Args:
        df (pd.DataFrame): The full dataset (features + target).
        test_size (float): Fraction of data to reserve for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    feature_cols = get_all_feature_columns()
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[Data] Total samples : {len(df)}")
    print(f"[Data] Training set  : X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"[Data] Test set      : X_test  {X_test.shape},  y_test  {y_test.shape}")

    return X_train, X_test, y_train, y_test
