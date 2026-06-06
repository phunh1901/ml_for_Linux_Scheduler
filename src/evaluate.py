"""
evaluate.py - Model Evaluation & Comparison Module for ML-Based Linux Scheduler

This module provides utilities to:
  - Evaluate individual supervised classifiers (accuracy, precision, recall, F1).
  - Evaluate K-Means clustering (ARI, silhouette, homogeneity).
  - Compare all trained models in a formatted table.
  - Extract feature importances from tree-based or linear models.
  - Generate CSV comparison reports and text summaries.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    adjusted_rand_score,
    silhouette_score,
    homogeneity_score,
)

from src import LABEL_NAMES


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name: str = 'Model') -> dict:
    """
    Evaluate a single supervised classifier on the test set.

    Computes accuracy, weighted precision, weighted recall, and weighted F1.

    Args:
        model: A fitted scikit-learn classifier.
        X_test (array-like): Test feature matrix.
        y_test (array-like): True target labels.
        model_name (str): Human-readable name for display.

    Returns:
        dict: Evaluation metrics keyed by metric name.
    """
    y_pred = model.predict(X_test)

    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }

    print(f"\n{'=' * 50}")
    print(f"  Evaluation Results: {model_name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}  (weighted)")
    print(f"  Recall    : {metrics['recall']:.4f}  (weighted)")
    print(f"  F1 Score  : {metrics['f1_score']:.4f}  (weighted)")
    print(f"{'=' * 50}")

    return metrics


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_all_models(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate every trained model and return a comparison DataFrame.

    K-Means is skipped here (use ``evaluate_kmeans`` separately).

    Args:
        trained_models (dict): Mapping of model_name -> fitted estimator.
        X_test (array-like): Test feature matrix.
        y_test (array-like): True target labels.

    Returns:
        pd.DataFrame: Results with columns
            [model_name, accuracy, precision, recall, f1_score],
            sorted by accuracy descending.
    """
    results = []

    for name, model in trained_models.items():
        if name == 'K-Means':
            print(f"\n[Skip] K-Means is unsupervised – evaluate separately "
                  f"with evaluate_kmeans().")
            continue

        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False).reset_index(drop=True)

    # Print formatted comparison table
    print(f"\n{'=' * 72}")
    print(f"  MODEL COMPARISON (sorted by accuracy)")
    print(f"{'=' * 72}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-' * 65}")
    for _, row in results_df.iterrows():
        print(f"  {row['model_name']:<25} "
              f"{row['accuracy']:>10.4f} "
              f"{row['precision']:>10.4f} "
              f"{row['recall']:>10.4f} "
              f"{row['f1_score']:>10.4f}")
    print(f"{'=' * 72}\n")

    return results_df


# ---------------------------------------------------------------------------
# K-Means evaluation
# ---------------------------------------------------------------------------

def evaluate_kmeans(kmeans_model, X_test, y_test) -> dict:
    """
    Evaluate K-Means clustering quality using external and internal metrics.

    Args:
        kmeans_model: A fitted KMeans estimator.
        X_test (array-like): Test feature matrix.
        y_test (array-like): True target labels (used for external metrics).

    Returns:
        dict: Metrics keyed by name (adjusted_rand, silhouette, homogeneity).
    """
    cluster_labels = kmeans_model.predict(X_test)

    metrics = {
        'adjusted_rand_score': adjusted_rand_score(y_test, cluster_labels),
        'silhouette_score': silhouette_score(X_test, cluster_labels),
        'homogeneity_score': homogeneity_score(y_test, cluster_labels),
    }

    print(f"\n{'=' * 50}")
    print(f"  K-Means Clustering Evaluation")
    print(f"{'=' * 50}")
    print(f"  Adjusted Rand Index : {metrics['adjusted_rand_score']:.4f}")
    print(f"  Silhouette Score    : {metrics['silhouette_score']:.4f}")
    print(f"  Homogeneity Score   : {metrics['homogeneity_score']:.4f}")
    print(f"{'=' * 50}\n")

    return metrics


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(model, feature_names) -> pd.Series:
    """
    Extract feature importances from a trained model.

    Supports:
      - Tree-based models via ``feature_importances_`` attribute.
      - Logistic Regression via ``coef_`` (mean absolute coefficients
        across classes for multi-class).

    Args:
        model: A fitted scikit-learn estimator.
        feature_names (list[str]): Names corresponding to each feature column.

    Returns:
        pd.Series: Feature importances sorted descending, or None if
            the model type is not supported.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        series = pd.Series(importances, index=feature_names)
        return series.sort_values(ascending=False)

    if hasattr(model, 'coef_'):
        # For multi-class LogReg, coef_ has shape (n_classes, n_features).
        # Take the mean of absolute values across classes.
        coef = np.abs(model.coef_)
        if coef.ndim > 1:
            coef = coef.mean(axis=0)
        series = pd.Series(coef, index=feature_names)
        return series.sort_values(ascending=False)

    # Unsupported model type (e.g. KNN, K-Means)
    return None


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_comparison_report(results_df: pd.DataFrame,
                               output_dir: str = 'output/results') -> str:
    """
    Save a comparison DataFrame to CSV and generate a short text report.

    Args:
        results_df (pd.DataFrame): Evaluation results (from evaluate_all_models).
        output_dir (str): Directory where the report files are written.

    Returns:
        str: Absolute path to the saved CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'model_comparison.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"[Report] Comparison CSV saved to: {csv_path}")

    # Generate a plain-text summary
    best = results_df.iloc[0]
    report_lines = [
        "=" * 60,
        "  MODEL COMPARISON REPORT",
        "=" * 60,
        f"  Total models evaluated : {len(results_df)}",
        f"  Best model             : {best['model_name']}",
        f"  Best accuracy          : {best['accuracy']:.4f}",
        f"  Best F1 (weighted)     : {best['f1_score']:.4f}",
        "=" * 60,
    ]

    report_path = os.path.join(output_dir, 'model_comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"[Report] Text report saved to: {report_path}")

    return csv_path


# ---------------------------------------------------------------------------
# Classification report wrapper
# ---------------------------------------------------------------------------

def get_classification_report_text(model, X_test, y_test) -> str:
    """
    Return a formatted sklearn classification report as a string.

    Uses the project's standard label names:
        ['IMMEDIATELY', 'NEXT', 'LATELY']

    Args:
        model: A fitted scikit-learn classifier.
        X_test (array-like): Test feature matrix.
        y_test (array-like): True target labels.

    Returns:
        str: The classification report text.
    """
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=['IMMEDIATELY', 'NEXT', 'LATELY'],
    )
    return report
