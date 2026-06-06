"""
visualize.py - Visualization Utilities

Generates publication-quality plots for exploratory data analysis,
model evaluation, and feature importance inspection.
All plots can optionally be saved to disk as PNG files.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from src import LABEL_NAMES, TARGET_COLUMN


# ---------------------------------------------------------------------------
# Global style configuration
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_or_show(fig, save_path: str = None) -> None:
    """Save the figure to *save_path* if provided, otherwise display it."""
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[PLOT] Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 1. CPU Distribution
# ---------------------------------------------------------------------------

def plot_cpu_distribution(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot a histogram with KDE overlay for CPU utilization.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a ``cpu_utilization`` column.
    save_path : str, optional
        If provided, save the figure to this path instead of showing it.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        df["cpu_utilization"], kde=True, color="blue", bins=40, ax=ax
    )
    ax.set_title("CPU Utilization Distribution")
    ax.set_xlabel("CPU Utilization (%)")
    ax.set_ylabel("Frequency")
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 2. Memory Distribution
# ---------------------------------------------------------------------------

def plot_memory_distribution(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot a histogram with KDE overlay for memory utilization.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a ``memory_utilization`` column.
    save_path : str, optional
        If provided, save the figure to this path instead of showing it.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        df["memory_utilization"], kde=True, color="green", bins=40, ax=ax
    )
    ax.set_title("Memory Utilization Distribution")
    ax.set_xlabel("Memory Utilization (%)")
    ax.set_ylabel("Frequency")
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 3. CPU vs Memory Scatter
# ---------------------------------------------------------------------------

def plot_cpu_vs_memory_scatter(
    df: pd.DataFrame, save_path: str = None
) -> None:
    """
    Scatter plot of CPU utilization versus memory utilization, coloured by
    scheduling label.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``cpu_utilization``, ``memory_utilization``, and
        ``target_schedule`` columns.
    save_path : str, optional
        If provided, save the figure to this path instead of showing it.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="cpu_utilization",
        y="memory_utilization",
        hue=TARGET_COLUMN,
        palette="Set1",
        alpha=0.6,
        ax=ax,
    )
    ax.set_title("CPU vs Memory Utilization by Schedule Class")
    ax.set_xlabel("CPU Utilization (%)")
    ax.set_ylabel("Memory Utilization (%)")
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Class Distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(
    df: pd.DataFrame, save_path: str = None
) -> None:
    """
    Bar chart (count plot) showing the distribution of scheduling labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``target_schedule`` column.
    save_path : str, optional
        If provided, save the figure to this path instead of showing it.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(x=TARGET_COLUMN, data=df, palette="viridis", ax=ax)
    ax.set_xticks(range(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_title("Scheduling Class Distribution")
    ax.set_xlabel("Schedule Class")
    ax.set_ylabel("Count")
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 5. Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame, save_path: str = None
) -> None:
    """
    Heatmap of the Pearson correlation matrix for all numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric feature columns.
    save_path : str, optional
        If provided, save the figure to this path instead of showing it.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 6. Feature Importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importances,
    feature_names,
    title: str = "Feature Importance",
    save_path: str = None,
) -> None:
    """
    Horizontal bar chart of feature importances, sorted from highest to lowest.

    Parameters
    ----------
    importances : array-like
        Importance values (e.g., from a tree-based model).
    feature_names : list of str
        Corresponding feature names.
    title : str, optional
        Plot title.
    save_path : str, optional
        If provided, save the figure to this path instead of showing it.
    """
    # Sort by importance
    indices = np.argsort(importances)
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = np.array(importances)[indices]

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.5)))
    ax.barh(sorted_names, sorted_importances, color="teal")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 7. Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    title: str = "Confusion Matrix",
    save_path: str = None,
) -> None:
    """
    Heatmap of the confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : list of str, optional
        Display labels for each class. Defaults to LABEL_NAMES.
    title : str, optional
        Plot title.
    save_path : str, optional
        If provided, save the figure to this path instead of showing it.
    """
    if labels is None:
        labels = LABEL_NAMES

    cm = sk_confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 8. Model Comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "accuracy",
    save_path: str = None,
) -> None:
    """
    Bar chart comparing multiple models on a chosen metric.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns ``model_name``, ``accuracy``, ``precision``,
        ``recall``, and ``f1_score``.
    metric : str, optional
        Which metric column to plot (default ``'accuracy'``).
    save_path : str, optional
        If provided, save the figure to this path instead of showing it.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        x="model_name",
        y=metric,
        data=results_df,
        palette="mako",
        ax=ax,
    )
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# 9. Generate All EDA Plots
# ---------------------------------------------------------------------------

def generate_all_eda_plots(
    df: pd.DataFrame,
    output_dir: str = "output/figures",
) -> None:
    """
    Generate and save all exploratory data analysis plots to *output_dir*.

    Calls:
    1. CPU distribution histogram
    2. Memory distribution histogram
    3. CPU vs Memory scatter
    4. Class distribution bar chart
    5. Correlation heatmap

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame containing feature and target columns.
    output_dir : str, optional
        Directory where PNG files will be saved (created if absent).
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[EDA] Generating all EDA plots -> {output_dir}")

    plot_cpu_distribution(
        df, save_path=os.path.join(output_dir, "cpu_distribution.png")
    )
    plot_memory_distribution(
        df, save_path=os.path.join(output_dir, "memory_distribution.png")
    )
    plot_cpu_vs_memory_scatter(
        df, save_path=os.path.join(output_dir, "cpu_vs_memory_scatter.png")
    )
    plot_class_distribution(
        df, save_path=os.path.join(output_dir, "class_distribution.png")
    )
    plot_correlation_heatmap(
        df, save_path=os.path.join(output_dir, "correlation_heatmap.png")
    )

    print(f"[EDA] All 5 EDA plots saved successfully.")
