"""Evaluation utilities for model comparison and final testing."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate_predictions(y_true, y_pred, task: str, model_name: str, split: str) -> dict:
    """Return the report metrics required for this assignment."""
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "task": task,
        "model": model_name,
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def save_confusion_matrix(y_true, y_pred, output_path: Path, title: str) -> None:
    """Save a confusion matrix chart for report and slide evidence."""
    labels = sorted(pd.concat([pd.Series(y_true), pd.Series(y_pred)]).astype(str).unique())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
