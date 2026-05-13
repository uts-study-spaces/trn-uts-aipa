"""Generate report and presentation assets from saved project outputs."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .config import DATA_PROCESSED_DIR, MODELS_DIR, PROJECT_ROOT, RESULTS_DIR
from .data_preprocessing import (
    build_modelling_frame,
    load_project_dataset,
    make_train_validation_test_split,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT_ASSETS_DIR = PROJECT_ROOT / "report_assets"


def _save_bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    hue: str | None = None,
    rotate_labels: bool = True,
) -> None:
    """Save a clean bar chart for slides and report figures."""
    plt.figure(figsize=(11, 6))
    sns.barplot(data=data, x=x, y=y, hue=hue)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rotate_labels:
        plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _load_model_ready_frame() -> tuple[pd.DataFrame, str, str]:
    """Use processed data when available, otherwise rebuild it."""
    processed_path = DATA_PROCESSED_DIR / "model_ready_tickets.csv"
    if processed_path.exists():
        frame = pd.read_csv(processed_path)
        return frame, "queue", "priority"

    raw = load_project_dataset()
    return build_modelling_frame(raw)


def create_dataset_charts(frame: pd.DataFrame) -> None:
    """Create dataset charts for EDA and presentation slides."""
    if "source_file" in frame.columns:
        source_counts = (
            frame["source_file"].value_counts().rename_axis("source_file").reset_index(name="tickets")
        )
        _save_bar_chart(
            source_counts,
            "source_file",
            "tickets",
            REPORT_ASSETS_DIR / "dataset_file_rows.png",
            "Tickets Used From Each Compatible CSV",
            "CSV file",
            "Tickets",
        )

    for column, filename, title in [
        ("language", "language_distribution.png", "Ticket Language Distribution"),
        ("queue", "queue_distribution.png", "Queue Label Distribution"),
        ("priority", "priority_distribution.png", "Priority Label Distribution"),
    ]:
        if column in frame.columns:
            counts = frame[column].value_counts().rename_axis(column).reset_index(name="tickets")
            _save_bar_chart(
                counts,
                column,
                "tickets",
                REPORT_ASSETS_DIR / filename,
                title,
                column.title(),
                "Tickets",
                rotate_labels=column == "queue",
            )


def create_model_comparison_chart() -> None:
    """Create a validation/test macro F1 comparison chart."""
    metrics_path = RESULTS_DIR / "metrics_summary.csv"
    metrics = pd.read_csv(metrics_path)
    metrics["label"] = metrics["task"] + " - " + metrics["model"] + " - " + metrics["split"]

    _save_bar_chart(
        metrics.sort_values(["task", "split", "macro_f1"], ascending=[True, True, False]),
        "label",
        "macro_f1",
        REPORT_ASSETS_DIR / "model_comparison_macro_f1.png",
        "Model Comparison by Macro F1",
        "Model and split",
        "Macro F1",
    )


def create_per_class_f1_outputs() -> None:
    """Extract per-class F1 scores from saved classification reports."""
    reports_path = RESULTS_DIR / "classification_reports.json"
    reports = json.loads(reports_path.read_text(encoding="utf-8"))
    rows = []

    for report_key, report in reports.items():
        if not report_key.endswith("_test"):
            continue
        task = report_key.split("_")[0]
        for label, values in report.items():
            if label in {"accuracy", "macro avg", "weighted avg"}:
                continue
            rows.append(
                {
                    "task": task,
                    "label": label,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1_score": values["f1-score"],
                    "support": values["support"],
                }
            )

    per_class = pd.DataFrame(rows)
    per_class.to_csv(RESULTS_DIR / "per_class_f1.csv", index=False)

    for task, task_data in per_class.groupby("task"):
        chart_data = task_data.sort_values("f1_score", ascending=False)
        _save_bar_chart(
            chart_data,
            "label",
            "f1_score",
            REPORT_ASSETS_DIR / f"per_class_f1_{task}.png",
            f"Per-Class F1 for {task.title()} Model",
            "Class",
            "F1-score",
        )


def _group_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def create_per_language_outputs(frame: pd.DataFrame, category_col: str, priority_col: str) -> None:
    """Evaluate final models separately for each ticket language."""
    if "language" not in frame.columns:
        return

    category_model = joblib.load(MODELS_DIR / "category_model.pkl")
    priority_model = joblib.load(MODELS_DIR / "priority_model.pkl")
    _, _, category_test = make_train_validation_test_split(frame, category_col)
    _, _, priority_test = make_train_validation_test_split(frame, priority_col)

    rows = []
    for task, test_frame, target_col, model in [
        ("category", category_test, category_col, category_model),
        ("priority", priority_test, priority_col, priority_model),
    ]:
        predictions = pd.Series(model.predict(test_frame["model_text"]), index=test_frame.index)
        for language, language_frame in test_frame.groupby("language"):
            language_predictions = predictions.loc[language_frame.index]
            metrics = _group_metrics(language_frame[target_col], language_predictions)
            rows.append(
                {
                    "task": task,
                    "language": language,
                    "test_rows": len(language_frame),
                    **metrics,
                }
            )

    per_language = pd.DataFrame(rows).sort_values(["task", "language"])
    per_language.to_csv(RESULTS_DIR / "per_language_metrics.csv", index=False)

    _save_bar_chart(
        per_language,
        "language",
        "macro_f1",
        REPORT_ASSETS_DIR / "per_language_macro_f1.png",
        "Macro F1 by Language",
        "Language",
        "Macro F1",
        hue="task",
        rotate_labels=False,
    )


def create_workflow_diagram() -> None:
    """Create a simple workflow figure for the presentation."""
    steps = [
        "Audit 5 Kaggle CSVs",
        "Merge 3 compatible files",
        "Clean text and metadata",
        "Compare TF-IDF & Embeddings",
        "Select Best representation",
        "Evaluate and explain",
        "Streamlit decision support",
    ]
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.axis("off")
    x_positions = [i / (len(steps) - 1) for i in range(len(steps))]

    for index, (x_pos, step) in enumerate(zip(x_positions, steps)):
        ax.text(
            x_pos,
            0.55,
            step,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eef5ff", "edgecolor": "#3b6ea8"},
        )
        if index < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[index + 1] - 0.06, 0.55),
                xytext=(x_pos + 0.06, 0.55),
                arrowprops={"arrowstyle": "->", "color": "#3b6ea8", "lw": 1.5},
            )

    plt.tight_layout()
    plt.savefig(REPORT_ASSETS_DIR / "workflow_diagram.png", dpi=180)
    plt.close()


def main() -> None:
    REPORT_ASSETS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    frame, category_col, priority_col = _load_model_ready_frame()
    create_dataset_charts(frame)
    create_model_comparison_chart()
    create_per_class_f1_outputs()
    create_per_language_outputs(frame, category_col, priority_col)
    create_workflow_diagram()

    print(f"Report assets saved to: {REPORT_ASSETS_DIR}")


if __name__ == "__main__":
    main()
