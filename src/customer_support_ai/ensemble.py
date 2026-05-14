"""Confidence-aware ensemble training for customer support triage.

This module is an optional extension to the main TF-IDF Linear SVM pipeline. It
trains soft-voting ensembles for queue and priority prediction, then reports
coverage-aware metrics that support responsible human-in-the-loop deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier

from .confidence import MEDIUM_CONFIDENCE_THRESHOLD, confidence_level
from .config import DATA_PROCESSED_DIR, DEFAULT_DATASET_PATH, MODELS_DIR, RESULTS_DIR
from .data_preprocessing import (
    build_modelling_frame,
    load_dataset,
    load_project_dataset,
    make_train_validation_test_split,
)


def build_confidence_ensemble() -> Pipeline:
    """Build a soft-voting text ensemble with probability-like outputs."""
    classifier = VotingClassifier(
        estimators=[
            (
                "logistic_regression",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
            (
                "calibrated_linear_svm",
                CalibratedClassifierCV(
                    LinearSVC(class_weight="balanced"),
                    cv=3,
                ),
            ),
            ("naive_bayes", MultinomialNB()),
        ],
        voting="soft",
    )
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)),
            ("model", classifier),
        ]
    )


def _evaluate_confidence(y_true, predictions, confidences, task: str, split: str) -> dict:
    """Return standard and confidence-aware metrics."""
    confidence_series = pd.Series(confidences, dtype=float)
    high_mask = confidence_series >= MEDIUM_CONFIDENCE_THRESHOLD
    low_mask = ~high_mask

    result = {
        "task": task,
        "model": "confidence_aware_voting_ensemble",
        "split": split,
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(y_true, predictions, average="macro", zero_division=0),
        "mean_confidence": float(confidence_series.mean()),
        "median_confidence": float(confidence_series.median()),
        "coverage_at_medium_confidence": float(high_mask.mean()),
        "human_review_rate": float(low_mask.mean()),
    }
    if high_mask.any():
        result["high_confidence_accuracy"] = accuracy_score(
            pd.Series(y_true).reset_index(drop=True)[high_mask],
            pd.Series(predictions).reset_index(drop=True)[high_mask],
        )
    else:
        result["high_confidence_accuracy"] = 0.0
    return result


def train_ensemble_task(frame: pd.DataFrame, target_column: str, task: str) -> dict:
    """Train one confidence-aware ensemble task and save the fitted model."""
    train, validation, test = make_train_validation_test_split(frame, target_column)
    train_validation = pd.concat([train, validation], axis=0)

    model = build_confidence_ensemble()
    model.fit(train_validation["model_text"], train_validation[target_column])

    predictions = model.predict(test["model_text"])
    probabilities = model.predict_proba(test["model_text"])
    confidences = probabilities.max(axis=1)

    output_path = MODELS_DIR / f"{task}_confidence_ensemble.pkl"
    joblib.dump(model, output_path)

    examples = pd.DataFrame(
        {
            "task": task,
            "ticket_text": test["ticket_text"].head(25).values,
            "actual_label": test[target_column].head(25).values,
            "predicted_label": predictions[:25],
            "confidence": confidences[:25],
            "confidence_level": [confidence_level(score) for score in confidences[:25]],
            "human_review_required": confidences[:25] < MEDIUM_CONFIDENCE_THRESHOLD,
        }
    )
    examples.to_csv(RESULTS_DIR / f"{task}_confidence_examples.csv", index=False)
    return _evaluate_confidence(test[target_column], predictions, confidences, task, "test")


def main() -> None:
    """Train queue and priority confidence ensembles."""
    parser = argparse.ArgumentParser(description="Train confidence-aware ensemble triage models.")
    parser.add_argument(
        "--data",
        default=None,
        help="Optional path to one CSV file. Leave blank to load the compatible Kaggle CSVs.",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(exist_ok=True)

    raw_data = load_dataset(args.data) if args.data else load_project_dataset()
    frame, category_column, priority_column = build_modelling_frame(raw_data)
    frame.to_csv(DATA_PROCESSED_DIR / "model_ready_tickets.csv", index=False)

    results = [
        train_ensemble_task(frame, category_column, "category"),
        train_ensemble_task(frame, priority_column, "priority"),
    ]
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "confidence_ensemble_results.csv", index=False)

    metadata = {
        "source_dataset": str(args.data or DEFAULT_DATASET_PATH),
        "confidence_threshold": MEDIUM_CONFIDENCE_THRESHOLD,
        "saved_models": [
            str(MODELS_DIR / "category_confidence_ensemble.pkl"),
            str(MODELS_DIR / "priority_confidence_ensemble.pkl"),
        ],
    }
    (RESULTS_DIR / "confidence_ensemble_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
