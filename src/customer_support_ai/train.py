"""Train and evaluate customer support AI models."""

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import DATA_PROCESSED_DIR, DEFAULT_DATASET_PATH, MODELS_DIR, RESULTS_DIR
from .data_preprocessing import (
    build_modelling_frame,
    load_dataset,
    load_project_dataset,
    make_train_validation_test_split,
)
from .evaluate import evaluate_predictions, save_confusion_matrix
from .vectorizers import Top2VecTransformer, Word2VecTransformer


def get_vectorizer(vectorizer_type: str):
    """Return the requested vectorizer instance."""
    if vectorizer_type == "tfidf":
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
    elif vectorizer_type == "word2vec":
        return Word2VecTransformer()
    elif vectorizer_type == "top2vec":
        return Top2VecTransformer()
    else:
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")


def build_models(vectorizer_type: str = "tfidf") -> dict[str, Pipeline]:
    """Keep models simple and explainable for assignment reporting."""
    
    models = {
        "logistic_regression": Pipeline(
            [
                ("vectorizer", get_vectorizer(vectorizer_type)),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("vectorizer", get_vectorizer(vectorizer_type)),
                ("model", LinearSVC(class_weight="balanced")),
            ]
        ),
    }
    
    # Naive Bayes doesn't work with negative values from embeddings
    if vectorizer_type == "tfidf":
        models["naive_bayes"] = Pipeline(
            [
                ("vectorizer", get_vectorizer(vectorizer_type)),
                ("model", MultinomialNB()),
            ]
        )
        
    return models


def train_task(frame: pd.DataFrame, target_column: str, task: str, vectorizer_type: str = "tfidf") -> list[dict]:
    """Train candidates on train, select on validation, report final test performance."""
    train, validation, test = make_train_validation_test_split(frame, target_column)
    results = []
    validation_results = []
    best_model = None
    best_name = None
    best_macro_f1 = -1.0

    for name, model in build_models(vectorizer_type).items():
        model.fit(train["model_text"], train[target_column])
        validation_predictions = model.predict(validation["model_text"])
        validation_result = evaluate_predictions(
            validation[target_column],
            validation_predictions,
            task=task,
            model_name=name,
            split="validation",
        )
        validation_results.append(validation_result)

        if validation_result["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_result["macro_f1"]
            best_model = model
            best_name = name

    final_train = pd.concat([train, validation], axis=0)
    best_model.fit(final_train["model_text"], final_train[target_column])
    test_predictions = best_model.predict(test["model_text"])
    test_result = evaluate_predictions(
        test[target_column],
        test_predictions,
        task=task,
        model_name=best_name,
        split="test",
    )

    model_path = MODELS_DIR / f"{task}_model.pkl"
    joblib.dump(best_model, model_path)
    save_confusion_matrix(
        test[target_column],
        test_predictions,
        RESULTS_DIR / f"confusion_matrix_{task}.png",
        f"{task.title()} Confusion Matrix",
    )
    results.extend(validation_results)
    results.append(test_result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train customer support AI models.")
    parser.add_argument(
        "--data",
        default=None,
        help="Optional path to one CSV file. Leave blank to load the three compatible Kaggle CSVs.",
    )
    parser.add_argument(
        "--vectorizer",
        choices=["tfidf", "word2vec", "top2vec"],
        default="tfidf",
        help="Text representation method to use.",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(exist_ok=True)

    raw_data = load_dataset(args.data) if args.data else load_project_dataset()
    frame, category_column, priority_column = build_modelling_frame(raw_data)
    frame.to_csv(DATA_PROCESSED_DIR / "model_ready_tickets.csv", index=False)

    profile = {
        "rows": len(frame),
        "dataset_files": raw_data.attrs.get("dataset_files", [str(args.data or DEFAULT_DATASET_PATH)]),
        "raw_rows_before_dedup": raw_data.attrs.get("raw_rows_before_dedup", len(raw_data)),
        "duplicate_rows_removed": raw_data.attrs.get("duplicate_rows_removed", 0),
        "text_column": frame.attrs.get("text_column"),
        "metadata_columns": frame.attrs.get("metadata_columns", []),
        "excluded_leakage_columns": frame.attrs.get("leakage_columns", []),
        "category_target": category_column,
        "priority_target": priority_column,
        "vectorizer_type": args.vectorizer,
    }
    (RESULTS_DIR / "dataset_profile.json").write_text(
        json.dumps(profile, indent=2),
        encoding="utf-8",
    )

    all_results = []
    all_results.extend(train_task(frame, category_column, "category", args.vectorizer))
    all_results.extend(train_task(frame, priority_column, "priority", args.vectorizer))

    metrics = pd.DataFrame(
        [
            {key: value for key, value in result.items() if key != "classification_report"}
            for result in all_results
        ]
    ).sort_values(["task", "macro_f1"], ascending=[True, False])
    metrics.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    reports = {
        f"{result['task']}_{result['model']}_{result['split']}": result["classification_report"]
        for result in all_results
    }
    (RESULTS_DIR / "classification_reports.json").write_text(
        json.dumps(reports, indent=2),
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
