"""Optional multilingual transformer-embedding benchmark.

This module is intentionally separate from the main training pipeline because
sentence-transformers adds heavier dependencies than the Streamlit demo needs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import RESULTS_DIR
from .data_preprocessing import (
    build_modelling_frame,
    load_project_dataset,
    make_train_validation_test_split,
)
from .evaluate import evaluate_predictions

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SAMPLE_SIZE = 2500
OUTPUT_CSV = RESULTS_DIR / "transformer_embedding_benchmark.csv"
OUTPUT_JSON = RESULTS_DIR / "transformer_embedding_benchmark.json"


def _require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for this optional benchmark. "
            "Install it with: python -m pip install -r requirements-transformer.txt"
        ) from exc
    return SentenceTransformer


def sample_frame(frame: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    """Return a reproducible language-balanced sample when possible."""
    if sample_size <= 0 or sample_size >= len(frame):
        return frame.copy()

    if "language" not in frame.columns:
        return frame.sample(n=sample_size, random_state=random_state).copy()

    language_counts = frame["language"].value_counts()
    proportions = language_counts / language_counts.sum()
    sampled_parts = []
    remaining = sample_size

    for language, proportion in proportions.items():
        language_frame = frame[frame["language"] == language]
        language_n = min(len(language_frame), max(1, int(round(sample_size * proportion))))
        language_n = min(language_n, remaining)
        if language_n <= 0:
            continue
        sampled_parts.append(language_frame.sample(n=language_n, random_state=random_state))
        remaining -= language_n

    sampled = pd.concat(sampled_parts, axis=0)
    if len(sampled) < sample_size:
        extras = frame.drop(sampled.index).sample(
            n=min(sample_size - len(sampled), len(frame) - len(sampled)),
            random_state=random_state,
        )
        sampled = pd.concat([sampled, extras], axis=0)

    return sampled.sample(frac=1, random_state=random_state).copy()


def encode_texts(
    texts: Iterable[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
):
    """Generate dense multilingual sentence embeddings."""
    SentenceTransformer = _require_sentence_transformers()
    model = SentenceTransformer(model_name)
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def run_benchmark(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    random_state: int = 42,
    tasks: tuple[str, ...] = ("category", "priority"),
) -> pd.DataFrame:
    """Compare transformer embeddings with a simple supervised classifier."""
    RESULTS_DIR.mkdir(exist_ok=True)
    raw = load_project_dataset()
    frame, category_column, priority_column = build_modelling_frame(raw)
    sampled = sample_frame(frame, sample_size=sample_size, random_state=random_state).reset_index(drop=True)

    embeddings = encode_texts(sampled["ticket_text"].astype(str), model_name=model_name)
    rows = []

    task_columns = {
        "category": category_column,
        "priority": priority_column,
    }
    for task in tasks:
        target_column = task_columns[task]
        train, validation, test = make_train_validation_test_split(sampled, target_column)

        classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        train_validation = pd.concat([train, validation], axis=0)
        classifier.fit(embeddings[train_validation.index], train_validation[target_column])
        predictions = classifier.predict(embeddings[test.index])

        result = evaluate_predictions(
            test[target_column],
            predictions,
            task=task,
            model_name="transformer_embeddings_logistic_regression",
            split="sampled_test",
        )
        rows.append({key: value for key, value in result.items() if key != "classification_report"})

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "embedding_model": model_name,
                "sample_size": len(sampled),
                "random_state": random_state,
                "tasks": list(tasks),
                "note": "Optional sampled benchmark; compare cautiously against full-data TF-IDF results.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional transformer-embedding benchmark.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["category", "priority"],
        default=["category", "priority"],
    )
    args = parser.parse_args()

    results = run_benchmark(
        sample_size=args.sample_size,
        model_name=args.model_name,
        random_state=args.random_state,
        tasks=tuple(args.tasks),
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
