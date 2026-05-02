"""Data loading and preprocessing for support-ticket modelling."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORY_COLUMN_CANDIDATES,
    COMPATIBLE_DATASET_PATHS,
    LEAKAGE_COLUMN_CANDIDATES,
    PRIORITY_COLUMN_CANDIDATES,
    RANDOM_STATE,
    SAFE_METADATA_CANDIDATES,
    SUBJECT_COLUMN_CANDIDATES,
    TEST_SIZE,
    TEXT_COLUMN_CANDIDATES,
    VALIDATION_SIZE,
)
from .features import clean_text, create_model_text


def normalise_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Use simple snake_case names so the pipeline is easier to read."""
    data = data.copy()
    data.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        for column in data.columns
    ]
    return data


def find_column(data: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    """Find the first available column from a list of expected dataset names."""
    normalised = {re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") for value in candidates}
    for column in data.columns:
        if column in normalised:
            return column
    if required:
        raise ValueError(f"Could not find any of these columns: {', '.join(candidates)}")
    return None


def identify_leakage_columns(data: pd.DataFrame) -> list[str]:
    """List columns that should not be used for initial ticket prediction."""
    normalised_leakage = {
        re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        for value in LEAKAGE_COLUMN_CANDIDATES
    }
    return [column for column in data.columns if column in normalised_leakage]


def build_modelling_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """Create the text feature and target columns used by the models."""
    data = normalise_columns(data)
    text_column = find_column(data, TEXT_COLUMN_CANDIDATES)
    subject_column = find_column(data, SUBJECT_COLUMN_CANDIDATES, required=False)
    category_column = find_column(data, CATEGORY_COLUMN_CANDIDATES)
    priority_column = find_column(data, PRIORITY_COLUMN_CANDIDATES)

    metadata_columns = [
        column
        for column in SAFE_METADATA_CANDIDATES
        if column in data.columns and column not in {text_column, subject_column}
    ]

    selected_columns = [text_column, category_column, priority_column, *metadata_columns]
    if subject_column:
        selected_columns.insert(0, subject_column)
    if "source_file" in data.columns:
        selected_columns.append("source_file")

    frame = data[selected_columns].copy()
    frame = frame.drop_duplicates()
    frame[text_column] = frame[text_column].fillna("")
    if subject_column:
        frame[subject_column] = frame[subject_column].fillna("")
    frame[category_column] = frame[category_column].astype(str).str.strip()
    frame[priority_column] = frame[priority_column].astype(str).str.strip()
    frame = frame[(frame[text_column].str.len() > 0) & (frame[category_column] != "")]

    if subject_column:
        frame["ticket_text"] = (frame[subject_column].astype(str) + " " + frame[text_column].astype(str)).str.strip()
    else:
        frame["ticket_text"] = frame[text_column].astype(str)

    frame["clean_text"] = frame["ticket_text"].map(clean_text)
    frame["model_text"] = create_model_text(frame, "ticket_text", metadata_columns)
    frame.attrs["text_column"] = text_column
    frame.attrs["subject_column"] = subject_column
    frame.attrs["metadata_columns"] = metadata_columns
    frame.attrs["leakage_columns"] = identify_leakage_columns(data)
    return frame, category_column, priority_column


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset from the Kaggle export or a compatible sample."""
    return pd.read_csv(path)


def load_project_dataset(paths: list[str | Path] | None = None) -> pd.DataFrame:
    """Load compatible Kaggle CSVs and remove duplicate ticket texts."""
    dataset_paths = [Path(path) for path in (paths or COMPATIBLE_DATASET_PATHS)]
    missing = [path.name for path in dataset_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Place these compatible CSV files in data/raw/: " + ", ".join(missing)
        )

    frames = []
    for path in dataset_paths:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True, sort=False)
    data = normalise_columns(data)
    subject_column = find_column(data, SUBJECT_COLUMN_CANDIDATES, required=False)
    text_column = find_column(data, TEXT_COLUMN_CANDIDATES)

    subject_text = data[subject_column].fillna("").astype(str) if subject_column else ""
    body_text = data[text_column].fillna("").astype(str)
    data["_dedupe_key"] = (subject_text + " || " + body_text).str.lower().str.strip()
    data = data.drop_duplicates("_dedupe_key", keep="first").drop(columns="_dedupe_key")
    data.attrs["dataset_files"] = [path.name for path in dataset_paths]
    data.attrs["raw_rows_before_dedup"] = sum(len(frame) for frame in frames)
    data.attrs["rows_after_dedup"] = len(data)
    data.attrs["duplicate_rows_removed"] = data.attrs["raw_rows_before_dedup"] - len(data)
    return data


def _can_stratify(labels: pd.Series, requested_rows: int) -> bool:
    label_counts = labels.value_counts()
    return label_counts.min() >= 2 and requested_rows >= label_counts.shape[0]


def make_train_validation_test_split(frame: pd.DataFrame, target_column: str):
    """Create train, validation, and test splits for fair model selection."""
    y = frame[target_column]
    test_rows = max(1, int(len(frame) * TEST_SIZE))
    first_stratify = y if _can_stratify(y, test_rows) else None

    train_validation, test = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=first_stratify,
    )

    validation_fraction = VALIDATION_SIZE / (1 - TEST_SIZE)
    validation_rows = max(1, int(len(train_validation) * validation_fraction))
    second_y = train_validation[target_column]
    second_stratify = second_y if _can_stratify(second_y, validation_rows) else None

    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction,
        random_state=RANDOM_STATE,
        stratify=second_stratify,
    )
    return train, validation, test


def make_train_test_split(frame: pd.DataFrame, target_column: str):
    """Backward-compatible train/test split used by older notebooks."""
    train, validation, test = make_train_validation_test_split(frame, target_column)
    train_validation = pd.concat([train, validation], axis=0)
    return (
        train_validation["model_text"],
        test["model_text"],
        train_validation[target_column],
        test[target_column],
    )
