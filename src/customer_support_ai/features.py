"""NLP cleaning and feature engineering helpers."""

from __future__ import annotations

import re

import pandas as pd


def clean_text(value: object) -> str:
    """Clean ticket text while preserving important support-domain words."""
    text = "" if pd.isna(value) else str(value)
    text = text.lower()
    text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", " emailaddress ", text)
    text = re.sub(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b", " phonenumber ", text)
    text = re.sub(r"\b\d{3,}\b", " number ", text)
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def metadata_to_tokens(row: pd.Series) -> str:
    """Represent structured metadata as readable text tokens for TF-IDF."""
    tokens = []
    for name, value in row.items():
        safe_name = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
        safe_value = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
        if safe_value:
            tokens.append(f"{safe_name}_{safe_value}")
    return " ".join(tokens)


def create_model_text(frame: pd.DataFrame, text_column: str, metadata_columns: list[str]) -> pd.Series:
    """Combine cleaned ticket text with safe pre-resolution metadata."""
    clean_ticket_text = frame[text_column].map(clean_text)
    if not metadata_columns:
        return clean_ticket_text

    metadata_tokens = frame[metadata_columns].fillna("unknown").apply(metadata_to_tokens, axis=1)
    return (clean_ticket_text + " " + metadata_tokens).str.strip()
