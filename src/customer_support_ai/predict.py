"""Prediction workflow used by scripts and the Streamlit demo."""

from __future__ import annotations

from pathlib import Path

import joblib

from .config import MODELS_DIR
from .explainability import top_tfidf_terms
from .features import clean_text
from .routing_rules import escalation_required, recommend_team
from .summarisation import summarise_ticket


def load_model(path: str | Path):
    return joblib.load(path)


def analyse_ticket(text: str, category_model, priority_model) -> dict:
    """Return decision-support outputs using already-loaded models."""
    model_text = clean_text(text)

    category = str(category_model.predict([model_text])[0])
    priority = str(priority_model.predict([model_text])[0])

    return {
        "category": category,
        "priority": priority,
        "recommended_team": recommend_team(category, text),
        "summary": summarise_ticket(text, category, priority),
        "escalation_required": escalation_required(priority, text),
        "category_explanation_terms": top_tfidf_terms(category_model, model_text),
        "priority_explanation_terms": top_tfidf_terms(priority_model, model_text),
    }


def predict_ticket(
    text: str,
    category_model_path: str | Path = MODELS_DIR / "category_model.pkl",
    priority_model_path: str | Path = MODELS_DIR / "priority_model.pkl",
) -> dict:
    """Return all decision-support outputs for one ticket."""
    category_model = load_model(category_model_path)
    priority_model = load_model(priority_model_path)
    return analyse_ticket(text, category_model, priority_model)
