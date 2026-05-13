"""Prediction workflow used by scripts and the Streamlit demo."""

from __future__ import annotations

from pathlib import Path

import joblib

from .config import MODELS_DIR
from .confidence import estimate_prediction_confidence, confidence_level, needs_human_review
from .explainability import top_tfidf_terms
from .features import clean_text
from .routing_rules import escalation_required, recommend_team
from .summarisation import summarise_ticket
from .rl_routing_agent import RoutingAgent as _RoutingAgent

# Shared agent instance
_rl_agent = _RoutingAgent()


def load_model(path: str | Path):
    return joblib.load(path)


def analyse_ticket(text: str, category_model, priority_model) -> dict:
    """Return decision-support outputs using already-loaded models."""
    model_text = clean_text(text)

    category = str(category_model.predict([model_text])[0])
    priority = str(priority_model.predict([model_text])[0])
    rl_action = _rl_agent.route(category, priority)
    queue_confidence = estimate_prediction_confidence(category_model, model_text)
    priority_confidence = estimate_prediction_confidence(priority_model, model_text)
    escalation = escalation_required(priority, text)
    review_required, review_reason = needs_human_review(
        queue_confidence,
        priority_confidence,
        escalation,
    )

    return {
        "category": category,
        "priority": priority,
        "recommended_team": recommend_team(category, text),
        "summary": summarise_ticket(text, category, priority),
        "escalation_required": escalation,
        "category_explanation_terms": top_tfidf_terms(category_model, model_text),
        "priority_explanation_terms": top_tfidf_terms(priority_model, model_text),
        "rl_routing_action": rl_action,
        "queue_confidence": queue_confidence,
        "priority_confidence": priority_confidence,
        "queue_confidence_level": confidence_level(queue_confidence),
        "priority_confidence_level": confidence_level(priority_confidence),
        "human_review_required": review_required,
        "human_review_reason": review_reason,
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
