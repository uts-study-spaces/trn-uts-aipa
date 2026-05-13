"""Confidence and human-review helpers for AI triage outputs."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any

import numpy as np


HIGH_CONFIDENCE_THRESHOLD = 0.70
MEDIUM_CONFIDENCE_THRESHOLD = 0.50


@dataclass(frozen=True)
class ConfidenceDecision:
    """Structured confidence result used by the app and evaluation scripts."""

    predicted_label: str
    confidence_score: float
    confidence_level: str
    human_review_required: bool
    review_reason: str


def _normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Convert arbitrary class scores into a stable 0-1 distribution."""
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size == 0:
        return scores
    shifted = scores - np.max(scores)
    exps = np.exp(shifted)
    total = float(np.sum(exps))
    if total == 0:
        return np.full(scores.shape, 1.0 / scores.size)
    return exps / total


def _positive_margin_confidence(scores: np.ndarray) -> float:
    """Estimate confidence for binary LinearSVC decision-function output."""
    margin = float(np.asarray(scores, dtype=float).ravel()[0])
    return 1.0 / (1.0 + exp(-abs(margin)))


def estimate_prediction_confidence(model: Any, text: str) -> float:
    """Estimate confidence from a fitted classifier pipeline.

    Models with ``predict_proba`` use the highest predicted probability. Linear
    margin models, such as ``LinearSVC``, use a softmax or logistic transform of
    the decision scores. The value is intended for human-review guidance, not as
    a fully calibrated probability.
    """
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba([text])[0], dtype=float)
        return float(np.max(probabilities))

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function([text]), dtype=float)
        if scores.ndim == 1 and scores.size == 1:
            return _positive_margin_confidence(scores)
        return float(np.max(_normalise_scores(scores)))

    return 0.0


def confidence_level(score: float) -> str:
    """Map a confidence score into a presentation-friendly level."""
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "High"
    if score >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "Medium"
    return "Low"


def needs_human_review(
    queue_confidence: float,
    priority_confidence: float,
    escalation_required: bool,
) -> tuple[bool, str]:
    """Decide whether the triage output should be reviewed by a human."""
    if escalation_required:
        return True, "Escalation or risk terms were detected."
    if queue_confidence < MEDIUM_CONFIDENCE_THRESHOLD:
        return True, "Queue confidence is below the safe review threshold."
    if priority_confidence < MEDIUM_CONFIDENCE_THRESHOLD:
        return True, "Priority confidence is below the safe review threshold."
    return False, "Confidence is sufficient for standard agent review."


def build_confidence_decision(
    predicted_label: str,
    confidence_score: float,
    human_review_required: bool,
    review_reason: str,
) -> ConfidenceDecision:
    """Create a reusable confidence decision object."""
    return ConfidenceDecision(
        predicted_label=str(predicted_label),
        confidence_score=float(confidence_score),
        confidence_level=confidence_level(confidence_score),
        human_review_required=bool(human_review_required),
        review_reason=review_reason,
    )
