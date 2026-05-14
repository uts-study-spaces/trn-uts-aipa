from customer_support_ai.confidence import (
    confidence_level,
    estimate_prediction_confidence,
    needs_human_review,
)


class ProbabilityModel:
    def predict_proba(self, texts):
        return [[0.1, 0.9]]


class MarginModel:
    def decision_function(self, texts):
        return [[-1.0, 2.0, 0.5]]


def test_confidence_level_thresholds():
    assert confidence_level(0.72) == "High"
    assert confidence_level(0.55) == "Medium"
    assert confidence_level(0.20) == "Low"


def test_probability_confidence_uses_highest_probability():
    assert estimate_prediction_confidence(ProbabilityModel(), "ticket") == 0.9


def test_margin_confidence_returns_normalised_score():
    score = estimate_prediction_confidence(MarginModel(), "ticket")
    assert 0.0 < score <= 1.0


def test_human_review_is_required_for_low_confidence():
    required, reason = needs_human_review(0.30, 0.85, False)
    assert required is True
    assert "Queue confidence" in reason


def test_human_review_is_required_for_escalation():
    required, reason = needs_human_review(0.95, 0.95, True)
    assert required is True
    assert "Escalation" in reason
