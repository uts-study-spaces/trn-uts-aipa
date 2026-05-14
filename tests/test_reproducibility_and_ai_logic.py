from pathlib import Path

import pandas as pd

from customer_support_ai.features import clean_text
from customer_support_ai.repository_health import has_conflict_markers, validate_metrics_file
from customer_support_ai.routing_rules import escalation_required, recommend_team
from customer_support_ai.summarisation import anonymise_text, summarise_ticket


def test_clean_text_masks_contact_details():
    cleaned = clean_text("Email me at user@example.com or call +61 400 111 222.")
    assert "emailaddress" in cleaned
    assert "phonenumber" in cleaned
    assert "user@example.com" not in cleaned


def test_security_terms_route_to_security_team():
    team = recommend_team("Billing and Payments", "Unknown device and unauthorised access detected.")
    assert team == "Security Team"


def test_urgent_security_text_triggers_escalation():
    assert escalation_required("low", "Urgent fraud and data breach concern.") is True


def test_summary_masks_private_information():
    summary = summarise_ticket(
        "Please contact customer@example.com or +61 400 111 222 about the refund.",
        "Billing and Payments",
        "high",
    )
    assert "[email]" in summary
    assert "[phone]" in summary


def test_anonymise_text_masks_email_and_phone():
    text = anonymise_text("Reach me at alpha@example.com and +61 411 222 333.")
    assert "alpha@example.com" not in text
    assert "[email]" in text
    assert "[phone]" in text


def test_metrics_validation_detects_conflict_markers(tmp_path: Path):
    metrics = tmp_path / "metrics_summary.csv"
    metrics.write_text("task,model\n<<<<<<< HEAD\n", encoding="utf-8")
    assert has_conflict_markers(metrics) is True
    assert validate_metrics_file(metrics)


def test_metrics_validation_accepts_required_columns(tmp_path: Path):
    metrics = tmp_path / "metrics_summary.csv"
    pd.DataFrame(
        [
            {
                "task": "category",
                "model": "linear_svm",
                "split": "test",
                "accuracy": 0.5,
                "macro_precision": 0.5,
                "macro_recall": 0.5,
                "macro_f1": 0.5,
                "weighted_f1": 0.5,
            }
        ]
    ).to_csv(metrics, index=False)
    assert validate_metrics_file(metrics) == []
