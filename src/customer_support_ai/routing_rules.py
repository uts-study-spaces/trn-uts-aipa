"""Transparent routing and escalation rules."""

from __future__ import annotations

SECURITY_TERMS = [
    "hacked",
    "unauthorised",
    "unauthorized",
    "data breach",
    "unknown device",
    "password changed",
]

URGENT_TERMS = [
    "urgent",
    "critical",
    "immediately",
    "asap",
    "breach",
    "fraud",
    "outage",
]

TEAM_BY_CATEGORY = {
    "technical support": "Technical Support",
    "product support": "Product Support",
    "customer service": "Customer Service",
    "it support": "IT Support",
    "billing and payments": "Billing and Payments",
    "returns and exchanges": "Returns and Exchanges",
    "service outages and maintenance": "Service Outages and Maintenance",
    "sales and pre sales": "Sales and Pre-Sales",
    "sales and pre-sales": "Sales and Pre-Sales",
    "human resources": "Human Resources",
    "general inquiry": "General Inquiry",
    "login issue": "Account Support",
    "account access": "Account Support",
    "payment problem": "Billing Team",
    "billing": "Billing Team",
    "refund request": "Billing Team",
    "security concern": "Security Team",
    "bug report": "Technical Support",
    "performance issue": "Technical Support",
    "feature request": "Product Team",
    "data sync issue": "Integration Support",
}


def normalise_label(label: str) -> str:
    return str(label).strip().lower().replace("_", " ")


def recommend_team(category: str, text: str = "") -> str:
    """Map model output and clear keywords to a support team."""
    lowered = text.lower()
    if any(term in lowered for term in SECURITY_TERMS):
        return "Security Team"

    category_key = normalise_label(category)
    for known_category, team in TEAM_BY_CATEGORY.items():
        if known_category in category_key:
            return team
    return "General Support"


def escalation_required(priority: str, text: str = "") -> bool:
    """Flag tickets that need fast human review."""
    lowered = text.lower()
    priority_key = normalise_label(priority)
    return priority_key in {"high", "critical", "urgent"} or any(
        term in lowered for term in URGENT_TERMS + SECURITY_TERMS
    )
