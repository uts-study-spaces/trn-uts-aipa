"""Small privacy-aware ticket summarisation helper."""

from __future__ import annotations

import re


def anonymise_text(text: str) -> str:
    """Mask obvious personal data before a summary is displayed or sent elsewhere."""
    text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "[email]", text)
    text = re.sub(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b", "[phone]", text)
    return text


def summarise_ticket(text: str, category: str, priority: str) -> str:
    """Create a concise summary without relying on an external LLM."""
    safe_text = anonymise_text(text).strip()
    first_sentence = re.split(r"(?<=[.!?])\s+", safe_text)[0] if safe_text else ""
    if len(first_sentence) > 220:
        first_sentence = first_sentence[:217].rstrip() + "..."
    return f"{priority} priority {category} ticket: {first_sentence}"
