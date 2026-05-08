"""Gemini-powered agent that orchestrates the ML pipeline via tool use."""

from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from .explainability import top_tfidf_terms
from .features import clean_text
from .predict import analyse_ticket
from .routing_rules import escalation_required, recommend_team

load_dotenv()

GEMINI_MODEL = "gemini-3.1-flash-lite"


def _resolve_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError(
            "No Gemini API key found. Add GEMINI_API_KEY to your .env file or enter one in the app."
        )
    return key

_SYSTEM_PROMPT = """You are a customer support triage assistant backed by trained machine learning models.

If the user sends a general question or message (e.g. asking how something works, or chatting), answer it conversationally without calling any tools.

Only call the tools when the user provides an actual customer support ticket to triage. Tickets may arrive in any format - a Subject/Body structure, plain prose, a pasted email, or informal text. Before calling the tools, combine all relevant fields (subject, body, description, etc.) into a single block of text and pass that combined text to each tool. In that case, call all four tools in order:
1. classify_category - predict the queue/category
2. classify_priority - predict the urgency level
3. get_recommended_team - determine the handling team
4. check_escalation - check whether escalation is needed
5. get_explanation_terms - retrieve the key terms that drove the predictions

After collecting all five results, respond with two clear sections:

**Triage Results**
Present each result (category, priority, team, escalation status, and explanation terms) in a readable format.

**Suggested Response**
Draft a professional, empathetic reply that the support agent can send directly to the customer.

Important: you may encourage human review and judgment on the results provided, but never suggest that this application, its models, or its outputs need improvement, retraining, or development. Treat the triage results as authoritative decision-support."""

_BATCH_SUMMARY_PROMPT = """You are a support team analyst. A batch of support tickets has just been processed by an ML triage pipeline.

Results summary:
{summary}

Write a short paragraph (3–5 sentences) summarising these results for a support team manager. Highlight any escalation concerns and the most common ticket types."""


def _make_tools(category_model, priority_model) -> list:
    """Create tool functions with trained models bound via closure."""

    def classify_category(ticket_text: str) -> str:
        """Classify a support ticket into its queue category using the trained ML model."""
        return str(category_model.predict([clean_text(ticket_text)])[0])

    def classify_priority(ticket_text: str) -> str:
        """Predict the urgency priority of a support ticket using the trained ML model."""
        return str(priority_model.predict([clean_text(ticket_text)])[0])

    def get_recommended_team(category: str, ticket_text: str) -> str:
        """Determine which support team should handle the ticket based on its category and content."""
        return recommend_team(category, ticket_text)

    def check_escalation(priority: str, ticket_text: str) -> str:
        """Check whether the ticket requires urgent escalation to senior staff."""
        return (
            "Escalation required — route to senior staff immediately."
            if escalation_required(priority, ticket_text)
            else "No escalation required."
        )

    def get_explanation_terms(ticket_text: str) -> str:
        """Return the most influential TF-IDF terms driving the category and priority predictions."""
        cleaned = clean_text(ticket_text)
        category_terms = ", ".join(t for t, _ in top_tfidf_terms(category_model, cleaned)) or "none"
        priority_terms = ", ".join(t for t, _ in top_tfidf_terms(priority_model, cleaned)) or "none"
        return f"Category terms: {category_terms}\nPriority terms: {priority_terms}"

    return [classify_category, classify_priority, get_recommended_team, check_escalation, get_explanation_terms]


def create_chat_session(category_model, priority_model, api_key: str | None = None) -> tuple:
    """Create a Gemini chat session with the four ML pipeline tools bound."""
    client = genai.Client(api_key=_resolve_key(api_key))
    tool_functions = _make_tools(category_model, priority_model)
    tool_map = {fn.__name__: fn for fn in tool_functions}
    chat = client.chats.create(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            tools=tool_functions,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
        ),
    )
    return client, chat, tool_map


def send_message(chat, tool_map: dict, message: str) -> str:
    """Send a message and run the tool-use loop manually.

    Gemini may respond by requesting one or more tool calls. We execute each
    tool ourselves, send the results back, and repeat until Gemini returns a
    plain text response.
    """
    response = chat.send_message(message)

    while True:
        function_calls = [
            part.function_call
            for part in response.candidates[0].content.parts
            if part.function_call is not None
        ]

        if not function_calls:
            return response.text or ""

        tool_response_parts = []
        for fc in function_calls:
            fn = tool_map.get(fc.name)
            result = fn(**dict(fc.args)) if fn is not None else f"Unknown tool: {fc.name}"
            tool_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result},
                    )
                )
            )

        response = chat.send_message(tool_response_parts)


def analyse_batch(
    tickets: list[str],
    category_model,
    priority_model,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Analyse a list of tickets with the ML pipeline and return results with an LLM narrative."""
    rows = []
    for text in tickets:
        if not text.strip():
            continue
        result = analyse_ticket(text, category_model, priority_model)
        rows.append({
            "ticket_text": text,
            "predicted_queue": result["category"],
            "predicted_priority": result["priority"],
            "recommended_team": result["recommended_team"],
            "escalation_required": result["escalation_required"],
            "summary": result["summary"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, "No valid tickets were found in the uploaded file."

    escalation_count = int(df["escalation_required"].sum())
    summary_text = (
        f"Total tickets processed: {len(df)}\n"
        f"Queue distribution: {df['predicted_queue'].value_counts().to_dict()}\n"
        f"Priority distribution: {df['predicted_priority'].value_counts().to_dict()}\n"
        f"Teams involved: {df['recommended_team'].value_counts().to_dict()}\n"
        f"Tickets requiring escalation: {escalation_count} of {len(df)}"
    )
    prompt = _BATCH_SUMMARY_PROMPT.format(summary=summary_text)

    client = genai.Client(api_key=_resolve_key(api_key))
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return df, response.text
