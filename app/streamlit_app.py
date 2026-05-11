"""Streamlit dashboard for the customer support AI project."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from customer_support_ai.config import DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_DIR
from customer_support_ai.predict import analyse_ticket, load_model

GROUP_LABEL = "36121 Artificial Intelligence Principles and Applications - AT3 - Group 2"
APP_TITLE = "Ticket Routing Intelligence"
APP_SUBTITLE = "Explainable Multilingual Queue and Priority Prediction"


st.set_page_config(page_title=APP_TITLE, page_icon="🎧", layout="wide")


def inject_style() -> None:
    """Add presentation styling and make Streamlit controls easier to use."""
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] {
            height: 0rem;
            background: transparent;
        }
        div[data-testid="stToolbar"] {
            display: none;
        }
        .stApp {
            background:
                radial-gradient(circle at 8% 3%, rgba(56, 189, 248, 0.24), transparent 22rem),
                radial-gradient(circle at 92% 10%, rgba(45, 212, 191, 0.20), transparent 24rem),
                linear-gradient(180deg, #071423 0%, #0b1f33 48%, #10263a 100%);
            color: #e8f1f9;
        }
        .block-container {
            padding-top: 0.35rem;
            padding-bottom: 3rem;
            max-width: 1250px;
        }
        h1, h2, h3, h4, h5, h6, p, label, span {
            color: inherit;
        }
        div[data-testid="stTabs"] {
            margin-top: 0.2rem;
        }
        div[data-testid="stTabs"] [role="tablist"] {
            gap: 0.55rem;
            border-bottom: 1px solid rgba(170, 211, 243, 0.16);
        }
        div[data-testid="stTabs"] [role="tab"] {
            min-height: 46px;
            padding: 0.75rem 1.05rem;
            border-radius: 8px 8px 0 0;
            border: 1px solid rgba(170, 211, 243, 0.12);
            background: rgba(255, 255, 255, 0.055);
            color: #d9eaf8;
            cursor: pointer;
        }
        div[data-testid="stTabs"] [role="tab"] p {
            font-weight: 800;
            font-size: 1rem;
        }
        div[data-testid="stTabs"] [aria-selected="true"] {
            background: rgba(94, 234, 212, 0.12);
            border-color: rgba(94, 234, 212, 0.35);
            color: #5eead4;
        }
        .hero {
            padding: 1.65rem 2rem;
            border: 1px solid rgba(148, 210, 255, 0.18);
            border-radius: 8px;
            background: linear-gradient(135deg, rgba(15, 32, 51, 0.96) 0%, rgba(17, 55, 73, 0.94) 55%, rgba(20, 69, 66, 0.92) 100%);
            box-shadow: 0 22px 55px rgba(0, 0, 0, 0.28);
        }
        .hero h1 {
            font-size: 3.65rem;
            line-height: 1.02;
            margin-bottom: 0.25rem;
            color: #f7fbff;
            letter-spacing: 0;
        }
        .subtitle {
            color: #c8e6f3;
            font-size: 1.22rem;
            margin-bottom: 0.75rem;
            font-weight: 650;
        }
        .group-label {
            color: #a9c1d6;
            font-size: 0.94rem;
            margin-bottom: 0.95rem;
        }
        .pill {
            display: inline-block;
            padding: 0.35rem 0.65rem;
            margin: 0 0.35rem 0.35rem 0;
            border-radius: 999px;
            background: rgba(45, 212, 191, 0.15);
            border: 1px solid rgba(45, 212, 191, 0.34);
            color: #bff7ec;
            font-size: 0.85rem;
            font-weight: 700;
        }
        .section-card {
            background: rgba(255, 255, 255, 0.075);
            border: 1px solid rgba(170, 211, 243, 0.18);
            border-radius: 8px;
            padding: 1.15rem;
            margin: 0.75rem 0;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
        }
        .mini-card-title {
            color: #f7fbff;
            font-weight: 850;
            margin-bottom: 0.25rem;
        }
        .muted {
            color: #bad0e1;
            font-size: 0.93rem;
        }
        .workflow-row {
            display: grid;
            grid-template-columns: repeat(7, minmax(112px, 1fr));
            gap: 0.65rem;
            margin: 0.9rem 0 1.2rem 0;
        }
        .workflow-step {
            min-height: 118px;
            padding: 0.85rem;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(94, 234, 212, 0.22);
            text-align: center;
        }
        .workflow-icon {
            font-size: 1.9rem;
            line-height: 1.9rem;
            margin-bottom: 0.5rem;
        }
        .workflow-title {
            font-weight: 850;
            color: #f7fbff;
            margin-bottom: 0.2rem;
        }
        .workflow-copy {
            color: #bad0e1;
            font-size: 0.82rem;
            line-height: 1.25rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(170, 211, 243, 0.28);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.18);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #f4f8fb;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: clamp(1.1rem, 1.45vw, 1.55rem);
            line-height: 1.1;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        .stTextArea label, .stSelectbox label, .stSlider label, .stFileUploader label {
            color: #f2f7fb !important;
            font-weight: 800;
        }
        .stTextArea textarea {
            background: #f8fafc !important;
            color: #111827 !important;
            border: 1px solid rgba(170, 211, 243, 0.55) !important;
            caret-color: #111827 !important;
        }
        .stTextArea textarea::placeholder {
            color: #64748b !important;
        }
        .stSelectbox div[data-baseweb="select"] {
            background: #f8fafc !important;
            color: #111827 !important;
            border-color: rgba(170, 211, 243, 0.55) !important;
        }
        .stSelectbox div[data-baseweb="select"] * {
            color: #111827 !important;
        }
        div[data-testid="stFileUploader"] section {
            background: rgba(255, 255, 255, 0.10);
            border: 1px dashed rgba(94, 234, 212, 0.48);
            border-radius: 8px;
            min-height: 132px;
            display: flex;
            align-items: center;
            padding: 1.05rem;
        }
        div[data-testid="stFileUploader"] section button {
            background: #ff4b4b !important;
            color: white !important;
            border: 0 !important;
            border-radius: 8px !important;
            font-weight: 800 !important;
            font-size: 1.05rem !important;
            padding: 0.65rem 1rem !important;
        }
        div[data-testid="stFileUploader"] small,
        div[data-testid="stFileUploader"] section > div > span {
            display: none !important;
        }
        div[data-testid="stFileUploader"] section::after {
            content: "Click Upload or drag a ticket file here";
            color: #e8f1f9;
            font-weight: 800;
            margin-left: 1rem;
        }
        div[data-testid="stButton"] button[kind="secondary"] {
            color: #f7fbff !important;
            background: rgba(255, 255, 255, 0.10) !important;
            border: 1px solid rgba(170, 211, 243, 0.34) !important;
            border-radius: 8px !important;
            font-weight: 800 !important;
        }
        div[data-testid="stDownloadButton"] button {
            color: #ffffff !important;
            background: #ff4b4b !important;
            border: 0 !important;
            border-radius: 8px !important;
            font-weight: 850 !important;
            padding: 0.65rem 1rem !important;
            box-shadow: 0 8px 18px rgba(255, 75, 75, 0.18);
        }
        div[data-testid="stDownloadButton"] button:hover {
            background: #ff5f5f !important;
            border: 0 !important;
            color: #ffffff !important;
        }
        div[data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.075);
            border: 1px solid rgba(170, 211, 243, 0.18);
            border-radius: 8px;
        }
        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] details[open],
        div[data-testid="stExpander"] summary {
            background: rgba(255, 255, 255, 0.075) !important;
            border-radius: 8px !important;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary p {
            color: #f4f8fb !important;
            font-weight: 800;
        }
        div[role="radiogroup"] label,
        div[role="radiogroup"] p,
        div[role="radiogroup"] span {
            color: #f4f8fb !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.075);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 8px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.16);
            padding: 0.7rem;
        }
        div[data-testid="stVegaLiteChart"] {
            background: transparent !important;
            border: 0 !important;
            padding: 0 !important;
            box-shadow: none !important;
        }
        .result-card {
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(170, 211, 243, 0.30);
            border-radius: 8px;
            padding: 1rem;
            min-height: 120px;
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.18);
        }
        .result-label {
            color: #f4f8fb;
            font-size: 0.92rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }
        .result-value {
            color: #ffffff;
            font-size: clamp(1.25rem, 1.7vw, 1.75rem);
            line-height: 1.14;
            font-weight: 650;
            overflow-wrap: anywhere;
        }
        .chart-title {
            color: #f7fbff;
            font-weight: 850;
            font-size: 1.03rem;
            margin: 0 0 0.35rem 0;
        }
        .dark-table {
            width: 100%;
            border-collapse: collapse;
            overflow: hidden;
            border-radius: 8px;
            font-size: 0.9rem;
        }
        .dark-table th {
            background: rgba(94, 234, 212, 0.14);
            color: #eafffb;
            padding: 0.7rem;
            text-align: left;
            border-bottom: 1px solid rgba(170, 211, 243, 0.18);
        }
        .dark-table td {
            background: rgba(255, 255, 255, 0.055);
            color: #e8f1f9;
            padding: 0.65rem 0.7rem;
            border-bottom: 1px solid rgba(170, 211, 243, 0.10);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def suppress_streamlit_copy_hotkey() -> None:
    """Reduce Streamlit's cache shortcut popup when Ctrl+C is used on the page."""
    components.html(
        """
        <script>
        window.parent.document.addEventListener('keydown', function(event) {
          if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'c') {
            const tag = window.parent.document.activeElement?.tagName?.toLowerCase();
            if (!['input', 'textarea'].includes(tag)) {
              event.stopImmediatePropagation();
            }
          }
        }, true);
        </script>
        """,
        height=0,
        width=0,
    )


def chart_theme(chart: alt.Chart) -> alt.Chart:
    """Apply a consistent dark chart theme."""
    return chart.configure_view(
        fill="transparent",
        strokeOpacity=0,
    ).configure_axis(
        labelColor="#d8e8f5",
        titleColor="#f4f8fb",
        gridColor="rgba(255,255,255,0.16)",
        domainColor="rgba(255,255,255,0.28)",
        tickColor="rgba(255,255,255,0.28)",
    ).configure_legend(
        labelColor="#d8e8f5",
        titleColor="#f4f8fb",
        orient="bottom",
    ).configure_title(
        color="#f4f8fb",
        fontSize=16,
        anchor="start",
    ).properties(
        background="transparent",
    )


def load_models():
    """Load models once per browser session without Streamlit cache shortcuts."""
    if "category_model" not in st.session_state:
        st.session_state["category_model"] = load_model(MODELS_DIR / "category_model.pkl")
    if "priority_model" not in st.session_state:
        st.session_state["priority_model"] = load_model(MODELS_DIR / "priority_model.pkl")
    return st.session_state["category_model"], st.session_state["priority_model"]


def load_metrics() -> pd.DataFrame:
    metrics_path = RESULTS_DIR / "metrics_summary.csv"
    return pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()


def load_dataset_profile() -> dict:
    profile_path = RESULTS_DIR / "dataset_profile.json"
    if not profile_path.exists():
        return {}
    return json.loads(profile_path.read_text(encoding="utf-8"))


def load_optional_csv(filename: str) -> pd.DataFrame:
    path = RESULTS_DIR / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_model_ready_data() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "model_ready_tickets.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    """Read a CSV or Excel upload for batch ticket analysis."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Upload a CSV or Excel file.")


def find_text_columns(frame: pd.DataFrame) -> list[str]:
    """Find likely ticket text columns in an uploaded batch file."""
    preferred_names = ["ticket_text", "body", "description", "message", "text", "subject"]
    normalised = {str(column).strip().lower(): column for column in frame.columns}

    candidates = [
        normalised[name]
        for name in preferred_names
        if name in normalised
    ]

    for column in frame.columns:
        if column in candidates:
            continue
        series = frame[column]
        if pd.api.types.is_string_dtype(series) or series.dtype == "object":
            candidates.append(column)

    return candidates


def render_dark_table(data: pd.DataFrame, max_rows: int | None = None) -> None:
    """Render a compact dark HTML table instead of the default white grid."""
    if data.empty:
        st.info("No table data is available.")
        return
    display_data = data.head(max_rows).copy() if max_rows else data.copy()
    for column in display_data.select_dtypes(include="number").columns:
        display_data[column] = display_data[column].map(lambda value: f"{value:.4g}")
    html = display_data.to_html(index=False, escape=False, classes="dark-table")
    st.markdown(html, unsafe_allow_html=True)


def bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    sort: str | list[str] | None = "-y",
    height: int = 320,
) -> None:
    """Render a polished Altair bar chart."""
    if data.empty:
        st.info("No chart data is available.")
        return

    encoding = {
        "x": alt.X(f"{x}:N", sort=sort, axis=alt.Axis(labelAngle=-30)),
        "y": alt.Y(f"{y}:Q"),
        "tooltip": list(data.columns),
    }
    if color:
        encoding["color"] = alt.Color(f"{color}:N", scale=alt.Scale(scheme="set2"))
    else:
        encoding["color"] = alt.value("#5eead4")

    chart = alt.Chart(data).mark_bar(
        cornerRadiusTopLeft=5,
        cornerRadiusTopRight=5,
        opacity=0.92,
    ).encode(**encoding).properties(title=title, height=height)

    st.altair_chart(chart_theme(chart), use_container_width=True)


def chart_card(title: str, data: pd.DataFrame, x: str, y: str, color: str | None = None, height: int = 300) -> None:
    """Render a chart inside a bordered dashboard panel."""
    with st.container(border=True):
        st.markdown(f"<div class='chart-title'>{title}</div>", unsafe_allow_html=True)
        bar_chart(data, x, y, title="", color=color, height=height)


def render_result(result: dict) -> None:
    result_items = [
        ("Predicted queue", result["category"]),
        ("Predicted priority", result["priority"]),
        ("Recommended team", result["recommended_team"]),
        ("Escalation", "Required" if result["escalation_required"] else "Not required"),
    ]
    columns = st.columns(4)
    for column, (label, value) in zip(columns, result_items):
        with column:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">{label}</div>
                    <div class="result-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    col1, col2 = st.columns([1.05, 0.95])
    with col1:
        st.markdown("#### Summary")
        st.markdown(f"<div class='section-card'>{result['summary']}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("#### Explanation Terms")
        category_terms = ", ".join(term for term, _ in result["category_explanation_terms"]) or "No terms available"
        priority_terms = ", ".join(term for term, _ in result["priority_explanation_terms"]) or "No terms available"
        st.markdown(
            f"""
            <div class="section-card">
            <b>Queue:</b> {category_terms}<br><br>
            <b>Priority:</b> {priority_terms}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_workflow() -> None:
    """Render a one-line project workflow."""
    st.markdown(
        """
        <div class="workflow-row">
            <div class="workflow-step"><div class="workflow-icon">📁</div><div class="workflow-title">Audit Data</div><div class="workflow-copy">Inspect all 5 Kaggle CSV files.</div></div>
            <div class="workflow-step"><div class="workflow-icon">🧹</div><div class="workflow-title">Prepare Text</div><div class="workflow-copy">Merge, deduplicate, clean, and mask text.</div></div>
            <div class="workflow-step"><div class="workflow-icon">🔎</div><div class="workflow-title">Build Features</div><div class="workflow-copy">TF-IDF text plus safe metadata tokens.</div></div>
            <div class="workflow-step"><div class="workflow-icon">🧠</div><div class="workflow-title">Train Models</div><div class="workflow-copy">Compare baseline NLP classifiers.</div></div>
            <div class="workflow-step"><div class="workflow-icon">📊</div><div class="workflow-title">Evaluate</div><div class="workflow-copy">Macro F1, confusion matrices, language checks.</div></div>
            <div class="workflow-step"><div class="workflow-icon">🧭</div><div class="workflow-title">Route</div><div class="workflow-copy">Recommend team and escalation review.</div></div>
            <div class="workflow-step"><div class="workflow-icon">💬</div><div class="workflow-title">Explain</div><div class="workflow-copy">Show summaries and influential terms.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def overview_tab(profile: dict, metrics: pd.DataFrame) -> None:
    """Home and insights combined into one page."""
    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_TITLE}</h1>
            <div class="subtitle">{APP_SUBTITLE}</div>
            <div class="group-label">{GROUP_LABEL}</div>
            <span class="pill">NLP classification</span>
            <span class="pill">Priority prediction</span>
            <span class="pill">Routing rules</span>
            <span class="pill">Escalation support</span>
            <span class="pill">Explainable AI</span>
            <br><br>
            <p>
                A decision-support system for first-pass customer support triage: predict the queue,
                estimate urgency, recommend a team, summarise the issue, and explain the decision.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows = profile.get("rows", 0)
    duplicates = profile.get("duplicate_rows_removed", 0)
    files_used = len(profile.get("dataset_files", []))
    test_rows = metrics[metrics["split"] == "test"] if not metrics.empty else pd.DataFrame()
    best_macro = test_rows["macro_f1"].max() if not test_rows.empty else 0

    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Usable tickets", f"{rows:,}")
    c2.metric("Compatible CSVs", files_used)
    c3.metric("Duplicates removed", f"{duplicates:,}")
    c4.metric("Best test macro F1", f"{best_macro:.3f}" if best_macro else "N/A")

    st.markdown("### AI Workflow")
    render_workflow()

    st.markdown("### Project Evidence")
    st.markdown(
        """
        <div class="section-card">
        <div class="mini-card-title">Dataset and modelling decision</div>
        <div class="muted">
        We audited five Kaggle CSV files, merged the three compatible multilingual files,
        and excluded the two German-normalized files because their queue/category labels use
        a different taxonomy. The model uses subject + body text, safe metadata, and excludes
        the post-response answer field to avoid leakage.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    frame = load_model_ready_data()
    per_language = load_optional_csv("per_language_metrics.csv")
    per_class = load_optional_csv("per_class_f1.csv")
    transformer_benchmark = load_optional_csv("transformer_embedding_benchmark.csv")

    left, right = st.columns(2)
    with left:
        if not frame.empty and "language" in frame.columns:
            language_counts = frame["language"].value_counts().rename_axis("language").reset_index(name="tickets")
            chart_card("Ticket Language Distribution", language_counts, "language", "tickets", height=280)
    with right:
        if not frame.empty and "priority" in frame.columns:
            priority_counts = frame["priority"].value_counts().rename_axis("priority").reset_index(name="tickets")
            chart_card("Priority Distribution", priority_counts, "priority", "tickets", height=280)

    left, right = st.columns(2)
    with left:
        chart_sources = [data for data in [metrics, transformer_benchmark] if not data.empty]
        if chart_sources:
            chart_data = pd.concat(chart_sources, ignore_index=True)
            chart_data = chart_data.assign(label=chart_data["task"] + " | " + chart_data["model"] + " | " + chart_data["split"])
            chart_card("Model Comparison by Macro F1", chart_data, "label", "macro_f1", color="task", height=330)
    with right:
        if not per_language.empty:
            chart_card("Macro F1 by Language", per_language, "language", "macro_f1", color="task", height=330)

    with st.expander("Detailed Report Tables"):
        table_choice = st.radio(
            "Detailed table choice",
            ["Final test metrics", "Transformer benchmark", "Per-language metrics", "Per-class F1"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if table_choice == "Final test metrics":
            render_dark_table(metrics[metrics["split"] == "test"] if not metrics.empty else pd.DataFrame())
        elif table_choice == "Transformer benchmark":
            render_dark_table(transformer_benchmark)
        elif table_choice == "Per-language metrics":
            render_dark_table(per_language)
        else:
            render_dark_table(per_class)


def solution_tab() -> None:
    """Single and batch prediction workflows."""
    st.markdown("### Try the Solution")
    st.caption("Analyse one ticket directly or upload a small batch file for multiple predictions.")

    render_label_space()

    single_mode, batch_mode = st.tabs(["Single ticket", "Upload batch"])
    with single_mode:
        render_single_ticket_form()
    with batch_mode:
        render_batch_upload_form()


def render_label_space() -> None:
    """Show the labels and outputs users can expect from the solution."""
    frame = load_model_ready_data()
    st.markdown("#### Output Labels")
    q1, q2, q3 = st.columns(3)
    with q1:
        if not frame.empty and "queue" in frame.columns:
            queues = sorted(frame["queue"].dropna().astype(str).unique())
            st.markdown(
                "<div class='section-card'><div class='mini-card-title'>Predicted Queues / Categories</div><div class='muted'>"
                + ", ".join(queues)
                + "</div></div>",
                unsafe_allow_html=True,
            )
    with q2:
        if not frame.empty and "priority" in frame.columns:
            priorities = sorted(frame["priority"].dropna().astype(str).unique())
            st.markdown(
                "<div class='section-card'><div class='mini-card-title'>Predicted Priorities</div><div class='muted'>"
                + ", ".join(priorities)
                + "</div></div>",
                unsafe_allow_html=True,
            )
    with q3:
        st.markdown(
            """
            <div class="section-card">
            <div class="mini-card-title">Queue vs Team</div>
            <div class="muted">
            The predicted queue is the model label. The recommended team is the operational
            routing suggestion produced from the queue and escalation keywords.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_single_ticket_form() -> None:
    """Render the single-ticket analysis form."""
    st.markdown("#### Single Ticket Analysis")
    st.caption("Paste one customer support ticket and analyse the queue, priority, routing, summary, and explanation terms.")

    examples = {
        "Security incident": "Subject: Possible unauthorised access to admin portal\n\nBody: Several staff members received unexpected password reset emails this morning. One admin account shows login attempts from an unknown location. We need urgent help checking whether customer records or billing details were accessed.",
        "Billing problem": "Subject: Duplicate invoice charged to account\n\nBody: Our company subscription was charged twice for the same billing period. The invoice numbers are different but both payments were taken from the same card. Please review the account and reverse the duplicate payment.",
        "Service outage": "Subject: Dashboard unavailable for multiple users\n\nBody: The customer reporting dashboard has been unavailable for our support team for the last hour. Clearing the browser cache and restarting devices did not fix the issue. Multiple users are blocked from viewing live case data.",
        "Product question": "Subject: Compatibility question before upgrade\n\nBody: We are planning to upgrade our analytics dashboard and need to confirm whether the new version supports Salesforce integration, single sign-on, and scheduled PDF exports for management reports.",
    }

    selected_example = st.selectbox("Demo example", ["Custom text", *examples.keys()])
    ticket_text = st.text_area(
        "Support ticket",
        value=examples.get(selected_example, ""),
        height=210,
        placeholder="Paste a customer support ticket description here...",
    )

    if st.button("Analyse ticket", type="primary"):
        if not ticket_text.strip():
            st.warning("Enter a support ticket first.")
            return
        try:
            category_model, priority_model = load_models()
            result = analyse_ticket(ticket_text, category_model, priority_model)
        except FileNotFoundError:
            st.error("Model files are missing. Train the models first with: python -m customer_support_ai.train")
            return
        render_result(result)


def render_batch_upload_form() -> None:
    """Render the batch upload workflow."""
    st.markdown("#### Upload Multiple Tickets")
    st.markdown(
        """
        <div class="section-card muted">
        <b>Expected file format</b><br>
        Upload a CSV, XLSX, or XLS file with one row per ticket and a text column such as
        <b>ticket_text</b>, <b>body</b>, <b>description</b>, <b>message</b>, or <b>text</b>.
        The demo analyses up to 200 rows at a time. Maximum upload size: 200MB per file.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "upload_widget_key" not in st.session_state:
        st.session_state["upload_widget_key"] = 0

    st.caption("Choose a file once. Streamlit loads it immediately after selection; use Clear selected file to reset.")
    uploaded_file = st.file_uploader(
        "Upload file",
        type=["csv", "xlsx", "xls"],
        key=f"ticket_upload_{st.session_state['upload_widget_key']}",
    )
    if uploaded_file is None:
        return

    left, _ = st.columns([0.22, 0.78])
    with left:
        if st.button("Clear selected file"):
            st.session_state["upload_widget_key"] += 1
            st.session_state.pop("batch_result_df", None)
            st.rerun()

    try:
        uploaded_df = read_uploaded_table(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return

    text_candidates = find_text_columns(uploaded_df)
    if not text_candidates:
        st.error(
            "No ticket text column was found. Please include a column named ticket_text, body, "
            "description, message, text, or subject."
        )
        return

    lower_candidates = {str(column).strip().lower(): column for column in text_candidates}
    preferred = next(
        (
            lower_candidates[name]
            for name in ["ticket_text", "body", "description", "message", "text", "subject"]
            if name in lower_candidates
        ),
        text_candidates[0],
    )
    if len(text_candidates) == 1:
        text_column = text_candidates[0]
        st.caption(f"Using `{text_column}` as the ticket text column.")
    else:
        text_column = st.selectbox("Ticket text column", text_candidates, index=text_candidates.index(preferred))
    row_limit = min(200, len(uploaded_df))
    if row_limit == 1:
        max_rows = 1
    else:
        max_rows = st.slider("Rows to analyse", 1, row_limit, min(25, row_limit))

    if st.button("Analyse uploaded tickets", type="primary"):
        try:
            category_model, priority_model = load_models()
        except FileNotFoundError:
            st.error("Model files are missing. Train the models first with: python -m customer_support_ai.train")
            return

        rows = []
        subset = uploaded_df.head(max_rows).copy()
        for text in subset[text_column].fillna("").astype(str):
            if text.strip():
                result = analyse_ticket(text, category_model, priority_model)
                rows.append(
                    {
                        "ticket_text": text,
                        "predicted_queue": result["category"],
                        "predicted_priority": result["priority"],
                        "recommended_team": result["recommended_team"],
                        "escalation_required": result["escalation_required"],
                        "summary": result["summary"],
                    }
                )

        result_df = pd.DataFrame(rows)
        st.session_state["batch_result_df"] = result_df

    if "batch_result_df" in st.session_state:
        result_df = st.session_state["batch_result_df"]
        if not result_df.empty:
            st.success(f"Analysed {len(result_df)} tickets.")
            render_dark_table(result_df, max_rows=12)
            st.download_button(
                "Download predictions CSV",
                result_df.to_csv(index=False).encode("utf-8"),
                file_name="ticket_predictions.csv",
                mime="text/csv",
                type="primary",
            )


def main() -> None:
    inject_style()
    suppress_streamlit_copy_hotkey()
    metrics = load_metrics()
    profile = load_dataset_profile()

    overview, solution = st.tabs(["Overview", "Try Solution"])
    with overview:
        overview_tab(profile, metrics)
    with solution:
        solution_tab()


if __name__ == "__main__":
    main()
