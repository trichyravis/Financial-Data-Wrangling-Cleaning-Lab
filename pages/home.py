"""Home page — overview and learning objectives"""

import streamlit as st
from utils.components import render_hero, render_footer, section_header, card, defn_box, insight_box, badge, metric_row


def render():
    render_hero(
        title="Financial Data Wrangling & Cleaning",
        subtitle="Session 2 · Financial Analytics · BITS Pilani WILP · MBA ZG517 / PDBA / PDFI / PDFT",
        quote="Clean Data is the Foundation of Every Credible Financial Model",
        badges=[
            {"text": "MBA ZG517", "style": "gold"},
            {"text": "Session 2", "style": "blue"},
            {"text": "Interactive Lab", "style": "green"},
        ],
    )

    # ── Intro cards ──────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        section_header("🎯", "What This Lab Covers")
        modules = [
            ("🔍", "Missing Data", "MCAR · MAR · MNAR classification, diagnosis & treatment", "blue"),
            ("📊", "Outlier Detection", "Z-score, MAD, IQR, Isolation Forest on live NSE data", "gold"),
            ("📅", "Time Series Formatting", "DatetimeIndex, business calendars, resampling, OHLCV", "blue"),
            ("⚠️", "Invalid Values", "OHLC logic checks, stale prices, duplicate timestamps", "gold"),
            ("⚙️", "Full Cleaning Pipeline", "Production-grade FinancialDataCleaner class, end-to-end", "green"),
            ("📚", "Indian Market Case Studies", "Demonetisation · IL&FS · NSE Data Feed Outage", "blue"),
        ]
        for icon, title, desc, style in modules:
            st.markdown(
                f"""
<div class="mp-card" style="display:flex; align-items:flex-start; gap:1rem; padding:1rem 1.3rem; margin-bottom:0.7rem;">
  <span style="font-size:1.4rem; flex-shrink:0;">{icon}</span>
  <div>
    <div style="font-weight:700; color:#ADD8E6; font-size:0.95rem;">{title}</div>
    <div style="color:#8892b0; font-size:0.83rem; margin-top:0.15rem;">{desc}</div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    with col2:
        section_header("📌", "Learning Objectives")
        objectives = [
            "Diagnose & classify data quality problems in real financial datasets",
            "Select & implement appropriate missing data imputation strategies",
            "Structure financial time series with correct indexing & frequency handling",
            "Apply statistical & domain-specific methods to detect & manage outliers",
            "Build a reproducible Python data-cleaning pipeline",
            "Critically evaluate downstream impact of data quality decisions on model outputs",
        ]
        for i, obj in enumerate(objectives, 1):
            st.markdown(
                f"""
<div class="step-row" style="margin-bottom:0.8rem;">
  <span class="step-badge">{i}</span>
  <span class="step-content" style="font-size:0.88rem; color:#e6f1ff;">{obj}</span>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        defn_box(
            "Key Python Libraries",
            "<code>pandas</code> · <code>numpy</code> · <code>scipy.stats</code> · <code>sklearn.impute</code> · <code>statsmodels</code> · <code>matplotlib</code> · <code>seaborn</code> · <code>plotly</code>",
        )

    # ── GIGO section ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("💥", "Why Data Quality Matters — The GIGO Principle")

    metric_row([
        {"val": "USD 440M", "lbl": "Knight Capital Loss (2012)"},
        {"val": "6.1%",     "lbl": "NIFTY Drop – Demonetisation"},
        {"val": "~18%",     "lbl": "Typical NSE Missing Rate"},
        {"val": "6",        "lbl": "Dimensions of Data Quality"},
    ])

    col1, col2, col3 = st.columns(3)
    with col1:
        defn_box("Complete", "All required observations and fields are present")
        defn_box("Accurate", "Values correctly represent the underlying financial reality")
    with col2:
        defn_box("Consistent", "No contradictions across fields, time periods, or sources")
        defn_box("Timely", "Reflects the relevant point in time (point-in-time databases)")
    with col3:
        defn_box("Valid", "Values fall within expected domain ranges and business rules")
        defn_box("Unique", "No duplicate records that would double-count observations")

    # ── Knight Capital callout ────────────────────────────────────────────────
    insight_box(
        "<b>Knight Capital Incident (August 1, 2012):</b> Knight Capital Group lost <b>USD 440 million in 45 minutes</b> "
        "due to a software deployment error where stale, erroneous order data triggered unintended trading. "
        "The root cause included failure to validate and clean stale routing data before deployment. "
        "<b>Lesson:</b> A single undetected invalid value can cascade into catastrophic financial loss."
    )

    # ── Pipeline overview ─────────────────────────────────────────────────────
    section_header("🔄", "The Data Quality Pipeline")
    st.markdown(
        """
<div class="mp-card-blue" style="padding:1.8rem;">
<div style="display:flex; align-items:center; gap:0; flex-wrap:wrap; justify-content:center;">
""" + "".join([
    f"""
  <div style="text-align:center; padding:0.8rem 1rem;">
    <div style="font-size:1.5rem;">{icon}</div>
    <div style="font-size:0.8rem; font-weight:700; color:#FFD700; margin:0.3rem 0 0.1rem;">{step}</div>
    <div style="font-size:0.72rem; color:#8892b0; max-width:90px;">{desc}</div>
  </div>
  {'<div style="color:#FFD700; font-size:1.2rem; padding-top:0.5rem;">→</div>' if i < 6 else ''}
"""
    for i, (icon, step, desc) in enumerate([
        ("📥", "Ingest",    "Raw data from APIs, files, DB"),
        ("✔️",  "Validate", "Schema, types, ranges, rules"),
        ("🔍", "Missing",   "Detect, classify, impute"),
        ("📊", "Outliers",  "Detect, review, manage"),
        ("📅", "Format",    "Types, index, frequency"),
        ("📤", "Output",    "Analysis-ready dataset"),
    ], 1)
]) + """
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Quick start ───────────────────────────────────────────────────────────
    st.info("👈 **Use the sidebar** to navigate to each module. Each section includes interactive demos, live Python code, and worked examples with Indian financial market data.")

    render_footer()
