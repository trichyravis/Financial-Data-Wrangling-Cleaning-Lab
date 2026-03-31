"""Indian Financial Market Case Studies"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.components import (
    render_hero, render_footer, section_header,
    defn_box, insight_box, warning_box, success_box, metric_row,
)
import utils.theme  # noqa


def render():
    render_hero(
        title="Indian Financial Market Case Studies",
        subtitle="Demonetisation Shock · IL&FS Default · NSE Data Feed Outage",
        badges=[
            {"text": "Nov 2016",  "style": "red"},
            {"text": "Sep 2018",  "style": "orange"},
            {"text": "Feed MCAR", "style": "blue"},
        ],
    )

    tabs = st.tabs([
        "📚 Case 1: Demonetisation (2016)",
        "📚 Case 2: IL&FS Default (2018)",
        "📚 Case 3: NSE Feed Outage",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        section_header("⚡", "Case Study 1: Demonetisation Shock — Outlier or Structural Break?")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
<div class="mp-card-gold">
<div style="font-weight:700; color:#FFD700; font-size:1.05rem; margin-bottom:0.6rem;">
  📅 Event: 8 November 2016
</div>
<div style="font-size:0.9rem; color:#e6f1ff;">
  The Government of India announced demonetisation of ₹500 and ₹1,000 currency notes.
  <b>NIFTY 50 fell 6.1% intraday and closed down 4.0%</b> — the largest single-day fall of 2016.
</div>
</div>
""", unsafe_allow_html=True)

        with col2:
            metric_row([
                {"val": "−4.0%", "lbl": "NIFTY Close Return"},
                {"val": "−5.2σ", "lbl": "Z-Score (2016 data)"},
            ])

        # ── Simulated NIFTY returns around demonetisation ─────────────────────
        rng = np.random.default_rng(42)
        n = 252
        dates = pd.bdate_range("2016-01-04", periods=n)
        rets = rng.normal(0.04/252, 0.12/np.sqrt(252), n)
        # Inject demonetisation shock ~Nov 8 2016 (approx day 215)
        rets[215] = -0.040
        rets[216] = -0.018
        rets[217] = -0.012
        rets[218] =  0.008

        series = pd.Series(rets * 100, index=dates, name="NIFTY_Return_pct")

        z_scores = np.abs((series - series.mean()) / series.std())
        outlier_mask = z_scores > 3

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=["NIFTY 50 Daily Returns (%)", "Z-Score (|Z| > 3 flagged)"],
                             row_heights=[0.65, 0.35])
        fig.add_trace(go.Bar(x=series.index, y=series,
                              marker_color=["#dc3545" if v < 0 else "#28a745" for v in series],
                              name="Daily Return"), row=1, col=1)
        # Highlight demonetisation
        fig.add_vrect(x0="2016-11-08", x1="2016-11-11",
                       fillcolor="rgba(255,215,0,0.12)", line_color="rgba(255,215,0,0.5)",
                       annotation_text="Demonetisation", annotation_position="top left",
                       row=1, col=1)
        fig.add_trace(go.Scatter(x=series.index, y=z_scores, mode="lines",
                                  line=dict(color="#ADD8E6", width=1.5), name="|Z-Score|"), row=2, col=1)
        fig.add_hline(y=3, line_dash="dash", line_color="#dc3545", row=2, col=1)
        fig.update_layout(height=450, showlegend=False,
                           title="NIFTY 50 — 2016 Returns with Demonetisation Shock")
        st.plotly_chart(fig, use_container_width=True)

        section_header("🔍", "Analytical Decision Framework")

        decisions = [
            ("VaR / ES Models", "🟢 RETAIN",
             "This observation IS the tail event the model must price. Removing it understates tail risk.",
             "green"),
            ("CAPM / Factor Regression", "🟡 FLAG",
             "Add a demonetisation indicator dummy variable (D_nov2016=1) rather than removing the observation.",
             "gold"),
            ("Volatility Modelling (GARCH)", "🟢 RETAIN",
             "The volatility clustering post-demonetisation is a key stylised fact GARCH must capture.",
             "green"),
            ("Cross-Sectional PE Analysis", "🟡 VERIFY",
             "Return series not used directly; PE ratios on that date should be verified for data feed errors from volatility.",
             "gold"),
            ("Stress Testing", "🟢 RETAIN",
             "Demonetisation is a prime stress scenario. Historical simulation VaR must include this event.",
             "green"),
        ]

        for use_case, action, rationale, style in decisions:
            colour = {"green": "#28a745", "gold": "#FFD700"}[style]
            st.markdown(f"""
<div class="mp-card" style="border-left:4px solid {colour}; margin-bottom:0.6rem; display:flex; gap:1rem; align-items:flex-start;">
  <div style="min-width:200px;">
    <div style="font-weight:700; color:#e6f1ff; font-size:0.88rem;">{use_case}</div>
    <div style="font-weight:700; color:{colour}; font-size:0.85rem; margin-top:0.2rem;">{action}</div>
  </div>
  <div style="font-size:0.86rem; color:#e6f1ff;">{rationale}</div>
</div>
""", unsafe_allow_html=True)

        warning_box(
            "<b>Key Lesson:</b> A mechanical |Z| > 3 rule would incorrectly flag COVID-crash and "
            "Demonetisation observations for removal, gutting the tail risk signal from historical data. "
            "<b>Domain knowledge must always override blind statistical rules.</b>"
        )

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        section_header("🚨", "Case Study 2: IL&FS Default — Missing Data as a Warning Signal")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
<div class="mp-card" style="border-color:rgba(220,53,69,0.3);">
<div style="font-weight:700; color:#dc3545; font-size:1.05rem; margin-bottom:0.6rem;">
  📅 Event: September 2018
</div>
<div style="font-size:0.9rem; color:#e6f1ff;">
  Infrastructure Leasing &amp; Financial Services (IL&amp;FS) defaulted on commercial paper obligations,
  triggering a <b>liquidity crisis across NBFCs and credit markets</b>.
</div>
</div>
""", unsafe_allow_html=True)

        with col2:
            metric_row([
                {"val": "MNAR",   "lbl": "Missingness Type"},
                {"val": "₹94K Cr", "lbl": "IL&FS Debt (Approx.)"},
            ])

        # ── Simulated data quality degradation ────────────────────────────────
        rng2 = np.random.default_rng(7)
        quarters = ["Q1 2017", "Q2 2017", "Q3 2017", "Q4 2017",
                    "Q1 2018", "Q2 2018", "Q3 2018 (Default)", "Q4 2018"]
        n_q = len(quarters)

        # Missing rate increases as distress grows
        missing_rate = [0.05, 0.06, 0.07, 0.09, 0.14, 0.22, 0.45, 0.65]
        cashflow_reported = [rng2.normal(200, 20), rng2.normal(195, 22), rng2.normal(180, 25),
                              rng2.normal(160, 30), rng2.normal(120, 40), np.nan, np.nan, np.nan]
        credit_rating = ["AAA", "AAA", "AAA", "AAA", "AA+", "AA", "D", "D"]
        icr = [2.8, 2.6, 2.3, 2.0, 1.5, 0.9, np.nan, np.nan]

        fig = make_subplots(rows=1, cols=2,
                             subplot_titles=["Data Missingness Rate Over Time", "Operating Cash Flow (₹ Cr)"])
        fig.add_trace(go.Bar(x=quarters, y=[m * 100 for m in missing_rate],
                              marker_color=["#28a745" if m < 0.10 else "#FFD700" if m < 0.25 else "#dc3545"
                                             for m in missing_rate],
                              name="Missing %"), row=1, col=1)
        fig.add_hline(y=10, line_dash="dash", line_color="#FFD700", row=1, col=1,
                       annotation_text="Warning threshold 10%")
        fig.add_trace(go.Scatter(x=quarters, y=cashflow_reported, mode="lines+markers",
                                  line=dict(color="#ADD8E6", width=2),
                                  marker=dict(color="#FFD700", size=8),
                                  name="CFO"), row=1, col=2)
        fig.add_trace(go.Scatter(x=["Q2 2018", "Q3 2018 (Default)"], y=[np.nan, np.nan],
                                  mode="markers", marker=dict(color="#dc3545", size=14, symbol="x-thin",
                                                               line=dict(width=3)),
                                  name="Missing — MNAR signal"), row=1, col=2)
        fig.update_layout(height=380, showlegend=False,
                           title="IL&FS: Missing Data Pattern as Early Warning Signal")
        st.plotly_chart(fig, use_container_width=True)

        section_header("💡", "Key Analytical Lessons")

        lessons = [
            ("1", "Always investigate WHY data is missing before deciding HOW to handle it.",
             "The mechanism (MNAR vs MAR) fundamentally changes both the analytical approach and the inference."),
            ("2", "In credit analysis, treat systematic missing data in financially stressed entities as a RED FLAG.",
             "Not a neutral data gap — missing disclosure is itself a distress signal."),
            ("3", "Implement 'missingness flags' as model features — not just for imputation.",
             "Binary flags (cash_flow_missing=1) can be powerful early-warning predictors in ML credit models."),
            ("4", "Analysts who assumed MCAR/MAR missed the signal.",
             "Those who investigated the pattern of missingness — asking 'why is this missing?' — correctly identified emerging distress."),
        ]

        for num, lesson, detail in lessons:
            st.markdown(f"""
<div class="mp-card-gold" style="margin-bottom:0.7rem;">
  <div style="display:flex; gap:1rem;">
    <span class="step-badge">{num}</span>
    <div>
      <div style="font-weight:700; color:#e6f1ff; font-size:0.9rem;">{lesson}</div>
      <div style="font-size:0.83rem; color:#8892b0; margin-top:0.3rem;">{detail}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.code("""
# Missingness as a predictive feature in credit models
df_credit['cash_flow_missing']   = df_credit['Operating_CFO'].isnull().astype(int)
df_credit['revenue_missing']     = df_credit['Revenue'].isnull().astype(int)
df_credit['disclosure_missing']  = df_credit['ESG_Score'].isnull().astype(int)

# These binary flags can be powerful early-warning features
# in ML credit models (XGBoost, Random Forest)
# Higher missingness → higher default probability (for MNAR patterns)
""", language="python")

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        section_header("📡", "Case Study 3: NSE Data Feed Outage — Block Missing Data")

        st.markdown("""
<div class="mp-card-blue">
<div style="font-weight:700; color:#ADD8E6; font-size:1.05rem; margin-bottom:0.6rem;">
  📅 Scenario: 3-Hour Data Feed Outage (10:30–13:30 IST)
</div>
<div style="font-size:0.9rem; color:#e6f1ff;">
  A quantitative fund experiences a data feed outage from <b>10:30 to 13:30 IST</b> on a regular trading day.
  The missing data is <b>MCAR</b> — a random technical failure unrelated to market conditions.
</div>
</div>
""", unsafe_allow_html=True)

        # ── Simulated intraday data with outage ───────────────────────────────
        rng3 = np.random.default_rng(99)
        timestamps = pd.date_range("2024-01-15 09:15", "2024-01-15 15:30", freq="1min")
        prices = 21500 + np.cumsum(rng3.normal(0, 15, len(timestamps)))
        s = pd.Series(prices, index=timestamps, name="NIFTY_Intraday")

        # Mask outage window
        outage_mask = (s.index >= "2024-01-15 10:30") & (s.index <= "2024-01-15 13:30")
        s_with_gap = s.copy()
        s_with_gap[outage_mask] = np.nan

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=s, name="True Price (hypothetical)",
                                  line=dict(color="#28a745", width=1, dash="dot"), opacity=0.5))
        fig.add_trace(go.Scatter(x=s_with_gap.index, y=s_with_gap, name="Observed (with outage)",
                                  line=dict(color="#ADD8E6", width=2)))
        fig.add_vrect(x0="2024-01-15 10:30", x1="2024-01-15 13:30",
                       fillcolor="rgba(220,53,69,0.12)", line_color="rgba(220,53,69,0.5)",
                       annotation_text="Data Feed Outage (MCAR)", annotation_position="top left")
        fig.update_layout(title="NIFTY 50 Intraday — 3-Hour Data Feed Outage",
                          xaxis_title="Time (IST)", yaxis_title="Index Value", height=380)
        st.plotly_chart(fig, use_container_width=True)

        section_header("📋", "Treatment Decision Matrix")

        decisions2 = [
            ("End-of-Day P&L", "Forward-fill missing ticks; use closing prices",
             "Final closing prices available; intraday not needed for EOD P&L", "#28a745"),
            ("Intraday VaR Calculation", "Flag and exclude the 3-hour window",
             "Missing high-frequency data cannot be reliably imputed for risk calculations", "#dc3545"),
            ("Intraday Momentum Signal", "Do NOT trade during outage window",
             "Stale prices produce false signals → potential large losses", "#dc3545"),
            ("VWAP Calculation", "Partial VWAP using available trades",
             "Use available volume-price pairs; note incomplete window in attribution", "#FFD700"),
            ("Backtesting Historical Data", "Fill with exchange-published data",
             "Source from NSE/BSE historical tick data download for complete records", "#FFD700"),
        ]

        for use_case, treatment, rationale, colour in decisions2:
            st.markdown(f"""
<div class="mp-card" style="border-left:4px solid {colour}; margin-bottom:0.6rem;">
  <div style="display:flex; gap:1.5rem; align-items:flex-start; flex-wrap:wrap;">
    <div style="min-width:180px;">
      <div style="font-weight:700; color:#ADD8E6; font-size:0.88rem;">{use_case}</div>
    </div>
    <div style="flex:1; min-width:180px;">
      <div style="font-size:0.78rem; color:#8892b0; text-transform:uppercase; letter-spacing:0.05em;">Treatment</div>
      <div style="font-weight:600; color:{colour}; font-size:0.86rem;">{treatment}</div>
    </div>
    <div style="flex:2; min-width:200px;">
      <div style="font-size:0.78rem; color:#8892b0; text-transform:uppercase; letter-spacing:0.05em;">Rationale</div>
      <div style="font-size:0.84rem; color:#e6f1ff;">{rationale}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        insight_box(
            "<b>Best Practice — Data Quality Gates:</b> Quantitative funds implement automated checks "
            "that halt or modify signal generation when data completeness falls below a threshold "
            "(e.g., < 90% of expected ticks received in the last 5 minutes). "
            "This prevents stale prices from triggering false trading signals during outages."
        )

        st.code("""
# Data Quality Gate — halt signal generation during feed outage
def check_data_quality_gate(latest_ticks, expected_freq_sec=60, lookback_min=5):
    \"\"\"
    Returns True (gate OPEN = safe to trade) or False (gate CLOSED = halt signals)
    \"\"\"
    cutoff = pd.Timestamp.now(tz='Asia/Kolkata') - pd.Timedelta(minutes=lookback_min)
    recent = latest_ticks[latest_ticks.index >= cutoff]
    
    expected_ticks = lookback_min * 60 / expected_freq_sec
    completeness = len(recent) / expected_ticks
    
    if completeness < 0.90:
        print(f"⚠️  DATA GATE CLOSED: Only {completeness:.0%} ticks received. Signals halted.")
        return False
    
    return True  # Safe to trade
""", language="python")

    render_footer()
