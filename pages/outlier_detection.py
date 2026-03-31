"""Outlier Detection — Statistical & Domain Methods"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import IsolationForest
import streamlit as st

from utils.components import (
    render_hero, render_footer, section_header,
    defn_box, insight_box, warning_box, success_box, metric_row,
)
from utils.data_gen import make_fat_returns
import utils.theme  # noqa


def render():
    render_hero(
        title="Outlier Detection in Financial Data",
        subtitle="Z-Score · Modified Z-Score (MAD) · IQR · Isolation Forest · Domain Rules",
        badges=[
            {"text": "Error Outliers",         "style": "red"},
            {"text": "Genuine Extreme Events", "style": "gold"},
            {"text": "Structural Breaks",      "style": "blue"},
        ],
    )

    tabs = st.tabs(["📘 Concepts", "🔬 Interactive Detection", "⚖️ Method Comparison", "📏 Domain Rules"])

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        section_header("📘", "Three Types of Financial Outliers")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
<div class="mp-card" style="border-color:rgba(220,53,69,0.35); height:100%;">
  <div style="text-align:center; font-size:1.6rem; margin-bottom:0.5rem;">🐛</div>
  <h3 style="color:#dc3545 !important;">Error Outliers</h3>
  <div style="font-size:0.88rem; color:#e6f1ff; margin-bottom:0.8rem;">
    Data errors — wrong decimal placement, feed errors, sign flips.
  </div>
  <div style="font-size:0.83rem; color:#8892b0;">Examples:</div>
  <ul style="font-size:0.83rem; color:#e6f1ff; padding-left:1.2rem; margin:0.4rem 0;">
    <li>Negative stock price</li>
    <li>PE Ratio = 50,000</li>
    <li>Volume = 0 on active day</li>
    <li>Return = −99% in liquid large-cap</li>
  </ul>
  <div style="margin-top:0.8rem; padding:0.6rem; background:rgba(220,53,69,0.08); border-radius:6px; font-size:0.82rem;">
    <b style="color:#dc3545;">Action:</b> Correct or remove. These are NOT real events.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
<div class="mp-card-gold" style="height:100%;">
  <div style="text-align:center; font-size:1.6rem; margin-bottom:0.5rem;">⚡</div>
  <h3>Genuine Extreme Events</h3>
  <div style="font-size:0.88rem; color:#e6f1ff; margin-bottom:0.8rem;">
    Real financial events — crashes, circuit breakers, policy shocks. Must be preserved.
  </div>
  <div style="font-size:0.83rem; color:#8892b0;">Examples:</div>
  <ul style="font-size:0.83rem; color:#e6f1ff; padding-left:1.2rem; margin:0.4rem 0;">
    <li>NIFTY −38% (COVID Mar 2020)</li>
    <li>NIFTY −4% (Demonetisation Nov 2016)</li>
    <li>Short squeezes on small-caps</li>
    <li>RBI rate shock days</li>
  </ul>
  <div style="margin-top:0.8rem; padding:0.6rem; background:rgba(255,215,0,0.08); border-radius:6px; font-size:0.82rem;">
    <b style="color:#FFD700;">Action:</b> Preserve for risk modelling & stress-testing. Document.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                """
<div class="mp-card-blue" style="height:100%;">
  <div style="text-align:center; font-size:1.6rem; margin-bottom:0.5rem;">🔀</div>
  <h3>Structural Breaks</h3>
  <div style="font-size:0.88rem; color:#e6f1ff; margin-bottom:0.8rem;">
    One-time shifts in the data-generating process. May require regime-splitting.
  </div>
  <div style="font-size:0.83rem; color:#8892b0;">Examples:</div>
  <ul style="font-size:0.83rem; color:#e6f1ff; padding-left:1.2rem; margin:0.4rem 0;">
    <li>Merger / demerger</li>
    <li>Regulatory change (Basel III → IV)</li>
    <li>Index reconstitution</li>
    <li>Currency regime change</li>
  </ul>
  <div style="margin-top:0.8rem; padding:0.6rem; background:rgba(107,174,214,0.08); border-radius:6px; font-size:0.82rem;">
    <b style="color:#ADD8E6;">Action:</b> Regime-split or add regime dummy variable. Don't simply remove.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("📐", "Statistical Detection Methods")

        methods_info = [
            ("Z-Score", "Zi = (Xi − X̄) / σ", "Flag |Zi| > 3", "Simple; assumes normality", "Limitation: Financial returns are NOT normally distributed. Fat tails inflate σ, making Z-score too lenient for tail events."),
            ("Modified Z-Score (MAD)", "Mi = 0.6745 · |Xi − X̃| / MAD", "Flag |Mi| > 3.5", "Robust; MAD unaffected by outliers themselves", "Preferred for return series, accounting ratios, and any fat-tailed financial variable."),
            ("IQR (Tukey's Fence)", "Lower = Q1 − k·IQR, Upper = Q3 + k·IQR", "k=1.5 moderate; k=3.0 extreme", "Non-parametric; no distribution assumption", "Standard in academic finance: winsorise at 1%/99% percentiles (k ≈ 2.2 equivalent)."),
            ("Isolation Forest", "Anomaly score s(x,n) = 2^(−E[h(x)]/c(n))", "Score ≈ 1 → anomalous; ≈ 0.5 → normal", "Multivariate; no distribution assumption", "Ideal for detecting anomalous trading patterns across multiple features simultaneously."),
        ]

        for name, formula, rule, strength, note in methods_info:
            with st.expander(f"**{name}**"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Formula:** `{formula}`")
                    st.markdown(f"**Rule:** {rule}")
                    st.markdown(f"**Strength:** {strength}")
                with col2:
                    insight_box(note)

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        section_header("🔬", "Interactive Outlier Detection")

        col_ctrl, col_main = st.columns([1, 3])
        with col_ctrl:
            n_obs = st.slider("Observations", 100, 1000, 500, step=50, key="out_n")
            n_shocks = st.slider("Injected Shock Events", 0, 20, 10)
            z_thresh = st.slider("Z-Score Threshold", 1.5, 5.0, 3.0, step=0.1)
            mad_thresh = st.slider("MAD Threshold", 1.5, 6.0, 3.5, step=0.1)
            contamination = st.slider("IF Contamination", 0.01, 0.10, 0.02, step=0.005)

        rng = np.random.default_rng(42)
        rets = rng.normal(0.05 / 252, 0.18 / np.sqrt(252), n_obs)
        if n_shocks > 0:
            shock_idx = rng.choice(n_obs, n_shocks, replace=False)
            rets[shock_idx] = rng.choice([-0.08, -0.06, -0.05, 0.06, 0.07, 0.09], n_shocks)
        dates = pd.bdate_range("2022-01-03", periods=n_obs)
        returns = pd.Series(rets, index=dates, name="Daily_Return")

        # ── Z-Score ───────────────────────────────────────────────────────────
        z_scores = np.abs(stats.zscore(returns))
        out_z = returns[z_scores > z_thresh]

        # ── MAD ───────────────────────────────────────────────────────────────
        median = returns.median()
        mad = np.median(np.abs(returns - median))
        mod_z = 0.6745 * np.abs(returns - median) / (mad + 1e-10)
        out_mad = returns[mod_z > mad_thresh]

        # ── IQR ───────────────────────────────────────────────────────────────
        Q1, Q3 = returns.quantile(0.25), returns.quantile(0.75)
        IQR = Q3 - Q1
        out_iqr = returns[(returns < Q1 - 3.0 * IQR) | (returns > Q3 + 3.0 * IQR)]

        # ── Isolation Forest ──────────────────────────────────────────────────
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso_pred = iso.fit_predict(returns.values.reshape(-1, 1))
        out_iso = returns[iso_pred == -1]

        # ── Consensus ─────────────────────────────────────────────────────────
        all_idx = set(out_z.index) | set(out_mad.index) | set(out_iqr.index) | set(out_iso.index)
        consensus_idx = [
            idx for idx in all_idx
            if sum([idx in set(o.index) for o in [out_z, out_mad, out_iqr, out_iso]]) >= 2
        ]

        metric_row([
            {"val": str(len(out_z)),         "lbl": "Z-Score Flags"},
            {"val": str(len(out_mad)),        "lbl": "MAD Flags"},
            {"val": str(len(out_iqr)),        "lbl": "IQR Flags"},
            {"val": str(len(out_iso)),        "lbl": "IF Flags"},
            {"val": str(len(consensus_idx)), "lbl": "Consensus (≥2)"},
        ])

        with col_main:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=returns.index, y=returns * 100, mode="lines",
                                     line=dict(color="#ADD8E6", width=1), name="Daily Returns", opacity=0.7))
            fig.add_trace(go.Scatter(x=out_z.index, y=out_z * 100, mode="markers",
                                     marker=dict(color="#FFD700", size=8, symbol="circle-open", line=dict(width=2)),
                                     name=f"Z-Score (|z|>{z_thresh})"))
            fig.add_trace(go.Scatter(x=out_mad.index, y=out_mad * 100, mode="markers",
                                     marker=dict(color="#fd7e14", size=8, symbol="triangle-up"), name="MAD"))
            fig.add_trace(go.Scatter(
                x=[returns.index[i] for i in range(len(returns)) if returns.index[i] in set(out_iso.index)],
                y=[returns[idx] * 100 for idx in returns.index if idx in set(out_iso.index)],
                mode="markers", marker=dict(color="#dc3545", size=9, symbol="x", line=dict(width=2)),
                name="Isolation Forest",
            ))
            if consensus_idx:
                con_rets = returns[returns.index.isin(consensus_idx)]
                fig.add_trace(go.Scatter(x=con_rets.index, y=con_rets * 100, mode="markers",
                                         marker=dict(color="#fff", size=14, symbol="star",
                                                     line=dict(color="#FFD700", width=2)),
                                         name="Consensus (≥2 methods)"))

            fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)")
            fig.update_layout(title="Multi-Method Outlier Detection on NSE Return Data",
                              xaxis_title="Date", yaxis_title="Return (%)", height=420,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

        if consensus_idx:
            st.subheader("🌟 Consensus Outlier Dates")
            con_df = pd.DataFrame({
                "Date": [idx.date() for idx in sorted(consensus_idx)],
                "Return (%)": [round(returns[idx] * 100, 2) for idx in sorted(consensus_idx)],
                "Flagged by": [
                    " | ".join([
                        n for n, s in [("Z-Score", out_z), ("MAD", out_mad), ("IQR", out_iqr), ("IF", out_iso)]
                        if idx in set(s.index)
                    ])
                    for idx in sorted(consensus_idx)
                ],
            })
            st.dataframe(con_df, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        section_header("⚖️", "Method Comparison Summary")

        compare = pd.DataFrame({
            "Method": ["Z-Score", "Modified Z-Score (MAD)", "IQR Method", "Isolation Forest", "DBSCAN"],
            "Distribution Assumption": ["Normal", "None (robust)", "None", "None", "None"],
            "Handles Fat Tails": ["❌ No", "✅ Yes", "✅ Yes", "✅ Yes", "✅ Yes"],
            "Multivariate": ["❌ No", "❌ No", "❌ No", "✅ Yes", "✅ Yes"],
            "Financial Use Case": [
                "Quick scan; symmetric data",
                "Return series; accounting ratios",
                "Cross-sectional winsorisation",
                "Market surveillance; multi-feature anomaly",
                "Cluster-level anomaly detection",
            ],
            "Python": [
                "scipy.stats.zscore",
                "Custom MAD function",
                "Series.quantile()",
                "sklearn.ensemble.IsolationForest",
                "sklearn.cluster.DBSCAN",
            ],
        })
        st.dataframe(compare, use_container_width=True, hide_index=True)

        # ── Sensitivity analysis callout ──────────────────────────────────────
        section_header("📋", "Sensitivity Analysis — The Gold Standard Decision Tool")
        insight_box(
            """When the decision to include or exclude outliers is genuinely uncertain, run <b>sensitivity analysis</b>:
            <ol>
            <li>Run analysis <b>with</b> outliers (full sample)</li>
            <li>Run analysis <b>without</b> outliers (trimmed/winsorised)</li>
            <li>Run with robust methods (Huber regression, Theil-Sen)</li>
            <li>If results are <b>materially different</b>: investigate further; report both; document assumptions</li>
            <li>If results are <b>similar</b>: outliers do not materially affect conclusions</li>
            </ol>
            <b>Regulatory note:</b> SEBI ICDR and RBI model risk guidelines require banks and AMCs to document 
            outlier treatment methodology and demonstrate stability of results to outlier assumptions."""
        )

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        section_header("📏", "Domain-Specific Financial Outlier Rules")

        rules = [
            ("Circuit Breaker Check", "NSE imposes 10%/15%/20% daily price limits. A move exceeding circuit limits is either a legitimate event (verify) or a data error (if claimed trade occurred outside limits).", "gold"),
            ("Earnings Surprise Threshold", "EPS > 5σ above/below consensus deserves verification — genuine surprise or data vendor error (mismatched denominator, per-share vs. total)?", "blue"),
            ("Volume Spike Rules", "Volume > 5× the 20-day average: corporate announcement, block deal, or data error. Cross-reference NSE bulk/block deal data.", "blue"),
            ("Financial Ratio Bounds", "PE < 0: negative earnings (valid). PE > 500: check extreme growth stock or error? Debt/Equity > 20: flag for leveraged sector verification.", "gold"),
            ("Bid-Ask Spread Outliers", "Spread > 5% for liquid large-cap → likely feed error. Spread > 20% for illiquid micro-cap → verify thin book.", "blue"),
            ("Stale Price Detection", "Price unchanged for 5+ consecutive days in a liquid stock → feed disconnection. Use rolling std-dev ≈ 0 alert.", "gold"),
        ]

        for title, body, style in rules:
            st.markdown(
                f"""
<div class="mp-card-{'blue' if style=='blue' else 'gold'}" style="margin-bottom:0.8rem;">
  <div style="font-weight:700; color:{'#ADD8E6' if style=='blue' else '#FFD700'}; margin-bottom:0.4rem;">
    📌 {title}
  </div>
  <div style="font-size:0.89rem; color:#e6f1ff;">{body}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        # Winsorisation demo
        section_header("✂️", "Winsorisation — Interactive Demo")

        col1, col2 = st.columns(2)
        with col1:
            lower_pct = st.slider("Lower Percentile", 0.5, 5.0, 1.0, step=0.5, key="w_lo")
            upper_pct = st.slider("Upper Percentile", 95.0, 99.5, 99.0, step=0.5, key="w_hi")

        rng2 = np.random.default_rng(42)
        pe = np.concatenate([rng2.lognormal(3.3, 0.6, 490),
                             np.array([500, 800, -50, -30, 0.5, 1200, 2000, 5000, 0.8, 1000])])
        s_pe = pd.Series(pe, name="PE_Ratio")

        lo = s_pe.quantile(lower_pct / 100)
        hi = s_pe.quantile(upper_pct / 100)
        s_wins = s_pe.clip(lo, hi)

        with col2:
            metric_row([
                {"val": f"{(s_pe < lo).sum()}",  "lbl": f"Capped at lower ({lo:.1f})"},
                {"val": f"{(s_pe > hi).sum()}",  "lbl": f"Capped at upper ({hi:.1f})"},
                {"val": f"{s_pe.std():.1f}",     "lbl": "Std Dev Before"},
                {"val": f"{s_wins.std():.1f}",   "lbl": "Std Dev After"},
            ])

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Original PE Distribution", "Winsorised PE Distribution"])
        fig.add_trace(go.Histogram(x=s_pe.clip(-100, 1000), nbinsx=60, marker_color="#ADD8E6", opacity=0.75, name="Original"), row=1, col=1)
        fig.add_trace(go.Histogram(x=s_wins, nbinsx=60, marker_color="#FFD700", opacity=0.75, name="Winsorised"), row=1, col=2)
        fig.add_vline(x=lo, line_dash="dash", line_color="#dc3545", row=1, col=2)
        fig.add_vline(x=hi, line_dash="dash", line_color="#dc3545", row=1, col=2)
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    render_footer()
