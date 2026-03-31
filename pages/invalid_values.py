"""Invalid Values — Detection and Treatment"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.components import (
    render_hero, render_footer, section_header,
    defn_box, insight_box, warning_box, success_box, metric_row,
)
import utils.theme  # noqa


def validate_ohlcv(df):
    issues = []
    for col in ["Open", "High", "Low", "Close"]:
        bad = df[df[col] <= 0]
        for idx in bad.index:
            issues.append({"Date": idx.date(), "Field": col, "Value": df.loc[idx, col], "Issue": "Non-positive price"})

    violations = df[
        (df["High"] < df["Low"]) | (df["High"] < df["Close"]) |
        (df["High"] < df["Open"]) | (df["Low"] > df["Close"]) | (df["Low"] > df["Open"])
    ]
    for idx in violations.index:
        issues.append({"Date": idx.date(), "Field": "OHLC Logic",
                       "Value": f"O={df.loc[idx,'Open']:.0f} H={df.loc[idx,'High']:.0f} L={df.loc[idx,'Low']:.0f} C={df.loc[idx,'Close']:.0f}",
                       "Issue": "High < Low or OHLC violation"})

    zero_vol = df[df["Volume"] == 0]
    for idx in zero_vol.index:
        issues.append({"Date": idx.date(), "Field": "Volume", "Value": 0, "Issue": "Zero volume on trading day"})

    df["ret"] = df["Close"].pct_change()
    extreme = df[df["ret"].abs() > 0.20].dropna()
    for idx in extreme.index:
        issues.append({"Date": idx.date(), "Field": "Return",
                       "Value": f"{df.loc[idx,'ret']*100:.1f}%",
                       "Issue": "Extreme move >20% (verify circuit breaker)"})

    price_changes = df["Close"] != df["Close"].shift(1)
    stale = price_changes.rolling(5).sum()
    stale_days = df[stale == 0]
    for idx in stale_days.index:
        issues.append({"Date": idx.date(), "Field": "Close",
                       "Value": df.loc[idx, "Close"],
                       "Issue": "Price unchanged for 5+ days (stale feed)"})

    dupes = df[df.index.duplicated(keep=False)]
    for idx in dupes.index:
        issues.append({"Date": idx.date(), "Field": "Index", "Value": "DUPLICATE", "Issue": "Duplicate timestamp"})

    return pd.DataFrame(issues)


def render():
    render_hero(
        title="Identifying & Treating Invalid Values",
        subtitle="OHLC Logic Checks · Negative Prices · Stale Data · Duplicate Timestamps",
        badges=[
            {"text": "OHLC Validation",   "style": "red"},
            {"text": "Stale Detection",   "style": "gold"},
            {"text": "Circuit Breakers",  "style": "blue"},
        ],
    )

    tabs = st.tabs(["📘 Taxonomy", "🔬 OHLCV Validator", "📋 Treatment Guide", "⚙️ Live Code"])

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        section_header("📘", "What Are Invalid Values?")

        defn_box(
            "Invalid Value",
            "A non-null observation that violates one or more of: "
            "<b>Domain constraints</b> (Price ≤ 0), "
            "<b>Logical constraints</b> (High < Low in OHLC), "
            "<b>Referential integrity</b> (Trade date after settlement date), "
            "<b>Business rules</b> (Volume = 0 during trading session), or "
            "<b>Relational constraints</b> (Assets ≠ Liabilities + Equity).",
        )

        section_header("📋", "Taxonomy of Invalid Values in Finance")

        taxonomy = [
            ("Negative Prices", "NSE price = −50", "Data feed error; sign flip", "df[col] > 0 assertion", "red"),
            ("Impossible OHLC", "High=2800, Low=3100", "Data entry / feed error", "H ≥ max(O,C) ≥ L check", "red"),
            ("Zero Volume on Trading Day", "Volume = 0 on active day", "Missing trade aggregation", "Market calendar cross-check", "orange"),
            ("Crossed Bid-Ask", "Bid=105 > Ask=103", "Quote data latency / error", "bid < ask assertion", "orange"),
            ("Future-Dated Entries", "Trade date in 2026 in 2024 DB", "System clock error", "Max date validation", "orange"),
            ("Accounting Identity Breach", "Assets ≠ Liabilities + Equity", "Consolidation error", "Cross-field formula check", "red"),
            ("Percentage > 100%", "Ownership = 150%", "Data entry error", "Range validation", "orange"),
            ("Duplicate Timestamps", "Two entries same tick/time", "Double-firing; race condition", "Deduplication check", "orange"),
            ("Stale Prices", "30 days zero-change in liquid stock", "Feed disconnection", "Rolling std-dev ≈ 0 alert", "orange"),
        ]

        for name, example, cause, detection, severity in taxonomy:
            colour = "#dc3545" if severity == "red" else "#fd7e14"
            st.markdown(
                f"""
<div class="mp-card" style="border-left:4px solid {colour}; margin-bottom:0.6rem; padding:0.9rem 1.2rem;">
  <div style="display:flex; gap:1.5rem; flex-wrap:wrap; align-items:flex-start;">
    <div style="min-width:180px;">
      <div style="font-weight:700; color:{colour}; font-size:0.9rem;">{name}</div>
      <div style="font-family:'Fira Code',monospace; font-size:0.78rem; color:#ADD8E6; margin-top:0.2rem;">{example}</div>
    </div>
    <div style="flex:1; min-width:150px;">
      <div style="font-size:0.78rem; color:#8892b0; text-transform:uppercase; letter-spacing:0.05em;">Root Cause</div>
      <div style="font-size:0.85rem; color:#e6f1ff;">{cause}</div>
    </div>
    <div style="flex:1; min-width:150px;">
      <div style="font-size:0.78rem; color:#8892b0; text-transform:uppercase; letter-spacing:0.05em;">Detection</div>
      <div style="font-family:'Fira Code',monospace; font-size:0.78rem; color:#6BAED6;">{detection}</div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        section_header("🔬", "Interactive OHLCV Validator")

        col1, col2, col3 = st.columns(3)
        with col1:
            n_days = st.slider("Trading Days", 30, 252, 80, key="iv_n")
            neg_price_day = st.checkbox("Inject: Negative Price (Day 6)", value=True)
        with col2:
            ohlc_violation = st.checkbox("Inject: OHLC Violation (Day 11)", value=True)
            zero_vol_day   = st.checkbox("Inject: Zero Volume (Day 16)", value=True)
        with col3:
            extreme_drop  = st.checkbox("Inject: Extreme Drop −99% (Day 21)", value=True)
            stale_prices  = st.checkbox("Inject: Stale Prices (Days 31–35)", value=True)

        rng = np.random.default_rng(99)
        dates = pd.bdate_range("2024-01-02", periods=n_days)
        close = 2800 + np.cumsum(rng.normal(0, 30, n_days))

        df_test = pd.DataFrame({
            "Open":   close * (1 + rng.normal(0, 0.005, n_days)),
            "High":   close * (1 + np.abs(rng.normal(0, 0.008, n_days))),
            "Low":    close * (1 - np.abs(rng.normal(0, 0.008, n_days))),
            "Close":  close,
            "Volume": rng.integers(500_000, 3_000_000, n_days),
        }, index=dates)

        if neg_price_day and n_days >= 6:
            df_test.loc[dates[5], "Close"] = -500
        if ohlc_violation and n_days >= 11:
            df_test.loc[dates[10], "High"] = df_test.loc[dates[10], "Low"] - 100
        if zero_vol_day and n_days >= 16:
            df_test.loc[dates[15], "Volume"] = 0
        if extreme_drop and n_days >= 21:
            df_test.loc[dates[20], "Close"] *= 0.01
        if stale_prices and n_days >= 36:
            for d in dates[30:35]:
                df_test.loc[d, "Close"] = df_test.loc[dates[29], "Close"]

        issues_df = validate_ohlcv(df_test)

        metric_row([
            {"val": str(len(issues_df)),                             "lbl": "Total Issues Found"},
            {"val": str((issues_df["Issue"].str.contains("Non-positive") if len(issues_df) > 0 else pd.Series([])).sum()), "lbl": "Negative Prices"},
            {"val": str((issues_df["Issue"].str.contains("OHLC") if len(issues_df) > 0 else pd.Series([])).sum()),        "lbl": "OHLC Violations"},
            {"val": str((issues_df["Issue"].str.contains("Extreme") if len(issues_df) > 0 else pd.Series([])).sum()),     "lbl": "Extreme Moves"},
            {"val": str((issues_df["Issue"].str.contains("stale") if len(issues_df) > 0 else pd.Series([])).sum()),       "lbl": "Stale Price Days"},
        ])

        # ── Price chart with anomaly markers ─────────────────────────────────
        fig = go.Figure()
        close_series = df_test["Close"].copy()
        display_close = close_series.abs().clip(upper=10000)  # avoid −500 distorting scale

        fig.add_trace(go.Scatter(x=df_test.index, y=display_close, mode="lines+markers",
                                  line=dict(color="#ADD8E6", width=1.5),
                                  marker=dict(size=4), name="Close Price"))

        if len(issues_df) > 0:
            for _, row in issues_df.iterrows():
                try:
                    date = pd.Timestamp(row["Date"])
                    if date in df_test.index:
                        fig.add_vline(x=date, line_dash="dash", line_color="#dc3545", opacity=0.4)
                except Exception:
                    pass

        fig.update_layout(title="RELIANCE.NS — Price with Detected Anomalies (Red lines = issues)",
                          xaxis_title="Date", yaxis_title="Close (₹)", height=350)
        st.plotly_chart(fig, use_container_width=True)

        if len(issues_df) > 0:
            st.subheader("⚠️ Validation Issues Found")
            st.dataframe(issues_df, use_container_width=True, hide_index=True)
        else:
            success_box("✅ All validations PASSED — no issues found in the dataset!")

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        section_header("📋", "Treatment Decision Guide")

        treatments = [
            ("Convert to NaN", "Clear data error (negative price, impossible value)", "df[mask] = np.nan; then apply imputation", "green"),
            ("Rule-based Correction", "Known systematic error (sign flip, decimal error)", "df['Price'] = df['Price'].abs()", "green"),
            ("Replace with Adjacent Valid Value", "OHLC logic violation where correction is obvious", "Forward-fill or arithmetic correction", "blue"),
            ("Flag and Quarantine", "Unknown origin; needs human review before model entry", "Create is_suspect column; exclude from model", "gold"),
            ("Winsorise", "Valid but extreme values in cross-sectional analysis", "Clip to percentile bounds (1%–99%)", "blue"),
            ("Consult Source", "Ambiguous case requiring cross-verification", "Cross-check NSE/BSE/Bloomberg; log action taken", "gold"),
        ]

        for treat, when, impl, style in treatments:
            colour = {"green": "#28a745", "blue": "#6BAED6", "gold": "#FFD700"}[style]
            st.markdown(
                f"""
<div class="mp-card" style="border-left:4px solid {colour}; margin-bottom:0.7rem;">
  <div style="font-weight:700; color:{colour}; margin-bottom:0.4rem;">🔧 {treat}</div>
  <div style="display:flex; gap:2rem; flex-wrap:wrap;">
    <div style="flex:1; min-width:200px;">
      <div style="font-size:0.75rem; color:#8892b0; text-transform:uppercase; letter-spacing:0.05em;">When Appropriate</div>
      <div style="font-size:0.87rem; color:#e6f1ff; margin-top:0.2rem;">{when}</div>
    </div>
    <div style="flex:1; min-width:200px;">
      <div style="font-size:0.75rem; color:#8892b0; text-transform:uppercase; letter-spacing:0.05em;">Implementation</div>
      <div style="font-family:'Fira Code',monospace; font-size:0.8rem; color:#6BAED6; margin-top:0.2rem;">{impl}</div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        section_header("📊", "Data Type Validation Quick Reference")

        dtype_table = pd.DataFrame({
            "Financial Variable": ["Price / NAV", "Volume", "Date", "ISIN / Ticker", "PE Ratio", "Rating", "Boolean Flags", "Currency Amount"],
            "Correct dtype": ["float64", "int64", "datetime64[ns]", "string / category", "float64", "category (ordered)", "bool", "float64"],
            "Common Error": ["Stored as object (string with commas)", "Float with decimals", "String 'DD-Mon-YY'", "Mixed case", "'N/A' stored as string", "object string", "0/1 integer or 'Y'/'N'", "Mixed Rs/USD"],
            "Fix": ["pd.to_numeric(); str.replace(',')", ".astype('Int64')", "pd.to_datetime()", ".str.upper().str.strip()", "pd.to_numeric(errors='coerce')", "pd.Categorical(...,ordered=True)", "Map and cast explicitly", "Currency column + normalise"],
        })
        st.dataframe(dtype_table, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        section_header("⚙️", "OHLCV Validation — Python Implementation")

        st.code("""
import pandas as pd
import numpy as np

def validate_ohlcv(df, symbol="STOCK"):
    \"\"\"
    Comprehensive OHLCV validation for financial data.
    Returns DataFrame of validation failures.
    \"\"\"
    issues = []
    
    # 1. Negative or zero prices
    for col in ['Open', 'High', 'Low', 'Close']:
        bad = df[df[col] <= 0]
        for idx in bad.index:
            issues.append({'Date': idx, 'Field': col,
                           'Value': df.loc[idx, col],
                           'Issue': 'Non-positive price'})
    
    # 2. OHLC logic violations
    violations = df[
        (df['High'] < df['Low']) |
        (df['High'] < df['Close']) |
        (df['High'] < df['Open']) |
        (df['Low'] > df['Close']) |
        (df['Low'] > df['Open'])
    ]
    for idx in violations.index:
        issues.append({'Date': idx, 'Field': 'OHLC Logic',
                       'Issue': 'High < Low or other OHLC violation'})
    
    # 3. Zero volume on trading days
    zero_vol = df[df['Volume'] == 0]
    for idx in zero_vol.index:
        issues.append({'Date': idx, 'Field': 'Volume',
                       'Value': 0, 'Issue': 'Zero volume on trading day'})
    
    # 4. Extreme single-day price move (>20% — circuit breaker check)
    df['ret'] = df['Close'].pct_change()
    extreme_moves = df[df['ret'].abs() > 0.20].dropna()
    for idx in extreme_moves.index:
        issues.append({'Date': idx, 'Field': 'Return',
                       'Value': f"{df.loc[idx, 'ret']*100:.1f}%",
                       'Issue': 'Extreme move >20% (verify circuit breaker/ex-div)'})
    
    # 5. Stale price detection (unchanged for 5+ consecutive days)
    price_changes = (df['Close'] != df['Close'].shift(1))
    stale_run = price_changes.rolling(5).sum()
    stale_days = df[stale_run == 0]
    for idx in stale_days.index:
        issues.append({'Date': idx, 'Field': 'Close',
                       'Value': df.loc[idx, 'Close'],
                       'Issue': 'Price unchanged for 5+ days (stale/data issue)'})
    
    # 6. Duplicate timestamps
    dupes = df[df.index.duplicated(keep=False)]
    for idx in dupes.index:
        issues.append({'Date': idx, 'Field': 'Index',
                       'Value': 'DUPLICATE', 'Issue': 'Duplicate timestamp'})
    
    return pd.DataFrame(issues)

# Usage
issues_df = validate_ohlcv(df_nse, "RELIANCE.NS")
print(f"Total issues: {len(issues_df)}")
""", language="python")

    render_footer()
