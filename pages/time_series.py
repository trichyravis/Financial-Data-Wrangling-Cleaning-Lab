"""Time Series Formatting — DatetimeIndex, Resampling, Corporate Actions"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
import streamlit as st

from utils.components import (
    render_hero, render_footer, section_header,
    defn_box, insight_box, warning_box, success_box,
)
from utils.data_gen import GSEC_MATURITIES, GSEC_YIELDS
import utils.theme  # noqa


def render():
    render_hero(
        title="Time Series Formatting for Financial Data",
        subtitle="DatetimeIndex · Business Calendars · Resampling · Corporate Actions · Timezone Handling",
        badges=[
            {"text": "NSE Calendar",    "style": "blue"},
            {"text": "OHLCV Resample",  "style": "gold"},
            {"text": "Yield Curve",     "style": "green"},
        ],
    )

    tabs = st.tabs(["📘 Five Dimensions", "📅 DatetimeIndex Builder", "📈 OHLCV Resampler", "🎯 Yield Curve Interpolation", "🌍 Timestamp Pipeline"])

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        section_header("📘", "Five Dimensions of Financial Time Series Formatting")

        dimensions = [
            ("1️⃣", "Index Type", "#ADD8E6",
             "DatetimeIndex MUST be used — string-based date indices cause silent errors in resampling and rolling calculations.",
             "❌ Wrong: df.index = ['2024-01-15', '2024-01-16']  ✅ Right: df.index = pd.to_datetime([...])"),
            ("2️⃣", "Timezone Handling", "#FFD700",
             "NSE trades IST (UTC+5:30); NYSE trades ET (UTC-5/UTC-4). Cross-market analysis requires tz-aware timestamps.",
             "pd.date_range('2024-01-15 09:15', freq='min', tz='Asia/Kolkata').tz_convert('America/New_York')"),
            ("3️⃣", "Business Calendar", "#ADD8E6",
             "Market holidays must be explicitly accounted for. pd.bdate_range() uses Mon–Fri but misses Indian holidays (Holi, Diwali, etc.)",
             "Use CustomBusinessDay with NSEHolidayCalendar for accurate trading day counts (~244/year)."),
            ("4️⃣", "Frequency", "#FFD700",
             "Tick, minute, hourly, daily, weekly, monthly, quarterly — each requires appropriate aggregation rules.",
             "OHLCV: Open→first, High→max, Low→min, Close→last, Volume→sum. Returns must be COMPOUNDED not summed."),
            ("5️⃣", "Corporate Actions", "#ADD8E6",
             "Price discontinuities from dividends, splits, bonuses, and rights issues create spurious return signals.",
             "Always use backward-adjusted close prices. Unadjusted TCS 1:1 bonus creates apparent −50% return."),
        ]

        for num, title, color, body, example in dimensions:
            st.markdown(
                f"""
<div class="mp-card" style="margin-bottom:0.9rem;">
  <div style="display:flex; align-items:flex-start; gap:1rem;">
    <div style="font-size:1.6rem; flex-shrink:0;">{num}</div>
    <div style="flex:1;">
      <div style="font-weight:700; color:{color}; font-size:1rem; margin-bottom:0.3rem;">{title}</div>
      <div style="font-size:0.88rem; color:#e6f1ff; margin-bottom:0.5rem;">{body}</div>
      <div style="font-family:'Fira Code',monospace; font-size:0.78rem; background:#060d1a;
                  padding:0.5rem 0.8rem; border-radius:6px; color:#6BAED6; border:1px solid rgba(100,140,200,0.15);">
        {example}
      </div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        # ── Resampling rules table ────────────────────────────────────────────
        section_header("📋", "Resampling Aggregation Rules")
        rules = pd.DataFrame({
            "Data Type": ["Open Price", "High Price", "Low Price", "Close Price", "Volume", "Returns", "Ratios (PE, EV)", "Macro (GDP)"],
            "Aggregation Rule": ["First observation", "Maximum", "Minimum", "Last observation", "Sum", "Compound: ∏(1+r) − 1", "Last or mean", "Last or sum"],
            "Pandas Method": [".first()", ".max()", ".min()", ".last()", ".sum()", "custom compound()", ".last()", "context-dependent"],
            "Rationale": [
                "Opening price of the period",
                "Highest point reached",
                "Lowest point reached",
                "Closing price of the period",
                "Total volume traded",
                "Compound, not sum",
                "Period-end valuation",
                "Annual → quarterly: sum",
            ],
        })
        st.dataframe(rules, use_container_width=True, hide_index=True)
        warning_box("Never sum returns when resampling! Use compound returns: (1 + r₁)(1 + r₂)...(1 + rₙ) − 1")

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        section_header("📅", "Financial DatetimeIndex Builder")

        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input("Start Date", value=pd.Timestamp("2024-01-01"))
            end   = st.date_input("End Date",   value=pd.Timestamp("2024-12-31"))
        with col2:
            cal_type = st.selectbox("Calendar Type", ["Mon–Fri (Standard Business Days)", "NSE Custom Holiday Calendar"])
            tz = st.selectbox("Timezone", ["Asia/Kolkata (IST)", "America/New_York (ET)", "UTC", "Europe/London (GMT)"])

        nse_holidays_2024 = [
            "2024-01-26", "2024-03-25", "2024-04-14", "2024-04-17",
            "2024-05-01", "2024-06-17", "2024-08-15", "2024-10-02",
            "2024-11-01", "2024-11-15", "2024-12-25",
        ]

        if cal_type == "Mon–Fri (Standard Business Days)":
            date_idx = pd.bdate_range(str(start), str(end))
        else:
            from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
            from pandas.tseries.offsets import CustomBusinessDay
            class NSECal(AbstractHolidayCalendar):
                rules = [Holiday("H" + str(i), year=pd.Timestamp(d).year,
                                  month=pd.Timestamp(d).month, day=pd.Timestamp(d).day)
                         for i, d in enumerate(nse_holidays_2024)]
            cbd = CustomBusinessDay(calendar=NSECal())
            date_idx = pd.bdate_range(str(start), str(end), freq=cbd)

        tz_map = {
            "Asia/Kolkata (IST)": "Asia/Kolkata",
            "America/New_York (ET)": "America/New_York",
            "UTC": "UTC",
            "Europe/London (GMT)": "Europe/London",
        }
        selected_tz = tz_map[tz]

        st.success(f"✅ **{len(date_idx)} trading days** from {start} to {end} using {cal_type}")

        if cal_type == "NSE Custom Holiday Calendar":
            removed = len(pd.bdate_range(str(start), str(end))) - len(date_idx)
            st.info(f"📅 {removed} NSE holidays removed from Mon–Fri calendar")

        # Show cross-market timezone example
        st.markdown("**Cross-Market Timezone Conversion:**")
        nse_open = pd.Timestamp("2024-01-15 09:15:00", tz="Asia/Kolkata")
        nse_close = pd.Timestamp("2024-01-15 15:30:00", tz="Asia/Kolkata")
        target = nse_open.tz_convert(selected_tz)
        target_close = nse_close.tz_convert(selected_tz)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
<div class="mp-card-blue" style="text-align:center;">
  <div style="color:#8892b0; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em;">NSE Market Hours (IST)</div>
  <div style="font-size:1.2rem; color:#FFD700; font-weight:700; margin:0.5rem 0;">09:15 — 15:30</div>
  <div style="font-size:0.83rem; color:#ADD8E6;">Asia/Kolkata (UTC+5:30)</div>
</div>
""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
<div class="mp-card-gold" style="text-align:center;">
  <div style="color:#8892b0; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em;">{selected_tz}</div>
  <div style="font-size:1.2rem; color:#FFD700; font-weight:700; margin:0.5rem 0;">
    {target.strftime('%H:%M')} — {target_close.strftime('%H:%M')}
  </div>
  <div style="font-size:0.83rem; color:#ADD8E6;">{selected_tz}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        # Date format validation
        section_header("🔍", "Date Format Validation Demo")
        raw_dates = ["2024-01-15", "15/01/24", "2024-13-01", "N/A", "01-01-2025", "1/1/25", "Jan 15 2024", "15 Jan 2024"]
        parsed_results = []
        for d in raw_dates:
            try:
                parsed = pd.to_datetime(d, dayfirst=True)
                if parsed.year < 2000 or parsed.year > 2030:
                    status = "⚠️ Year out of range"
                else:
                    status = f"✅ {parsed.strftime('%Y-%m-%d')}"
            except Exception:
                status = "❌ Unparseable"
            parsed_results.append({"Raw Input": d, "Result": status})
        st.dataframe(pd.DataFrame(parsed_results), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        section_header("📈", "OHLCV Resampler — Interactive")

        from_freq = st.selectbox("Source Frequency", ["Daily"], index=0)
        to_freq   = st.selectbox("Resample To", ["Weekly (W-FRI)", "Monthly (ME)", "Quarterly (QE)"], index=1)
        n_days    = st.slider("Number of Trading Days", 50, 504, 252, step=10)

        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2024-01-02", periods=n_days)
        close = 21000 + np.cumsum(rng.normal(10, 150, n_days))
        df = pd.DataFrame({
            "Open":   close * (1 + rng.normal(0, 0.005, n_days)),
            "High":   close * (1 + np.abs(rng.normal(0, 0.008, n_days))),
            "Low":    close * (1 - np.abs(rng.normal(0, 0.008, n_days))),
            "Close":  close,
            "Volume": rng.integers(300_000, 2_000_000, n_days),
        }, index=dates)

        agg_map = {"Weekly (W-FRI)": "W-FRI", "Monthly (ME)": "ME", "Quarterly (QE)": "QE"}
        freq_code = agg_map[to_freq]
        resampled = df.resample(freq_code).agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
        }).dropna()

        def compound_return(s):
            return (1 + s).prod() - 1

        daily_ret = df["Close"].pct_change().dropna()
        resampled_ret = daily_ret.resample(freq_code).apply(compound_return).dropna()
        resampled["Return (%)"] = (resampled_ret * 100).round(2)

        # OHLCV chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=resampled.index,
            open=resampled["Open"], high=resampled["High"],
            low=resampled["Low"],  close=resampled["Close"],
            name="NIFTY 50",
            increasing_line_color="#28a745", decreasing_line_color="#dc3545",
        ))
        fig.update_layout(
            title=f"NIFTY 50 — {to_freq} OHLCV Candlestick",
            xaxis_title="Period", yaxis_title="Index Value", height=420,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"📋 {to_freq} OHLCV (first 8 periods)")
            st.dataframe(resampled[["Open", "High", "Low", "Close", "Volume"]].head(8).round(1),
                         use_container_width=True)
        with col2:
            st.subheader(f"📊 Compounded Returns")
            fig2 = go.Figure(go.Bar(
                x=resampled_ret.index, y=resampled_ret * 100,
                marker_color=["#28a745" if r >= 0 else "#dc3545" for r in resampled_ret],
            ))
            fig2.update_layout(title="Correctly Compounded Returns (%)", height=280, xaxis_title="Period", yaxis_title="%")
            st.plotly_chart(fig2, use_container_width=True)

        success_box("Returns are <b>compounded</b> correctly using ∏(1+rᵢ) − 1. Summing daily returns would overstate multi-period performance.")

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        section_header("🎯", "G-Sec Yield Curve — Cubic Spline Interpolation")

        st.markdown("**Interpolate missing maturities on the Indian Government Securities yield curve:**")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("""
<div class="mp-card-blue" style="font-size:0.85rem;">
<b style="color:#ADD8E6;">Observed G-Sec Maturities</b><br><br>
<table style="width:100%; font-size:0.82rem;">
<tr><th style="color:#FFD700; text-align:left;">Tenor (Yr)</th><th style="color:#FFD700; text-align:left;">Yield (%)</th></tr>
""" + "".join(f"<tr><td style='color:#e6f1ff;'>{m}</td><td style='color:#28a745;'>{y:.2f}%</td></tr>"
               for m, y in zip(GSEC_MATURITIES, GSEC_YIELDS)) + """
</table>
</div>
""", unsafe_allow_html=True)

        with col2:
            cs = CubicSpline(GSEC_MATURITIES, GSEC_YIELDS)
            dense = np.linspace(0.25, 30, 300)
            y_dense = cs(dense)

            missing_tenors = st.multiselect(
                "Select missing tenors to interpolate:",
                [4, 6, 8, 9, 11, 12, 15, 18, 20, 25],
                default=[4, 6, 8, 12, 20],
            )
            y_missing = cs(np.array(missing_tenors)) if missing_tenors else []

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dense, y=y_dense, mode="lines",
                                     line=dict(color="#003366", width=2.5),
                                     name="Cubic Spline Interpolation"))
            fig.add_trace(go.Scatter(x=GSEC_MATURITIES, y=GSEC_YIELDS, mode="markers",
                                     marker=dict(color="#ADD8E6", size=10, symbol="circle",
                                                  line=dict(color="#003366", width=2)),
                                     name="Observed G-Sec Yields"))
            if missing_tenors:
                fig.add_trace(go.Scatter(
                    x=list(missing_tenors), y=list(y_missing), mode="markers",
                    marker=dict(color="#FFD700", size=13, symbol="diamond",
                                 line=dict(color="#003366", width=2)),
                    name="Interpolated Tenors",
                ))
            fig.update_layout(
                title="Indian G-Sec Yield Curve — Cubic Spline Interpolation",
                xaxis_title="Maturity (Years)", yaxis_title="Yield (% p.a.)",
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

        if missing_tenors:
            col1, col2 = st.columns(2)
            with col1:
                interp_df = pd.DataFrame({
                    "Missing Tenor (Yr)": missing_tenors,
                    "Interpolated Yield (%)": [round(float(y), 4) for y in y_missing],
                })
                st.dataframe(interp_df, use_container_width=True, hide_index=True)
            with col2:
                insight_box(
                    "FIMMDA and CCIL use <b>cubic spline / Nelson-Siegel-Svensson interpolation</b> "
                    "to construct the complete G-Sec yield curve from available liquid benchmark yields. "
                    "This is standard practice for bond valuation, duration analysis, and pricing of interest rate derivatives."
                )

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[4]:
        section_header("🌍", "Timestamp Handling Pipeline")

        raw_timestamps = [
            ("2024-01-15 09:15:00", "ISO format (NSE intraday)"),
            ("15/01/2024 09:16:00", "DD/MM/YYYY (Bloomberg export)"),
            ("Jan 15 2024 9:17 AM",  "Text format (some vendors)"),
            ("1705292280",            "Unix timestamp (seconds)"),
            ("2024-01-15T09:19:00+05:30", "ISO 8601 with IST timezone"),
            ("",                      "Missing timestamp!"),
            ("2024-01-32 09:20:00",  "Invalid date!"),
        ]

        results = []
        for raw, desc in raw_timestamps:
            try:
                if not raw:
                    status = "❌ Missing (NaT)"
                    parsed_str = "NaT"
                elif raw.isdigit():
                    ts = pd.Timestamp(int(raw), unit="s", tz="UTC").tz_convert("Asia/Kolkata")
                    status = "✅ Parsed"
                    parsed_str = str(ts)
                else:
                    ts = pd.to_datetime(raw, dayfirst=True)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("Asia/Kolkata")
                    else:
                        ts = ts.tz_convert("Asia/Kolkata")
                    status = "✅ Parsed"
                    parsed_str = str(ts)
            except Exception:
                status = "❌ Invalid"
                parsed_str = "NaT"

            results.append({"Raw Input": raw, "Description": desc, "Status": status, "IST Timestamp": parsed_str})

        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

        warning_box(
            "<b>5 of 7</b> raw timestamps parse successfully; 2 fail (missing + invalid date). "
            "Always use <code>safe_parse_timestamp()</code> with try/except to handle mixed-format inputs "
            "from multiple data vendors gracefully."
        )

    render_footer()
