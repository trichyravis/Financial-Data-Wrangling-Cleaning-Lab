"""Missing Data — Classification, Diagnosis & Treatment"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from utils.components import (
    render_hero, render_footer, section_header,
    defn_box, insight_box, warning_box, success_box, metric_row,
)
from utils.data_gen import make_nse_prices, inject_missing
import utils.theme  # noqa: F401 — registers template


def render():
    render_hero(
        title="Missing Data — Diagnosis & Treatment",
        subtitle="Classification · Detection · Imputation Strategies for Financial Datasets",
        badges=[
            {"text": "MCAR", "style": "green"},
            {"text": "MAR",  "style": "gold"},
            {"text": "MNAR", "style": "red"},
        ],
    )

    tabs = st.tabs(["📘 Theory", "🔬 Interactive Diagnosis", "🛠️ Imputation Methods", "📐 Method Comparison"])

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        section_header("📘", "The Three Mechanisms of Missingness")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
<div class="mp-card-blue" style="height:100%;">
  <div style="text-align:center; font-size:1.6rem;">🎲</div>
  <h3>MCAR</h3>
  <div style="font-size:0.82rem; color:#8892b0; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.8rem;">Missing Completely at Random</div>
  <div class="badge badge-green" style="margin-bottom:0.8rem;">Best Case</div>
  <div style="font-size:0.9rem; color:#e6f1ff; margin-bottom:0.8rem;">
    Missingness is <b>unrelated</b> to any observed or unobserved data.
  </div>
  <div style="background:rgba(10,15,30,0.5); border-radius:8px; padding:0.8rem; font-size:0.82rem; color:#8892b0; font-family:'Fira Code',monospace;">
    P(missing) = P(missing | X_obs, X_miss)
  </div>
  <div style="margin-top:0.8rem; font-size:0.83rem;">
    <b style="color:#ADD8E6;">NSE Example:</b> Random packet-loss on 0.1% of trading days due to network errors.
  </div>
  <div style="margin-top:0.8rem; font-size:0.83rem;">
    <b style="color:#28a745;">Fix:</b> Any imputation method produces unbiased estimates. Listwise deletion valid.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
<div class="mp-card-gold" style="height:100%;">
  <div style="text-align:center; font-size:1.6rem;">📊</div>
  <h3>MAR</h3>
  <div style="font-size:0.82rem; color:#8892b0; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.8rem;">Missing at Random</div>
  <div class="badge badge-orange" style="margin-bottom:0.8rem;">Manageable</div>
  <div style="font-size:0.9rem; color:#e6f1ff; margin-bottom:0.8rem;">
    Missingness depends on <b>observed data</b>, not on the missing value itself.
  </div>
  <div style="background:rgba(10,15,30,0.5); border-radius:8px; padding:0.8rem; font-size:0.82rem; color:#8892b0; font-family:'Fira Code',monospace;">
    P(missing | X_obs, X_miss) = P(missing | X_obs)
  </div>
  <div style="margin-top:0.8rem; font-size:0.83rem;">
    <b style="color:#ADD8E6;">NSE Example:</b> Small-caps more likely to have missing analyst EPS estimates (missingness related to market cap, not EPS itself).
  </div>
  <div style="margin-top:0.8rem; font-size:0.83rem;">
    <b style="color:#FFD700;">Fix:</b> MICE, KNN, or model-based imputation using observed predictors.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                """
<div class="mp-card" style="height:100%; border-color:rgba(220,53,69,0.3);">
  <div style="text-align:center; font-size:1.6rem;">🚨</div>
  <h3 style="color:#dc3545 !important;">MNAR</h3>
  <div style="font-size:0.82rem; color:#8892b0; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.8rem;">Missing Not at Random</div>
  <div class="badge badge-red" style="margin-bottom:0.8rem;">Most Dangerous</div>
  <div style="font-size:0.9rem; color:#e6f1ff; margin-bottom:0.8rem;">
    Missingness depends on the <b>missing value itself</b>.
  </div>
  <div style="background:rgba(10,15,30,0.5); border-radius:8px; padding:0.8rem; font-size:0.82rem; color:#8892b0; font-family:'Fira Code',monospace;">
    P(missing | X_obs, X_miss) depends on X_miss
  </div>
  <div style="margin-top:0.8rem; font-size:0.83rem;">
    <b style="color:#ADD8E6;">NSE Example:</b> Company revenue missing <em>because revenues were catastrophically low</em> and management delayed filing.
  </div>
  <div style="margin-top:0.8rem; font-size:0.83rem;">
    <b style="color:#dc3545;">Fix:</b> No standard statistical fix. Requires domain expertise + sensitivity analysis.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("📋", "Patterns of Missing Data in Financial Datasets")

        patterns = {
            "Pattern": ["Monotone Missingness", "Block Missingness", "Intermittent", "Unit Non-Response", "Item Non-Response", "Censored Data"],
            "Description": [
                "Once missing, stays missing",
                "Entire block/period missing",
                "Scattered missing values",
                "Entire rows/entities missing",
                "Specific variables missing",
                "Value exists but truncated",
            ],
            "Financial Context": [
                "Company delisted or defaults",
                "Market closure / data feed outage",
                "Holidays, thin trading days",
                "No ESG data for a sector",
                "No PE ratio for unprofitable firms",
                "Credit score below 300 not reported",
            ],
            "Detection Method": [
                "Sort by time; check tail",
                "Heatmap of missingness",
                "df.isnull().sum()",
                "Cross-sectional coverage",
                "Column-level null count",
                "Distribution analysis",
            ],
        }
        df_patterns = pd.DataFrame(patterns)
        st.dataframe(df_patterns, use_container_width=True, hide_index=True)

        section_header("🔀", "Decision Framework — Choosing Your Strategy")

        decision = {
            "% Missing": ["< 5%", "5–15%", "> 15%", "Any", "> 50%", "Monotone tail"],
            "Mechanism": ["MCAR", "MAR", "MAR", "MNAR", "Any", "Known event"],
            "Variable Type": ["Price/Return", "Financial ratio", "Any", "Any", "Any", "Price"],
            "Recommended Strategy": [
                "Forward-fill or mean imputation",
                "Regression / KNN imputation",
                "Multiple Imputation (MICE)",
                "Domain-adjusted; sensitivity analysis",
                "Exclude variable; flag and document",
                "Mark as delisted/terminated; exclude",
            ],
        }
        df_dec = pd.DataFrame(decision)
        st.dataframe(df_dec, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        section_header("🔬", "Interactive Missing Data Diagnosis")

        col_ctrl, col_main = st.columns([1, 3])
        with col_ctrl:
            n_obs = st.slider("Trading Days", 100, 504, 252, step=10)
            block_pct = st.slider("Block Missing %", 0, 20, 5)
            random_pct = st.slider("Random Missing %", 0, 20, 8)
            monotone_tail = st.slider("Monotone Tail (days)", 0, 80, 30)

        df_raw = make_nse_prices(n=n_obs)
        df_missing = df_raw.copy()

        # Inject user-controlled missing
        if block_pct > 0:
            block_size = max(2, int(n_obs * block_pct / 100))
            df_missing.iloc[50 : 50 + block_size, 0] = np.nan

        if random_pct > 0:
            rng = np.random.default_rng(7)
            idx = rng.choice(n_obs, int(n_obs * random_pct / 100), replace=False)
            df_missing.iloc[idx, 1] = np.nan

        if monotone_tail > 0:
            df_missing.iloc[-monotone_tail :, 2] = np.nan

        total_missing = df_missing.isnull().sum().sum()
        total_cells = df_missing.size
        pct_missing = total_missing / total_cells * 100

        metric_row([
            {"val": f"{pct_missing:.1f}%", "lbl": "Overall Missing"},
            {"val": str(total_missing),     "lbl": "Missing Cells"},
            {"val": str(n_obs),             "lbl": "Trading Days"},
            {"val": "5",                    "lbl": "Stocks"},
        ])

        with col_main:
            # ── Heatmap ──────────────────────────────────────────────────────
            z = df_missing.isnull().astype(int).values.T
            fig = go.Figure(
                go.Heatmap(
                    z=z,
                    x=[str(d.date()) for d in df_missing.index],
                    y=df_missing.columns.tolist(),
                    colorscale=[[0, "#0f1a2e"], [1, "#FFD700"]],
                    showscale=True,
                    colorbar=dict(title="Missing", tickvals=[0, 1], ticktext=["Present", "Missing"]),
                )
            )
            fig.update_layout(
                title="Missing Data Heatmap (Yellow = Missing)",
                height=280,
                xaxis=dict(showticklabels=False),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Per-column diagnostics ────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            miss_pct = (df_missing.isnull().sum() / len(df_missing) * 100).reset_index()
            miss_pct.columns = ["Stock", "Pct_Missing"]
            miss_pct["colour"] = miss_pct["Pct_Missing"].apply(
                lambda v: "#28a745" if v < 5 else ("#FFD700" if v < 15 else "#dc3545")
            )
            fig2 = go.Figure(
                go.Bar(
                    x=miss_pct["Pct_Missing"],
                    y=miss_pct["Stock"],
                    orientation="h",
                    marker_color=miss_pct["colour"].tolist(),
                    text=[f"{v:.1f}%" for v in miss_pct["Pct_Missing"]],
                    textposition="outside",
                )
            )
            fig2.add_vline(x=5, line_dash="dash", line_color="#dc3545", annotation_text="5% threshold")
            fig2.update_layout(title="Missing % by Stock", height=300, xaxis_title="% Missing")
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            cum_miss = df_missing.isnull().sum(axis=1).cumsum()
            fig3 = go.Figure(
                go.Scatter(
                    x=df_missing.index,
                    y=cum_miss,
                    fill="tozeroy",
                    line=dict(color="#FFD700", width=2),
                    fillcolor="rgba(255,215,0,0.08)",
                )
            )
            fig3.update_layout(title="Cumulative Missing Observations Over Time", height=300)
            st.plotly_chart(fig3, use_container_width=True)

        # ── Diagnostic report table ───────────────────────────────────────────
        st.subheader("📋 Diagnostic Report")
        report = pd.DataFrame({
            "Total Missing": df_missing.isnull().sum(),
            "% Missing": (df_missing.isnull().sum() / len(df_missing) * 100).round(2),
            "Max Consecutive": df_missing.isnull().apply(
                lambda col: col.groupby((col != col.shift()).cumsum()).sum().max()
            ),
            "Likely Mechanism": ["Block MCAR", "Random MCAR/MAR", "Monotone MNAR", "MAR", "Clean"],
        })
        st.dataframe(report, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        section_header("🛠️", "Imputation Methods — Live Comparison")

        st.markdown("**Select a stock and method to see imputation in action:**")

        df_raw2 = make_nse_prices(n=252)
        df_miss2 = inject_missing(df_raw2)

        col1, col2 = st.columns(2)
        with col1:
            stock = st.selectbox("Stock", df_miss2.columns.tolist(), index=1)
        with col2:
            method = st.selectbox("Imputation Method", ["Forward Fill (LOCF)", "Backward Fill (NOCB)", "Linear Interpolation", "Spline Interpolation", "Mean Imputation", "Median Imputation", "KNN Imputation"])

        s_raw  = df_miss2[stock]
        s_true = df_raw2[stock]

        if method == "Forward Fill (LOCF)":
            s_imp = s_raw.ffill()
        elif method == "Backward Fill (NOCB)":
            s_imp = s_raw.bfill()
        elif method == "Linear Interpolation":
            s_imp = s_raw.interpolate(method="linear")
        elif method == "Spline Interpolation":
            s_imp = s_raw.interpolate(method="spline", order=3)
        elif method == "Mean Imputation":
            s_imp = s_raw.fillna(s_raw.mean())
        elif method == "Median Imputation":
            s_imp = s_raw.fillna(s_raw.median())
        else:  # KNN
            scaler = StandardScaler()
            X = df_miss2.values
            X_sc = scaler.fit_transform(X)
            knn = KNNImputer(n_neighbors=7)
            X_imp = scaler.inverse_transform(knn.fit_transform(X_sc))
            df_imp = pd.DataFrame(X_imp, columns=df_miss2.columns, index=df_miss2.index)
            s_imp = df_imp[stock]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s_true.index, y=s_true, name="True Price", line=dict(color="#28a745", width=1.5, dash="dot"), opacity=0.6))
        fig.add_trace(go.Scatter(x=s_raw.index,  y=s_raw,  name="Observed (with gaps)", line=dict(color="#ADD8E6", width=2)))
        fig.add_trace(go.Scatter(x=s_imp.index,  y=s_imp,  name=f"Imputed ({method})", line=dict(color="#FFD700", width=2)))

        # Highlight missing zones
        missing_mask = s_raw.isnull()
        for i in range(len(missing_mask) - 1):
            if missing_mask.iloc[i]:
                fig.add_vrect(
                    x0=s_raw.index[i], x1=s_raw.index[min(i + 1, len(s_raw) - 1)],
                    fillcolor="rgba(220,53,69,0.08)", line_width=0,
                )

        fig.update_layout(
            title=f"{stock} — {method} Imputation",
            xaxis_title="Date", yaxis_title="Price (₹)",
            height=420, legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Return distortion warning
        returns_raw  = s_imp.pct_change()
        zero_ret_pct = (returns_raw == 0).mean() * 100
        if method in ["Forward Fill (LOCF)", "Backward Fill (NOCB)"]:
            warning_box(
                f"<b>Forward/Backward Fill Distortion:</b> {zero_ret_pct:.1f}% of returns are exactly zero — "
                "artificial staleness that understates true volatility. "
                "<b>Never use forward-filled prices for return-based risk models!</b>"
            )
        elif method == "Mean Imputation":
            warning_box(
                f"<b>Mean Imputation Warning:</b> All {s_raw.isnull().sum()} missing values set to "
                f"₹{s_raw.mean():.1f}. This <b>reduces variance</b> of the imputed series, "
                "biasing VaR calculations and correlation estimates."
            )
        else:
            success_box(
                f"<b>{method}</b> preserves the temporal structure better than simple fill methods. "
                "Still verify imputed values against adjacent observations and market context."
            )

        # MICE description
        with st.expander("📐 MICE — Multiple Imputation by Chained Equations (Gold Standard)"):
            st.markdown("""
The **MICE algorithm** is the gold standard for handling missing data in research-grade financial analysis:

1. Fill missing values with simple initial estimates (e.g., mean)
2. For each variable $X_j$ with missing values: fit a regression model using all other variables as predictors
3. Draw imputations from the posterior predictive distribution
4. Cycle through all variables $m$ times (typically $m$ = 5–20 iterations)
5. Create $M$ complete datasets (typically $M$ = 5–10)
6. Run analysis on each complete dataset separately
7. **Combine using Rubin's Rules:**

$$\\hat{\\theta} = \\frac{1}{M}\\sum_{i=1}^{M}\\hat{\\theta}_i$$

$$Var(\\hat{\\theta}) = \\underbrace{\\frac{1}{M}\\sum W_i}_{\\text{within-imputation}} + \\underbrace{\\left(1 + \\frac{1}{M}\\right)\\cdot B}_{\\text{between-imputation}}$$

**Python:** `sklearn.impute.IterativeImputer` (experimental) or `miceforest` package.
""")

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        section_header("📐", "Method Comparison — Distribution Impact")

        df_raw3 = make_nse_prices(n=252)
        df_miss3 = inject_missing(df_raw3)
        stock3 = st.selectbox("Select Stock for Comparison", df_miss3.columns, index=0, key="comp_stock")

        s_raw3 = df_miss3[stock3]
        methods_out = {
            "True (no missing)":       df_raw3[stock3].pct_change().dropna(),
            "Forward Fill":            s_raw3.ffill().pct_change().dropna(),
            "Linear Interpolation":    s_raw3.interpolate("linear").pct_change().dropna(),
            "Mean Imputation":         s_raw3.fillna(s_raw3.mean()).pct_change().dropna(),
        }

        fig = make_subplots(rows=1, cols=4, subplot_titles=list(methods_out.keys()), shared_yaxes=True)
        colours = ["#28a745", "#FFD700", "#ADD8E6", "#dc3545"]
        for i, (name, rets) in enumerate(methods_out.items(), 1):
            fig.add_trace(
                go.Histogram(x=rets, name=name, marker_color=colours[i-1], opacity=0.75, nbinsx=40, showlegend=False),
                row=1, col=i,
            )

        fig.update_layout(title="Return Distribution: True vs Imputed Methods", height=380)
        st.plotly_chart(fig, use_container_width=True)

        # Stats comparison
        stats_data = {}
        for name, rets in methods_out.items():
            stats_data[name] = {
                "Mean (%)":  round(rets.mean() * 100, 4),
                "Std Dev (%)": round(rets.std() * 100, 4),
                "Skewness":  round(rets.skew(), 3),
                "Kurtosis":  round(rets.kurt(), 3),
                "Zero Returns": int((rets == 0).sum()),
            }
        st.dataframe(pd.DataFrame(stats_data).T, use_container_width=True)

        insight_box(
            "Mean imputation produces the <b>lowest variance</b> (visible in the narrower histogram). "
            "This directly understates VaR and portfolio risk. "
            "Forward Fill creates a spike at zero returns (stale price effect). "
            "Linear Interpolation best approximates the true distribution shape."
        )

    render_footer()
