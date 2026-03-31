"""Full End-to-End Financial Data Cleaning Pipeline"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import streamlit as st

from utils.components import (
    render_hero, render_footer, section_header,
    defn_box, insight_box, warning_box, success_box, metric_row,
)
import utils.theme  # noqa


class FinancialDataCleaner:
    def __init__(self, config=None):
        self.config = config or {
            "missing_threshold": 0.30,
            "winsorise_level": 0.01,
            "zscore_threshold": 4.0,
            "knn_neighbors": 7,
            "outlier_action": "winsorise",
        }
        self.cleaning_log = []

    def _log(self, step, detail):
        self.cleaning_log.append({"Step": step, "Detail": detail})

    def validate_schema(self, df):
        self._log("VALIDATE", f"Shape: {df.shape}")
        for col in df.select_dtypes(include="object"):
            n_num = pd.to_numeric(df[col], errors="coerce").notna().sum()
            if n_num > len(df) * 0.8:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                self._log("TYPE-FIX", f"Column '{col}': coerced to numeric")
        if df.index.dtype != "datetime64[ns]":
            try:
                df.index = pd.to_datetime(df.index)
                self._log("INDEX", "Converted index to DatetimeIndex")
            except Exception:
                pass
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            self._log("SORT", "Sorted index in ascending order")
        n_dupes = df.index.duplicated().sum()
        if n_dupes > 0:
            df = df[~df.index.duplicated(keep="last")]
            self._log("DEDUP", f"Removed {n_dupes} duplicate index entries")
        return df

    def treat_missing(self, df):
        miss = df.isnull().mean()
        self._log("MISSING", f"Overall missing rate: {df.isnull().values.mean()*100:.2f}%")
        drop_cols = miss[miss > self.config["missing_threshold"]].index.tolist()
        if drop_cols:
            df = df.drop(columns=drop_cols)
            self._log("DROP-COLS", f"Dropped {len(drop_cols)} columns > {self.config['missing_threshold']*100:.0f}% missing")
        for col in df.columns:
            pct = df[col].isnull().mean()
            if pct == 0:
                continue
            elif pct < 0.02:
                df[col] = df[col].ffill().bfill()
                self._log("FFILL", f"'{col}': {pct*100:.1f}% → ffill")
            elif pct >= 0.15:
                med = df[col].median()
                flag_col = f"{col}_imputed"
                df[flag_col] = df[col].isnull().astype(int)
                df[col] = df[col].fillna(med)
                self._log("MEDIAN", f"'{col}': {pct*100:.1f}% → median imputation + flag")

        mod_cols = [c for c in df.columns if 0.02 <= df[c].isnull().mean() <= 0.15
                    and df[c].dtype in ["float64", "int64"]]
        if mod_cols:
            scaler = StandardScaler()
            X = df[mod_cols]
            X_sc = pd.DataFrame(scaler.fit_transform(X), columns=mod_cols, index=df.index)
            knn = KNNImputer(n_neighbors=self.config["knn_neighbors"])
            X_imp = pd.DataFrame(
                scaler.inverse_transform(knn.fit_transform(X_sc)),
                columns=mod_cols, index=df.index,
            )
            df[mod_cols] = X_imp
            self._log("KNN", f"KNN imputed {len(mod_cols)} columns: {mod_cols}")
        return df

    def treat_outliers(self, df):
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if not c.endswith("_imputed")]
        action = self.config["outlier_action"]
        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 30:
                continue
            median = series.median()
            mad = np.median(np.abs(series - median))
            if mad == 0:
                continue
            mod_z = 0.6745 * np.abs(df[col] - median) / mad
            mask = mod_z > self.config["zscore_threshold"]
            n_out = mask.sum()
            if n_out > 0:
                if action == "winsorise":
                    lo = df[col].quantile(self.config["winsorise_level"])
                    hi = df[col].quantile(1 - self.config["winsorise_level"])
                    df[col] = df[col].clip(lo, hi)
                    self._log("WINSORISE", f"'{col}': {n_out} outliers clipped to [{lo:.2f}, {hi:.2f}]")
                elif action == "flag":
                    df[f"{col}_outlier"] = mask.astype(int)
                    self._log("FLAG", f"'{col}': {n_out} outliers flagged")
                elif action == "remove":
                    df.loc[mask, col] = np.nan
                    df[col] = df[col].ffill()
                    self._log("REMOVE", f"'{col}': {n_out} outliers set NaN + ffilled")
        return df

    def clean(self, df):
        df_clean = df.copy()
        df_clean = self.validate_schema(df_clean)
        df_clean = self.treat_missing(df_clean)
        df_clean = self.treat_outliers(df_clean)
        return df_clean


def render():
    render_hero(
        title="End-to-End Financial Data Cleaning Pipeline",
        subtitle="Production-Grade FinancialDataCleaner · Schema Validation → Missing → Outliers → Report",
        badges=[
            {"text": "Production Code", "style": "green"},
            {"text": "KNN Imputation",  "style": "blue"},
            {"text": "Winsorisation",   "style": "gold"},
        ],
    )

    tabs = st.tabs(["🏗️ Pipeline Architecture", "⚙️ Interactive Runner", "📋 Cleaning Report", "💻 Full Source Code"])

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        section_header("🏗️", "Pipeline Architecture — 4 Stages")

        stages = [
            ("1️⃣", "VALIDATE", "#28a745",
             "Schema & Structure",
             ["Detect mixed-type columns", "Ensure DatetimeIndex", "Sort by temporal order", "Remove duplicate index entries"],
             "validate_schema(df)"),
            ("2️⃣", "MISSING", "#FFD700",
             "Missing Data Treatment",
             ["Drop columns > 30% missing", "< 2% missing → forward fill", "2–15% missing → KNN imputation (k=7)", "> 15% missing → median + flag column"],
             "treat_missing(df)"),
            ("3️⃣", "OUTLIERS", "#fd7e14",
             "Outlier Treatment",
             ["Modified Z-Score (MAD) detection", "Configurable threshold (default 4.0)", "Actions: winsorise / flag / remove", "Skip imputation flag columns"],
             "treat_outliers(df)"),
            ("4️⃣", "REPORT", "#ADD8E6",
             "Cleaning Audit Trail",
             ["Shape before vs after", "Missing count before vs after", "Step-by-step cleaning log", "Documentation for model governance"],
             "cleaning_report(df_orig, df_clean)"),
        ]

        for num, step, colour, title, points, method in stages:
            st.markdown(
                f"""
<div class="mp-card" style="border-left:5px solid {colour}; margin-bottom:1rem;">
  <div style="display:flex; gap:1.2rem; align-items:flex-start;">
    <div style="font-size:1.8rem; flex-shrink:0;">{num}</div>
    <div style="flex:1;">
      <div style="display:flex; align-items:baseline; gap:0.8rem; margin-bottom:0.5rem;">
        <span style="font-family:'Fira Code',monospace; font-size:0.75rem; color:{colour};
                     background:rgba(0,0,0,0.3); padding:0.1rem 0.5rem; border-radius:4px;">{step}</span>
        <span style="font-weight:700; color:#e6f1ff; font-size:1rem;">{title}</span>
      </div>
      <ul style="margin:0; padding-left:1.4rem; color:#e6f1ff; font-size:0.87rem;">
        {''.join(f"<li style='margin-bottom:0.2rem;'>{p}</li>" for p in points)}
      </ul>
      <div style="margin-top:0.6rem; font-family:'Fira Code',monospace; font-size:0.78rem;
                  color:#6BAED6; background:#060d1a; padding:0.3rem 0.7rem; border-radius:5px; display:inline-block;">
        cleaner.{method}
      </div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        insight_box(
            "The <code>FinancialDataCleaner</code> class accepts a <b>config dictionary</b> allowing full customisation: "
            "adjust missing thresholds, KNN neighbours, winsorisation levels, and outlier actions per project. "
            "All steps are logged to <code>cleaning_log</code> for model governance documentation."
        )

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        section_header("⚙️", "Interactive Pipeline Runner")

        st.markdown("**Configure your pipeline and run it on a synthetic NSE multi-stock dataset:**")

        col1, col2, col3 = st.columns(3)
        with col1:
            miss_thresh = st.slider("Drop Column if Missing > (%)", 10, 60, 25, step=5)
            wins_level  = st.slider("Winsorisation Level (%)", 0.5, 5.0, 1.0, step=0.5)
        with col2:
            zscore_thresh = st.slider("MAD Z-Score Threshold", 2.0, 6.0, 3.5, step=0.25)
            knn_k         = st.slider("KNN Neighbours", 3, 15, 7)
        with col3:
            out_action = st.selectbox("Outlier Action", ["winsorise", "flag", "remove"])
            n_days     = st.slider("Trading Days", 100, 756, 504, step=50)

        # Generate raw dataset with problems
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2022-01-03", periods=n_days)
        n = n_days

        df_raw = pd.DataFrame({
            "RELIANCE":    2800 + np.cumsum(rng.normal(0.5, 25, n)),
            "TCS":         3500 + np.cumsum(rng.normal(0.3, 40, n)),
            "HDFCBANK":    1600 + np.cumsum(rng.normal(0.2, 18, n)),
            "PE_RELIANCE": rng.lognormal(3.4, 0.3, n),
            "PE_TCS":      rng.lognormal(3.5, 0.4, n),
        }, index=dates)

        # Inject problems
        df_raw.iloc[50:55, 0] = np.nan                        # Block missing
        idx_rand = rng.choice(n, 20, replace=False)
        df_raw.iloc[idx_rand, 2] = np.nan                     # Random missing
        df_raw.iloc[100, 0] = -2800                           # Invalid negative
        df_raw.iloc[150, 4] = 5000                            # Outlier PE
        df_raw.iloc[200, 3] = 0.001                           # Outlier low PE

        config = {
            "missing_threshold": miss_thresh / 100,
            "winsorise_level":   wins_level / 100,
            "zscore_threshold":  zscore_thresh,
            "knn_neighbors":     knn_k,
            "outlier_action":    out_action,
        }

        if st.button("▶️  Run Cleaning Pipeline", use_container_width=True):
            cleaner = FinancialDataCleaner(config=config)
            df_clean = cleaner.clean(df_raw)
            st.session_state["df_raw"]   = df_raw
            st.session_state["df_clean"] = df_clean
            st.session_state["log"]      = cleaner.cleaning_log

        if "df_clean" in st.session_state:
            df_raw2   = st.session_state["df_raw"]
            df_clean2 = st.session_state["df_clean"]
            log       = st.session_state["log"]

            orig_miss  = df_raw2.isnull().sum().sum()
            clean_miss = df_clean2.isnull().sum().sum()

            metric_row([
                {"val": str(df_raw2.shape),   "lbl": "Original Shape"},
                {"val": str(df_clean2.shape), "lbl": "Cleaned Shape"},
                {"val": str(orig_miss),       "lbl": "Missing Before"},
                {"val": str(clean_miss),      "lbl": "Missing After"},
                {"val": str(len(log)),        "lbl": "Cleaning Steps"},
            ])

            # Before vs After chart
            col1, col2 = st.columns(2)
            stock_cols = [c for c in df_clean2.columns if c in df_raw2.columns and "imputed" not in c and c.isupper()]
            plot_stock = stock_cols[0] if stock_cols else df_clean2.columns[0]

            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_raw2.index, y=df_raw2[plot_stock].abs(),
                                          name="Before", line=dict(color="#dc3545", width=1.5)))
                fig.add_trace(go.Scatter(x=df_clean2.index, y=df_clean2.get(plot_stock, df_clean2.iloc[:, 0]),
                                          name="After", line=dict(color="#28a745", width=1.5)))
                fig.update_layout(title=f"{plot_stock} — Before vs After", height=320)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                miss_before = (df_raw2.isnull().sum() / len(df_raw2) * 100).reset_index()
                miss_before.columns = ["Column", "Missing %"]
                fig2 = go.Figure(go.Bar(x=miss_before["Column"], y=miss_before["Missing %"],
                                         marker_color="#dc3545", name="Before", opacity=0.8))
                miss_after = (df_clean2.isnull().sum() / len(df_clean2) * 100).reset_index()
                miss_after.columns = ["Column", "Missing %"]
                fig2.add_trace(go.Bar(x=miss_after["Column"], y=miss_after["Missing %"],
                                       marker_color="#28a745", name="After", opacity=0.8))
                fig2.update_layout(title="Missing Data: Before vs After (%)", barmode="group", height=320)
                st.plotly_chart(fig2, use_container_width=True)

            # Cleaning log
            st.subheader("📋 Cleaning Audit Log")
            log_df = pd.DataFrame(log)
            st.dataframe(log_df, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        section_header("📋", "Cleaning Report — Before vs After Statistics")
        if "df_clean" in st.session_state:
            df_r = st.session_state["df_raw"]
            df_c = st.session_state["df_clean"]

            stats_rows = []
            for col in [c for c in df_r.columns if "imputed" not in c][:6]:
                if col in df_c.columns:
                    stats_rows.append({
                        "Column": col,
                        "Mean (Before)": round(df_r[col].mean(), 2),
                        "Mean (After)": round(df_c[col].mean(), 2),
                        "Std (Before)": round(df_r[col].std(), 2),
                        "Std (After)": round(df_c[col].std(), 2),
                        "Min (Before)": round(df_r[col].min(), 2),
                        "Min (After)": round(df_c[col].min(), 2),
                        "Missing (Before)": df_r[col].isnull().sum(),
                        "Missing (After)": df_c[col].isnull().sum(),
                    })
            if stats_rows:
                st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)
        else:
            st.info("▶️ Run the pipeline in the **Interactive Runner** tab first.")

    # ─────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        section_header("💻", "Full Source Code — FinancialDataCleaner")

        st.code("""
class FinancialDataCleaner:
    \"\"\"
    Production-grade pipeline for cleaning financial time series
    and cross-sectional datasets.
    BITS WILP | MBA ZG517 | Prof. V. Ravichandran | themountainpathacademy.com
    \"\"\"
    def __init__(self, config=None):
        self.config = config or {
            'missing_threshold': 0.30,   # Drop columns >30% missing
            'winsorise_level':   0.01,   # 1%/99% winsorisation
            'zscore_threshold':  4.0,    # MAD Z-score threshold
            'knn_neighbors':     7,      # KNN imputation neighbours
            'outlier_action':    'winsorise'  # 'remove', 'winsorise', 'flag'
        }
        self.cleaning_log = []

    def validate_schema(self, df):
        # Coerce numeric strings, ensure DatetimeIndex, sort, deduplicate
        ...

    def treat_missing(self, df):
        # Drop high-missing cols, ffill sparse, KNN moderate, median+flag heavy
        ...

    def treat_outliers(self, df):
        # Modified Z-Score per column → winsorise / flag / remove
        ...

    def cleaning_report(self, df_original, df_clean):
        # Print shape, missing counts, all log entries
        ...

    def clean(self, df):
        df_clean = df.copy()
        df_clean = self.validate_schema(df_clean)
        df_clean = self.treat_missing(df_clean)
        df_clean = self.treat_outliers(df_clean)
        self.cleaning_report(df, df_clean)
        return df_clean

# ── Usage ──────────────────────────────────────────────────────────────────
cleaner = FinancialDataCleaner(config={
    'missing_threshold': 0.25,
    'winsorise_level':   0.01,
    'zscore_threshold':  3.5,
    'knn_neighbors':     7,
    'outlier_action':    'winsorise'
})
df_cleaned = cleaner.clean(df_raw)
""", language="python")

    render_footer()
