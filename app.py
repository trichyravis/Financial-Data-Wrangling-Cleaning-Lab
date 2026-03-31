"""
=============================================================================
THE MOUNTAIN PATH - WORLD OF FINANCE
Financial Data Wrangling & Cleaning
Interactive Learning Platform
Prof. V. Ravichandran | themountainpathacademy.com
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.interpolate import CubicSpline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Data Wrangling | The Mountain Path",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# DESIGN TOKENS  (exact match to Benford app / themountainpathacademy.com)
# ─────────────────────────────────────────────────────────────
DARK_BLUE  = "#003366"
MID_BLUE   = "#004d80"
GOLD       = "#FFD700"
CARD_BG    = "#112240"
TXT        = "#e6f1ff"
MUTED      = "#8892b0"
GREEN      = "#28a745"
RED        = "#dc3545"
LIGHT_BLUE = "#ADD8E6"
BG_GRAD    = "linear-gradient(135deg,#1a2332,#243447,#2a3f5f)"

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;900&family=Source+Sans+3:wght@300;400;600&family=Fira+Code:wght@400;500&display=swap');

.stApp {{
  background: {BG_GRAD} !important;
  color: {TXT} !important;
  font-family: 'Source Sans 3', 'Segoe UI', Arial, sans-serif;
}}
.block-container {{ padding-top: 1.5rem; max-width:1180px; }}

section[data-testid="stSidebar"] {{
  background: #0a1628 !important;
  border-right: 3px solid {GOLD} !important;
}}
section[data-testid="stSidebar"] * {{ color: #ffffff !important; }}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stRadio p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {{
  color: #ffffff !important; font-size: 13px !important;
}}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {{
  background: {DARK_BLUE}aa !important; border-radius: 6px; color: {GOLD} !important;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {{ color: {GOLD} !important; }}
section[data-testid="stSidebar"] a {{ color: {GOLD} !important; text-decoration: none; }}
section[data-testid="stSidebar"] a:hover {{ color: {LIGHT_BLUE} !important; text-decoration: underline; }}
section[data-testid="stSidebar"] hr {{ border-color: {GOLD}55 !important; }}
section[data-testid="stSidebar"] .stRadio > label {{
  color: {GOLD} !important; font-weight: 700 !important;
  font-size: 13px !important; letter-spacing: 0.5px; text-transform: uppercase;
}}

h1, h2, h3, h4 {{ color: {GOLD} !important; font-family: 'Playfair Display', serif !important; }}
p, li, span {{ color: {TXT}; }}

[data-testid="metric-container"] {{
  background: #0d1b2e !important; border: 1px solid {GOLD}55 !important;
  border-radius: 10px !important; padding: 14px !important;
}}
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {{
  color: {LIGHT_BLUE} !important; font-size: 12px !important;
  font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.5px;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
  color: {GOLD} !important; font-size: 26px !important; font-weight: 800 !important;
}}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
  color: {MUTED} !important; font-size: 11px !important;
}}

.stTabs [data-baseweb="tab-list"] {{
  background: #0d1b2e !important; border-radius: 8px 8px 0 0;
  gap: 3px; padding: 4px 4px 0; border-bottom: 2px solid {GOLD}33;
}}
.stTabs [data-baseweb="tab"] {{
  color: {LIGHT_BLUE} !important; font-weight: 600 !important; font-size: 13px !important;
  border-radius: 6px 6px 0 0; padding: 8px 16px; background: #152035 !important;
  border: 1px solid {GOLD}22 !important; border-bottom: none !important; transition: all 0.2s;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: {GOLD} !important; background: {DARK_BLUE} !important; }}
.stTabs [aria-selected="true"] {{
  background: {DARK_BLUE} !important; color: {GOLD} !important;
  border-bottom: 3px solid {GOLD} !important; font-weight: 700 !important;
}}

div[data-testid="stInfo"] {{
  background: #0d2240 !important; border-left: 4px solid {LIGHT_BLUE} !important; color: {TXT} !important;
}}
div[data-testid="stSuccess"] {{
  background: #0a2a14 !important; border-left: 4px solid {GREEN} !important; color: #c3f0ca !important;
}}
div[data-testid="stWarning"] {{
  background: #2a1f00 !important; border-left: 4px solid {GOLD} !important; color: #ffe9a0 !important;
}}
div[data-testid="stError"] {{
  background: #2a0a0a !important; border-left: 4px solid {RED} !important; color: #ffb3b3 !important;
}}

.stSelectbox [data-baseweb="select"] div,
.stSelectbox [data-baseweb="select"] span {{
  background: #0d1b2e !important; color: {TXT} !important; border-color: {GOLD}44 !important;
}}
.stTextArea textarea {{
  background: #0d1b2e !important; color: {TXT} !important;
  border: 1px solid {GOLD}44 !important; border-radius: 6px !important;
  font-family: 'Fira Code', monospace;
}}
.main .stRadio label p, .main .stRadio label span {{ color: {TXT} !important; }}
.stCheckbox label p, .stCheckbox label span {{ color: {TXT} !important; }}

[data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}
.stDataFrame thead tr th {{
  background: {DARK_BLUE} !important; color: {GOLD} !important; font-weight: 700 !important;
}}
.stDataFrame tbody tr td {{ color: {TXT} !important; background: #0d1b2e !important; }}
.stDataFrame tbody tr:nth-child(even) td {{ background: #112240 !important; }}

[data-testid="stFileUploader"] {{
  background: #0d1b2e !important; border: 2px dashed {GOLD}55 !important; border-radius: 8px !important;
}}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {{ color: {LIGHT_BLUE} !important; }}

.streamlit-expanderHeader {{
  background: #0d1b2e !important; color: {GOLD} !important;
  border-radius: 6px !important; font-weight: 600 !important;
}}
.streamlit-expanderContent {{
  background: #0a1628 !important; border: 1px solid {GOLD}22 !important;
}}

.stButton > button {{
  background: {DARK_BLUE} !important; color: {GOLD} !important;
  border: 2px solid {GOLD} !important; border-radius: 8px !important;
  font-weight: 700 !important; font-size: 14px !important;
  padding: 10px 24px !important; transition: all 0.2s !important;
}}
.stButton > button:hover {{
  background: {GOLD} !important; color: {DARK_BLUE} !important;
}}
.stButton > button[kind="primary"] {{
  background: {GOLD} !important; color: {DARK_BLUE} !important;
}}

/* ══ DOWNLOAD BUTTONS — match app design ══ */
[data-testid="stDownloadButton"] > button {{
  background: {DARK_BLUE} !important;
  color: {GOLD} !important;
  border: 2px solid {GOLD} !important;
  border-radius: 8px !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  padding: 10px 24px !important;
  width: 100% !important;
  transition: all 0.2s !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
  background: {GOLD} !important;
  color: {DARK_BLUE} !important;
  border-color: {GOLD} !important;
}}
[data-testid="stDownloadButton"] > button p,
[data-testid="stDownloadButton"] > button span,
[data-testid="stDownloadButton"] > button div {{
  color: {GOLD} !important;
  font-weight: 700 !important;
}}
[data-testid="stDownloadButton"] > button:hover p,
[data-testid="stDownloadButton"] > button:hover span,
[data-testid="stDownloadButton"] > button:hover div {{
  color: {DARK_BLUE} !important;
}}

.mp-card {{
  background: #0d1b2e; border: 1px solid {GOLD}44;
  border-left: 4px solid {GOLD}; border-radius: 10px;
  padding: 18px 22px; margin-bottom: 14px; color: {TXT};
}}
.mp-card-red {{
  background: #1a0a0a; border: 1px solid {RED}66;
  border-left: 4px solid {RED}; border-radius: 10px;
  padding: 18px 22px; margin-bottom: 14px; color: {TXT};
}}
.mp-card-green {{
  background: #0a1a0a; border: 1px solid {GREEN}66;
  border-left: 4px solid {GREEN}; border-radius: 10px;
  padding: 18px 22px; margin-bottom: 14px; color: {TXT};
}}
.mp-card-blue {{
  background: #0a1428; border: 1px solid {LIGHT_BLUE}66;
  border-left: 4px solid {LIGHT_BLUE}; border-radius: 10px;
  padding: 18px 22px; margin-bottom: 14px; color: {TXT};
}}
.hero-wrap {{
  background: linear-gradient(135deg,{DARK_BLUE},{MID_BLUE});
  border: 2px solid {GOLD}; border-radius: 14px;
  padding: 28px 34px; margin-bottom: 22px;
}}
.badge {{
  display: inline-block; background: {GOLD}; color: {DARK_BLUE};
  font-weight: 700; font-size: 11px; padding: 3px 10px;
  border-radius: 20px; margin: 2px; user-select: none;
}}
.badge-green {{ background: {GREEN}; color: #ffffff; }}
.badge-red   {{ background: {RED};   color: #ffffff; }}
.badge-blue  {{ background: {LIGHT_BLUE}33; color: {LIGHT_BLUE}; border: 1px solid {LIGHT_BLUE}55; }}
.formula-box {{
  background: linear-gradient(135deg,#001a40,{DARK_BLUE});
  border: 2px solid {GOLD}; border-radius: 10px;
  padding: 18px 24px; text-align: center; margin: 14px 0;
}}
.verdict-ok {{
  background: linear-gradient(90deg,#0d2e0d,#0a1628);
  border-left: 5px solid {GREEN}; border-radius: 8px;
  padding: 14px 20px; color: #c3f0ca;
  font-weight: 700; font-size: 15px; margin: 10px 0;
}}
.verdict-warn {{
  background: linear-gradient(90deg,#2a1f00,#0a1628);
  border-left: 5px solid {GOLD}; border-radius: 8px;
  padding: 14px 20px; color: {GOLD};
  font-weight: 700; font-size: 15px; margin: 10px 0;
}}
.verdict-bad {{
  background: linear-gradient(90deg,#2a0000,#0a1628);
  border-left: 5px solid {RED}; border-radius: 8px;
  padding: 14px 20px; color: #ffb3b3;
  font-weight: 700; font-size: 15px; margin: 10px 0;
}}
a {{ color: {GOLD} !important; text-decoration: none; }}
a:hover {{ color: {LIGHT_BLUE} !important; text-decoration: underline; }}
code, pre {{
  font-family: 'Fira Code', monospace !important;
  background: #060d1a !important; color: #a8d8ea !important;
  border: 1px solid {GOLD}22 !important; border-radius: 6px !important;
  font-size: 0.84rem !important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SHARED PLOTLY THEME
# ─────────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD_BG,
    font=dict(color=TXT, family="Source Sans 3, Segoe UI, Arial"),
    margin=dict(l=40, r=20, t=50, b=40),
)

# ─────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────
GSEC_M = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 14, 30])
GSEC_Y = np.array([6.55, 6.60, 6.68, 6.72, 6.79, 6.85, 6.91, 6.96, 7.02, 7.08])

def nse_prices(n=252, seed=42):
    rng = np.random.default_rng(seed)
    dt  = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({
        s: b + np.cumsum(rng.normal(0.3, sg, n))
        for s, b, sg in [("RELIANCE",2800,50),("TCS",3500,80),
                          ("HDFCBANK",1600,30),("INFY",1450,60),("ICICIBANK",950,25)]
    }, index=dt)

def inject_miss(df, seed=42):
    rng = np.random.default_rng(seed); df = df.copy()
    df.iloc[50:54, 0] = np.nan
    df.iloc[rng.choice(len(df), 15, replace=False), 1] = np.nan
    df.iloc[220:, 2] = np.nan
    df.iloc[rng.choice(range(150, len(df)), 8, replace=False), 3] = np.nan
    return df

def validate_ohlcv(df):
    iss = []
    for col in ["Open","High","Low","Close"]:
        for idx in df[df[col]<=0].index:
            iss.append({"Date":idx.date(),"Field":col,"Value":df.loc[idx,col],"Issue":"Non-positive price"})
    v = df[(df.High<df.Low)|(df.High<df.Close)|(df.High<df.Open)|(df.Low>df.Close)|(df.Low>df.Open)]
    for idx in v.index:
        iss.append({"Date":idx.date(),"Field":"OHLC","Value":f"H={df.loc[idx,'High']:.0f} L={df.loc[idx,'Low']:.0f}","Issue":"OHLC violation"})
    for idx in df[df.Volume==0].index:
        iss.append({"Date":idx.date(),"Field":"Volume","Value":0,"Issue":"Zero volume on trading day"})
    df2=df.copy(); df2["r"]=df2.Close.pct_change()
    for idx in df2[df2.r.abs()>0.20].dropna().index:
        iss.append({"Date":idx.date(),"Field":"Return","Value":f"{df2.loc[idx,'r']*100:.1f}%","Issue":"Extreme move >20%"})
    pc = df.Close!=df.Close.shift(1)
    for idx in df[pc.rolling(5).sum()==0].index:
        iss.append({"Date":idx.date(),"Field":"Close","Value":df.loc[idx,"Close"],"Issue":"Stale price 5+ days"})
    return pd.DataFrame(iss)

# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────
def hero(title, subtitle, badges=None):
    b = "".join(f'<span class="badge">{x}</span> ' for x in (badges or []))
    st.markdown(f"""
    <div class="hero-wrap">
      <div style="font-size:36px;color:{GOLD};font-weight:900;
           font-family:'Playfair Display',serif;letter-spacing:1px;">{title}</div>
      <div style="font-size:15px;color:{TXT};margin:8px 0 12px;">{subtitle}</div>
      {b}
    </div>""", unsafe_allow_html=True)

def shdr(icon, title):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:1.4rem 0 0.5rem;">
      <span style="font-size:1.3rem;">{icon}</span>
      <span style="font-family:'Playfair Display',serif;font-size:1.25rem;
                   font-weight:700;color:{LIGHT_BLUE};">{title}</span>
    </div>
    <div style="height:2px;background:linear-gradient(90deg,{GOLD},transparent);margin-bottom:1rem;"></div>
    """, unsafe_allow_html=True)

def mrow(items):
    cols = st.columns(len(items))
    for c, m in zip(cols, items): c.metric(m["lbl"], m["val"])

def footer():
    st.markdown(f"""
    <div style="text-align:center;color:{MUTED};font-size:12px;
         padding:24px 0 10px;border-top:1px solid {GOLD}33;margin-top:32px;">
      <b style="color:{GOLD};font-family:'Playfair Display',serif;">
        The Mountain Path – World of Finance</b><br>
      Prof. V. Ravichandran &nbsp;|&nbsp;
      <a href="https://themountainpathacademy.com">themountainpathacademy.com</a>
      &nbsp;|&nbsp;<a href="https://www.linkedin.com/in/trichyravis">LinkedIn</a>
      &nbsp;|&nbsp;<a href="https://github.com/trichyravis">GitHub</a>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:18px 10px 12px;background:#001428;
         border-radius:10px;margin-bottom:12px;border:1px solid {GOLD}55;">
      <div style="font-size:22px;color:{GOLD};font-weight:900;letter-spacing:1px;">
        🏔️ THE MOUNTAIN PATH</div>
      <div style="font-size:11px;color:#ADD8E6;margin-top:4px;letter-spacing:2px;text-transform:uppercase;">
        World of Finance</div>
      <div style="height:2px;background:linear-gradient(90deg,transparent,{GOLD},transparent);margin:10px 0;"></div>
      <div style="font-size:12px;color:#ffffff;font-weight:600;">Prof. V. Ravichandran</div>
      <div style="font-size:11px;color:#ADD8E6;margin-top:3px;">
        <a href="https://themountainpathacademy.com" target="_blank"
           style="color:{GOLD} !important;">themountainpathacademy.com</a></div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style="font-size:11px;color:{GOLD};font-weight:700;
         text-transform:uppercase;letter-spacing:1px;padding:4px 6px 6px;margin-bottom:4px;">
      📚 NAVIGATE</div>""", unsafe_allow_html=True)

    page = st.radio("nav", [
        "🏠 Home",
        "🔍 Missing Data",
        "📊 Outlier Detection",
        "📅 Time Series Formatting",
        "⚠️ Invalid Values",
        "⚙️ Full Cleaning Pipeline",
        "📚 Case Studies",
        "❓ Quiz & Assessment",
    ], label_visibility="collapsed")

    st.markdown(f"""
    <div style="margin-top:14px;padding:12px 14px;background:#001428;
         border-radius:8px;border:1px solid {GOLD}33;">
      <div style="font-size:11px;font-weight:700;color:{GOLD};
           text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">🔑 Key Topics</div>
      <div style="font-size:12px;color:#e6f1ff;line-height:1.9;">
        <span style="color:{GOLD};">▸</span> MCAR · MAR · MNAR<br>
        <span style="color:{GOLD};">▸</span> KNN &amp; MICE Imputation<br>
        <span style="color:{GOLD};">▸</span> Z-Score · MAD · IQR<br>
        <span style="color:{GOLD};">▸</span> Isolation Forest ML<br>
        <span style="color:{GOLD};">▸</span> OHLCV Validation<br>
        <span style="color:{GOLD};">▸</span> G-Sec Yield Curve<br>
        <span style="color:{GOLD};">▸</span> NSE Business Calendar<br>
        <span style="color:{GOLD};">▸</span> Winsorisation &amp; Transforms
      </div>
    </div>
    <div style="height:1px;background:{GOLD}33;margin:14px 0;"></div>
    <div style="font-size:11px;color:#ADD8E6;text-align:center;line-height:1.8;">
      <span style="color:#ffffff;font-weight:600;">© 2025 The Mountain Path</span><br>
      <a href="https://www.linkedin.com/in/trichyravis" target="_blank"
         style="color:{GOLD} !important;font-weight:600;">LinkedIn</a>
      <span style="color:{GOLD};">  |  </span>
      <a href="https://github.com/trichyravis" target="_blank"
         style="color:{GOLD} !important;font-weight:600;">GitHub</a>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════
if page == "🏠 Home":
    hero("Financial Data Wrangling & Cleaning",
         "An Interactive Learning Platform — The Mountain Path · World of Finance",
         ["Missing Data","Outlier Detection","Time Series","Invalid Values","ML Pipeline","Python"])

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("GIGO Principle",      "Garbage In → Out",  delta="Clean data = credible model",  delta_color="off")
    c2.metric("Knight Capital Loss", "USD 440M",          delta="45 minutes, Aug 2012",          delta_color="off")
    c3.metric("Imputation Methods",  "6+",                delta="LOCF · KNN · MICE · Spline",    delta_color="off")
    c4.metric("Outlier Methods",     "4 Statistical",     delta="+ Domain Rules + ML",            delta_color="off")

    st.markdown("---")
    col1, col2 = st.columns([3,2])
    with col1:
        shdr("📌","What This Lab Covers")
        for icon,title,desc in [
            ("🔍","Missing Data","MCAR · MAR · MNAR classification, diagnosis & imputation (LOCF, KNN, MICE)"),
            ("📊","Outlier Detection","Z-Score, Modified Z-Score (MAD), IQR, Isolation Forest on live NSE data"),
            ("📅","Time Series Formatting","DatetimeIndex, NSE business calendar, OHLCV resampler, G-Sec yield curve"),
            ("⚠️","Invalid Values","OHLC logic validation, stale price detection, duplicate timestamps"),
            ("⚙️","Full Cleaning Pipeline","Production-grade FinancialDataCleaner class — end-to-end with audit trail"),
            ("📚","Indian Market Cases","Demonetisation 2016 · IL&FS Default 2018 · NSE Feed Outage"),
        ]:
            st.markdown(f"""
            <div class="mp-card" style="display:flex;align-items:flex-start;
                 gap:14px;padding:14px 18px;margin-bottom:8px;">
              <span style="font-size:1.4rem;flex-shrink:0;">{icon}</span>
              <div>
                <div style="font-weight:700;color:{LIGHT_BLUE};font-size:0.95rem;">{title}</div>
                <div style="color:{MUTED};font-size:0.82rem;margin-top:2px;">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with col2:
        shdr("🎯","Learning Objectives")
        for i,obj in enumerate([
            "Diagnose & classify data quality problems in real financial datasets",
            "Select & implement appropriate missing data imputation strategies",
            "Structure financial time series with correct indexing & frequency handling",
            "Apply statistical & domain-specific methods to detect & manage outliers",
            "Build a reproducible Python data-cleaning pipeline",
            "Critically evaluate downstream impact of data quality decisions on model outputs",
        ], 1):
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:10px;">
              <div style="width:28px;height:28px;background:{GOLD};color:{DARK_BLUE};
                   border-radius:50%;display:flex;align-items:center;justify-content:center;
                   font-weight:700;font-size:13px;flex-shrink:0;">{i}</div>
              <span style="font-size:0.87rem;color:{TXT};padding-top:4px;">{obj}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="formula-box" style="margin-top:14px;">
        <div style="color:{MUTED};font-size:11px;margin-bottom:6px;">THE PIPELINE</div>
        <div style="color:{GOLD};font-size:15px;font-weight:700;
             font-family:'Playfair Display',serif;line-height:1.8;">
          Ingest → Validate → Missing<br>→ Outliers → Format → Output
        </div>
        <div style="color:{LIGHT_BLUE};font-size:11px;margin-top:6px;">"Garbage In, Garbage Out" — GIGO</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""<div class="mp-card-red">
    <b style="color:{RED};font-size:15px;">⚡ Knight Capital Incident — 1 August 2012</b><br><br>
    Knight Capital Group lost <b style="color:{GOLD};">USD 440 million in 45 minutes</b> due to a
    software deployment error where stale, erroneous order data triggered unintended trading.
    Root cause: failure to validate and clean stale routing data before deployment.<br><br>
    <b>Lesson:</b> A single undetected invalid value can cascade into catastrophic financial loss.
    </div>""", unsafe_allow_html=True)
    st.info("👈 **Use the sidebar** to navigate between modules.")
    footer()


# ═════════════════════════════════════════════════════════════
# PAGE: MISSING DATA
# ═════════════════════════════════════════════════════════════
elif page == "🔍 Missing Data":
    hero("Missing Data — Diagnosis & Treatment",
         "Classification · Detection · Imputation Strategies for Financial Datasets",
         ["MCAR","MAR","MNAR","KNN","MICE"])

    tab1,tab2,tab3,tab4 = st.tabs([
        "📘 Theory & Taxonomy","🔬 Interactive Diagnosis",
        "🛠️ Imputation Methods","📐 Method Comparison"])

    with tab1:
        shdr("📘","The Three Mechanisms of Missingness")
        c1,c2,c3 = st.columns(3)
        for col,clr,icon,nm,tag,ex,fix in [
            (c1,LIGHT_BLUE,"🎲","MCAR","Best Case — any fix works",
             "NSE random packet-loss on 0.1% of trading days.",
             "Any imputation valid. Listwise deletion acceptable."),
            (c2,GOLD,"📊","MAR","Manageable — model-based imputation",
             "Small-caps missing EPS estimates (related to market cap, not EPS itself).",
             "MICE, KNN, or regression using observed predictors."),
            (c3,RED,"🚨","MNAR","Most Dangerous — domain expertise required",
             "Company revenue missing because revenues were catastrophically low.",
             "No standard fix. Domain expertise + sensitivity analysis."),
        ]:
            col.markdown(f"""
            <div class="mp-card" style="border-left:4px solid {clr};height:100%;">
              <div style="text-align:center;font-size:1.6rem;margin-bottom:6px;">{icon}</div>
              <div style="font-weight:900;color:{clr};font-size:1.05rem;
                   font-family:'Playfair Display',serif;margin-bottom:4px;">{nm}</div>
              <div style="font-size:0.75rem;color:{MUTED};text-transform:uppercase;
                   letter-spacing:0.06em;margin-bottom:8px;">{tag}</div>
              <div style="font-size:0.87rem;color:{TXT};margin-bottom:8px;">{ex}</div>
              <div style="font-size:0.82rem;background:#060d1a;padding:8px;
                   border-radius:6px;color:{LIGHT_BLUE};">✔ {fix}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        shdr("📋","Decision Framework")
        st.dataframe(pd.DataFrame({
            "% Missing":["<5%","5–15%",">15%","Any",">50%","Monotone tail"],
            "Mechanism":["MCAR","MAR","MAR","MNAR","Any","Known event"],
            "Variable":["Price/Return","Financial ratio","Any","Any","Any","Price"],
            "Recommended Strategy":[
                "Forward-fill or mean imputation","Regression / KNN imputation",
                "Multiple Imputation (MICE)","Domain-adjusted; sensitivity analysis",
                "Exclude; flag and document","Mark as delisted; exclude"],
        }), use_container_width=True, hide_index=True)

        st.markdown(f"""<div class="mp-card-red">
        <b style="color:{RED};">Mean Imputation Warning:</b>
        Var(X<sub>imputed</sub>) &lt; Var(X<sub>true</sub>) — artificially reduces variance.
        VaR calculations <b>understate risk</b>. Correlation estimates biased toward zero.
        <b>Never use for risk models or where distribution shape matters.</b>
        </div>""", unsafe_allow_html=True)

    with tab2:
        shdr("🔬","Interactive Missing Data Diagnosis")
        cc,cm = st.columns([1,3])
        with cc:
            n  = st.slider("Trading Days",100,504,252,10)
            bp = st.slider("Block Missing %",0,20,5)
            rp = st.slider("Random Missing %",0,20,8)
            mt = st.slider("Monotone Tail (days)",0,80,30)
        df0 = nse_prices(n=n); df1 = df0.copy()
        if bp>0: df1.iloc[50:50+max(2,int(n*bp/100)),0]=np.nan
        if rp>0: df1.iloc[np.random.default_rng(7).choice(n,int(n*rp/100),replace=False),1]=np.nan
        if mt>0: df1.iloc[-mt:,2]=np.nan
        tot=df1.isnull().sum().sum(); pct=tot/df1.size*100
        mrow([{"val":f"{pct:.1f}%","lbl":"Overall Missing"},
              {"val":str(tot),"lbl":"Missing Cells"},
              {"val":str(n),"lbl":"Trading Days"},
              {"val":"5","lbl":"NSE Stocks"}])
        with cm:
            fig=go.Figure(go.Heatmap(z=df1.isnull().astype(int).values.T,
                x=[str(d.date()) for d in df1.index],y=df1.columns.tolist(),
                colorscale=[[0,"#0f1a2e"],[1,GOLD]],showscale=True,
                colorbar=dict(title="Missing",tickvals=[0,1],ticktext=["Present","Missing"])))
            fig.update_layout(**PL,title=dict(text="<b>Missing Data Heatmap (Gold=Missing)</b>",
                font=dict(color=GOLD,size=13)),height=260,xaxis=dict(showticklabels=False))
            st.plotly_chart(fig,use_container_width=True)
        c1,c2=st.columns(2)
        with c1:
            mp=(df1.isnull().sum()/n*100).reset_index(); mp.columns=["S","P"]
            fig2=go.Figure(go.Bar(x=mp.P,y=mp.S,orientation="h",
                marker_color=[GREEN if v<5 else(GOLD if v<15 else RED) for v in mp.P],
                text=[f"{v:.1f}%" for v in mp.P],textposition="outside",textfont=dict(color=TXT)))
            fig2.add_vline(x=5,line_dash="dash",line_color=RED)
            fig2.update_layout(**PL,height=280,title=dict(text="<b>Missing % by Stock</b>",font=dict(color=GOLD,size=13)))
            st.plotly_chart(fig2,use_container_width=True)
        with c2:
            fig3=go.Figure(go.Scatter(x=df1.index,y=df1.isnull().sum(axis=1).cumsum(),
                fill="tozeroy",line=dict(color=GOLD,width=2),fillcolor="rgba(255,215,0,0.08)"))
            fig3.update_layout(**PL,height=280,title=dict(text="<b>Cumulative Missing Over Time</b>",font=dict(color=GOLD,size=13)))
            st.plotly_chart(fig3,use_container_width=True)

    with tab3:
        shdr("🛠️","Live Imputation — Method Selector")
        df2=nse_prices(n=252); df3=inject_miss(df2)
        c1,c2=st.columns(2)
        with c1: stk=st.selectbox("Stock",df3.columns.tolist(),index=1)
        with c2: meth=st.selectbox("Method",["Forward Fill (LOCF)","Backward Fill (NOCB)",
                   "Linear Interpolation","Spline Interpolation",
                   "Mean Imputation","Median Imputation","KNN Imputation"])
        sr=df3[stk]; st2=df2[stk]
        if   meth=="Forward Fill (LOCF)":    si=sr.ffill()
        elif meth=="Backward Fill (NOCB)":   si=sr.bfill()
        elif meth=="Linear Interpolation":   si=sr.interpolate("linear")
        elif meth=="Spline Interpolation":   si=sr.interpolate("spline",order=3)
        elif meth=="Mean Imputation":        si=sr.fillna(sr.mean())
        elif meth=="Median Imputation":      si=sr.fillna(sr.median())
        else:
            sc=StandardScaler(); X=sc.fit_transform(df3.values)
            Xp=sc.inverse_transform(KNNImputer(n_neighbors=7).fit_transform(X))
            si=pd.Series(Xp[:,df3.columns.get_loc(stk)],index=df3.index)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=st2.index,y=st2,name="True",line=dict(color=GREEN,width=1.5,dash="dot"),opacity=0.6))
        fig.add_trace(go.Scatter(x=sr.index,y=sr,name="Observed",line=dict(color=LIGHT_BLUE,width=2)))
        fig.add_trace(go.Scatter(x=si.index,y=si,name=f"Imputed ({meth})",line=dict(color=GOLD,width=2)))
        fig.update_layout(**PL,height=400,title=dict(text=f"<b>{stk} — {meth}</b>",font=dict(color=GOLD,size=13)),
            legend=dict(orientation="h"))
        st.plotly_chart(fig,use_container_width=True)
        if "Fill" in meth:
            st.markdown(f'<div class="verdict-warn">⚠️ {meth} creates artificial zero-return days — understates volatility. Never use for risk models.</div>',unsafe_allow_html=True)
        elif meth=="Mean Imputation":
            st.markdown(f'<div class="verdict-warn">⚠️ Mean imputation reduces variance — biases VaR and correlation estimates.</div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-ok">✅ {meth} better preserves temporal structure.</div>',unsafe_allow_html=True)

    with tab4:
        shdr("📐","Return Distribution — True vs Imputed")
        df4=nse_prices(n=252); df5=inject_miss(df4)
        s4=st.selectbox("Stock",df5.columns,key="cst")
        s5=df5[s4]
        methods={
            "True":df4[s4].pct_change().dropna(),
            "Forward Fill":s5.ffill().pct_change().dropna(),
            "Linear Interp":s5.interpolate("linear").pct_change().dropna(),
            "Mean Imputation":s5.fillna(s5.mean()).pct_change().dropna(),
        }
        fig=make_subplots(rows=1,cols=4,subplot_titles=list(methods.keys()),shared_yaxes=True)
        for i,(nm,r) in enumerate(methods.items(),1):
            fig.add_trace(go.Histogram(x=r,name=nm,
                marker_color=[GREEN,GOLD,LIGHT_BLUE,RED][i-1],opacity=0.75,nbinsx=40,showlegend=False),row=1,col=i)
        fig.update_layout(**PL,height=360,title=dict(text="<b>Return Distribution: True vs Imputed</b>",font=dict(color=GOLD,size=13)))
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(pd.DataFrame({nm:{
            "Mean (%)":round(r.mean()*100,4),"Std Dev (%)":round(r.std()*100,4),
            "Skewness":round(r.skew(),3),"Kurtosis":round(r.kurt(),3),"Zero Returns":int((r==0).sum()),
        } for nm,r in methods.items()}).T,use_container_width=True)
        st.markdown(f"""<div class="mp-card-red">
        Mean imputation produces the <b>lowest variance</b> — directly understating VaR.
        Forward Fill spikes at zero returns. Linear Interpolation best approximates true distribution.
        </div>""",unsafe_allow_html=True)
    footer()


# ═════════════════════════════════════════════════════════════
# PAGE: OUTLIER DETECTION
# ═════════════════════════════════════════════════════════════
elif page == "📊 Outlier Detection":
    hero("Outlier Detection in Financial Data",
         "Z-Score · Modified Z-Score (MAD) · IQR · Isolation Forest · Winsorisation",
         ["Error Outliers","Genuine Extreme Events","Structural Breaks"])

    tab1,tab2,tab3,tab4 = st.tabs([
        "📘 Three Types","🔬 Interactive Detection",
        "⚖️ Method Comparison","✂️ Winsorisation Lab"])

    with tab1:
        c1,c2,c3=st.columns(3)
        for col,clr,icon,title,exs,action in [
            (c1,RED,"🐛","Error Outliers",
             ["Negative stock price","PE=50,000","Volume=0 on active day","Return=−99% large-cap"],
             f"<b style='color:{RED}'>CORRECT or REMOVE. Not real events.</b>"),
            (c2,GOLD,"⚡","Genuine Extreme Events",
             ["NIFTY −38% (COVID Mar 2020)","NIFTY −4% (Demonetisation)","RBI rate shock days","Short squeezes"],
             f"<b style='color:{GOLD}'>PRESERVE for risk modelling & stress-testing.</b>"),
            (c3,LIGHT_BLUE,"🔀","Structural Breaks",
             ["Merger / demerger","Basel III→IV regulatory change","Index reconstitution","Currency regime change"],
             f"<b style='color:{LIGHT_BLUE}'>REGIME-SPLIT or add dummy variable.</b>"),
        ]:
            col.markdown(f"""
            <div class="mp-card" style="border-left:4px solid {clr};height:100%;">
              <div style="text-align:center;font-size:1.5rem;">{icon}</div>
              <div style="font-weight:900;color:{clr};font-family:'Playfair Display',serif;
                   font-size:1rem;margin:6px 0;">{title}</div>
              <ul style="font-size:0.83rem;color:{TXT};padding-left:18px;margin:6px 0;">
                {"".join(f'<li style="margin-bottom:3px;">{e}</li>' for e in exs)}
              </ul>
              <div style="font-size:0.83rem;margin-top:8px;padding:8px;
                   background:#060d1a;border-radius:6px;">{action}</div>
            </div>""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        shdr("📐","Statistical Methods Overview")
        st.dataframe(pd.DataFrame({
            "Method":["Z-Score","Modified Z-Score (MAD)","IQR (Tukey)","Isolation Forest"],
            "Formula":["Zᵢ=(Xᵢ−X̄)/σ","Mᵢ=0.6745·|Xᵢ−X̃|/MAD","Fences=Q1±k·IQR","s(x)=2^(−E[h(x)]/c(n))"],
            "Flag Rule":["|Z|>3","|M|>3.5","k=1.5 or k=3","Score≈1→anomalous"],
            "Fat Tails":["❌","✅","✅","✅"],
            "Best For":["Quick scan","Return series; ratios","Winsorisation bounds","Market surveillance"],
        }),use_container_width=True,hide_index=True)

    with tab2:
        shdr("🔬","Multi-Method Outlier Detection")
        cc,cm=st.columns([1,3])
        with cc:
            n2=st.slider("Observations",100,1000,500,50,key="on")
            ns=st.slider("Injected Shocks",0,20,10)
            zt=st.slider("Z-Score Threshold",1.5,5.0,3.0,0.1)
            mt=st.slider("MAD Threshold",1.5,6.0,3.5,0.1)
            ct=st.slider("IF Contamination",0.01,0.10,0.02,0.005)
        rng=np.random.default_rng(42)
        ret=rng.normal(0.05/252,0.18/np.sqrt(252),n2)
        if ns>0: ret[rng.choice(n2,ns,replace=False)]=rng.choice([-0.08,-0.06,-0.05,0.06,0.07,0.09],ns)
        dt2=pd.bdate_range("2022-01-03",periods=n2); returns=pd.Series(ret,index=dt2)
        zs=np.abs(stats.zscore(returns)); oz=returns[zs>zt]
        med=returns.median(); mad=np.median(np.abs(returns-med))
        mz=0.6745*np.abs(returns-med)/(mad+1e-10); om=returns[mz>mt]
        Q1,Q3=returns.quantile(0.25),returns.quantile(0.75); IQR=Q3-Q1
        oi=returns[(returns<Q1-3*IQR)|(returns>Q3+3*IQR)]
        iso=IsolationForest(contamination=ct,random_state=42)
        ip=iso.fit_predict(returns.values.reshape(-1,1)); ois=returns[ip==-1]
        ai=set(oz.index)|set(om.index)|set(oi.index)|set(ois.index)
        con=[ix for ix in ai if sum([ix in set(o.index) for o in [oz,om,oi,ois]])>=2]
        mrow([{"val":str(len(oz)),"lbl":"Z-Score"},{"val":str(len(om)),"lbl":"MAD"},
              {"val":str(len(oi)),"lbl":"IQR"},{"val":str(len(ois)),"lbl":"IF"},
              {"val":str(len(con)),"lbl":"Consensus (≥2)"}])
        with cm:
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=returns.index,y=returns*100,mode="lines",
                line=dict(color=LIGHT_BLUE,width=1),name="Daily Returns",opacity=0.7))
            fig.add_trace(go.Scatter(x=oz.index,y=oz*100,mode="markers",
                marker=dict(color=GOLD,size=8,symbol="circle-open",line=dict(width=2)),name=f"Z-Score"))
            fig.add_trace(go.Scatter(x=ois.index,y=ois*100,mode="markers",
                marker=dict(color=RED,size=9,symbol="x",line=dict(width=2)),name="Isolation Forest"))
            if con:
                cr=returns[returns.index.isin(con)]
                fig.add_trace(go.Scatter(x=cr.index,y=cr*100,mode="markers",
                    marker=dict(color="white",size=14,symbol="star",line=dict(color=GOLD,width=2)),
                    name="Consensus (≥2)"))
            fig.update_layout(**PL,height=400,
                title=dict(text="<b>Multi-Method Outlier Detection — NSE Returns</b>",font=dict(color=GOLD,size=13)),
                legend=dict(orientation="h",y=1.02))
            st.plotly_chart(fig,use_container_width=True)
        if con:
            st.dataframe(pd.DataFrame({
                "Date":[ix.date() for ix in sorted(con)],
                "Return (%)":[round(returns[ix]*100,2) for ix in sorted(con)],
                "Methods":[" | ".join([n for n,s in [("Z",oz),("MAD",om),("IQR",oi),("IF",ois)]
                           if ix in set(s.index)]) for ix in sorted(con)],
            }),use_container_width=True,hide_index=True)

    with tab3:
        shdr("⚖️","Method Comparison & Sensitivity Analysis")
        st.dataframe(pd.DataFrame({
            "Method":["Z-Score","Modified Z-Score (MAD)","IQR Method","Isolation Forest","DBSCAN"],
            "Distribution":["Normal","None (robust)","None","None","None"],
            "Fat Tails":["❌","✅","✅","✅","✅"],
            "Multivariate":["❌","❌","❌","✅","✅"],
            "Python":["scipy.stats.zscore","Custom MAD","Series.quantile()","sklearn IsolationForest","sklearn DBSCAN"],
        }),use_container_width=True,hide_index=True)
        st.markdown(f"""<div class="mp-card-blue">
        <b style="color:{LIGHT_BLUE};">Sensitivity Analysis — Gold Standard</b><br><br>
        1. Run with outliers (full sample) &nbsp; 2. Run without (winsorised) &nbsp;
        3. Run with robust methods<br>
        If results are <b>materially different</b>: investigate; report both; document assumptions.<br>
        If results are <b>similar</b>: outliers do not materially affect conclusions.<br><br>
        <b>Regulatory note:</b> SEBI ICDR and RBI model risk guidelines require documentation of
        outlier treatment methodology.
        </div>""",unsafe_allow_html=True)

    with tab4:
        shdr("✂️","Winsorisation — Live Demo")
        c1,c2=st.columns(2)
        with c1:
            lo_p=st.slider("Lower Percentile (%)",0.5,5.0,1.0,0.5)
            hi_p=st.slider("Upper Percentile (%)",95.0,99.5,99.0,0.5)
        rng2=np.random.default_rng(42)
        pe=np.concatenate([rng2.lognormal(3.3,0.6,490),np.array([500,800,-50,-30,0.5,1200,2000,5000,0.8,1000])])
        spe=pd.Series(pe); lo=spe.quantile(lo_p/100); hi=spe.quantile(hi_p/100)
        sw=spe.clip(lo,hi)
        with c2:
            mrow([{"val":str((spe<lo).sum()),"lbl":f"Capped at lower ({lo:.0f})"},
                  {"val":str((spe>hi).sum()),"lbl":f"Capped at upper ({hi:.0f})"},
                  {"val":f"{spe.std():.1f}","lbl":"Std Dev Before"},
                  {"val":f"{sw.std():.1f}","lbl":"Std Dev After"}])
        fig=make_subplots(rows=1,cols=2,subplot_titles=["Original PE","Winsorised PE"])
        fig.add_trace(go.Histogram(x=spe.clip(-100,1000),nbinsx=60,marker_color=LIGHT_BLUE,opacity=0.75,name="Orig"),row=1,col=1)
        fig.add_trace(go.Histogram(x=sw,nbinsx=60,marker_color=GOLD,opacity=0.75,name="Wins"),row=1,col=2)
        fig.add_vline(x=lo,line_dash="dash",line_color=RED,row=1,col=2)
        fig.add_vline(x=hi,line_dash="dash",line_color=RED,row=1,col=2)
        fig.update_layout(**PL,height=340,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
        st.markdown(f"""<div class="mp-card">
        <b style="color:{GOLD};">Formula:</b>
        X<sup>wins</sup>ᵢ = Q<sub>α</sub> if X<sub>i</sub>&lt;Q<sub>α</sub> |
        Xᵢ if Q<sub>α</sub>≤Xᵢ≤Q<sub>1−α</sub> |
        Q<sub>1−α</sub> if Xᵢ&gt;Q<sub>1−α</sub><br><br>
        Standard in academic finance: winsorise at <b>1% and 99%</b>.
        For time-series: use <b>rolling percentiles</b> to avoid look-ahead bias.
        </div>""",unsafe_allow_html=True)
    footer()


# ═════════════════════════════════════════════════════════════
# PAGE: TIME SERIES
# ═════════════════════════════════════════════════════════════
elif page == "📅 Time Series Formatting":
    hero("Time Series Formatting for Financial Data",
         "DatetimeIndex · Business Calendars · OHLCV Resampling · Yield Curve Interpolation",
         ["NSE Calendar","OHLCV","G-Sec Yield Curve","Timezone"])

    tab1,tab2,tab3,tab4 = st.tabs([
        "📘 Five Dimensions","📅 DatetimeIndex Builder",
        "📈 OHLCV Resampler","🎯 Yield Curve"])

    with tab1:
        shdr("📘","Five Dimensions of Financial Time Series Formatting")
        for num,title,clr,body,ex in [
            ("1️⃣","Index Type",LIGHT_BLUE,
             "DatetimeIndex MUST be used — string-based indices cause silent errors in resampling and rolling calculations.",
             "df.index = pd.to_datetime(df.index)"),
            ("2️⃣","Timezone Handling",GOLD,
             "NSE trades IST (UTC+5:30); NYSE trades ET. Cross-market analysis requires tz-aware timestamps.",
             "nse_ts.tz_localize('Asia/Kolkata').tz_convert('America/New_York')"),
            ("3️⃣","Business Calendar",LIGHT_BLUE,
             "pd.bdate_range() misses Indian holidays. Use CustomBusinessDay with NSEHolidayCalendar (~244 days/year).",
             "CustomBusinessDay(calendar=NSEHolidayCalendar())"),
            ("4️⃣","Frequency",GOLD,
             "OHLCV: Open→first, High→max, Low→min, Close→last, Volume→sum. Returns must be COMPOUNDED not summed.",
             "df.resample('ME').agg({'Open':'first','High':'max','Volume':'sum',...})"),
            ("5️⃣","Corporate Actions",LIGHT_BLUE,
             "TCS 1:1 bonus creates apparent −50% return on ex-date. Always use backward-adjusted close prices.",
             "yfinance: auto_adjust=True  |  NSE official adjusted data"),
        ]:
            st.markdown(f"""
            <div class="mp-card" style="margin-bottom:10px;">
              <div style="display:flex;align-items:flex-start;gap:12px;">
                <div style="font-size:1.5rem;flex-shrink:0;">{num}</div>
                <div style="flex:1;">
                  <div style="font-weight:700;color:{clr};font-size:0.97rem;margin-bottom:3px;">{title}</div>
                  <div style="font-size:0.86rem;color:{TXT};margin-bottom:6px;">{body}</div>
                  <code style="font-size:0.78rem;">{ex}</code>
                </div>
              </div>
            </div>""",unsafe_allow_html=True)

        shdr("📋","Resampling Aggregation Rules")
        st.dataframe(pd.DataFrame({
            "Data Type":["Open","High","Low","Close","Volume","Returns","Ratios","Macro"],
            "Rule":[".first()",".max()",".min()",".last()",".sum()","∏(1+rᵢ)−1",".last()",".last()/.sum()"],
            "Rationale":["Opening of period","Highest reached","Lowest reached","Closing of period",
                         "Total traded","Compound not sum","Period-end valuation","Context-dependent"],
        }),use_container_width=True,hide_index=True)
        st.markdown(f'<div class="mp-card-red"><b style="color:{RED};">Never sum returns!</b> Correct: ∏(1+rᵢ)−1</div>',unsafe_allow_html=True)

    with tab2:
        shdr("📅","Financial DatetimeIndex Builder")
        c1,c2=st.columns(2)
        with c1:
            start=st.date_input("Start",pd.Timestamp("2024-01-01"))
            end  =st.date_input("End",  pd.Timestamp("2024-12-31"))
        with c2:
            cal=st.selectbox("Calendar",["Mon–Fri (Standard)","NSE Custom Holiday Calendar"])
            tz =st.selectbox("Timezone",["Asia/Kolkata (IST)","America/New_York (ET)","UTC","Europe/London"])
        nse_h=["2024-01-26","2024-03-25","2024-04-14","2024-04-17",
               "2024-05-01","2024-06-17","2024-08-15","2024-10-02",
               "2024-11-01","2024-11-15","2024-12-25"]
        if cal=="Mon–Fri (Standard)":
            idx=pd.bdate_range(str(start),str(end))
        else:
            from pandas.tseries.holiday import AbstractHolidayCalendar,Holiday
            from pandas.tseries.offsets import CustomBusinessDay
            class NSECal(AbstractHolidayCalendar):
                rules=[Holiday("H"+str(i),year=pd.Timestamp(d).year,
                    month=pd.Timestamp(d).month,day=pd.Timestamp(d).day) for i,d in enumerate(nse_h)]
            idx=pd.bdate_range(str(start),str(end),freq=CustomBusinessDay(calendar=NSECal()))
        st.success(f"✅ **{len(idx)} trading days** from {start} to {end} using {cal}")
        if "NSE" in cal:
            st.info(f"📅 {len(pd.bdate_range(str(start),str(end)))-len(idx)} NSE holidays removed")
        tz_m={"Asia/Kolkata (IST)":"Asia/Kolkata","America/New_York (ET)":"America/New_York",
              "UTC":"UTC","Europe/London":"Europe/London"}
        to=pd.Timestamp("2024-01-15 09:15:00",tz="Asia/Kolkata").tz_convert(tz_m[tz])
        tc=pd.Timestamp("2024-01-15 15:30:00",tz="Asia/Kolkata").tz_convert(tz_m[tz])
        c1,c2=st.columns(2)
        c1.markdown(f"""<div class="formula-box"><div style="color:{MUTED};font-size:11px;">NSE Market Hours (IST)</div>
        <div style="color:{GOLD};font-size:22px;font-weight:800;">09:15 — 15:30</div>
        <div style="color:{LIGHT_BLUE};font-size:12px;">Asia/Kolkata (UTC+5:30)</div></div>""",unsafe_allow_html=True)
        c2.markdown(f"""<div class="formula-box"><div style="color:{MUTED};font-size:11px;">{tz}</div>
        <div style="color:{GOLD};font-size:22px;font-weight:800;">{to.strftime('%H:%M')} — {tc.strftime('%H:%M')}</div>
        <div style="color:{LIGHT_BLUE};font-size:12px;">{tz}</div></div>""",unsafe_allow_html=True)

    with tab3:
        shdr("📈","OHLCV Resampler — Interactive")
        tf=st.selectbox("Resample To",["Weekly (W-FRI)","Monthly (ME)","Quarterly (QE)"])
        nd=st.slider("Trading Days",50,504,252,10,key="nd3")
        rng3=np.random.default_rng(42); dt3=pd.bdate_range("2024-01-02",periods=nd)
        cl3=21000+np.cumsum(rng3.normal(10,150,nd))
        df3=pd.DataFrame({"Open":cl3*(1+rng3.normal(0,0.005,nd)),"High":cl3*(1+np.abs(rng3.normal(0,0.008,nd))),
            "Low":cl3*(1-np.abs(rng3.normal(0,0.008,nd))),"Close":cl3,"Volume":rng3.integers(300_000,2_000_000,nd)},index=dt3)
        fc={"Weekly (W-FRI)":"W-FRI","Monthly (ME)":"ME","Quarterly (QE)":"QE"}[tf]
        rs=df3.resample(fc).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
        dr=df3.Close.pct_change().dropna()
        rs["Return(%)"]=((dr.resample(fc).apply(lambda s:(1+s).prod()-1))*100).round(2)
        fig=go.Figure(go.Candlestick(x=rs.index,open=rs.Open,high=rs.High,low=rs.Low,close=rs.Close,
            increasing_line_color=GREEN,decreasing_line_color=RED))
        fig.update_layout(**PL,height=400,title=dict(text=f"<b>NIFTY 50 — {tf} OHLCV</b>",font=dict(color=GOLD,size=13)),
            xaxis_rangeslider_visible=False)
        st.plotly_chart(fig,use_container_width=True)
        c1,c2=st.columns(2)
        with c1: st.dataframe(rs[["Open","High","Low","Close","Volume"]].head(8).round(1),use_container_width=True)
        with c2:
            fig2=go.Figure(go.Bar(x=rs.index,y=rs["Return(%)"],
                marker_color=[GREEN if r>=0 else RED for r in rs["Return(%)"]]))
            fig2.update_layout(**PL,height=280,title=dict(text="<b>Compounded Returns (%)</b>",font=dict(color=GOLD,size=13)))
            st.plotly_chart(fig2,use_container_width=True)
        st.markdown(f'<div class="verdict-ok">✅ Returns compounded correctly: ∏(1+rᵢ) − 1. Summing would overstate performance.</div>',unsafe_allow_html=True)

    with tab4:
        shdr("🎯","G-Sec Yield Curve — Cubic Spline Interpolation")
        c1,c2=st.columns([1,2])
        with c1:
            rows="".join(f"<tr><td style='color:{TXT};'>{m}Y</td><td style='color:{GREEN};'>{y:.2f}%</td></tr>"
                         for m,y in zip(GSEC_M,GSEC_Y))
            st.markdown(f"""<div class="mp-card-blue">
            <b style="color:{LIGHT_BLUE};">Observed G-Sec Maturities</b><br><br>
            <table style="width:100%;font-size:0.83rem;">
            <tr><th style="color:{GOLD};text-align:left;">Tenor</th>
                <th style="color:{GOLD};text-align:left;">Yield</th></tr>{rows}</table></div>""",unsafe_allow_html=True)
        with c2:
            cs=CubicSpline(GSEC_M,GSEC_Y); dn=np.linspace(0.25,30,300)
            mt=st.multiselect("Interpolate missing tenors (years):",
                [4,6,8,9,11,12,15,18,20,25],default=[4,6,8,12,20])
            ym=cs(np.array(mt)) if mt else []
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=dn,y=cs(dn),mode="lines",
                line=dict(color=DARK_BLUE,width=2.5),name="Cubic Spline"))
            fig.add_trace(go.Scatter(x=GSEC_M,y=GSEC_Y,mode="markers",
                marker=dict(color=LIGHT_BLUE,size=10,line=dict(color=DARK_BLUE,width=2)),name="Observed G-Sec"))
            if mt:
                fig.add_trace(go.Scatter(x=list(mt),y=list(ym),mode="markers",
                    marker=dict(color=GOLD,size=14,symbol="diamond",line=dict(color=DARK_BLUE,width=2)),
                    name="Interpolated"))
            fig.update_layout(**PL,height=380,
                title=dict(text="<b>Indian G-Sec Yield Curve — Cubic Spline</b>",font=dict(color=GOLD,size=13)),
                xaxis_title="Maturity (Years)",yaxis_title="Yield (% p.a.)")
            st.plotly_chart(fig,use_container_width=True)
        if mt:
            st.dataframe(pd.DataFrame({"Tenor (Yr)":mt,"Interpolated Yield (%)":
                [round(float(y),4) for y in ym]}),use_container_width=True,hide_index=True)
            st.markdown(f"""<div class="mp-card-blue">
            <b style="color:{LIGHT_BLUE};">FIMMDA & CCIL</b> use cubic spline / Nelson-Siegel-Svensson
            to construct the complete G-Sec yield curve from available liquid benchmark yields.
            </div>""",unsafe_allow_html=True)
    footer()


# ═════════════════════════════════════════════════════════════
# PAGE: INVALID VALUES
# ═════════════════════════════════════════════════════════════
elif page == "⚠️ Invalid Values":
    hero("Identifying & Treating Invalid Values",
         "OHLC Logic Checks · Negative Prices · Stale Data · Duplicate Timestamps",
         ["OHLCV Validation","Stale Detection","Circuit Breakers"])

    tab1,tab2,tab3=st.tabs(["📘 Taxonomy","🔬 Live OHLCV Validator","📋 Treatment Guide"])

    with tab1:
        shdr("📘","What Are Invalid Values?")
        st.markdown(f"""<div class="mp-card">
        An invalid value is a <b>non-null observation</b> that violates:
        <b style="color:{GOLD};">Domain constraints</b> (Price≤0) ·
        <b style="color:{GOLD};">Logical constraints</b> (High&lt;Low) ·
        <b style="color:{GOLD};">Referential integrity</b> (Trade date after settlement) ·
        <b style="color:{GOLD};">Business rules</b> (Volume=0 during session) ·
        <b style="color:{GOLD};">Relational constraints</b> (Assets≠Liabilities+Equity)
        </div>""",unsafe_allow_html=True)
        shdr("📋","Taxonomy of Invalid Values")
        for nm,ex,cause,detect,sev in [
            ("Negative Prices","NSE price=−50","Data feed; sign flip","df[col]>0 assertion",RED),
            ("Impossible OHLC","High=2800, Low=3100","Data entry / feed error","H≥max(O,C)≥L check",RED),
            ("Zero Volume on Trading Day","Volume=0 on active day","Missing trade aggregation","Market calendar cross-check","#fd7e14"),
            ("Crossed Bid-Ask","Bid=105 > Ask=103","Quote data latency","bid<ask assertion","#fd7e14"),
            ("Future-Dated Entries","Trade 2026 in 2024 DB","System clock error","Max date validation","#fd7e14"),
            ("Accounting Identity Breach","Assets≠Liabilities+Equity","Consolidation error","Cross-field formula",RED),
            ("Stale Prices","30 days unchanged in liquid stk","Feed disconnection","Rolling std-dev≈0","#fd7e14"),
        ]:
            st.markdown(f"""
            <div class="mp-card" style="border-left:4px solid {sev};margin-bottom:7px;
                 display:flex;gap:16px;padding:12px 16px;flex-wrap:wrap;">
              <div style="min-width:170px;font-weight:700;color:{sev};">{nm}</div>
              <div style="flex:1;min-width:120px;"><code style="font-size:0.8rem;">{ex}</code></div>
              <div style="flex:1;min-width:120px;font-size:0.83rem;color:{TXT};">{cause}</div>
              <div style="flex:1;min-width:120px;"><code style="font-size:0.78rem;">{detect}</code></div>
            </div>""",unsafe_allow_html=True)

    with tab2:
        shdr("🔬","Interactive OHLCV Validator")
        c1,c2,c3=st.columns(3)
        with c1:
            nd2=st.slider("Trading Days",30,252,80,key="nd2"); neg=st.checkbox("Inject: Negative Price",True)
        with c2:
            ov=st.checkbox("Inject: OHLC Violation",True); zv=st.checkbox("Inject: Zero Volume",True)
        with c3:
            ed=st.checkbox("Inject: Extreme Drop −99%",True); st_=st.checkbox("Inject: Stale Prices",True)
        rng4=np.random.default_rng(99); dt4=pd.bdate_range("2024-01-02",periods=nd2)
        cl4=2800+np.cumsum(rng4.normal(0,30,nd2))
        df4=pd.DataFrame({"Open":cl4*(1+rng4.normal(0,0.005,nd2)),"High":cl4*(1+np.abs(rng4.normal(0,0.008,nd2))),
            "Low":cl4*(1-np.abs(rng4.normal(0,0.008,nd2))),"Close":cl4.copy(),"Volume":rng4.integers(500_000,3_000_000,nd2)},index=dt4)
        if neg and nd2>=6:  df4.loc[dt4[5],"Close"]=-500
        if ov and nd2>=11:  df4.loc[dt4[10],"High"]=df4.loc[dt4[10],"Low"]-100
        if zv and nd2>=16:  df4.loc[dt4[15],"Volume"]=0
        if ed and nd2>=21:  df4.loc[dt4[20],"Close"]*=0.01
        if st_ and nd2>=36:
            for d in dt4[30:35]: df4.loc[d,"Close"]=df4.loc[dt4[29],"Close"]
        iss=validate_ohlcv(df4)
        mrow([{"val":str(len(iss)),"lbl":"Total Issues"},
              {"val":str((iss["Issue"].str.contains("Non-positive") if len(iss)>0 else pd.Series([])).sum()),"lbl":"Negative Prices"},
              {"val":str((iss["Issue"].str.contains("OHLC") if len(iss)>0 else pd.Series([])).sum()),"lbl":"OHLC Violations"},
              {"val":str((iss["Issue"].str.contains("Stale") if len(iss)>0 else pd.Series([])).sum()),"lbl":"Stale Prices"}])
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=df4.index,y=df4.Close.abs().clip(upper=10000),
            mode="lines+markers",line=dict(color=LIGHT_BLUE,width=1.5),marker=dict(size=3)))
        if len(iss)>0:
            for _,row in iss.iterrows():
                try:
                    ts=pd.Timestamp(str(row["Date"]))
                    if ts in df4.index: fig.add_vline(x=ts,line_dash="dash",line_color=RED,opacity=0.4)
                except: pass
        fig.update_layout(**PL,height=340,title=dict(text="<b>RELIANCE.NS — Anomalies (Red=Issues)</b>",font=dict(color=GOLD,size=13)))
        st.plotly_chart(fig,use_container_width=True)
        if len(iss)>0: st.dataframe(iss,use_container_width=True,hide_index=True)
        else: st.markdown(f'<div class="verdict-ok">✅ All validations PASSED — no issues found.</div>',unsafe_allow_html=True)

    with tab3:
        shdr("📋","Treatment Decision Guide")
        for treat,when,impl,clr in [
            ("Convert to NaN","Clear data error (negative price)","df[mask]=np.nan  →  then impute",GREEN),
            ("Rule-Based Correction","Known systematic error (sign flip)","df['Price']=df['Price'].abs()",GREEN),
            ("Replace with Adjacent Valid","OHLC logic violation","Forward-fill or arithmetic correction",LIGHT_BLUE),
            ("Flag and Quarantine","Unknown origin; needs review","df['is_suspect']=1; exclude from model",GOLD),
            ("Winsorise","Valid but extreme cross-sectional","df[col].clip(lower=lo, upper=hi)",LIGHT_BLUE),
            ("Consult Source","Ambiguous; cross-verify needed","NSE / BSE / Bloomberg; log action taken",GOLD),
        ]:
            st.markdown(f"""
            <div class="mp-card" style="border-left:4px solid {clr};margin-bottom:7px;display:flex;gap:16px;flex-wrap:wrap;">
              <div style="min-width:170px;font-weight:700;color:{clr};">🔧 {treat}</div>
              <div style="flex:1;font-size:0.85rem;color:{TXT};">{when}</div>
              <div style="flex:1;"><code style="font-size:0.78rem;">{impl}</code></div>
            </div>""",unsafe_allow_html=True)
    footer()


# ═════════════════════════════════════════════════════════════
# PAGE: FULL PIPELINE
# ═════════════════════════════════════════════════════════════
elif page == "⚙️ Full Cleaning Pipeline":
    hero("End-to-End Financial Data Cleaning Pipeline",
         "Production-Grade FinancialDataCleaner — Validate → Missing → Outliers → Audit Trail",
         ["Production Code","KNN Imputation","Winsorisation","Audit Trail"])

    tab1,tab2,tab3=st.tabs(["🏗️ Architecture","⚙️ Interactive Runner","💻 Source Code"])

    with tab1:
        shdr("🏗️","Four-Stage Pipeline Architecture")
        for num,step,clr,title,pts,method in [
            ("1️⃣","VALIDATE",GREEN,"Schema & Structure",
             ["Detect mixed-type columns","Ensure DatetimeIndex","Sort temporally","Remove duplicate entries"],
             "cleaner.validate_schema(df)"),
            ("2️⃣","MISSING",GOLD,"Missing Data Treatment",
             ["Drop columns >30% missing","<2% → forward fill","2–15% → KNN imputation (k=7)",">15% → median + flag column"],
             "cleaner.treat_missing(df)"),
            ("3️⃣","OUTLIERS","#fd7e14","Outlier Treatment",
             ["Modified Z-Score (MAD) detection","Configurable threshold (default 4.0)","Actions: winsorise / flag / remove","Skip flag columns"],
             "cleaner.treat_outliers(df)"),
            ("4️⃣","REPORT",LIGHT_BLUE,"Cleaning Audit Trail",
             ["Shape before vs after","Missing count before vs after","Step-by-step log","Model governance documentation"],
             "cleaner.cleaning_report(df_orig, df_clean)"),
        ]:
            st.markdown(f"""
            <div class="mp-card" style="border-left:5px solid {clr};margin-bottom:10px;">
              <div style="display:flex;gap:12px;align-items:flex-start;">
                <span style="font-size:1.7rem;">{num}</span>
                <div style="flex:1;">
                  <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:6px;">
                    <code style="font-size:0.72rem;color:{clr};background:#060d1a;padding:2px 7px;border-radius:4px;">{step}</code>
                    <span style="font-weight:700;color:{TXT};">{title}</span>
                  </div>
                  <ul style="margin:0;padding-left:18px;color:{TXT};">
                    {"".join(f'<li style="margin-bottom:2px;font-size:0.84rem;">{p}</li>' for p in pts)}
                  </ul>
                  <code style="font-size:0.78rem;color:{LIGHT_BLUE};margin-top:6px;display:inline-block;">{method}</code>
                </div>
              </div>
            </div>""",unsafe_allow_html=True)

    with tab2:
        shdr("⚙️","Interactive Pipeline Runner")
        c1,c2,c3=st.columns(3)
        with c1: mt=st.slider("Drop if Missing>(%)",10,60,25,5); wl=st.slider("Winsorisation Level(%)",0.5,5.0,1.0,0.5)
        with c2: zt=st.slider("MAD Z-Score Threshold",2.0,6.0,3.5,0.25); kk=st.slider("KNN Neighbours",3,15,7)
        with c3: oa=st.selectbox("Outlier Action",["winsorise","flag","remove"]); nd3=st.slider("Trading Days",100,756,504,50)
        rng5=np.random.default_rng(42); n5=nd3; dt5=pd.bdate_range("2022-01-03",periods=n5)
        df5=pd.DataFrame({"RELIANCE":2800+np.cumsum(rng5.normal(0.5,25,n5)),
            "TCS":3500+np.cumsum(rng5.normal(0.3,40,n5)),"HDFCBANK":1600+np.cumsum(rng5.normal(0.2,18,n5)),
            "PE_REL":rng5.lognormal(3.4,0.3,n5),"PE_TCS":rng5.lognormal(3.5,0.4,n5)},index=dt5)
        df5.iloc[50:55,0]=np.nan; df5.iloc[rng5.choice(n5,20,replace=False),2]=np.nan
        df5.iloc[100,0]=-2800; df5.iloc[150,4]=5000; df5.iloc[200,3]=0.001

        if st.button("▶️  Run Cleaning Pipeline",type="primary"):
            log=[]
            def _l(s,d): log.append({"Step":s,"Detail":d})
            dc=df5.copy()
            _l("VALIDATE",f"Shape: {dc.shape}"); dc.index=pd.to_datetime(dc.index)
            if not dc.index.is_monotonic_increasing: dc=dc.sort_index(); _l("SORT","Sorted")
            nd=dc.index.duplicated().sum()
            if nd>0: dc=dc[~dc.index.duplicated(keep="last")]; _l("DEDUP",f"{nd} dupes removed")
            ms=dc.isnull().mean(); _l("MISSING",f"Overall: {dc.isnull().values.mean()*100:.2f}%")
            dr=ms[ms>mt/100].index.tolist()
            if dr: dc=dc.drop(columns=dr); _l("DROP-COLS",f"Dropped {len(dr)}: {dr}")
            for col in dc.columns:
                pm=dc[col].isnull().mean()
                if pm==0: continue
                elif pm<0.02: dc[col]=dc[col].ffill().bfill(); _l("FFILL",f"'{col}': {pm*100:.1f}%→ffill")
                elif pm>=0.15:
                    dc[f"{col}_imputed"]=dc[col].isnull().astype(int)
                    dc[col]=dc[col].fillna(dc[col].median()); _l("MEDIAN",f"'{col}': {pm*100:.1f}%→median+flag")
            mc=[c for c in dc.columns if 0.02<=dc[c].isnull().mean()<=0.15 and dc[c].dtype in["float64","int64"]]
            if mc:
                sc=StandardScaler(); X=dc[mc]; Xs=pd.DataFrame(sc.fit_transform(X),columns=mc,index=dc.index)
                Xi=pd.DataFrame(sc.inverse_transform(KNNImputer(n_neighbors=kk).fit_transform(Xs)),columns=mc,index=dc.index)
                dc[mc]=Xi; _l("KNN",f"KNN imputed {len(mc)} cols")
            nc=[c for c in dc.select_dtypes(include=[np.number]).columns if not c.endswith("_imputed")]
            for col in nc:
                s=dc[col].dropna()
                if len(s)<30: continue
                med=s.median(); mad=np.median(np.abs(s-med))
                if mad==0: continue
                mz=0.6745*np.abs(dc[col]-med)/mad; mask=mz>zt
                if mask.sum()>0:
                    if oa=="winsorise":
                        lo2=dc[col].quantile(wl/100); hi2=dc[col].quantile(1-wl/100)
                        dc[col]=dc[col].clip(lo2,hi2); _l("WINSORISE",f"'{col}': {mask.sum()}→[{lo2:.1f},{hi2:.1f}]")
                    elif oa=="flag":
                        dc[f"{col}_outlier"]=mask.astype(int); _l("FLAG",f"'{col}': {mask.sum()} flagged")
                    else:
                        dc.loc[mask,col]=np.nan; dc[col]=dc[col].ffill(); _l("REMOVE",f"'{col}': {mask.sum()} removed")
            st.session_state.update({"drp":df5,"dcp":dc,"lgp":log})

        if "dcp" in st.session_state:
            dr2=st.session_state["drp"]; dc2=st.session_state["dcp"]; lg=st.session_state["lgp"]
            mrow([{"val":str(dr2.shape),"lbl":"Original Shape"},{"val":str(dc2.shape),"lbl":"Cleaned Shape"},
                  {"val":str(dr2.isnull().sum().sum()),"lbl":"Missing Before"},
                  {"val":str(dc2.isnull().sum().sum()),"lbl":"Missing After"},{"val":str(len(lg)),"lbl":"Steps"}])
            c1,c2=st.columns(2)
            with c1:
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=dr2.index,y=dr2.RELIANCE.abs(),name="Before",line=dict(color=RED,width=1.5)))
                fig.add_trace(go.Scatter(x=dc2.index,y=dc2.get("RELIANCE",dc2.iloc[:,0]),name="After",line=dict(color=GREEN,width=1.5)))
                fig.update_layout(**PL,height=300,title=dict(text="<b>RELIANCE — Before vs After</b>",font=dict(color=GOLD,size=13)))
                st.plotly_chart(fig,use_container_width=True)
            with c2:
                mb=(dr2.isnull().sum()/len(dr2)*100).reset_index(); mb.columns=["C","P"]
                ma=(dc2.isnull().sum()/len(dc2)*100).reset_index(); ma.columns=["C","P"]
                fig2=go.Figure()
                fig2.add_trace(go.Bar(x=mb.C,y=mb.P,name="Before",marker_color=RED,opacity=0.8))
                fig2.add_trace(go.Bar(x=ma.C,y=ma.P,name="After",marker_color=GREEN,opacity=0.8))
                fig2.update_layout(**PL,height=300,barmode="group",title=dict(text="<b>Missing % Before vs After</b>",font=dict(color=GOLD,size=13)))
                st.plotly_chart(fig2,use_container_width=True)
            st.dataframe(pd.DataFrame(lg),use_container_width=True,hide_index=True)
        else:
            st.info("Click **▶️ Run Cleaning Pipeline** to execute.")

    with tab3:
        shdr("💻","FinancialDataCleaner — Key Methods")
        code_str = '''"""
FinancialDataCleaner — Production-Grade Pipeline
Prof. V. Ravichandran | themountainpathacademy.com
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


class FinancialDataCleaner:
    """
    Production-grade pipeline for cleaning financial time series
    and cross-sectional datasets.

    Stages:
        1. validate_schema  — types, DatetimeIndex, sort, dedupe
        2. treat_missing    — ffill / KNN / median+flag by % missing
        3. treat_outliers   — MAD-based detection → winsorise/flag/remove
        4. cleaning_report  — before/after summary + full audit log
    """

    def __init__(self, config=None):
        self.config = config or {
            "missing_threshold": 0.30,   # Drop columns with > 30% missing
            "winsorise_level":   0.01,   # Winsorise at 1st / 99th percentile
            "zscore_threshold":  4.0,    # Modified Z-Score (MAD) flag threshold
            "knn_neighbors":     7,      # K for KNN imputation
            "outlier_action":    "winsorise",  # "winsorise" | "flag" | "remove"
        }
        self.cleaning_log = []

    def _log(self, step, detail):
        self.cleaning_log.append({"Step": step, "Detail": detail})
        print(f"  [{step:<12}] {detail}")

    # ─────────────────────────────────────────────────────────
    # STAGE 1: Schema Validation
    # ─────────────────────────────────────────────────────────
    def validate_schema(self, df):
        """
        Coerce numeric strings, ensure DatetimeIndex,
        sort chronologically, remove duplicate timestamps.
        """
        self._log("VALIDATE", f"Input shape: {df.shape}")

        # 1a. Coerce object columns that are mostly numeric
        for col in df.select_dtypes(include="object").columns:
            n_num = pd.to_numeric(df[col], errors="coerce").notna().sum()
            if n_num > len(df) * 0.8:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                self._log("TYPE-FIX", f"'{col}': coerced to numeric")

        # 1b. Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
                self._log("INDEX", "Converted index to DatetimeIndex")
            except Exception as e:
                self._log("INDEX-WARN", f"Could not parse index: {e}")

        # 1c. Sort chronologically
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            self._log("SORT", "Sorted index ascending (chronological)")

        # 1d. Remove duplicate timestamps (keep last)
        n_dupes = df.index.duplicated().sum()
        if n_dupes > 0:
            df = df[~df.index.duplicated(keep="last")]
            self._log("DEDUP", f"Removed {n_dupes} duplicate timestamp(s)")

        return df

    # ─────────────────────────────────────────────────────────
    # STAGE 2: Missing Data Treatment
    # ─────────────────────────────────────────────────────────
    def treat_missing(self, df):
        """
        Three-tier strategy based on % missing per column:
            < 2%         -> Forward-fill (LOCF) then backward-fill
            2 – 15%      -> KNN imputation (batch, StandardScaler)
            > 15%        -> Median imputation + binary flag column
            > threshold  -> Column dropped entirely
        """
        miss_pct = df.isnull().mean()
        overall  = df.isnull().values.mean() * 100
        self._log("MISSING", f"Overall missing rate: {overall:.2f}%")

        # Drop columns exceeding missing_threshold
        drop_cols = miss_pct[
            miss_pct > self.config["missing_threshold"]
        ].index.tolist()
        if drop_cols:
            df = df.drop(columns=drop_cols)
            self._log("DROP-COLS",
                f"Dropped {len(drop_cols)} col(s) "
                f"(>{self.config['missing_threshold']*100:.0f}% missing): {drop_cols}")
            miss_pct = df.isnull().mean()

        # Per-column strategy
        for col in df.columns:
            pm = df[col].isnull().mean()
            if pm == 0:
                continue
            elif pm < 0.02:
                df[col] = df[col].ffill().bfill()
                self._log("FFILL", f"'{col}': {pm*100:.1f}% -> forward/backward fill")
            elif pm >= 0.15:
                med = df[col].median()
                df[f"{col}_imputed"] = df[col].isnull().astype(int)
                df[col] = df[col].fillna(med)
                self._log("MEDIAN",
                    f"'{col}': {pm*100:.1f}% -> median ({med:.2f}); "
                    f"flag col '{col}_imputed' added")
            # 2–15%: handled in batch KNN block below

        # Batch KNN for moderate-missing numeric columns
        mod_cols = [
            c for c in df.columns
            if 0.02 <= df[c].isnull().mean() <= 0.15
            and df[c].dtype in ["float64", "int64"]
            and not c.endswith("_imputed")
        ]
        if mod_cols:
            scaler   = StandardScaler()
            X        = df[mod_cols]
            Xs       = pd.DataFrame(scaler.fit_transform(X),
                                    columns=mod_cols, index=df.index)
            imputer  = KNNImputer(n_neighbors=self.config["knn_neighbors"],
                                  weights="distance")
            Xi       = pd.DataFrame(
                           scaler.inverse_transform(imputer.fit_transform(Xs)),
                           columns=mod_cols, index=df.index)
            df[mod_cols] = Xi
            self._log("KNN",
                f"KNN (k={self.config['knn_neighbors']}) imputed "
                f"{len(mod_cols)} col(s): {mod_cols}")

        return df

    # ─────────────────────────────────────────────────────────
    # STAGE 3: Outlier Treatment
    # ─────────────────────────────────────────────────────────
    def treat_outliers(self, df, numeric_cols=None):
        """
        Modified Z-Score (MAD) — robust to outliers in the data itself,
        unlike standard Z-Score where sigma is inflated by extreme values.

            MAD        = Median |Xi - X_tilde|
            Modified Z = 0.6745 * |Xi - X_tilde| / MAD

        Actions:
            "winsorise" -> clip to [lo_pct, hi_pct] percentile bounds
            "flag"      -> add <col>_outlier binary column
            "remove"    -> set to NaN then forward-fill
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols
                        if not c.endswith(("_imputed", "_outlier"))]

        action = self.config["outlier_action"]

        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) < 30:
                continue
            median = s.median()
            mad    = np.median(np.abs(s - median))
            if mad == 0:
                continue
            mod_z = 0.6745 * np.abs(df[col] - median) / mad
            mask  = mod_z > self.config["zscore_threshold"]
            n_out = mask.sum()
            if n_out == 0:
                continue

            if action == "winsorise":
                lo = df[col].quantile(self.config["winsorise_level"])
                hi = df[col].quantile(1 - self.config["winsorise_level"])
                df[col] = df[col].clip(lower=lo, upper=hi)
                self._log("WINSORISE",
                    f"'{col}': {n_out} outlier(s) clipped to [{lo:.2f}, {hi:.2f}]")
            elif action == "flag":
                df[f"{col}_outlier"] = mask.astype(int)
                self._log("FLAG", f"'{col}': {n_out} outlier(s) flagged")
            elif action == "remove":
                df.loc[mask, col] = np.nan
                df[col] = df[col].ffill()
                self._log("REMOVE", f"'{col}': {n_out} outlier(s) -> NaN + ffill")

        return df

    # ─────────────────────────────────────────────────────────
    # STAGE 4: Cleaning Report
    # ─────────────────────────────────────────────────────────
    def cleaning_report(self, df_original, df_clean):
        """Print before/after summary and return log as DataFrame."""
        print("\\n" + "=" * 60)
        print("  FINANCIAL DATA CLEANING REPORT")
        print("=" * 60)
        print(f"  Original shape  : {df_original.shape}")
        print(f"  Cleaned  shape  : {df_clean.shape}")
        print(f"  Rows delta      : {df_clean.shape[0] - df_original.shape[0]:+d}")
        print(f"  Cols delta      : {df_clean.shape[1] - df_original.shape[1]:+d}")
        print(f"  Missing (before): {df_original.isnull().sum().sum()}")
        print(f"  Missing (after) : {df_clean.isnull().sum().sum()}")
        print(f"  Cleaning steps  : {len(self.cleaning_log)}")
        print("\\nCleaning log:")
        for e in self.cleaning_log:
            print(f"  {e['Step']:<14} {e['Detail']}")
        return pd.DataFrame(self.cleaning_log)

    # ─────────────────────────────────────────────────────────
    # MASTER METHOD
    # ─────────────────────────────────────────────────────────
    def clean(self, df):
        """Run all four stages and return the cleaned DataFrame."""
        print("\\nStarting Financial Data Cleaning Pipeline...")
        print("-" * 50)
        self.cleaning_log = []   # reset for re-runs
        df_clean = df.copy()
        df_clean = self.validate_schema(df_clean)
        df_clean = self.treat_missing(df_clean)
        df_clean = self.treat_outliers(df_clean)
        return df_clean


# ──────────────────────────────────────────────────────────────
# USAGE EXAMPLE
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    dates  = pd.bdate_range("2022-01-03", "2024-12-31")
    n      = len(dates)
    df_raw = pd.DataFrame({
        "RELIANCE" : 2800 + np.cumsum(np.random.normal(0.5, 25, n)),
        "TCS"      : 3500 + np.cumsum(np.random.normal(0.3, 40, n)),
        "HDFCBANK" : 1600 + np.cumsum(np.random.normal(0.2, 18, n)),
        "PE_REL"   : np.random.lognormal(3.4, 0.3, n),
        "PE_TCS"   : np.random.lognormal(3.5, 0.4, n),
    }, index=dates)

    # Inject realistic problems
    df_raw.iloc[50:55, 0]                   = np.nan    # block missing
    df_raw.iloc[np.random.choice(n, 20), 2] = np.nan    # random missing
    df_raw.iloc[100, 0]                     = -2800     # negative price
    df_raw.iloc[150, 4]                     = 5000      # extreme outlier PE
    df_raw.iloc[200, 3]                     = 0.001     # extreme low PE

    cleaner = FinancialDataCleaner(config={
        "missing_threshold" : 0.25,
        "winsorise_level"   : 0.01,
        "zscore_threshold"  : 3.5,
        "knn_neighbors"     : 7,
        "outlier_action"    : "winsorise",
    })

    df_cleaned = cleaner.clean(df_raw)
    cleaner.cleaning_report(df_raw, df_cleaned)
'''
        st.code(code_str, language="python")
    footer()


# ═════════════════════════════════════════════════════════════
# PAGE: CASE STUDIES
# ═════════════════════════════════════════════════════════════
elif page == "📚 Case Studies":
    hero("Indian Financial Market Case Studies",
         "Demonetisation Shock · IL&FS Default · NSE Data Feed Outage",
         ["Nov 2016","Sep 2018","Feed MCAR"])

    # ── Helper: styled download button ────────────────────────
    def dl_button(csv_bytes, filename, label):
        st.download_button(
            label=label,
            data=csv_bytes,
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
        )

    def dl_info(rows, cols, note):
        st.markdown(f"""
        <div class="mp-card-blue" style="padding:12px 16px;margin-bottom:0;">
          <div style="font-size:0.78rem;color:{MUTED};margin-bottom:4px;text-transform:uppercase;letter-spacing:0.05em;">Dataset Info</div>
          <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <span style="font-size:0.85rem;color:{TXT};">📏 <b style="color:{GOLD};">{rows}</b> rows</span>
            <span style="font-size:0.85rem;color:{TXT};">🗂️ <b style="color:{GOLD};">{cols}</b> columns</span>
          </div>
          <div style="font-size:0.82rem;color:{LIGHT_BLUE};margin-top:6px;">ℹ️ {note}</div>
        </div>""", unsafe_allow_html=True)

    # ── Build all three datasets upfront (same seeds as charts) ──

    # CASE 1 ─ Demonetisation: full 2016 daily return series
    rng6 = np.random.default_rng(42)
    n6   = 252
    dt6  = pd.bdate_range("2016-01-04", periods=n6)
    ret6 = rng6.normal(0.04/252, 0.12/np.sqrt(252), n6)
    ret6[215]=-0.040; ret6[216]=-0.018; ret6[217]=-0.012; ret6[218]=0.008
    s6   = pd.Series(ret6*100, index=dt6)
    zs6  = np.abs((s6-s6.mean())/s6.std())
    # Enrich with price level and event flag
    price6 = 7946 * (1 + s6/100).cumprod()   # start near actual 2016 NIFTY open
    df_cs1 = pd.DataFrame({
        "Date":              dt6.strftime("%Y-%m-%d"),
        "NIFTY50_Close":     price6.round(2).values,
        "Daily_Return_Pct":  s6.round(4).values,
        "Z_Score":           zs6.round(4).values,
        "Outlier_Flag":      (zs6 > 3).astype(int).values,
        "Demonetisation":    [1 if d.strftime("%Y-%m-%d") in
                              ["2016-11-08","2016-11-09","2016-11-10","2016-11-11"] else 0
                              for d in dt6],
        "Event_Note":        ["Demonetisation Announced" if d.strftime("%Y-%m-%d")=="2016-11-08"
                              else ("Post-Demonetisation Shock" if d.strftime("%Y-%m-%d") in
                              ["2016-11-09","2016-11-10","2016-11-11"] else "")
                              for d in dt6],
    })

    # CASE 2 ─ IL&FS: quarterly financials panel with missingness
    rng7 = np.random.default_rng(7)
    qtrs = ["Q1 2017","Q2 2017","Q3 2017","Q4 2017",
            "Q1 2018","Q2 2018","Q3 2018","Q4 2018"]
    mr   = [0.05, 0.06, 0.07, 0.09, 0.14, 0.22, 0.45, 0.65]
    cfo_v= [round(float(rng7.normal(200,20)),2), round(float(rng7.normal(195,22)),2),
             round(float(rng7.normal(180,25)),2), round(float(rng7.normal(160,30)),2),
             round(float(rng7.normal(120,40)),2), None, None, None]
    rev_v= [round(float(rng7.normal(1800,80)),2), round(float(rng7.normal(1780,90)),2),
             round(float(rng7.normal(1740,100)),2), round(float(rng7.normal(1690,120)),2),
             round(float(rng7.normal(1580,150)),2), round(float(rng7.normal(1420,200)),2),
             None, None]
    debt_v=[round(float(rng7.normal(9200,150)),2), round(float(rng7.normal(9450,160)),2),
             round(float(rng7.normal(9750,180)),2), round(float(rng7.normal(10100,200)),2),
             round(float(rng7.normal(10600,250)),2), None, None, None]
    icr_v= [round(float(rng7.normal(1.85,0.08)),3), round(float(rng7.normal(1.72,0.09)),3),
             round(float(rng7.normal(1.61,0.10)),3), round(float(rng7.normal(1.44,0.12)),3),
             round(float(rng7.normal(1.18,0.15)),3), None, None, None]
    df_cs2 = pd.DataFrame({
        "Quarter":               qtrs,
        "Data_Missingness_Pct":  [round(m*100,1) for m in mr],
        "Operating_CFO_Cr":      cfo_v,
        "Revenue_Cr":            rev_v,
        "Total_Debt_Cr":         debt_v,
        "Interest_Coverage":     icr_v,
        "CFO_Missing_Flag":      [1 if v is None else 0 for v in cfo_v],
        "Revenue_Missing_Flag":  [1 if v is None else 0 for v in rev_v],
        "Debt_Missing_Flag":     [1 if v is None else 0 for v in debt_v],
        "Distress_Label":        [0,0,0,0,0,1,1,1],
        "Event_Note":            ["Normal","Normal","Liquidity squeeze emerging",
                                  "Covenant breach reported","Credit downgrade warning",
                                  "CP default — disclosure halted",
                                  "NCLT proceedings","Resolution in progress"],
    })

    # CASE 3 ─ NSE Feed Outage: 1-min intraday with gap
    rng8 = np.random.default_rng(99)
    ts8  = pd.date_range("2024-01-15 09:15", "2024-01-15 15:30", freq="1min")
    pr8  = 21500 + np.cumsum(rng8.normal(0, 15, len(ts8)))
    vol8 = rng8.integers(5000, 80000, len(ts8))
    gap  = (ts8 >= "2024-01-15 10:30") & (ts8 <= "2024-01-15 13:30")
    pr8g = pr8.copy().astype(float); pr8g[gap] = np.nan
    vol8g= vol8.copy().astype(float); vol8g[gap] = np.nan
    df_cs3 = pd.DataFrame({
        "Timestamp_IST":      ts8.strftime("%Y-%m-%d %H:%M"),
        "NIFTY50_Price_True": pr8.round(2),
        "NIFTY50_Price_Obs":  pr8g.round(2),   # NaN during outage
        "Volume_True":        vol8,
        "Volume_Obs":         vol8g,
        "Feed_Outage_Flag":   gap.astype(int),
        "Missingness_Mechanism": ["MCAR" if g else "Complete" for g in gap],
    })

    # ── TABS ──────────────────────────────────────────────────
    tab1,tab2,tab3 = st.tabs([
        "📚 Case 1: Demonetisation (2016)",
        "📚 Case 2: IL&FS Default (2018)",
        "📚 Case 3: NSE Feed Outage"])

    # ══════════════════════════════════════════════════════════
    with tab1:
        st.markdown(f"""<div class="hero-wrap" style="padding:20px 28px;">
        <span class="badge">Case Study 1</span>
        <span class="badge badge-red">GENUINE EXTREME EVENT</span>
        <h3 style="color:{GOLD};margin:10px 0 6px;font-family:'Playfair Display',serif;">
          Demonetisation Shock — Outlier or Structural Break?</h3>
        <p style="color:{TXT};margin:0;">8 November 2016: NIFTY 50 fell 6.1% intraday, closed −4.0%.
          Z-score ≈ −5.2σ. The question: should this be removed from the dataset?</p>
        </div>""", unsafe_allow_html=True)

        sub1a, sub1b = st.tabs(["📊 Analysis", "📥 Download Raw Data"])

        with sub1a:
            zs6_  = np.abs((s6-s6.mean())/s6.std())
            fig=make_subplots(rows=2,cols=1,shared_xaxes=True,
                subplot_titles=["NIFTY 50 Daily Returns (%)","Z-Score"],row_heights=[0.65,0.35])
            fig.add_trace(go.Bar(x=s6.index,y=s6,
                marker_color=[RED if v<0 else GREEN for v in s6]),row=1,col=1)
            fig.add_vrect(x0="2016-11-08",x1="2016-11-11",
                fillcolor="rgba(255,215,0,0.09)",line_color="rgba(255,215,0,0.4)",
                annotation_text="Demonetisation",annotation_position="top left",row=1,col=1)
            fig.add_trace(go.Scatter(x=s6.index,y=zs6_,mode="lines",
                line=dict(color=LIGHT_BLUE,width=1.5)),row=2,col=1)
            fig.add_hline(y=3,line_dash="dash",line_color=RED,row=2,col=1)
            fig.update_layout(**PL,height=430,showlegend=False,
                title=dict(text="<b>NIFTY 50 — 2016 Returns with Demonetisation Shock</b>",
                           font=dict(color=GOLD,size=13)))
            st.plotly_chart(fig,use_container_width=True)

            for uc,action,reason,clr in [
                ("VaR / ES Models","🟢 RETAIN","This IS the tail event the model must price.",GREEN),
                ("CAPM / Factor Regression","🟡 FLAG + DUMMY","Add D_nov2016=1 indicator instead of removing.",GOLD),
                ("GARCH Volatility","🟢 RETAIN","Post-demonetisation volatility clustering is a key stylised fact.",GREEN),
                ("Stress Testing","🟢 RETAIN","Prime stress scenario. Historical simulation VaR must include this.",GREEN),
            ]:
                st.markdown(f"""
                <div class="mp-card" style="border-left:4px solid {clr};margin-bottom:6px;display:flex;gap:14px;">
                  <div style="min-width:170px;">
                    <div style="font-weight:700;color:{TXT};font-size:0.87rem;">{uc}</div>
                    <div style="font-weight:700;color:{clr};font-size:0.84rem;">{action}</div>
                  </div>
                  <div style="font-size:0.85rem;color:{TXT};">{reason}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown(f'<div class="verdict-bad">⚠️ A mechanical |Z|>3 rule would flag Demonetisation for removal — gutting the tail risk signal. Domain knowledge must override blind statistical rules.</div>',unsafe_allow_html=True)

        with sub1b:
            shdr("📥","Case 1 Raw Data — NIFTY 50 Daily Returns (2016)")
            st.markdown(f"""<div class="mp-card">
            This dataset contains the full 2016 NIFTY 50 daily return series used in the analysis,
            enriched with Z-scores, outlier flags, and the Demonetisation event marker.
            Use it to practise outlier detection, VaR back-testing, and GARCH volatility modelling.
            </div>""", unsafe_allow_html=True)

            # Preview table
            st.dataframe(df_cs1.head(20), use_container_width=True, hide_index=True)

            c1,c2 = st.columns(2)
            with c1:
                dl_info(len(df_cs1), len(df_cs1.columns),
                    "Simulated NSE-calibrated data · Seed 42 · 252 business days · 2016")
            with c2:
                st.markdown("<br>", unsafe_allow_html=True)
                dl_button(
                    df_cs1.to_csv(index=False).encode("utf-8"),
                    "case1_nifty50_demonetisation_2016.csv",
                    "⬇️  Download Case 1 CSV"
                )

            st.markdown("<br>", unsafe_allow_html=True)
            shdr("📋","Column Definitions")
            st.dataframe(pd.DataFrame({
                "Column":      ["Date","NIFTY50_Close","Daily_Return_Pct","Z_Score",
                                "Outlier_Flag","Demonetisation","Event_Note"],
                "Type":        ["date","float","float","float","int (0/1)","int (0/1)","string"],
                "Description": [
                    "Trading date (business days only, YYYY-MM-DD)",
                    "NIFTY 50 index level (simulated, calibrated to 2016 range)",
                    "Daily log return as a percentage",
                    "Standard Z-score of daily return (|Z| > 3 → flagged)",
                    "1 if |Z| > 3 (statistical outlier), else 0",
                    "1 on the four Demonetisation shock days (8–11 Nov 2016)",
                    "Human-readable event label for the date",
                ],
            }), use_container_width=True, hide_index=True)

            st.markdown(f"""<div class="mp-card-blue">
            <b style="color:{LIGHT_BLUE};">Suggested Exercises with this Dataset:</b><br>
            1. Compute rolling 21-day VaR at 95% and 99% — with and without the Demonetisation rows<br>
            2. Fit a GARCH(1,1) model on <code>Daily_Return_Pct</code> and observe the volatility clustering<br>
            3. Run a CAPM regression adding <code>Demonetisation</code> as a dummy variable<br>
            4. Compare the return distribution using QQ-plots before and after Winsorisation
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    with tab2:
        st.markdown(f"""<div class="hero-wrap" style="padding:20px 28px;">
        <span class="badge">Case Study 2</span>
        <span class="badge badge-red">MNAR — Danger Signal</span>
        <h3 style="color:{GOLD};margin:10px 0 6px;font-family:'Playfair Display',serif;">
          IL&FS Default — Missing Data as a Warning Signal</h3>
        <p style="color:{TXT};margin:0;">IL&FS defaulted on commercial paper in September 2018.
          Months before, subsidiaries began <b>selectively reporting</b> — the missingness was MNAR.</p>
        </div>""", unsafe_allow_html=True)

        sub2a, sub2b = st.tabs(["📊 Analysis", "📥 Download Raw Data"])

        with sub2a:
            fig=make_subplots(rows=1,cols=2,
                subplot_titles=["Missingness Rate Over Time","Operating Cash Flow (₹ Cr)"])
            fig.add_trace(go.Bar(x=qtrs,y=[m*100 for m in mr],
                marker_color=[GREEN if m<0.10 else(GOLD if m<0.25 else RED) for m in mr]),row=1,col=1)
            fig.add_hline(y=10,line_dash="dash",line_color=GOLD,
                annotation_text="Warning 10%",row=1,col=1)
            cfo_plot = [v if v is not None else np.nan for v in cfo_v]
            fig.add_trace(go.Scatter(x=qtrs,y=cfo_plot,mode="lines+markers",
                line=dict(color=LIGHT_BLUE,width=2)),row=1,col=2)
            fig.update_layout(**PL,height=360,showlegend=False,
                title=dict(text="<b>IL&FS: Missing Data as Early Warning Signal</b>",
                           font=dict(color=GOLD,size=13)))
            st.plotly_chart(fig,use_container_width=True)

            for i,(l,d) in enumerate([
                ("Investigate WHY before deciding HOW","The mechanism (MNAR vs MAR) changes the approach entirely."),
                ("Systematic missing = RED FLAG in credit","Missing disclosure is itself a distress signal."),
                ("Missingness flags as model features","Binary flag (cash_flow_missing=1) → powerful ML predictor."),
                ("MAR assumption missed the signal","Asking 'why?' correctly identified distress before default."),
            ],1):
                st.markdown(f"""
                <div class="mp-card" style="margin-bottom:7px;display:flex;gap:12px;">
                  <div style="width:28px;height:28px;background:{GOLD};color:{DARK_BLUE};border-radius:50%;
                       display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0;">{i}</div>
                  <div><div style="font-weight:700;color:{TXT};">{l}</div>
                  <div style="font-size:0.83rem;color:{MUTED};margin-top:2px;">{d}</div></div>
                </div>""", unsafe_allow_html=True)

        with sub2b:
            shdr("📥","Case 2 Raw Data — IL&FS Quarterly Financial Panel (2017–2018)")
            st.markdown(f"""<div class="mp-card">
            Quarterly financial data for the IL&FS group covering eight quarters (Q1 2017 – Q4 2018).
            The dataset deliberately includes MNAR-pattern missingness — financial disclosures that
            disappeared as the entity approached default. Includes binary flags for ML credit models.
            </div>""", unsafe_allow_html=True)

            st.dataframe(df_cs2, use_container_width=True, hide_index=True)

            c1,c2 = st.columns(2)
            with c1:
                dl_info(len(df_cs2), len(df_cs2.columns),
                    "Simulated MNAR panel · Calibrated to IL&FS public disclosures · 8 quarters")
            with c2:
                st.markdown("<br>", unsafe_allow_html=True)
                dl_button(
                    df_cs2.to_csv(index=False).encode("utf-8"),
                    "case2_ilfs_quarterly_mnar_panel.csv",
                    "⬇️  Download Case 2 CSV"
                )

            st.markdown("<br>", unsafe_allow_html=True)
            shdr("📋","Column Definitions")
            st.dataframe(pd.DataFrame({
                "Column": [
                    "Quarter","Data_Missingness_Pct","Operating_CFO_Cr","Revenue_Cr",
                    "Total_Debt_Cr","Interest_Coverage","CFO_Missing_Flag",
                    "Revenue_Missing_Flag","Debt_Missing_Flag","Distress_Label","Event_Note"],
                "Type": ["string","float","float (NaN)","float (NaN)","float (NaN)",
                         "float (NaN)","int","int","int","int (0/1)","string"],
                "Description": [
                    "Fiscal quarter label",
                    "% of financial fields missing for this quarter",
                    "Operating cash flow in ₹ Crore (NaN = not disclosed)",
                    "Revenue in ₹ Crore (NaN = not disclosed)",
                    "Total debt in ₹ Crore (NaN = not disclosed)",
                    "EBIT / Interest expense (NaN = not disclosed)",
                    "1 if CFO not disclosed, else 0 (use as ML feature)",
                    "1 if Revenue not disclosed, else 0",
                    "1 if Debt not disclosed, else 0",
                    "1 = distress / post-default quarter, 0 = normal",
                    "Human-readable event label",
                ],
            }), use_container_width=True, hide_index=True)

            st.markdown(f"""<div class="mp-card-blue">
            <b style="color:{LIGHT_BLUE};">Suggested Exercises with this Dataset:</b><br>
            1. Demonstrate MNAR: show that <code>CFO_Missing_Flag</code> correlates with <code>Distress_Label</code><br>
            2. Build a logistic regression using missing flags as features — compare AUC with and without flags<br>
            3. Apply KNN imputation (k=3) and compare imputed CFO with the true trajectory<br>
            4. Compute a distress scoring model: weighted sum of ICR, CFO trend, and missingness rate
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    with tab3:
        st.markdown(f"""<div class="hero-wrap" style="padding:20px 28px;">
        <span class="badge">Case Study 3</span>
        <span class="badge badge-blue">MCAR — Feed Outage</span>
        <h3 style="color:{GOLD};margin:10px 0 6px;font-family:'Playfair Display',serif;">
          NSE Data Feed Outage — Block Missing Treatment</h3>
        <p style="color:{TXT};margin:0;">A quant fund experiences a <b>3-hour outage (10:30–13:30 IST)</b>.
          Missing data is MCAR — random technical failure unrelated to market conditions.</p>
        </div>""", unsafe_allow_html=True)

        sub3a, sub3b = st.tabs(["📊 Analysis", "📥 Download Raw Data"])

        with sub3a:
            s8  = pd.Series(pr8,  index=ts8)
            s8g = pd.Series(pr8g, index=ts8)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=s8.index,y=s8,name="True (hypothetical)",
                line=dict(color=GREEN,width=1,dash="dot"),opacity=0.4))
            fig.add_trace(go.Scatter(x=s8g.index,y=s8g,name="Observed (with outage)",
                line=dict(color=LIGHT_BLUE,width=2)))
            fig.add_vrect(x0="2024-01-15 10:30",x1="2024-01-15 13:30",
                fillcolor="rgba(220,53,69,0.09)",line_color="rgba(220,53,69,0.33)",
                annotation_text="Feed Outage (MCAR)",annotation_position="top left")
            fig.update_layout(**PL,height=360,
                title=dict(text="<b>NIFTY 50 Intraday — 3-Hour Data Feed Outage</b>",
                           font=dict(color=GOLD,size=13)),
                xaxis_title="Time (IST)",yaxis_title="Index Value")
            st.plotly_chart(fig,use_container_width=True)

            for uc,treat,reason,clr in [
                ("End-of-Day P&L","Forward-fill; use closing prices","Final closing prices available",GREEN),
                ("Intraday VaR","Flag and EXCLUDE the 3-hour window","Cannot reliably impute for risk",RED),
                ("Momentum Signal","Do NOT trade during outage","Stale prices → false signals",RED),
                ("VWAP","Partial VWAP using available trades","Note incomplete window",GOLD),
                ("Historical Backtesting","Fill with exchange-published data","NSE/BSE historical tick downloads",GOLD),
            ]:
                st.markdown(f"""
                <div class="mp-card" style="border-left:4px solid {clr};margin-bottom:6px;
                     display:flex;gap:14px;flex-wrap:wrap;padding:10px 16px;">
                  <div style="min-width:150px;font-weight:700;color:{LIGHT_BLUE};font-size:0.87rem;">{uc}</div>
                  <div style="flex:1;font-weight:600;color:{clr};font-size:0.85rem;">{treat}</div>
                  <div style="flex:2;font-size:0.83rem;color:{TXT};">{reason}</div>
                </div>""", unsafe_allow_html=True)

        with sub3b:
            shdr("📥","Case 3 Raw Data — NIFTY 50 Intraday Feed (1-Min, 15 Jan 2024)")
            st.markdown(f"""<div class="mp-card">
            One full trading day of 1-minute NIFTY 50 tick data (09:15 – 15:30 IST) with a simulated
            3-hour MCAR feed outage injected (10:30 – 13:30). Includes both the true price series and
            the observed series with NaN gaps, plus volume data and outage flags for treatment exercises.
            </div>""", unsafe_allow_html=True)

            # Quick summary metrics
            n_obs   = int((~gap).sum())
            n_gap   = int(gap.sum())
            gap_pct = round(n_gap / len(ts8) * 100, 1)
            mrow([
                {"val": str(len(df_cs3)), "lbl": "Total 1-Min Bars"},
                {"val": str(n_obs),       "lbl": "Complete Observations"},
                {"val": str(n_gap),       "lbl": "Missing (Outage)"},
                {"val": f"{gap_pct}%",    "lbl": "% Data Lost"},
            ])
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df_cs3.head(30), use_container_width=True, hide_index=True)

            c1,c2 = st.columns(2)
            with c1:
                dl_info(len(df_cs3), len(df_cs3.columns),
                    "1-min intraday · MCAR outage 10:30–13:30 IST · 375 bars · Seed 99")
            with c2:
                st.markdown("<br>", unsafe_allow_html=True)
                dl_button(
                    df_cs3.to_csv(index=False).encode("utf-8"),
                    "case3_nse_intraday_feed_outage_2024.csv",
                    "⬇️  Download Case 3 CSV"
                )

            st.markdown("<br>", unsafe_allow_html=True)
            shdr("📋","Column Definitions")
            st.dataframe(pd.DataFrame({
                "Column": [
                    "Timestamp_IST","NIFTY50_Price_True","NIFTY50_Price_Obs",
                    "Volume_True","Volume_Obs","Feed_Outage_Flag","Missingness_Mechanism"],
                "Type": ["datetime","float","float (NaN)","int","float (NaN)","int (0/1)","string"],
                "Description": [
                    "1-minute bar timestamp in IST (YYYY-MM-DD HH:MM)",
                    "Hypothetical true price (unaffected by outage)",
                    "Observed price — NaN during 10:30–13:30 outage window",
                    "Simulated trade volume (true, unaffected)",
                    "Observed volume — NaN during outage (feed not received)",
                    "1 during feed outage window, 0 otherwise",
                    "'MCAR' during outage, 'Complete' otherwise",
                ],
            }), use_container_width=True, hide_index=True)

            st.markdown(f"""<div class="mp-card-blue">
            <b style="color:{LIGHT_BLUE};">Suggested Exercises with this Dataset:</b><br>
            1. Compute VWAP using <code>NIFTY50_Price_Obs</code> × <code>Volume_Obs</code> — note partial result vs true VWAP<br>
            2. Apply forward-fill to <code>NIFTY50_Price_Obs</code> and compute returns — count artificial zero-return bars<br>
            3. Build a "data quality gate": flag any 5-min window with completeness &lt;80%<br>
            4. Compare intraday volatility (std of returns) before and after the outage window
            </div>""", unsafe_allow_html=True)

    # ── Combined download of all three ──────────────────────
    st.markdown("---")
    shdr("📦","Download All Case Study Datasets")
    st.markdown(f"""<div class="mp-card">
    Download each dataset individually as CSV, or grab all three in a single ZIP archive
    — no additional packages required.
    </div>""", unsafe_allow_html=True)

    import io, zipfile

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("case1_nifty50_demonetisation_2016.csv",
                    df_cs1.to_csv(index=False))
        zf.writestr("case2_ilfs_quarterly_mnar_panel.csv",
                    df_cs2.to_csv(index=False))
        zf.writestr("case3_nse_intraday_feed_outage_2024.csv",
                    df_cs3.to_csv(index=False))
        zf.writestr("README.txt",
            "Mountain Path - World of Finance\n"
            "Case Study Datasets\n"
            "Prof. V. Ravichandran | themountainpathacademy.com\n\n"
            "Files:\n"
            "  case1_nifty50_demonetisation_2016.csv   -- 252 daily rows, 2016 NIFTY 50\n"
            "  case2_ilfs_quarterly_mnar_panel.csv     -- 8 quarterly rows, IL&FS (MNAR)\n"
            "  case3_nse_intraday_feed_outage_2024.csv -- 375 one-minute rows, NSE (MCAR)\n"
        )
    zip_buf.seek(0)

    c1, c2, c3 = st.columns(3)
    with c1:
        dl_button(df_cs1.to_csv(index=False).encode("utf-8"),
            "case1_nifty50_demonetisation_2016.csv", "\u2b07\ufe0f  Case 1 CSV")
    with c2:
        dl_button(df_cs2.to_csv(index=False).encode("utf-8"),
            "case2_ilfs_quarterly_mnar_panel.csv", "\u2b07\ufe0f  Case 2 CSV")
    with c3:
        dl_button(df_cs3.to_csv(index=False).encode("utf-8"),
            "case3_nse_intraday_feed_outage_2024.csv", "\u2b07\ufe0f  Case 3 CSV")

    st.download_button(
        label="\U0001f4e6  Download All Three Datasets - ZIP Archive (3 CSVs + README)",
        data=zip_buf,
        file_name="mountain_path_case_studies_data.zip",
        mime="application/zip",
        use_container_width=True,
    )
    footer()


# ═════════════════════════════════════════════════════════════
# PAGE: QUIZ
# ═════════════════════════════════════════════════════════════
elif page == "❓ Quiz & Assessment":
    hero("Quiz, Assessment & Study Guide",
         "Financial Data Wrangling & Cleaning — Learn · Practise · Master",
         ["Study Guide","8-Question Quiz","Instant Feedback","Detailed Explanations"])

    tab_study, tab_quiz = st.tabs([
        "📖 Study Guide — Q&A Format",
        "✏️ Take the Quiz",
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — STUDY GUIDE (Q&A format, all topics)
    # ══════════════════════════════════════════════════════════
    with tab_study:
        shdr("📖", "Complete Study Guide — Question & Answer Format")
        st.markdown(f"""<div class="mp-card">
        This study guide covers <b>every topic</b> in the course using a question-and-answer format.
        Click any question to reveal the full explanation. Work through each section before taking the quiz.
        </div>""", unsafe_allow_html=True)

        # ── HELPER: renders one Q&A card ──────────────────────
        def qa(q, a, tag="", formula=None, example=None, warning=None, tip=None):
            tag_html = f'<span class="badge" style="font-size:10px;margin-bottom:8px;">{tag}</span><br>' if tag else ""
            formula_html = f"""<div class="formula-box" style="margin:10px 0;padding:12px 18px;font-size:0.9rem;">
                {formula}</div>""" if formula else ""
            example_html = f"""<div class="mp-card-blue" style="margin:8px 0;padding:10px 14px;">
                <span style="font-size:0.75rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.05em;">
                Example</span><br><span style="font-size:0.85rem;">{example}</span></div>""" if example else ""
            warn_html = f"""<div class="mp-card-red" style="margin:8px 0;padding:10px 14px;">
                <span style="font-size:0.75rem;color:{RED};text-transform:uppercase;font-weight:700;">
                ⚠️ Common Mistake</span><br><span style="font-size:0.85rem;">{warning}</span></div>""" if warning else ""
            tip_html = f"""<div class="mp-card-green" style="margin:8px 0;padding:10px 14px;">
                <span style="font-size:0.75rem;color:{GREEN};text-transform:uppercase;font-weight:700;">
                💡 Exam Tip</span><br><span style="font-size:0.85rem;">{tip}</span></div>""" if tip else ""
            with st.expander(f"❓  {q}"):
                st.markdown(f"""{tag_html}
                <div style="font-size:0.93rem;color:{TXT};line-height:1.7;">{a}</div>
                {formula_html}{example_html}{warn_html}{tip_html}
                """, unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════
        # SECTION 1 — DATA QUALITY FUNDAMENTALS
        # ══════════════════════════════════════════════════════
        shdr("🏗️","Section 1 — Data Quality Fundamentals")

        qa("What is 'data quality' and why does it matter in finance?",
           f"""Data quality refers to the <b style="color:{GOLD};">fitness of data for its intended analytical purpose.</b>
           High-quality financial data must be: <b>Complete</b> (all required observations present),
           <b>Accurate</b> (values correctly represent financial reality), <b>Consistent</b> (no contradictions
           across fields or time), <b>Timely</b> (reflects the relevant point in time), <b>Valid</b>
           (values within expected domain ranges), and <b>Unique</b> (no duplicate records).<br><br>
           In finance, decisions involving billions of rupees are made based on quantitative models.
           The GIGO principle — <i>Garbage In, Garbage Out</i> — has profound consequences: a single
           undetected invalid value can cascade into catastrophic financial loss.""",
           tag="Fundamentals",
           example="Knight Capital Group lost USD 440 million in 45 minutes on 1 August 2012 due to stale, erroneous order data triggering unintended trading. Root cause: failure to validate and clean data before deployment.",
           tip="Remember the 6 dimensions: Complete · Accurate · Consistent · Timely · Valid · Unique")

        qa("What is the standard data quality pipeline in financial analytics?",
           f"""The pipeline has six stages in sequence:<br><br>
           <b style="color:{GOLD};">1. Ingest</b> — Load raw data from APIs, files, or databases<br>
           <b style="color:{GOLD};">2. Validate</b> — Check schema, data types, domain ranges, and business rules<br>
           <b style="color:{GOLD};">3. Missing</b> — Detect, classify (MCAR/MAR/MNAR), and impute missing values<br>
           <b style="color:{GOLD};">4. Outliers</b> — Detect using statistical methods; review and manage appropriately<br>
           <b style="color:{GOLD};">5. Format</b> — Fix data types, set correct index, handle frequency and timezone<br>
           <b style="color:{GOLD};">6. Output</b> — Produce the analysis-ready dataset with a cleaning audit trail""",
           tag="Fundamentals",
           tip="The pipeline is sequential — you MUST handle missing values BEFORE outlier detection, because outlier statistics (mean, std) are distorted by NaNs.")

        qa("What are the real-world consequences of poor data quality in finance?",
           f"""Different domains suffer different consequences:<br><br>
           • <b>Equity Research:</b> Missing corporate action adjustments → incorrect return calculations, wrong momentum signals<br>
           • <b>Credit Scoring:</b> Null income imputed as zero → healthy borrowers misclassified as high-risk<br>
           • <b>Risk Management:</b> Outlier VaR observation from data error → VaR massively overstated, excess capital held<br>
           • <b>Algorithmic Trading:</b> Stale prices from feed outage → algorithm trades on wrong prices, large losses<br>
           • <b>Regulatory Reporting:</b> Invalid date formats in CRILC filing → regulatory non-compliance, penalties<br>
           • <b>Macro Forecasting:</b> Outlier GDP revision not handled → model forecasts structurally biased""",
           tag="Fundamentals",
           warning="Regulatory data quality failures (CRILC, RBI filings) are treated as compliance violations, not just modelling errors. Penalties can be severe.")

        # ══════════════════════════════════════════════════════
        # SECTION 2 — MISSING DATA
        # ══════════════════════════════════════════════════════
        shdr("🔍","Section 2 — Missing Data — Classification & Treatment")

        qa("What are the three mechanisms of missingness? Explain each with a financial example.",
           f"""Formalised by Little & Rubin (1987) — the mechanism determines the correct treatment.<br><br>
           <b style="color:{LIGHT_BLUE};">MCAR — Missing Completely at Random</b><br>
           The probability of being missing is unrelated to any observed or unobserved data.
           <i>P(missing) = constant</i><br>
           → Any imputation method works. Listwise deletion is valid but wasteful.<br><br>
           <b style="color:{GOLD};">MAR — Missing at Random</b><br>
           The probability of being missing depends on <i>observed</i> data but NOT on the missing value itself.
           <i>P(missing | X_obs, X_miss) = P(missing | X_obs)</i><br>
           → MICE, KNN, or model-based imputation using observed predictors.<br><br>
           <b style="color:{RED};">MNAR — Missing Not at Random</b><br>
           The probability of being missing depends on the <i>missing value itself</i>.
           <i>P(missing | X_obs, X_miss) depends on X_miss</i><br>
           → No standard fix. Requires domain expertise and sensitivity analysis.""",
           tag="Missing Data",
           example="MCAR: NSE random packet-loss on 0.1% of days. MAR: Small-caps missing EPS estimates (missingness related to market cap, not EPS). MNAR: IL&FS stopped disclosing cash flows precisely because cash flows were catastrophically negative.",
           warning="MNAR is the most dangerous because analysts who assume MCAR or MAR and apply standard imputation will get biased results — and will not know it.",
           tip="Exam shortcut: MCAR = Best case · MAR = Manageable · MNAR = Most Dangerous")

        qa("How do you choose which imputation method to use?",
           f"""The decision depends on three factors: <b>% missing</b>, <b>missingness mechanism</b>, and <b>variable type</b>.<br><br>
           <table style="width:100%;font-size:0.83rem;border-collapse:collapse;">
           <tr style="color:{GOLD};border-bottom:1px solid {GOLD}33;">
             <th style="text-align:left;padding:4px 8px;">% Missing</th>
             <th style="text-align:left;padding:4px 8px;">Mechanism</th>
             <th style="text-align:left;padding:4px 8px;">Strategy</th>
           </tr>
           <tr style="color:{TXT};"><td style="padding:4px 8px;">&lt;2%</td><td>MCAR</td><td>Forward-fill (LOCF) or backward-fill</td></tr>
           <tr style="color:{TXT};background:#0a1428;"><td style="padding:4px 8px;">2–15%</td><td>MAR</td><td>KNN or regression-based imputation</td></tr>
           <tr style="color:{TXT};"><td style="padding:4px 8px;">&gt;15%</td><td>MAR</td><td>MICE (gold standard) or median + flag</td></tr>
           <tr style="color:{TXT};background:#0a1428;"><td style="padding:4px 8px;">Any</td><td>MNAR</td><td>Domain-adjusted; sensitivity analysis</td></tr>
           <tr style="color:{TXT};"><td style="padding:4px 8px;">&gt;50%</td><td>Any</td><td>Drop column; flag and document</td></tr>
           </table>""",
           tag="Missing Data",
           tip="In exams: 'moderate missing + MAR' → KNN. 'Research grade + MAR' → MICE. 'MNAR' → no standard fix.")

        qa("Why is mean imputation dangerous for financial data?",
           f"""Mean imputation has a critical flaw: it <b style="color:{RED};">artificially reduces the variance</b>
           of the imputed variable.<br><br>
           Mathematically: <b>Var(X_imputed) &lt; Var(X_true)</b><br><br>
           This creates three downstream problems:<br>
           1. <b>VaR calculations:</b> Understated risk — imputed return distributions have artificially thin tails<br>
           2. <b>Correlation estimates:</b> Biased toward zero — imputed values all equal X̄, reducing covariance<br>
           3. <b>Credit scoring:</b> Attenuates discriminatory power of imputed variables in models<br><br>
           The rule: <b>Never use mean imputation for variables entering variance/covariance calculations,
           risk models, or where distribution shape matters.</b>""",
           tag="Missing Data",
           example="If a stock has 20 missing return days imputed as the mean return (say +0.02%), those days contribute zero variance. A VaR model will see a narrower distribution than reality — dangerously underestimating tail risk.",
           warning="Mean imputation is so common in practice that examiners frequently test whether students understand its variance-reducing effect.")

        qa("What is Forward Fill (LOCF) and when should you NOT use it?",
           f"""<b>LOCF — Last Observation Carried Forward</b>: replaces a missing value with the most
           recent non-missing value in the series.<br><br>
           <b style="color:{GREEN};">When appropriate:</b> Price data on non-trading days, holiday gaps,
           thin trading days for illiquid securities — where the last traded price is a reasonable proxy
           for the current fair value.<br><br>
           <b style="color:{RED};">When NOT appropriate:</b><br>
           1. <b>Return calculations:</b> Forward-filled prices produce artificial zero-return days.
              The return series will contain spikes at zero that understate true volatility.<br>
           2. <b>Real-time signals:</b> Stale forward-filled prices in live trading systems trigger
              trades at wrong prices — the cause of many algorithmic trading losses.<br>
           3. <b>Macroeconomic data with trends:</b> LOCF on GDP or inflation can mask structural changes.""",
           tag="Missing Data",
           formula="Zero-return detection: <code style='font-size:0.85rem;'>returns_raw = prices_ffill.pct_change()<br>zero_flag = (returns_raw == 0) & prices.isnull().shift(1).fillna(False)</code>",
           tip="Always calculate returns on the ORIGINAL series with NaNs, not on the forward-filled series.")

        qa("What is KNN Imputation and how is it applied to financial data?",
           f"""KNN Imputation finds the <b>k most similar observations</b> (based on observed features)
           and imputes the missing value as a weighted average of the k neighbours:<br><br>
           <b>X̂_miss = Σ wⱼ · Xⱼ,miss / Σ wⱼ</b><br><br>
           where wⱼ = 1/d(i,j) — inverse distance weights, d = Euclidean distance on scaled features.<br><br>
           <b style="color:{GOLD};">Key implementation steps:</b><br>
           1. <b>Scale features first</b> — KNN is distance-based; unscaled features distort distances<br>
           2. Use StandardScaler before KNNImputer<br>
           3. Inverse-transform after imputation to restore original scale<br>
           4. Typical k = 5–10; tune via cross-validation on complete cases<br><br>
           <b style="color:{LIGHT_BLUE};">Financial application:</b> Imputing missing PE or EV/EBITDA ratios
           using comparable companies (same sector, size, profitability) as neighbours.""",
           tag="Missing Data",
           formula="<code>from sklearn.impute import KNNImputer<br>from sklearn.preprocessing import StandardScaler<br>scaler = StandardScaler()<br>X_scaled = scaler.fit_transform(df)<br>X_imp = KNNImputer(n_neighbors=7).fit_transform(X_scaled)<br>df_imputed = pd.DataFrame(scaler.inverse_transform(X_imp))</code>",
           warning="Never run KNNImputer on raw (unscaled) financial data. A stock price of ₹3,500 will dominate distances over a PE ratio of 25, making neighbour selection meaningless.")

        qa("What is MICE and why is it considered the gold standard?",
           f"""<b>MICE — Multiple Imputation by Chained Equations</b><br><br>
           <b>Algorithm:</b><br>
           1. Fill missing values with simple initial estimates (mean)<br>
           2. For each variable Xⱼ with missing values: fit a regression model using all other
              variables as predictors; draw imputations from the posterior predictive distribution<br>
           3. Cycle through all variables m times (m = 5–20 iterations)<br>
           4. Create M complete datasets (M = 5–10)<br>
           5. Run the full analysis on each complete dataset separately<br>
           6. Combine results using <b>Rubin's Rules</b>: θ̂ = (1/M) Σ θ̂ᵢ<br><br>
           <b style="color:{GOLD};">Why it's the gold standard:</b><br>
           • Correctly propagates uncertainty from imputation into final estimates<br>
           • Produces unbiased estimates under MAR<br>
           • Preserves the relationships between variables (unlike single imputation methods)<br>
           • Rubin's combining rule accounts for both within-imputation and between-imputation variance""",
           tag="Missing Data",
           formula="Rubin's Rule for variance: Var(θ̂) = (1/M)ΣWᵢ + (1 + 1/M)·B<br>where W = within-imputation variance, B = between-imputation variance",
           tip="Python: sklearn.impute.IterativeImputer (experimental) or the miceforest package")

        # ══════════════════════════════════════════════════════
        # SECTION 3 — OUTLIER DETECTION
        # ══════════════════════════════════════════════════════
        shdr("📊","Section 3 — Outlier Detection & Management")

        qa("What are the three types of outliers in financial data and how should each be treated?",
           f"""<b style="color:{RED};">1. Error Outliers</b> — Data errors (negative prices, wrong decimal,
           feed errors). These should be corrected or removed.<br>
           <i>Examples: NSE price = −50; Volume = 0 on active trading day; PE ratio = 50,000</i><br><br>
           <b style="color:{GOLD};">2. Genuine Extreme Events</b> — Real market events that represent
           legitimate tail risk. These must be <b>PRESERVED</b> for risk modelling and stress-testing.<br>
           <i>Examples: NIFTY −38% (COVID March 2020); NIFTY −4% (Demonetisation Nov 2016);
           RBI emergency rate decisions</i><br><br>
           <b style="color:{LIGHT_BLUE};">3. Structural Breaks</b> — One-time shifts in the data-generating
           process. Require regime-splitting or a dummy variable rather than outlier treatment.<br>
           <i>Examples: Merger/demerger; Basel III → IV regulatory change; Index reconstitution</i>""",
           tag="Outlier Detection",
           warning="The most dangerous mistake is applying a mechanical |Z| > 3 rule that removes genuine extreme events like COVID crashes from VaR datasets — gutting the very tail risk signal the model is supposed to capture.",
           tip="The key question: 'Is this a data error or a real event?' Answer requires domain knowledge, not just statistics.")

        qa("Explain the Z-Score method for outlier detection and its limitation in finance.",
           f"""<b>Z-Score</b> measures how many standard deviations an observation is from the mean:<br><br>
           <b>Zᵢ = (Xᵢ − X̄) / σ</b><br><br>
           Rule: Flag observations where |Zᵢ| > 3 (covers 99.73% of data under normality).<br><br>
           <b style="color:{RED};">Critical limitation for financial returns:</b><br>
           Financial returns are NOT normally distributed — they exhibit:
           <ul style="margin:8px 0;padding-left:18px;">
           <li><b>Fat tails (leptokurtosis):</b> Extreme events occur far more frequently than the normal distribution predicts</li>
           <li><b>Negative skewness:</b> Large negative returns are more common than large positive returns</li>
           <li><b>Volatility clustering:</b> Periods of high volatility cluster together</li>
           </ul>
           Additionally: σ itself is <b>inflated by the very outliers you are trying to detect</b>,
           making Z-scores less sensitive.""",
           tag="Outlier Detection",
           formula="Zᵢ = (Xᵢ − X̄) / σ &nbsp;&nbsp; Flag: |Z| > 3",
           tip="Z-Score is appropriate for quick screening of clearly non-financial data (e.g. data entry errors where a price is negative). For return series, always prefer Modified Z-Score (MAD).")

        qa("What is the Modified Z-Score (MAD method) and why is it preferred for financial data?",
           f"""The <b>Modified Z-Score</b> uses the <b>Median Absolute Deviation (MAD)</b>
           instead of the standard deviation — making it robust to outliers themselves:<br><br>
           <b>MAD = Median |Xᵢ − X̃|</b><br>
           <b>Mᵢ = 0.6745 × |Xᵢ − X̃| / MAD</b><br><br>
           The constant 0.6745 makes MAD comparable to σ for normally distributed data.<br>
           Rule: Flag |Mᵢ| > 3.5 as an outlier.<br><br>
           <b style="color:{GREEN};">Why preferred:</b><br>
           • The standard deviation (σ) is inflated by the extreme values you are trying to detect —
             making Z-scores less sensitive to genuine outliers<br>
           • MAD uses the <b>median</b>, which is completely unaffected by a few extreme values<br>
           • Works well for return series, accounting ratios, and any fat-tailed financial variable<br>
           • Preferred in the FinancialDataCleaner pipeline for this reason""",
           tag="Outlier Detection",
           formula="MAD = Median|Xᵢ − X̃| &nbsp;&nbsp; Mᵢ = 0.6745·|Xᵢ − X̃|/MAD &nbsp;&nbsp; Flag: |M| > 3.5",
           tip="Exam key: 'Modified Z-Score' = MAD method = robust to outliers. Standard Z-Score = not robust (σ inflated by outliers).")

        qa("What is the IQR method and how is it used for Winsorisation?",
           f"""The <b>Interquartile Range (IQR) method</b> uses quartile-based fences:<br><br>
           <b>IQR = Q3 − Q1</b><br>
           <b>Lower Fence = Q1 − k × IQR</b><br>
           <b>Upper Fence = Q3 + k × IQR</b><br><br>
           <b>k = 1.5</b> → moderate outliers (standard boxplot whiskers)<br>
           <b>k = 3.0</b> → extreme outliers (Tukey's outer fence)<br><br>
           <b style="color:{GOLD};">Winsorisation using IQR:</b><br>
           Values beyond the fences are <i>replaced</i> (not removed) with the fence values.
           Standard in academic finance: winsorise at <b>1st and 99th percentiles</b>
           for cross-sectional regressions on financial ratios.<br><br>
           <b style="color:{LIGHT_BLUE};">Winsorisation Formula:</b><br>
           X_wins = Qα if X &lt; Qα | X if Qα ≤ X ≤ Q₁₋α | Q₁₋α if X &gt; Q₁₋α""",
           tag="Outlier Detection",
           formula="IQR = Q3−Q1 &nbsp;|&nbsp; Fences = Q1±k·IQR &nbsp;|&nbsp; k=1.5 (moderate), k=3 (extreme)",
           warning="Winsorisation vs Trimming: Winsorisation REPLACES extreme values with percentile bounds (preserving row count). Trimming REMOVES them. Always use Winsorisation in time-series to avoid losing observations.",
           tip="Winsorisation at 1%/99% is the academic finance standard. This means clipping the top and bottom 1% to their respective boundary values.")

        qa("How does Isolation Forest detect outliers?",
           f"""<b>Isolation Forest</b> works on a fundamentally different principle from statistical methods:
           it builds an ensemble of random decision trees and measures how <i>easy</i> it is to isolate
           each observation.<br><br>
           <b style="color:{GOLD};">Key insight:</b> Anomalies are rare and different — they require
           <b>very few random splits</b> to isolate. Normal observations are densely packed and require
           many splits.<br><br>
           <b>Anomaly Score:</b> s(x, n) = 2^(−E[h(x)] / c(n))<br>
           where E[h(x)] = expected path length, c(n) = normalisation constant<br><br>
           • Score ≈ 1 → clearly anomalous (isolated in very few splits)<br>
           • Score ≈ 0.5 → normal (requires many splits to isolate)<br>
           • Score ≪ 0.5 → very dense normal point<br><br>
           <b style="color:{LIGHT_BLUE};">Financial application:</b> Detecting anomalous trading patterns
           across <b>multiple features simultaneously</b> (price + volume + spread + order imbalance) —
           unlike univariate Z-score methods.""",
           tag="Outlier Detection",
           formula="<code>from sklearn.ensemble import IsolationForest<br>iso = IsolationForest(contamination=0.02, random_state=42)<br>preds = iso.fit_predict(X)  # -1 = anomaly, +1 = normal</code>",
           tip="Isolation Forest is the only multivariate method in the course. Use it when you have multiple features (price, volume, spread) and want to detect observations that are unusual across ALL dimensions simultaneously.")

        qa("What is Sensitivity Analysis and why is it the 'gold standard' for outlier decisions?",
           f"""When the decision to include or exclude outliers is genuinely uncertain,
           sensitivity analysis provides the most defensible approach:<br><br>
           <b>Step 1:</b> Run the analysis <b>with</b> outliers (full sample)<br>
           <b>Step 2:</b> Run the analysis <b>without</b> outliers (trimmed/winsorised)<br>
           <b>Step 3:</b> Run with <b>robust methods</b> (Huber regression, Theil-Sen estimator)<br>
           <b>Step 4:</b> Compare results:<br>
           &nbsp;&nbsp;&nbsp;• If <b>materially different</b> → investigate further; report both; document assumptions clearly<br>
           &nbsp;&nbsp;&nbsp;• If <b>similar</b> → outliers do not materially affect conclusions<br><br>
           <b style="color:{GOLD};">Regulatory requirement:</b> SEBI ICDR and RBI model risk guidelines
           require banks and AMCs to document outlier treatment methodology and demonstrate
           stability of results to outlier assumptions.""",
           tag="Outlier Detection",
           tip="In any assignment or exam involving financial model building, always include a sensitivity analysis table showing results with and without extreme observations — this demonstrates professional rigour.")

        # ══════════════════════════════════════════════════════
        # SECTION 4 — TIME SERIES
        # ══════════════════════════════════════════════════════
        shdr("📅","Section 4 — Time Series Formatting")

        qa("What are the five dimensions of financial time series formatting?",
           f"""Financial time series are not generic tabular data — they require special handling across
           five dimensions:<br><br>
           <b style="color:{GOLD};">1. Index Type</b> — DatetimeIndex MUST be used.
           String-based date indices cause silent errors in resampling and rolling calculations.<br><br>
           <b style="color:{GOLD};">2. Timezone Handling</b> — NSE trades IST (UTC+5:30); NYSE trades ET (UTC−5).
           Cross-market analysis requires tz-aware timestamps to avoid off-by-one-day errors.<br><br>
           <b style="color:{GOLD};">3. Business Calendar</b> — pd.bdate_range() uses generic Mon–Fri but misses
           Indian market holidays. NSE has ~244 trading days/year vs 261 Mon–Fri days. Use CustomBusinessDay.<br><br>
           <b style="color:{GOLD};">4. Frequency</b> — Each data type has a specific aggregation rule when
           resampling. Returns must be compounded (not summed). Volume must be summed.<br><br>
           <b style="color:{GOLD};">5. Corporate Actions</b> — Price discontinuities from dividends, splits,
           and bonuses create spurious returns. Always use backward-adjusted close prices.""",
           tag="Time Series",
           example="TCS declared a 1:1 bonus in 2018. On the ex-date, the unadjusted price appeared to fall 50%. This would show as a −50% outlier and then apparent outperformance — completely misleading. Adjusted prices eliminate this discontinuity.",
           tip="'Backward adjustment' means: historical prices are revised downward to reflect the split/bonus, so the series is continuous and comparable across time.")

        qa("Why must returns be compounded when resampling — not summed?",
           f"""When aggregating daily returns to weekly or monthly frequency, the mathematically correct
           operation is <b>compounding</b>, not summing.<br><br>
           <b style="color:{RED};">WRONG (summing):</b> Monthly return = Σ rᵢ<br>
           <b style="color:{GREEN};">CORRECT (compounding):</b> Monthly return = ∏(1 + rᵢ) − 1<br><br>
           <b>Numerical example:</b> 3-day returns of +5%, −3%, +2%<br>
           • Sum: +5% − 3% + 2% = <b>+4.00%</b> (WRONG)<br>
           • Compound: (1.05)(0.97)(1.02) − 1 = <b>+3.937%</b> (CORRECT)<br><br>
           The error from summing grows with the number of periods, the size of returns, and
           the volatility of the series. Over monthly horizons, the bias is economically significant.""",
           tag="Time Series",
           formula="Monthly Return = ∏(1 + rᵢ) − 1<br><code>def compound_return(s): return (1 + s).prod() - 1<br>monthly_ret = daily_ret.resample('ME').apply(compound_return)</code>",
           warning="This is one of the most commonly tested topics — and one of the most common practitioner errors. Always compound returns; never sum them across periods.")

        qa("What is the correct OHLCV aggregation rule when resampling?",
           f"""Each OHLCV field has a specific aggregation that preserves financial meaning:<br><br>
           <table style="width:100%;font-size:0.83rem;border-collapse:collapse;">
           <tr style="color:{GOLD};border-bottom:1px solid {GOLD}33;">
             <th style="padding:4px 8px;text-align:left;">Field</th>
             <th style="padding:4px 8px;text-align:left;">Aggregation</th>
             <th style="padding:4px 8px;text-align:left;">Rationale</th>
           </tr>
           <tr style="color:{TXT};"><td style="padding:4px 8px;">Open</td><td>.first()</td><td>Opening price of the period</td></tr>
           <tr style="color:{TXT};background:#0a1428;"><td style="padding:4px 8px;">High</td><td>.max()</td><td>Highest price reached in period</td></tr>
           <tr style="color:{TXT};"><td style="padding:4px 8px;">Low</td><td>.min()</td><td>Lowest price reached in period</td></tr>
           <tr style="color:{TXT};background:#0a1428;"><td style="padding:4px 8px;">Close</td><td>.last()</td><td>Closing price of period</td></tr>
           <tr style="color:{TXT};"><td style="padding:4px 8px;">Volume</td><td>.sum()</td><td>Total shares traded in period</td></tr>
           <tr style="color:{TXT};background:#0a1428;"><td style="padding:4px 8px;">Returns</td><td>∏(1+rᵢ)−1</td><td>Compound — never sum!</td></tr>
           </table>""",
           tag="Time Series",
           formula="<code>ohlcv_agg = {'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}<br>weekly = df.resample('W-FRI').agg(ohlcv_agg)</code>",
           tip="Exam trick: If a question asks which OHLCV aggregation is wrong — it is always 'Returns = sum of daily returns'.")

        qa("What is cubic spline interpolation and how is it used for yield curves?",
           f"""When some government bond maturities are not actively traded (e.g., 4-year, 6-year),
           their yields must be <b>interpolated</b> from observable benchmark yields.<br><br>
           <b>Cubic Spline Interpolation</b> fits smooth polynomial curves of degree 3 between known
           data points. Unlike linear interpolation, it ensures smooth first and second derivatives at
           each knot — producing a smooth, economically realistic yield curve.<br><br>
           <b style="color:{GOLD};">Indian market context:</b> FIMMDA and CCIL use cubic spline /
           Nelson-Siegel-Svensson interpolation to construct the complete G-Sec yield curve from
           the ~10 liquid benchmark maturities (0.25Y to 30Y).<br><br>
           <b>Applications:</b> Bond valuation (present value calculations), duration and convexity analysis,
           interest rate derivative pricing (FRAs, IRS), regulatory capital computations.""",
           tag="Time Series",
           formula="<code>from scipy.interpolate import CubicSpline<br>cs = CubicSpline(maturities_observed, yields_observed)<br>yields_missing = cs(missing_tenors)</code>",
           tip="Spline is preferred over linear interpolation for yield curves because interest rates change smoothly — sudden 'kinks' in the yield curve at observable maturities would be economically unrealistic.")

        # ══════════════════════════════════════════════════════
        # SECTION 5 — INVALID VALUES
        # ══════════════════════════════════════════════════════
        shdr("⚠️","Section 5 — Invalid Values")

        qa("What is the difference between a missing value and an invalid value?",
           f"""This distinction is fundamental and often confused:<br><br>
           <b style="color:{LIGHT_BLUE};">Missing Value (NaN / NULL)</b><br>
           The observation <b>is absent</b> from the dataset entirely. The cell contains no value.
           Treatment: imputation methods (LOCF, KNN, MICE, etc.).<br><br>
           <b style="color:{RED};">Invalid Value</b><br>
           The observation <b>is present</b> but is logically, mathematically, or domain-impossibly wrong.
           The cell contains a value — but that value violates a rule.<br><br>
           <b>Categories of invalidity:</b><br>
           • <b>Domain constraints:</b> Price ≤ 0; Probability ∉ [0,1]<br>
           • <b>Logical constraints:</b> OHLC High &lt; Low; Ask &lt; Bid<br>
           • <b>Referential integrity:</b> Trade date after settlement date; Bond maturity before issue date<br>
           • <b>Business rules:</b> Volume = 0 during claimed active trading session<br>
           • <b>Relational constraints:</b> Total Assets ≠ Total Liabilities + Equity""",
           tag="Invalid Values",
           example="NSE price = −₹50 → Invalid (domain constraint: price must be positive). NSE price field is blank → Missing (NaN). Both need treatment but different treatment.",
           tip="Invalid values must be DETECTED before imputation. If you impute missing values first and then check for invalids, you may impute an invalid value into many rows.")

        qa("What is 'stale price detection' and why does it matter?",
           f"""A <b>stale price</b> occurs when a financial instrument's price shows zero change
           for an abnormally long consecutive period — indicating the data feed has disconnected
           or the price is not being updated.<br><br>
           <b style="color:{GOLD};">Detection rule:</b> If Close price is unchanged for 5+ consecutive
           trading days in a normally liquid security, flag as stale.<br><br>
           <b>Why it matters:</b><br>
           • Portfolio valuation becomes incorrect (marking to stale prices)<br>
           • Returns series shows artificial zeros → volatility understated<br>
           • Risk models think the security has zero volatility → VaR completely wrong<br>
           • Algorithmic momentum signals generate false signals<br><br>
           <b>Legitimate exceptions:</b> Auction stocks, circuit-breaker halted stocks, very illiquid
           securities — always verify against market calendar and circuit breaker records.""",
           tag="Invalid Values",
           formula="<code>price_unchanged = (df['Close'] != df['Close'].shift(1))<br>stale_flag = (price_unchanged.rolling(5).sum() == 0)</code>",
           warning="Do not automatically remove or impute all stale prices — first verify whether the stock was suspended, in an auction, or hit a circuit breaker. Removal of legitimate trading halts distorts historical data.")

        qa("What are the OHLC logic constraints and how do you validate them?",
           f"""The OHLC structure has rigid mathematical relationships that must always hold:<br><br>
           <b style="color:{GOLD};">Required invariants:</b><br>
           • High ≥ Low &nbsp;&nbsp; (always true by definition)<br>
           • High ≥ Open &nbsp;&nbsp; (the high must be at least the open)<br>
           • High ≥ Close &nbsp;&nbsp; (the high must be at least the close)<br>
           • Low ≤ Open &nbsp;&nbsp; (the low must be no more than the open)<br>
           • Low ≤ Close &nbsp;&nbsp; (the low must be no more than the close)<br><br>
           <b>Any violation is ALWAYS a data error</b> — never a legitimate market event.
           (Short squeezes, limit moves, and circuit breakers do not violate OHLC logic.)<br><br>
           <b>Treatment:</b> Convert to NaN → impute using adjacent valid observations, or flag for
           source data verification (cross-check NSE/BSE/Bloomberg).""",
           tag="Invalid Values",
           formula="<code>ohlc_invalid = df[(df.High < df.Low) | (df.High < df.Close) |<br>               (df.High < df.Open) | (df.Low > df.Close) |<br>               (df.Low > df.Open)]</code>",
           tip="High < Low is the most frequently tested OHLC validity rule. It is always an error — never valid market data under any circumstances.")

        # ══════════════════════════════════════════════════════
        # SECTION 6 — PIPELINE & CASE STUDIES
        # ══════════════════════════════════════════════════════
        shdr("⚙️","Section 6 — The Cleaning Pipeline & Case Studies")

        qa("What are the four stages of the FinancialDataCleaner pipeline?",
           f"""<b style="color:{GOLD};">Stage 1 — validate_schema()</b><br>
           Detect mixed-type columns, coerce numeric strings, ensure DatetimeIndex,
           sort chronologically, remove duplicate timestamps.<br><br>
           <b style="color:{GOLD};">Stage 2 — treat_missing()</b><br>
           Drop columns exceeding missing_threshold. Apply tiered strategy by % missing:
           &lt;2% → ffill, 2–15% → KNN, &gt;15% → median + binary flag column.<br><br>
           <b style="color:{GOLD};">Stage 3 — treat_outliers()</b><br>
           Modified Z-Score (MAD) per column. Action is configurable:
           'winsorise', 'flag', or 'remove' + ffill. Skips flag/indicator columns.<br><br>
           <b style="color:{GOLD};">Stage 4 — cleaning_report()</b><br>
           Produces a full before/after audit trail: shape change, missing count change,
           step-by-step log of every action taken. Required for model governance documentation.""",
           tag="Pipeline",
           formula="<code>cleaner = FinancialDataCleaner(config={'outlier_action':'winsorise', 'zscore_threshold':3.5})<br>df_clean = cleaner.clean(df_raw)<br>log_df = cleaner.cleaning_report(df_raw, df_clean)</code>",
           tip="In any assessment, always mention that the pipeline produces an audit trail (cleaning_report). Model governance and regulatory requirements demand documented, reproducible data cleaning decisions.")

        qa("Demonetisation (Nov 2016): Should NIFTY's −4% be removed as an outlier?",
           f"""<b style="color:{RED};">No — it must be retained in almost all financial models.</b><br><br>
           <b>Statistical assessment:</b> Z-score ≈ −5.2σ for a single trading day return in 2016.
           Mechanically, this would be flagged for removal by any |Z| > 3 rule.<br><br>
           <b>Domain assessment:</b> This is a <b>genuine extreme event</b> driven by a known policy shock,
           not a data error. The correct treatment depends on the model:<br><br>
           <table style="width:100%;font-size:0.83rem;">
           <tr style="color:{GOLD};"><th style="padding:4px 8px;text-align:left;">Model</th><th style="padding:4px 8px;text-align:left;">Treatment</th></tr>
           <tr style="color:{TXT};"><td style="padding:4px 8px;">Historical VaR / ES</td><td style="padding:4px 8px;">✅ RETAIN — this IS the tail event the model must price</td></tr>
           <tr style="color:{TXT};background:#0a1428;"><td style="padding:4px 8px;">CAPM / Factor Regression</td><td style="padding:4px 8px;">🟡 FLAG — add D_nov2016=1 dummy variable</td></tr>
           <tr style="color:{TXT};"><td style="padding:4px 8px;">GARCH Volatility</td><td style="padding:4px 8px;">✅ RETAIN — volatility clustering is a key stylised fact</td></tr>
           <tr style="color:{TXT};background:#0a1428;"><td style="padding:4px 8px;">Stress Testing</td><td style="padding:4px 8px;">✅ RETAIN — prime historical stress scenario</td></tr>
           </table>""",
           tag="Case Studies",
           warning="Removing Demonetisation from a VaR dataset would understate tail risk — making the bank's capital buffer look adequate when it may not be. This is precisely the type of error that leads to regulatory failures.",
           tip="The lesson: domain knowledge must ALWAYS override mechanical statistical rules. A model that removes all |Z|>3 events is systematically blind to the tail risks it is designed to measure.")

        qa("IL&FS Default (2018): What was the data quality lesson and how does MNAR apply?",
           f"""<b style="color:{GOLD};">What happened:</b> Infrastructure Leasing & Financial Services (IL&FS)
           defaulted on commercial paper obligations in September 2018, triggering a liquidity crisis
           across Indian NBFCs and credit markets.<br><br>
           <b style="color:{RED};">The MNAR pattern:</b> Months before the default, IL&FS subsidiaries
           began selectively reporting financial data. Cash flow metrics were delayed or missing —
           and the missingness was MNAR: the subsidiaries <b>in most financial stress had the most
           missing/delayed data</b>.<br><br>
           <b style="color:{LIGHT_BLUE};">Why MCAR/MAR assumptions failed:</b> Analysts who treated missing
           values as MCAR or MAR (and applied simple imputation) missed the distress signal entirely.
           Those who investigated <i>why</i> data was missing correctly identified emerging distress.<br><br>
           <b>The lesson — three key practices:</b><br>
           1. Always investigate <i>why</i> data is missing before deciding <i>how</i> to handle it<br>
           2. In credit analysis, systematic missing data in stressed entities is a RED FLAG, not a neutral gap<br>
           3. Add missingness as a feature: <code>df['cfo_missing'] = df['CFO'].isnull().astype(int)</code>
              — binary flags can be powerful early-warning predictors in ML credit models""",
           tag="Case Studies",
           tip="This case study is ideal for demonstrating the MNAR concept. IL&FS shows that 'missing data' was itself the signal — the absence of disclosure was more informative than the disclosure would have been.")

        qa("NSE Feed Outage: How does MCAR block missing differ from MNAR, and how should it be treated?",
           f"""<b style="color:{GREEN};">MCAR context:</b> A quantitative fund's data feed experiences a
           3-hour outage (10:30–13:30 IST) on a regular trading day. The missing data is MCAR —
           the outage is a random technical failure completely unrelated to market conditions
           (the market did not stop trading; only the firm's data feed disconnected).<br><br>
           <b>Treatment depends on the downstream use:</b><br><br>
           <table style="width:100%;font-size:0.83rem;">
           <tr style="color:{GOLD};"><th style="padding:4px 8px;text-align:left;">Use Case</th><th style="padding:4px 8px;text-align:left;">Treatment</th><th style="padding:4px 8px;text-align:left;">Rationale</th></tr>
           <tr style="color:{TXT};"><td style="padding:4px 8px;">End-of-Day P&L</td><td>Forward-fill</td><td>Closing prices available; intraday irrelevant</td></tr>
           <tr style="color:{TXT};background:#0a1428;"><td style="padding:4px 8px;">Intraday VaR</td><td>EXCLUDE window</td><td>Cannot reliably impute for risk calcs</td></tr>
           <tr style="color:{TXT};"><td style="padding:4px 8px;">Momentum Signal</td><td>Do not trade</td><td>Stale prices → false signals → losses</td></tr>
           <tr style="color:{TXT};background:#0a1428;"><td style="padding:4px 8px;">VWAP Calculation</td><td>Partial VWAP</td><td>Note incomplete window in reports</td></tr>
           <tr style="color:{TXT};"><td style="padding:4px 8px;">Historical Backtest</td><td>Fill from exchange</td><td>NSE/BSE historical tick data</td></tr>
           </table><br>
           <b>Best practice:</b> Quant funds implement <b>data quality gates</b> — automated checks
           that halt or modify signal generation when data completeness falls below 90% of expected
           ticks in the last 5 minutes.""",
           tag="Case Studies",
           tip="Contrast with IL&FS (MNAR): in MCAR outage, imputation is acceptable for non-risk uses (P&L valuation). In MNAR (IL&FS), imputation is dangerous and the pattern of missingness IS the signal.")

        # ══════════════════════════════════════════════════════
        # SECTION 7 — PYTHON QUICK REFERENCE
        # ══════════════════════════════════════════════════════
        shdr("💻","Section 7 — Python Quick Reference")

        qa("What are the essential Python one-liners every financial data analyst must know?",
           f"""Key pandas / sklearn commands for financial data cleaning:<br>""",
           tag="Python",
           formula="""<code style="font-size:0.78rem;line-height:2;">
# Missing data diagnosis
df.isnull().sum()                              # Count per column
df.isnull().mean() * 100                       # % missing per column
df.isnull().sum().sum()                        # Total missing cells

# Imputation
df[col].ffill()                                # Forward fill (LOCF)
df[col].interpolate('linear')                  # Linear interpolation
df[col].fillna(df[col].median())               # Median imputation
KNNImputer(n_neighbors=7).fit_transform(X)     # KNN imputation

# DatetimeIndex
df.index = pd.to_datetime(df.index)            # Convert to DatetimeIndex
df = df.sort_index()                           # Sort chronologically
df = df[~df.index.duplicated(keep='last')]     # Remove duplicate timestamps
df.resample('ME').agg({'Close':'last', ...})   # Resample to monthly

# Outlier detection
z = np.abs(stats.zscore(series))               # Standard Z-Score
mad = np.median(np.abs(s - s.median()))        # MAD
mod_z = 0.6745 * np.abs(s - s.median()) / mad # Modified Z-Score
IsolationForest(contamination=0.02).fit_predict(X) # Isolation Forest

# Winsorisation
series.clip(lower=series.quantile(0.01),
            upper=series.quantile(0.99))       # Winsorise at 1%/99%

# OHLCV validation
invalid = df[(df.High < df.Low) |
             (df.Close <= 0) |
             (df.Volume == 0)]                 # Flag invalid rows

# Compound returns
def compound_ret(s): return (1 + s).prod() - 1
monthly_ret = daily_ret.resample('ME').apply(compound_ret)
</code>""",
           tip="Memorise the Modified Z-Score formula: mod_z = 0.6745 * |x - median| / MAD. This appears in almost every outlier detection question.")

        st.markdown(f"""<div class="mp-card-green" style="margin-top:14px;">
        <b style="color:{GREEN};">✅ Study Guide Complete!</b><br><br>
        You have covered all 7 sections — Fundamentals, Missing Data (MCAR/MAR/MNAR),
        Outlier Detection (4 methods), Time Series Formatting, Invalid Values, the Cleaning Pipeline,
        and three Indian market case studies.<br><br>
        <b>Now test yourself</b> → Switch to the <b>✏️ Take the Quiz</b> tab.
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # TAB 2 — QUIZ
    # ══════════════════════════════════════════════════════════
    with tab_quiz:
        shdr("✏️","Knowledge Assessment — 8 Questions")

        if "quiz_score" not in st.session_state:
            st.session_state.quiz_score=0; st.session_state.quiz_answers={}; st.session_state.quiz_submitted=False

        Qs=[
            {"q":"1. Which missingness mechanism is most dangerous and cannot be corrected by standard statistical methods?",
             "opts":["MCAR — Missing Completely at Random","MAR — Missing at Random","MNAR — Missing Not at Random","Block Missingness"],
             "ans":2,"exp":"MNAR missingness depends on the unobserved missing value itself — no standard imputation corrects for it. IL&FS is a classic MNAR example where missing data signalled distress."},
            {"q":"2. A portfolio VaR model uses forward-filled prices to compute daily returns. What is the primary consequence?",
             "opts":["VaR is overstated due to inflated variance","VaR is understated because artificial zero-return days reduce measured volatility","Forward fill has no impact on VaR","The correlation matrix becomes non-positive definite"],
             "ans":1,"exp":"Forward-filled prices create artificial zero returns, reducing measured standard deviation — VaR is understated. This creates dangerously false security in risk management."},
            {"q":"3. For 8% missing values with MAR mechanism, what is the recommended imputation strategy?",
             "opts":["Listwise deletion","Mean imputation","KNN or regression-based imputation","Only MICE works"],
             "ans":2,"exp":"For 5–15% missing under MAR, KNN or regression imputation is recommended. Listwise deletion produces biased estimates under MAR; mean imputation reduces variance."},
            {"q":"4. Modified Z-Score (MAD) is preferred over standard Z-Score for financial returns because:",
             "opts":["MAD requires larger samples","Standard Z-Score needs sorted arrays","MAD is unaffected by outliers themselves — σ is inflated by the very values being detected","MAD assumes normality"],
             "ans":2,"exp":"σ is inflated by extreme values — making detection less sensitive. MAD uses the median, which is robust to the outliers you are trying to identify."},
            {"q":"5. NIFTY 50 fell 4.0% on 8 November 2016 (Demonetisation). For historical simulation VaR, you should:",
             "opts":["Remove it — statistical outlier (|Z|>3)","Winsorise it to 1st percentile","RETAIN it — genuine tail event the model must capture","Replace with surrounding day average"],
             "ans":2,"exp":"Historical simulation VaR MUST include this observation — it IS the tail risk. Removing it would understate tail risk in a way that creates false security."},
            {"q":"6. When resampling daily OHLCV to monthly frequency, which aggregation is INCORRECT?",
             "opts":["Open = first observation","High = maximum of daily highs","Volume = sum of daily volumes","Returns = sum of daily returns"],
             "ans":3,"exp":"Returns must be COMPOUNDED: ∏(1+rᵢ)−1. Summing daily returns overestimates multi-period performance — particularly over monthly horizons."},
            {"q":"7. An OHLCV dataset shows High = ₹2,800 and Low = ₹3,100 on the same trading day. This is:",
             "opts":["Valid — intraday prices can cross","Error outlier — High must always ≥ Low; correct it","Genuine extreme event — short squeezes cause this","Structural break needing regime analysis"],
             "ans":1,"exp":"By definition High ≥ Low always. When High < Low it is ALWAYS a data entry or feed error — never a real market observation. Must be corrected or flagged."},
            {"q":"8. FinancialDataCleaner uses Modified Z-Score (MAD) rather than standard Z-Score because:",
             "opts":["MAD is faster to compute","Standard Z-Score needs sorted arrays","MAD is robust to outliers in the data itself — σ is inflated by the same outliers being detected","MAD works only for normal distributions"],
             "ans":2,"exp":"σ is inflated by extreme values — making Z-score less sensitive. MAD (Median Absolute Deviation) is based on the median: robust to the outliers you are trying to find."},
        ]

        if not st.session_state.quiz_submitted:
            st.markdown(f"""<div class="mp-card">
            <b style="color:{GOLD};">📋 Instructions</b> — Answer all 8 questions, then click Submit.
            Each question carries 1 mark. Detailed explanations provided after submission.
            Haven't studied yet? Switch to the <b>📖 Study Guide</b> tab first!
            </div>""",unsafe_allow_html=True)
            for i,q in enumerate(Qs):
                sel=st.radio(q["q"],q["opts"],key=f"q{i}",index=None)
                if sel: st.session_state.quiz_answers[i]=q["opts"].index(sel)
            if st.button("✅ Submit Assessment",type="primary"):
                if len(st.session_state.quiz_answers)<len(Qs):
                    st.warning(f"Please answer all {len(Qs)} questions. ({len(st.session_state.quiz_answers)} answered)")
                else:
                    st.session_state.quiz_score=sum(1 for i,q in enumerate(Qs)
                        if st.session_state.quiz_answers.get(i)==q["ans"])
                    st.session_state.quiz_submitted=True; st.rerun()
        else:
            sc=st.session_state.quiz_score; pct=sc/len(Qs)*100
            if pct>=75: g,css="Excellent! 🎓","verdict-ok"
            elif pct>=50: g,css="Good effort! 👍","verdict-warn"
            else: g,css="Needs Review 📖 — revisit the Study Guide","verdict-bad"
            st.markdown(f'<div class="{css}">Your Score: {sc}/{len(Qs)} ({pct:.0f}%) — {g}</div>',unsafe_allow_html=True)
            st.progress(sc/len(Qs))

            st.markdown(f"### 📖 Detailed Explanations")
            for i,q in enumerate(Qs):
                ua=st.session_state.quiz_answers.get(i,-1); ok=ua==q["ans"]
                with st.expander(f"{'✅' if ok else '❌'} Q{i+1}: {q['q'][:70]}...",expanded=not ok):
                    st.markdown(f"**Your answer:** {q['opts'][ua] if ua>=0 else 'Not answered'}")
                    st.markdown(f"**Correct answer:** {q['opts'][q['ans']]}")
                    if ok: st.success(f"✅ Correct! {q['exp']}")
                    else:  st.error(f"❌ {q['exp']}")

            c1,c2=st.columns(2)
            with c1:
                if st.button("🔄 Retake Quiz"):
                    st.session_state.quiz_score=0; st.session_state.quiz_answers={}
                    st.session_state.quiz_submitted=False; st.rerun()
            with c2:
                st.markdown(f"""<div class="mp-card">
                <b style="color:{GOLD};">📚 Continue Learning</b><br>
                Visit <a href="https://themountainpathacademy.com">themountainpathacademy.com</a>
                for full session notes, Python notebooks, and extended case studies.
                </div>""",unsafe_allow_html=True)
    footer()
