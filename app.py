"""
Financial Data Wrangling & Cleaning — Interactive Learning Lab
The Mountain Path – World of Finance
Prof. V. Ravichandran | themountainpathacademy.com
BITS Pilani WILP | Financial Analytics MBA ZG517
"""

import streamlit as st

st.set_page_config(
    page_title="Financial Data Wrangling Lab | The Mountain Path",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.styles import inject_styles
from utils.components import render_sidebar, render_hero
from pages import (
    home,
    missing_data,
    outlier_detection,
    time_series,
    invalid_values,
    pipeline,
    case_studies,
)

inject_styles()

# ── Sidebar navigation ──────────────────────────────────────────────────────
page = render_sidebar()

# ── Page router ─────────────────────────────────────────────────────────────
if page == "🏠 Home":
    home.render()
elif page == "🔍 Missing Data":
    missing_data.render()
elif page == "📊 Outlier Detection":
    outlier_detection.render()
elif page == "📅 Time Series Formatting":
    time_series.render()
elif page == "⚠️ Invalid Values":
    invalid_values.render()
elif page == "⚙️ Full Cleaning Pipeline":
    pipeline.render()
elif page == "📚 Case Studies":
    case_studies.render()
