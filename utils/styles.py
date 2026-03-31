"""
Styles — The Mountain Path Design System
Colors: #003366 (dark blue), #FFD700 (gold), #ADD8E6 (light blue)
"""

import streamlit as st


def inject_styles():
    st.markdown(
        """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;900&family=Source+Sans+3:wght@300;400;600&family=Fira+Code:wght@400;500&display=swap');

/* ── CSS Variables ── */
:root {
    --navy:      #003366;
    --navy-mid:  #004d80;
    --gold:      #FFD700;
    --gold-dark: #C8A800;
    --sky:       #ADD8E6;
    --sky-deep:  #6BAED6;
    --bg:        #0a0f1e;
    --card:      #0f1a2e;
    --card2:     #112240;
    --txt:       #e6f1ff;
    --muted:     #8892b0;
    --green:     #28a745;
    --red:       #dc3545;
    --orange:    #fd7e14;
    --border:    rgba(100,140,200,0.18);
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--txt) !important;
}

/* Streamlit main area */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020915 0%, #051428 50%, #071e38 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--txt) !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 0.45rem 0.8rem !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    font-size: 0.9rem !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,215,0,0.08) !important;
    color: var(--gold) !important;
}

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--txt) !important;
}
h1 { font-size: 2.4rem !important; font-weight: 900 !important; }
h2 { font-size: 1.7rem !important; color: var(--sky) !important; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }
h3 { font-size: 1.2rem !important; color: var(--gold) !important; }

/* ── Gold accent line ── */
.gold-rule { height: 2px; background: linear-gradient(90deg, var(--gold), transparent); border: none; margin: 1rem 0; }

/* ── Cards ── */
.mp-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    transition: border-color 0.3s;
}
.mp-card:hover { border-color: rgba(255,215,0,0.3); }

.mp-card-gold {
    background: linear-gradient(135deg, rgba(255,215,0,0.07), rgba(0,51,102,0.4));
    border: 1px solid rgba(255,215,0,0.25);
    border-radius: 12px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 1.1rem;
}

.mp-card-blue {
    background: linear-gradient(135deg, rgba(0,77,128,0.3), rgba(0,51,102,0.5));
    border: 1px solid rgba(107,174,214,0.25);
    border-radius: 12px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 1.1rem;
}

/* ── Metric boxes ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.metric-box {
    flex: 1; min-width: 130px;
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-box .val { font-size: 1.8rem; font-weight: 700; color: var(--gold); font-family: 'Playfair Display', serif; }
.metric-box .lbl { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 0.22rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-gold  { background: rgba(255,215,0,0.15);  color: var(--gold);   border: 1px solid rgba(255,215,0,0.3); }
.badge-blue  { background: rgba(107,174,214,0.15); color: var(--sky);    border: 1px solid rgba(107,174,214,0.3); }
.badge-green { background: rgba(40,167,69,0.15);   color: var(--green);  border: 1px solid rgba(40,167,69,0.3); }
.badge-red   { background: rgba(220,53,69,0.15);   color: var(--red);    border: 1px solid rgba(220,53,69,0.3); }
.badge-orange{ background: rgba(253,126,20,0.15);  color: var(--orange); border: 1px solid rgba(253,126,20,0.3); }

/* ── Code blocks ── */
code, .stCode, pre {
    font-family: 'Fira Code', monospace !important;
    background: #060d1a !important;
    color: #a8d8ea !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 0.84rem !important;
}

/* ── Tables ── */
.stDataFrame, .stTable { border-radius: 10px !important; overflow: hidden !important; }
thead tr th {
    background: var(--navy) !important;
    color: var(--gold) !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--navy), var(--navy-mid)) !important;
    color: var(--gold) !important;
    border: 1px solid rgba(255,215,0,0.35) !important;
    border-radius: 8px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: all 0.25s !important;
    padding: 0.5rem 1.4rem !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--navy-mid), #006699) !important;
    border-color: var(--gold) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(255,215,0,0.15) !important;
}

/* ── Sliders / Selects ── */
.stSlider > div > div > div > div { background: var(--gold) !important; }
.stSelectbox > div > div { background: var(--card2) !important; border-color: var(--border) !important; }
.stMultiSelect > div > div { background: var(--card2) !important; border-color: var(--border) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card2) !important;
    border-radius: 10px 10px 0 0 !important;
    border-bottom: 2px solid var(--gold) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.3rem !important;
    border-radius: 8px 8px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(255,215,0,0.1) !important;
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--card2) !important;
    border-radius: 8px !important;
    color: var(--sky) !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
    border: 1px solid var(--border) !important;
}

/* ── Info / Warning / Success ── */
.stInfo    { background: rgba(107,174,214,0.1) !important; border-left: 4px solid var(--sky) !important; border-radius: 8px !important; }
.stWarning { background: rgba(255,215,0,0.08)  !important; border-left: 4px solid var(--gold) !important; border-radius: 8px !important; }
.stSuccess { background: rgba(40,167,69,0.08)  !important; border-left: 4px solid var(--green) !important; border-radius: 8px !important; }
.stError   { background: rgba(220,53,69,0.08)  !important; border-left: 4px solid var(--red) !important; border-radius: 8px !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, var(--navy), var(--gold)) !important; border-radius: 4px !important; }

/* ── Definition / insight boxes ── */
.defn-box {
    background: linear-gradient(135deg, rgba(173,216,230,0.08), rgba(0,51,102,0.25));
    border-left: 4px solid var(--sky);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.95rem;
}
.insight-box {
    background: linear-gradient(135deg, rgba(255,215,0,0.07), rgba(200,168,0,0.04));
    border-left: 4px solid var(--gold);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.95rem;
}
.warning-box {
    background: rgba(220,53,69,0.07);
    border-left: 4px solid var(--red);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.95rem;
}
.success-box {
    background: rgba(40,167,69,0.07);
    border-left: 4px solid var(--green);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.95rem;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #051428 0%, #0a1f3d 50%, #051428 100%);
    border: 1px solid rgba(255,215,0,0.2);
    border-radius: 16px;
    padding: 2.5rem 2.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(255,215,0,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.5rem !important;
    font-weight: 900 !important;
    color: #fff !important;
    line-height: 1.2 !important;
    margin: 0 0 0.5rem 0 !important;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--sky);
    margin: 0.4rem 0 1rem 0;
}
.hero-quote {
    font-style: italic;
    color: var(--gold);
    font-size: 1rem;
    border-left: 3px solid var(--gold);
    padding-left: 1rem;
    margin-top: 1.2rem;
}

/* ── Footer ── */
.mp-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
    color: var(--muted);
    font-size: 0.82rem;
}
.mp-footer a { color: var(--gold); text-decoration: none; }
.mp-footer a:hover { text-decoration: underline; }

/* ── Step indicators ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px; height: 32px;
    background: var(--gold);
    color: var(--navy);
    border-radius: 50%;
    font-weight: 700;
    font-size: 0.9rem;
    margin-right: 0.7rem;
    flex-shrink: 0;
}
.step-row { display: flex; align-items: flex-start; margin-bottom: 1rem; }
.step-content { flex: 1; padding-top: 0.3rem; }

/* ── Selectbox labels ── */
label[data-testid="stWidgetLabel"] {
    color: var(--sky) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* Streamlit plot backgrounds transparent */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""",
        unsafe_allow_html=True,
    )
