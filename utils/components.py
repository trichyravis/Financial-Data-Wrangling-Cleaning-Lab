"""
Reusable UI components — The Mountain Path design system
"""

import streamlit as st


# ── Navigation ───────────────────────────────────────────────────────────────
PAGES = [
    "🏠 Home",
    "🔍 Missing Data",
    "📊 Outlier Detection",
    "📅 Time Series Formatting",
    "⚠️ Invalid Values",
    "⚙️ Full Cleaning Pipeline",
    "📚 Case Studies",
]


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
<div style="text-align:center; padding: 1rem 0 0.5rem;">
  <div style="font-size:2.4rem;">⛰️</div>
  <div style="font-family:'Playfair Display',serif; font-size:1.05rem;
              font-weight:700; color:#FFD700; line-height:1.3;">
    The Mountain Path
  </div>
  <div style="font-size:0.72rem; color:#8892b0; text-transform:uppercase;
              letter-spacing:0.1em; margin-top:0.2rem;">
    World of Finance
  </div>
</div>
<hr style="border-color:rgba(100,140,200,0.2); margin:0.8rem 0;">
<div style="font-size:0.72rem; color:#8892b0; text-align:center;
            text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem;">
  Financial Analytics — Session 2
</div>
""",
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigate",
            PAGES,
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:rgba(100,140,200,0.15); margin:1.2rem 0 0.8rem;'>", unsafe_allow_html=True)
        st.markdown(
            """
<div style="font-size:0.75rem; color:#8892b0; padding:0 0.2rem;">
  <div style="color:#ADD8E6; font-weight:600; margin-bottom:0.4rem;">Prof. V. Ravichandran</div>
  28+ Yrs Corporate Finance<br>10+ Yrs Academic Excellence
  <div style="margin-top:0.8rem;">
    <span style="color:#6BAED6;">Visiting Faculty:</span><br>
    NMIMS Bangalore • BITS Pilani<br>RV University • GIM Goa
  </div>
  <div style="margin-top:0.8rem;">
    <a href="https://themountainpathacademy.com" target="_blank"
       style="color:#FFD700; text-decoration:none; font-size:0.73rem;">
      🌐 themountainpathacademy.com
    </a>
  </div>
  <div style="margin-top:0.4rem;">
    <a href="https://www.linkedin.com/in/trichyravis" target="_blank"
       style="color:#6BAED6; text-decoration:none; font-size:0.73rem;">
      💼 LinkedIn
    </a>
    &nbsp;|&nbsp;
    <a href="https://github.com/trichyravis" target="_blank"
       style="color:#6BAED6; text-decoration:none; font-size:0.73rem;">
      🐙 GitHub
    </a>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    return page


# ── Hero Banner ──────────────────────────────────────────────────────────────
def render_hero(title: str, subtitle: str, quote: str = "", badges: list = None):
    badges_html = ""
    if badges:
        badges_html = "".join(
            f'<span class="badge badge-{b["style"]}" style="margin-right:0.5rem;">{b["text"]}</span>'
            for b in badges
        )

    quote_html = ""
    if quote:
        quote_html = f'<div class="hero-quote">"{quote}"</div>'

    st.markdown(
        f"""
<div class="hero-banner">
  <div class="hero-title">{title}</div>
  <div class="hero-sub">{subtitle}</div>
  {badges_html}
  {quote_html}
</div>
""",
        unsafe_allow_html=True,
    )


# ── Card helpers ─────────────────────────────────────────────────────────────
def card(content_html: str, style: str = "default"):
    cls = {"default": "mp-card", "gold": "mp-card-gold", "blue": "mp-card-blue"}.get(style, "mp-card")
    st.markdown(f'<div class="{cls}">{content_html}</div>', unsafe_allow_html=True)


def defn_box(title: str, body: str):
    st.markdown(
        f'<div class="defn-box"><strong style="color:#ADD8E6;">📘 {title}</strong><br>{body}</div>',
        unsafe_allow_html=True,
    )


def insight_box(body: str):
    st.markdown(
        f'<div class="insight-box">💡 {body}</div>',
        unsafe_allow_html=True,
    )


def warning_box(body: str):
    st.markdown(
        f'<div class="warning-box">⚠️ {body}</div>',
        unsafe_allow_html=True,
    )


def success_box(body: str):
    st.markdown(
        f'<div class="success-box">✅ {body}</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, style: str = "blue"):
    return f'<span class="badge badge-{style}">{text}</span>'


def metric_row(metrics: list):
    """metrics = [{"val": "42%", "lbl": "Missing Rate"}, ...]"""
    items = "".join(
        f'<div class="metric-box"><div class="val">{m["val"]}</div><div class="lbl">{m["lbl"]}</div></div>'
        for m in metrics
    )
    st.markdown(f'<div class="metric-row">{items}</div>', unsafe_allow_html=True)


# ── Section header ────────────────────────────────────────────────────────────
def section_header(icon: str, title: str):
    st.markdown(
        f"""
<div style="display:flex; align-items:center; gap:0.7rem; margin:1.5rem 0 0.8rem;">
  <span style="font-size:1.4rem;">{icon}</span>
  <span style="font-family:'Playfair Display',serif; font-size:1.35rem;
               font-weight:700; color:#ADD8E6;">{title}</span>
</div>
<div class="gold-rule"></div>
""",
        unsafe_allow_html=True,
    )


# ── Footer ───────────────────────────────────────────────────────────────────
def render_footer():
    st.markdown(
        """
<div class="mp-footer">
  <b style="color:#ADD8E6; font-family:'Playfair Display',serif;">
    The Mountain Path – World of Finance
  </b><br>
  Financial Analytics | Session 2 — Data Quality &amp; Time Series Engineering<br>
  BITS Pilani WILP | MBA ZG517 / PDBA / PDFI / PDFT ZG517 | Second Semester 2025–2026<br>
  <div style="margin-top:0.5rem;">
    Prof. V. Ravichandran &nbsp;|&nbsp;
    <a href="https://themountainpathacademy.com" target="_blank">themountainpathacademy.com</a>
    &nbsp;|&nbsp;
    <a href="https://www.linkedin.com/in/trichyravis" target="_blank">LinkedIn</a>
    &nbsp;|&nbsp;
    <a href="https://github.com/trichyravis" target="_blank">GitHub</a>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
