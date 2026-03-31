"""
Shared Plotly theme — Mountain Path colour system
"""

import plotly.graph_objects as go
import plotly.io as pio

NAVY  = "#003366"
GOLD  = "#FFD700"
SKY   = "#ADD8E6"
SKY_D = "#6BAED6"
BG    = "rgba(0,0,0,0)"
CARD  = "#0f1a2e"
TXT   = "#e6f1ff"
MUTED = "#8892b0"
GREEN = "#28a745"
RED   = "#dc3545"
ORG   = "#fd7e14"

PALETTE = [GOLD, SKY, "#4fc3f7", GREEN, ORG, "#ab47bc", RED, "#ef5350"]

MP_TEMPLATE = go.layout.Template(
    layout=dict(
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(family="Source Sans 3, sans-serif", color=TXT, size=12),
        title=dict(font=dict(family="Playfair Display, serif", color=TXT, size=16)),
        xaxis=dict(
            gridcolor="rgba(100,140,200,0.12)",
            linecolor="rgba(100,140,200,0.3)",
            tickcolor=MUTED,
            zerolinecolor="rgba(100,140,200,0.2)",
        ),
        yaxis=dict(
            gridcolor="rgba(100,140,200,0.12)",
            linecolor="rgba(100,140,200,0.3)",
            tickcolor=MUTED,
            zerolinecolor="rgba(100,140,200,0.2)",
        ),
        legend=dict(
            bgcolor="rgba(10,15,30,0.8)",
            bordercolor="rgba(100,140,200,0.2)",
            borderwidth=1,
            font=dict(color=TXT),
        ),
        colorway=PALETTE,
        margin=dict(l=50, r=20, t=50, b=40),
    )
)
pio.templates["mountain_path"] = MP_TEMPLATE
pio.templates.default = "mountain_path"
