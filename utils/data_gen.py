"""
Shared data generation utilities for all pages
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ── NSE stock returns ─────────────────────────────────────────────────────────
def make_nse_prices(n=252, n_stocks=5, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    base = [2800, 3500, 1600, 1450, 950]
    sigma = [50, 80, 30, 60, 25]
    data = {
        s: base[i] + np.cumsum(rng.normal(0.3, sigma[i], n))
        for i, s in enumerate(stocks[:n_stocks])
    }
    df = pd.DataFrame(data, index=dates)
    return df


def inject_missing(df, seed=42):
    rng = np.random.default_rng(seed)
    df = df.copy()
    cols = df.columns.tolist()
    # Block missing
    df.iloc[50:54, 0] = np.nan
    # Intermittent random
    idx = rng.choice(len(df), 15, replace=False)
    df.iloc[idx, 1] = np.nan
    # Monotone tail (delisted)
    df.iloc[220:, 2] = np.nan
    # MAR: high-rate missing in later period
    idx2 = rng.choice(range(150, len(df)), 10, replace=False)
    df.iloc[idx2, 3] = np.nan
    return df


def make_returns(df):
    return df.pct_change().dropna()


# ── Financial ratios panel ───────────────────────────────────────────────────
def make_ratios(n=300, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Market_Cap_Cr": rng.lognormal(10, 2, n),
        "PE_Ratio":      rng.lognormal(3.3, 0.5, n),
        "EV_EBITDA":     rng.lognormal(2.5, 0.4, n),
        "ROE_pct":       rng.normal(15, 8, n),
        "Debt_Equity":   rng.exponential(0.8, n),
        "PAT_Margin":    rng.normal(12, 6, n),
    })
    return df


# ── G-Sec yield curve ────────────────────────────────────────────────────────
GSEC_MATURITIES = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 14, 30])
GSEC_YIELDS     = np.array([6.55, 6.60, 6.68, 6.72, 6.79, 6.85, 6.91, 6.96, 7.02, 7.08])

# ── Return series with fat tails ─────────────────────────────────────────────
def make_fat_returns(n=500, seed=42):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.05/252, 0.18/np.sqrt(252), n)
    shock_idx = rng.choice(n, 10, replace=False)
    rets[shock_idx] = rng.choice([-0.08, -0.06, -0.05, 0.06, 0.07, 0.09], 10)
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(rets, index=dates, name="Daily_Return")
