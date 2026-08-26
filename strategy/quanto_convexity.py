"""
strategy/quanto_convexity.py
----------------------------
QUANTO-CONVEXITY funding spread — Björk-derived (ch. 4 Itô, 13 quanto/currency
derivatives, 26 change of numeraire), futures-only.

THE THEORY (why a spread must exist). Binance lists two perps on the same USD
index: LINEAR (USDT-margined) and INVERSE (coin-margined, P&L settled in the
asset itself — a quanto). In the BTC numeraire the inverse position's equity
is linear where the USD book is convex, and change-of-numeraire (Girsanov with
the asset itself as payout currency ⇒ quanto correlation = 1) forces the fair
carry of the two contracts apart by the quanto adjustment

        fair spread (linear − inverse) ≈ σ² · Δt          per funding period.

THE VERIFIED P&L MODEL (exact contract arithmetic, no Taylor — see study log):
with both legs marked and converted to USD at the same close, the price legs
of {long $w linear, short $w inverse} cancel EXACTLY; what remains is the
FUNDING SPREAD minus costs. So the tradeable object is the spread itself, and
the theory supplies its fair value (σ²Δt) — the identity this study tests.

Everything causal: day-t spread uses prints stamped during day t (all known by
00:00 next day); a side decided at t earns day t+1's spread (shift in pnl).
"""

import numpy as np
import pandas as pd

COST_PER_SIDE_CHANGE = 4 * 0.0007   # flip = close 2 legs + open 2 legs @ 7bp
PPY = 365


def align_funding(lin_8h: pd.Series, inv_8h: pd.Series) -> pd.DataFrame:
    """
    Align linear vs inverse 8h funding prints. Exchange timestamps carry
    millisecond jitter on BOTH endpoints (e.g. 07:59:59.997, 00:00:00.001), so
    naive index alignment silently drops ~60% of prints — timestamps are
    ROUNDED to the nearest hour first (verified lossless: prints are 8h apart).
    Returns DataFrame [lin, inv] on the rounded print grid, inner-joined.
    """
    l = lin_8h.copy(); l.index = l.index.round("h")
    i = inv_8h.copy(); i.index = i.index.round("h")
    l = l[~l.index.duplicated(keep="last")]
    i = i[~i.index.duplicated(keep="last")]
    return pd.concat([l.rename("lin"), i.rename("inv")], axis=1).dropna()


def daily_spread(aligned: pd.DataFrame) -> pd.Series:
    """Day-t funding spread = Σ over day-t prints of (lin − inv). All three
    prints of day t are stamped ≤ 16:00 UTC, so the value is known at t close."""
    sp = aligned["lin"] - aligned["inv"]
    g = sp.groupby(sp.index.floor("D"))
    sums, counts = g.sum(), g.size()
    # a "day" with <3 matched prints (missing print on either leg, or the
    # partial fetch day) must not enter the identity/P&L as a full day
    return sums[counts == 3]


def fair_spread_daily(close_1h: pd.Series, rv_days: int = 30) -> pd.Series:
    """Theoretical fair DAILY spread = σ̂²/365, with σ̂² the trailing rv_days
    realized variance of daily log returns (annualized), lagged one day so the
    day-t value uses returns through t−1 only. Causal."""
    daily = close_1h.resample("1D").last()
    r = np.log(daily).diff()
    var_ann = (r.rolling(rv_days, min_periods=rv_days).var(ddof=1) * PPY)
    return (var_ann / PPY).shift(1)


def deviation_signal(spread_d: pd.Series, fair_d: pd.Series,
                     smooth: int = 5) -> pd.Series:
    """D_t = smoothed observed spread − fair spread (both known at t close).
    D > 0: the market pays MORE than the convexity is worth → receive the
    spread (long inverse + short linear ... the side collecting lin−inv).
    D < 0: spread too small vs theory → hold the opposite side."""
    obs = spread_d.rolling(smooth, min_periods=smooth).mean()
    return obs - fair_d.reindex(spread_d.index)


def side_from_signal(D: pd.Series, deadband: float = 0.0,
                     mode: str = "sign") -> pd.Series:
    """
    mode 'sign'    : side = +1 when D > +deadband (collect spread), −1 when
                     D < −deadband, else carry previous side (hysteresis).
    mode 'receive' : classic carry — side = sign of the smoothed spread itself
                     (ignore theory; always sit on the receiving side).
    +1 ≡ the book that RECEIVES (lin − inv); −1 the mirror.
    """
    if mode == "receive":
        s = np.sign(D)                      # caller passes smoothed spread as D
        return pd.Series(s, index=D.index).replace(0.0, np.nan).ffill().fillna(0.0)
    raw = pd.Series(np.nan, index=D.index)
    raw[D > deadband] = 1.0
    raw[D < -deadband] = -1.0
    return raw.ffill().fillna(0.0)


def pair_pnl(side: pd.Series, spread_d: pd.Series,
             cost_flip: float = COST_PER_SIDE_CHANGE) -> pd.Series:
    """
    Daily net P&L per $1-per-leg pair: side decided at t−1 earns day t's
    spread (no lookahead); each |Δside| charges the 4-leg flip cost pro rata.
        pnl_t = side_{t−1}·spread_t − |side_t − side_{t−1}|/2 · cost_flip
    (÷2: |Δside| of 2 = a full flip = one full close+open of both legs.)
    """
    s = side.reindex(spread_d.index).fillna(0.0)
    held = s.shift(1).fillna(0.0)
    turn = (s - held).abs() / 2.0
    return held * spread_d - turn * cost_flip
