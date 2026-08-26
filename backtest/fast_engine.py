"""
backtest/fast_engine.py
-----------------------
OPTIMIZATION engine — the speed-focused counterpart to engine.py.

Two engines, two jobs:
  • engine.py       → LEARNING. Keeps every series, builds a trade ledger,
                      renders a 6-panel dashboard. Slow, transparent, for one
                      configuration you want to understand deeply.
  • fast_engine.py  → OPTIMIZATION. Pure NumPy, no ledger, no plotting, no
                      DataFrame churn. Returns just the numbers an optimiser
                      needs, as fast as possible so thousands of parameter
                      combinations can be swept in seconds.

CRITICAL invariant: both engines MUST produce identical P&L for identical
inputs. The same two honesty rules are enforced here:
  1. No lookahead — positions are shifted one bar before earning returns.
  2. Costs are charged on every change in position.
The stress-test script asserts the two engines agree to floating-point
tolerance; if you change the P&L logic in one, change it in both (or better,
keep them in sync via these primitives).

Everything operates on raw NumPy arrays internally to avoid pandas overhead,
which is the single biggest speed win in a tight optimisation loop.
"""

import numpy as np
import pandas as pd

PERIODS_PER_YEAR_1H = 24 * 365  # 8760


# ---------------------------------------------------------------------------
# Core P&L primitive (shared mental model with engine.py)
# ---------------------------------------------------------------------------

def compute_pnl(
    prices: np.ndarray,
    target_positions: np.ndarray,
    fee: float,
    slippage: float,
    funding: np.ndarray | None = None,
):
    """
    Vectorised net-return computation. Pure NumPy.

    Parameters
    ----------
    prices : np.ndarray (T,)
    target_positions : np.ndarray (T,)
        Desired position decided at each bar's close (pre-shift).
    fee, slippage : float
        Per-side cost fractions.
    funding : np.ndarray (T,), optional
        Per-period perp FUNDING RATE accrued during each bar. The HELD
        (shifted) position pays it: a long PAYS when funding>0, a short
        RECEIVES it — funding_cost[t] = positions[t]·funding[t]. This is the
        third honesty rule for a futures/perp book: its P&L is price move ∓
        funding, not price move alone. None (default) → no funding, so the
        result is bit-for-bit identical to the pre-funding engine (backward
        compatible). NaNs in funding are treated as 0 (no accrual that bar).

    Returns
    -------
    dict of arrays: market_returns, positions (held), gross_returns,
                    costs, funding_costs, net_returns, equity
    """
    prices = np.asarray(prices, dtype=float)
    # Kernel contract: prices must be clean. Silently propagating a NaN here
    # would diverge from run_backtest (which drops NaN bars) and poison every
    # downstream metric — so fail loudly instead. The engines that wrap this
    # (fast_backtest / run_backtest) drop/align NaN before calling in.
    if not np.all(np.isfinite(prices)):
        raise ValueError("compute_pnl: prices contain non-finite values; "
                         "drop/align them before calling.")
    T = prices.shape[0]

    # Simple returns; first bar = 0
    market_returns = np.empty(T)
    market_returns[0] = 0.0
    market_returns[1:] = prices[1:] / prices[:-1] - 1.0

    # Rule 1: no lookahead — hold today what we targeted yesterday
    positions = np.empty(T)
    positions[0] = 0.0
    positions[1:] = target_positions[:-1]

    gross_returns = positions * market_returns

    # Rule 2: cost on every change in held position
    turnover = np.empty(T)
    turnover[0] = abs(positions[0])
    turnover[1:] = np.abs(np.diff(positions))
    costs = turnover * (fee + slippage)

    # Rule 3: funding accrual on the held position (perp/futures books only)
    if funding is None:
        funding_costs = np.zeros(T)
    else:
        f = np.asarray(funding, dtype=float)
        if f.shape[0] != T:
            raise ValueError(f"funding ({f.shape[0]}) must match prices ({T})")
        funding_costs = positions * np.nan_to_num(f, nan=0.0)

    net_returns = gross_returns - costs - funding_costs
    equity = np.cumprod(1.0 + net_returns)

    return {
        "market_returns": market_returns,
        "positions": positions,
        "gross_returns": gross_returns,
        "costs": costs,
        "funding_costs": funding_costs,
        "net_returns": net_returns,
        "equity": equity,
    }


# ---------------------------------------------------------------------------
# Fast metric helpers (NumPy, no pandas)
# ---------------------------------------------------------------------------

def _sharpe(net: np.ndarray, ppy: float) -> float:
    # ddof=1 (sample std) to match pandas .std() used in metrics.py — the two
    # engines must agree exactly, so the convention is pinned here.
    sd = net.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(net.mean() / sd * np.sqrt(ppy))


def _sortino(net: np.ndarray, ppy: float) -> float:
    downside = net[net < 0]
    if downside.size == 0:
        return 0.0
    dd = downside.std(ddof=1)
    if dd == 0 or not np.isfinite(dd):
        return 0.0
    return float(net.mean() / dd * np.sqrt(ppy))


def _max_drawdown(equity: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity)
    dd = equity / running_max - 1.0
    return float(dd.min())


def _cagr(equity: np.ndarray, ppy: float) -> float:
    n = equity.shape[0] - 1
    if n <= 0:
        return 0.0
    years = n / ppy
    growth = equity[-1] / equity[0] if equity[0] != 0 else 0.0
    if growth <= 0:
        return -1.0
    # Compute in log-space and guard against overflow on extreme (e.g.
    # perfect-foresight) equity curves over very short annualisation windows.
    with np.errstate(over="ignore"):
        log_cagr = np.log(growth) / years
        if log_cagr > 700:          # exp(700) is near float max
            return float("inf")
        return float(np.exp(log_cagr) - 1.0)


def _count_trades(positions: np.ndarray) -> int:
    """Number of trade openings: sign changes into a new non-zero direction."""
    sign = np.sign(positions)
    prev = np.concatenate(([0.0], sign[:-1]))
    opens = (sign != prev) & (sign != 0)
    return int(opens.sum())


def _exposure(positions: np.ndarray) -> float:
    return float((positions != 0).mean())


# ---------------------------------------------------------------------------
# Public fast backtest
# ---------------------------------------------------------------------------

def fast_backtest(
    prices,
    target_positions,
    fee: float = 0.001,
    slippage: float = 0.0005,
    periods_per_year: float = PERIODS_PER_YEAR_1H,
    funding=None,
) -> dict:
    """
    Run a fast backtest and return a metrics dict only.

    Accepts pandas Series or NumPy arrays. Designed to be called in a tight
    loop — minimal allocation, no ledger, no plotting. `funding` (optional,
    per-period funding rate; Series aligned to prices or array) charges perp
    funding on the held position — see compute_pnl. None ⇒ identical to before.

    Returns
    -------
    dict with: total_return, cagr, sharpe, sortino, calmar, max_drawdown,
               exposure, num_trades, turnover, final_equity
    """
    # Coerce to aligned NumPy arrays, with the SAME NaN policy as run_backtest:
    # drop NaN price bars and label-align target+funding onto the surviving
    # index, so the two engines agree bit-for-bit even on gappy inputs.
    if isinstance(prices, pd.Series):
        if funding is not None and prices.isna().any():
            raise ValueError(
                "prices contain NaN while funding is supplied; align both to a "
                "gapless index first — dropping bars would silently mis-accrue "
                "funding.")
        if not isinstance(target_positions, pd.Series):
            target_positions = pd.Series(np.asarray(target_positions, dtype=float),
                                         index=prices.index)
        keep = prices.dropna()
        target_positions = target_positions.reindex(keep.index).fillna(0.0)
        if funding is not None:
            if not isinstance(funding, pd.Series):
                raise ValueError("when prices is a Series, pass funding as a "
                                 "Series (label-aligned) or None, not a bare array.")
            funding = funding.reindex(keep.index).fillna(0.0).to_numpy(dtype=float)
        prices = keep.to_numpy(dtype=float)
    elif funding is not None and isinstance(funding, pd.Series):
        raise ValueError("when prices is an array, pass funding as an array or "
                         "None, not a Series (it cannot be label-aligned).")
    px = np.asarray(prices, dtype=float)
    tp = np.asarray(target_positions, dtype=float)
    fnd = None if funding is None else np.asarray(funding, dtype=float)

    if px.shape[0] != tp.shape[0]:
        raise ValueError(f"prices ({px.shape[0]}) and positions ({tp.shape[0]}) "
                         "must be the same length")
    if px.shape[0] < 2:
        # Degenerate: nothing to trade
        return {
            "total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
            "calmar": 0.0, "max_drawdown": 0.0, "exposure": 0.0,
            "num_trades": 0, "turnover": 0.0, "final_equity": 1.0,
        }

    out = compute_pnl(px, tp, fee, slippage, funding=fnd)
    net    = out["net_returns"]
    equity = out["equity"]
    pos    = out["positions"]

    total_ret = float(equity[-1] - 1.0)
    mdd = _max_drawdown(equity)
    cg  = _cagr(equity, periods_per_year)
    calmar = float(cg / abs(mdd)) if mdd != 0 else 0.0

    return {
        "total_return": total_ret,
        "cagr":         cg,
        "sharpe":       _sharpe(net, periods_per_year),
        "sortino":      _sortino(net, periods_per_year),
        "calmar":       calmar,
        "max_drawdown": mdd,
        "exposure":     _exposure(pos),
        "num_trades":   _count_trades(pos),
        "turnover":     float(out["costs"].sum() / (fee + slippage)) if (fee + slippage) > 0 else 0.0,
        "final_equity": float(equity[-1]),
    }
