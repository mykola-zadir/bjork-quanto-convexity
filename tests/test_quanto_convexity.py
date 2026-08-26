"""
tests/test_quanto_convexity.py
------------------------------
Run: venv/bin/python tests/test_quanto_convexity.py
Covers: (0) the EXACT contract-arithmetic identity the whole strategy rests on
(price legs of the USD-converted pair cancel; the r² term only appears under
entry-price conversion); (1) jitter-proof print alignment; (2) daily spread
aggregation + known-by-close timing; (3) fair-spread causality; (4) signal
hysteresis + both modes; (5) pair P&L hand case, shift, flip costs, causality.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy.quanto_convexity import (align_funding, daily_spread,
                                       fair_spread_daily, deviation_signal,
                                       side_from_signal, pair_pnl,
                                       COST_PER_SIDE_CHANGE)

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ✓ {name}")
    else:
        FAIL += 1; print(f"  ✗ {name}  {detail}")


def test_contract_identity():
    print("\n[0] exact contract arithmetic (the derivation itself)")
    S0, w = 100.0, 1.0
    for r in (0.05, -0.05, 0.20):
        S1 = S0 * (1 + r)
        lin = (w / S0) * (S1 - S0)                    # long $w linear
        inv_usd_same_day = w * (1/S0 - 1/S1) * S1     # long $w inverse, @S1
        check(f"r={r:+.2f}: same-day-converted legs cancel exactly",
              abs(lin - inv_usd_same_day) < 1e-12)
        inv_usd_entry = w * (1/S0 - 1/S1) * S0        # converted at entry px
        check(f"r={r:+.2f}: entry-price conversion exposes ≈w·r² convexity",
              abs((lin - inv_usd_entry) - w * r * r / (1 + r)) < 1e-12)


def test_alignment():
    print("\n[1] jitter-proof alignment")
    t0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    lin_idx = [t0 + pd.Timedelta(hours=8 * k, milliseconds=1) for k in range(9)]
    inv_idx = [t0 + pd.Timedelta(hours=8 * k) - pd.Timedelta(milliseconds=3)
               for k in range(9)]
    lin = pd.Series(0.0001, index=pd.DatetimeIndex(lin_idx))
    inv = pd.Series(0.0002, index=pd.DatetimeIndex(inv_idx))
    al = align_funding(lin, inv)
    check("all 9 jittered prints align (naive join would drop them)",
          len(al) == 9)
    check("columns carry the right series",
          (al["lin"] == 0.0001).all() and (al["inv"] == 0.0002).all())


def test_daily_spread():
    print("\n[2] daily spread aggregation")
    # day 1: complete (3 prints); day 2: PARTIAL (2 prints — missing 16:00)
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")
                            + pd.Timedelta(hours=h) for h in (0, 8, 16, 24, 32)])
    al = pd.DataFrame({"lin": [3e-4] * 5, "inv": [1e-4] * 5}, index=idx)
    sp = daily_spread(al)
    check("day spread = Σ 3 prints of (lin−inv)", np.isclose(sp.iloc[0], 6e-4))
    check("partial day (2 prints) is EXCLUDED, not counted as a full day",
          len(sp) == 1)


def test_fair_and_signal():
    print("\n[3] fair spread causality + [4] signal")
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=24 * 200, freq="1h", tz="UTC")
    px = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.005, len(idx))), index=idx)
    f = fair_spread_daily(px)
    f2 = fair_spread_daily(px.iloc[:24 * 150])
    common = f.iloc[:148].dropna().index.intersection(f2.dropna().index)
    check("fair spread causal (prefix-invariant, 1-day lag)",
          np.allclose(f.loc[common].values, f2.loc[common].values))
    D = pd.Series([1., 1., 0.1, -0.2, -1., -1., 0.2, 1.],
                  index=pd.date_range("2024-01-01", periods=8, freq="1D", tz="UTC"))
    s = side_from_signal(D, deadband=0.5)
    check("hysteresis: holds prior side inside the deadband",
          list(s.values) == [1, 1, 1, 1, -1, -1, -1, 1])
    sr = side_from_signal(D, mode="receive")
    check("receive mode = sign with ffill", list(sr.values[:2]) == [1, 1]
          and sr.iloc[4] == -1)


def test_pair_pnl():
    print("\n[5] pair P&L: shift, hand case, flip cost, causality")
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    spread = pd.Series([10e-4, 10e-4, -10e-4, 10e-4, 10e-4], index=idx)
    side = pd.Series([1., 1., -1., -1., 1.], index=idx)
    p = pair_pnl(side, spread)
    c = COST_PER_SIDE_CHANGE
    # t0: held 0 (entry cost half-flip? |1-0|/2 = 0.5 flips = open both legs)
    hand = [0 - 0.5 * c,
            1 * 10e-4 - 0,           # held +1 earns t1 spread, no change
            1 * -10e-4 - 1.0 * c,    # held +1 earns t2 (−10bp), full flip cost
            -1 * 10e-4 - 0,          # held −1 earns t3 (+10bp) → −10bp
            -1 * 10e-4 - 1.0 * c]    # held −1, flips at t4
    check("hand-computed path matches exactly", np.allclose(p.values, hand),
          f"{p.values} vs {hand}")
    check("no-lookahead: day-t spread earned by side decided t−1",
          np.isclose(p.iloc[1], 10e-4))
    # causality: truncating the input reproduces the prefix
    p2 = pair_pnl(side.iloc[:3], spread.iloc[:3])
    check("prefix-invariant", np.allclose(p.iloc[:3].values, p2.values))


if __name__ == "__main__":
    print("=" * 60)
    print("  Quanto-convexity tests")
    print("=" * 60)
    test_contract_identity()
    test_alignment()
    test_daily_spread()
    test_fair_and_signal()
    test_pair_pnl()
    print("\n" + "=" * 60)
    print(f"  RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)
