"""
verify_quanto_convexity.py
==========================
PRE-REGISTRATION (frozen before any trading P&L was computed)
-------------------------------------------------------------
BJÖRK-DERIVED STUDY (Arbitrage Theory in Continuous Time: ch. 4 Itô, ch. 13
quanto, ch. 26 change of numeraire). Two perps on the same USD index — linear
(USDT-margined) and inverse (coin-margined, P&L paid in the asset = a quanto
with payout-currency correlation 1) — must, under no-arbitrage, carry funding
rates that differ by the quanto convexity adjustment:

        fair (f_lin − f_inv) ≈ σ² Δt        per funding period.

Exact contract arithmetic (test [0] of tests/test_quanto_convexity.py) shows
the USD-converted pair's price legs cancel exactly, so the pair's P&L IS the
funding spread minus costs — making the spread the clean observable for the
identity, and the identity's failure a real theory-vs-market finding.

(I) IDENTITY TEST (the primary deliverable — a theory test, not a P&L claim):
    H₀ᴬ: mean daily spread = 0            (market prices NO convexity)
    H₀ᴮ: mean (spread − σ̂²/365) = 0       (market prices the FULL quanto adj.)
    Newey-West on both, BTC & ETH, plus clamp diagnostics and sub-periods.
    The two H₀ are mutually exclusive in expectation; whichever survives
    tells us WHERE the convexity premium lives (or that it is absent).

(II) TRADING (all trials → cumulative ledger): daily pair carry per $1/leg,
    flip cost 4 legs × 7bp. Rules (frozen):
      theory|db{0, 1e-4}: side = sign(D), D = 5d-smoothed spread − σ̂²₃₀/365,
                          with deadband hysteresis;
      receive|sm{5, 20}:  classic carry — side = sign of smoothed spread.
    × {BTC, ETH} = 8 trials. Certify best BTC rule (ETH twin replication,
    2024→now sub-period, returns_matrix, DSR vs cumulative ledger).

HONEST PRIOR. Identity: pre-fetch magnitudes suggest H₀ᴮ is rejected hard
(observed ≈ +1%/yr vs theory ≈ 26%/yr) — the funding CHANNEL cannot carry the
quanto premium because both funding formulas anchor to the same USD index;
the premium, if priced anywhere, must live in the mark-price basis (no public
history — documented limitation). Trading prior ~20-25%: spread std ~1bp/print
is real, but 28bp flip cost demands multi-week persistence per side.

Run (network-free, data committed): python verify_quanto_convexity.py
"""

import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd

from strategy.quanto_convexity import (align_funding, daily_spread,
                                       fair_spread_daily, deviation_signal,
                                       side_from_signal, pair_pnl)
from perf_utils import (perf, perf_row, PPY, log, log_header,
                        load_cached_perp, load_prior_trials)
from backtest.statistics import newey_west_mean
from backtest.certify import certify
from utils.data_fetch import widest_cache

SPLIT = "2024-01-01"
PAIRS = {"BTC": ("FUNDING_BTCUSDT_*.parquet", "DFUNDING_BTCUSD_PERP_*.parquet",
                 "BTCUSDT"),
         "ETH": ("FUNDING_ETHUSDT_*.parquet", "DFUNDING_ETHUSD_PERP_*.parquet",
                 "ETHUSDT")}
RULES = [("theory|db0", "sign", 0.0, 5), ("theory|db1e-4", "sign", 1e-4, 5),
         ("receive|sm5", "receive", 0.0, 5), ("receive|sm20", "receive", 0.0, 20),
         # v2 improvement (from the v1 diagnosis, charged to the ledger):
         # funding P&L is a FLOW — a level-convergence signal is category-
         # confused. Theory contributes exactly ONE bit: the SIGN of the fair
         # spread (linear > inverse). So hold that side permanently:
         # zero parameters, one entry, ~zero turnover.
         ("always|theoryside", "always", 0.0, 1)]


def load_pair(ccy):
    fl, fi, px_sym = PAIRS[ccy]
    # widest-span cache, NOT lexicographic [-1]: a 2021-start file must not
    # shadow a 2020-start one (drops ~140 days of valid pair history)
    lin = pd.read_parquet(widest_cache(
        [Path(x) for x in glob.glob(f"data/{fl}")]))["funding_rate"]
    inv = pd.read_parquet(widest_cache(
        [Path(x) for x in glob.glob(f"data/{fi}")]))["funding_rate"]
    al = align_funding(lin, inv)
    sp = daily_spread(al)
    px = load_cached_perp(px_sym, "1h")["close"]
    fair = fair_spread_daily(px, rv_days=30).reindex(sp.index)
    return al, sp, fair


def main():
    led = load_prior_trials()
    log_header("QUANTO CONVEXITY (Björk ch.4/13/26) — linear vs inverse perp "
               "funding spread")
    log("Identity: fair (f_lin − f_inv) = σ²Δt (change of numeraire, quanto "
        "corr = 1). Pair price legs cancel exactly (tested) → spread is the "
        f"observable. Ledger before: {len(led['sharpes'])}.")

    # ---------------- (I) IDENTITY ----------------
    log_header("(I) Identity test — does the market price the quanto adjustment?")
    log("```")
    log(f"{'':<4}{'prints':>7}{'pinned@1bp':>11}{'spread bp/d':>12}{'σ²/365 bp/d':>12}"
        f"{'H0A p':>8}{'H0B p':>8}")
    data = {}
    for ccy in PAIRS:
        al, sp, fair = load_pair(ccy)
        pin = ((al["inv"] - 0.0001).abs() < 1e-9).mean()
        gap = (sp - fair).dropna()
        a = newey_west_mean(sp.dropna().values)
        b = newey_west_mean(gap.values)
        data[ccy] = (al, sp, fair)
        log(f"{ccy:<4}{len(al):>7}{pin:>11.0%}{sp.mean()*1e4:>12.3f}"
            f"{fair.mean()*1e4:>12.2f}{a['p_two_sided']:>8.3f}{b['p_two_sided']:>8.3f}")
    log("```")
    for ccy in PAIRS:
        _, sp, fair = data[ccy]
        gap = (sp - fair).dropna()
        log(f"- {ccy}: observed spread {sp.mean()*PPY*100:+.2f}%/yr vs quanto "
            f"theory {fair.mean()*PPY*100:+.1f}%/yr → shortfall "
            f"{gap.mean()*PPY*100:+.1f}%/yr "
            f"(NW p={newey_west_mean(gap.values)['p_two_sided']:.1e}).")
    log("- READING: H0B rejected ⇒ the funding channel does NOT carry the "
        "quanto premium (both funding formulas anchor the SAME USD index, so "
        "they are mechanically pinned together); if priced anywhere it must "
        "sit in the mark-price basis, for which no public history exists "
        "(documented limitation).")

    # ---------------- (II) TRADING ----------------
    log_header("(II) Trading — pair carry rules (frozen grid, 8 trials)")
    trials, ret_cols, sides, new_sh = {}, {}, {}, []
    for ccy in PAIRS:
        _, sp, fair = data[ccy]
        for name, mode, db, sm in RULES:
            if mode == "always":
                side = pd.Series(1.0, index=sp.index)
            elif mode == "sign":
                D = deviation_signal(sp, fair, smooth=sm)
                side = side_from_signal(D, deadband=db, mode="sign")
            else:
                obs = sp.rolling(sm, min_periods=sm).mean()
                side = side_from_signal(obs, mode="receive")
            r = pair_pnl(side, sp)
            m = perf(r)
            if m is None:
                continue
            key = f"{ccy}|{name}"
            flips = float((side.diff().abs() / 2).sum())
            m["flips_yr"] = flips / (len(r) / PPY)
            trials[key] = m; ret_cols[key] = r; sides[key] = side
            new_sh.append(m["pp_sharpe"])
    log("```")
    log(f"{'trial':<20}{'ann.ret':>9}{'Sharpe':>8}{'  95% CI':<15}{'MaxDD':>8}"
        f"{'NW p':>7}{'flips/yr':>9}")
    for k, m in sorted(trials.items(), key=lambda kv: -kv[1]["sharpe"]):
        log(perf_row(k, m, w=20) + f"{m['flips_yr']:>9.1f}")
    log("```")

    all_sh = np.array(led["sharpes"] + new_sh)
    best_btc = max((k for k in trials if k.startswith("BTC")),
                   key=lambda k: trials[k]["sharpe"])
    twin = "ETH" + best_btc[3:]
    rb = ret_cols[best_btc].dropna()
    rep = certify(rb.values, trials=all_sh,
                  returns_matrix=pd.DataFrame(ret_cols).dropna(),
                  second_asset_returns=(ret_cols[twin].dropna().values
                                        if twin in ret_cols else None),
                  recent_returns=rb.loc[SPLIT:].values, periods_per_year=PPY)
    cr = rep["criteria"]
    pre, post = perf(rb.loc[:"2023-12-31"]), perf(rb.loc[SPLIT:])
    log(f"- BEST BTC {best_btc}: {rep['status']} (Sharpe-CI "
        f"{cr['sharpe_ci']['status']}, DSR {cr['dsr']['status']} vs "
        f"{len(all_sh)}, PBO {cr['pbo']['status']}, repl(ETH twin) "
        f"{cr['replication']['status']}, sub-period {cr['subperiod']['status']}).")
    log(f"- DIAGNOSIS: 2021-23 {pre['sharpe'] if pre else float('nan'):+.2f} vs "
        f"2024→now {post['sharpe'] if post else float('nan'):+.2f}; "
        f"{trials[best_btc]['flips_yr']:.1f} flips/yr.")

    log(f"- DSR luck benchmark charged against {len(all_sh)} cumulative trials "
        f"({len(led['sharpes'])} prior + {len(new_sh)} from this study).")
    with open("results/quanto_convexity.json", "w") as fj:
        json.dump({"trials": trials, "best": best_btc, "status": rep["status"]},
                  fj, indent=2, default=float)
    print("\nDone → results/quanto_convexity.json")


if __name__ == "__main__":
    main()
