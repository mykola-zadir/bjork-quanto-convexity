"""
backtest/certify.py
-------------------
certify() — ONE call that runs the whole significance battery and returns a
clean PASS/FAIL verdict per criterion plus a single overall status. This
standardises the certification bar across every strategy (vol book, directional
book, …) so each study no longer hand-rolls — and silently varies — its own
pass/fail logic.

THE BAR (an edge is "CERTIFIED" only if it clears ALL of these):
  • Sharpe CI       — 95% confidence interval EXCLUDES zero, on the positive side.
                      [Mertens skew/kurtosis-aware standard error for the Sharpe.]
  • PSR ≥ 0.95      — Probabilistic Sharpe Ratio: P(true Sharpe > 0 | data),
                      corrected for fat tails.
  • DSR ≥ 0.95      — Deflated Sharpe Ratio fed EVERY trial Sharpe (not just the
                      winner). This is the anti-data-mining criterion.
  • Permutation p≤.05 — does the TIMING of positions beat a luck-matched null
                      (circularly-rotated positions, identical exposure/turnover)?
  • Newey-West p≤.05 — HAC one-sided mean test (autocorrelation-robust), so an
                      autocorrelated/overlapping series is not judged by a naive t.
  • PBO < 0.5       — Probability of Backtest Overfitting (CSCV): how often the
                      in-sample-best config is below-median out-of-sample.
  • Replication     — re-passes the core tests on a SECOND asset (e.g. BTC→ETH).
  • Sub-period      — still positive & significant on a recent held-out window.

FAIL-CLOSED. The DSR cannot be computed honestly without the trial set, so if
no trials are supplied that criterion is recorded NO_TRIALS and the verdict is
never "CERTIFIED". Criteria whose inputs were not supplied are SKIPPED, and a
run with any SKIPPED required check is "INCOMPLETE", never "CERTIFIED" — strict
by design, so a half-run battery is never mistaken for a clean pass.
"""

import numpy as np
import pandas as pd

from backtest.fast_engine import PERIODS_PER_YEAR_1H
from backtest.statistics import (
    sharpe_confidence_interval, probabilistic_sharpe_ratio,
    deflated_sharpe_ratio, probability_of_backtest_overfitting,
    permutation_test, newey_west_mean)

DEFAULT_THRESHOLDS = {"psr": 0.95, "dsr": 0.95, "pbo": 0.5,
                      "perm_p": 0.05, "nw_p": 0.05}

# the criteria that make up a FULL certification (all must PASS, none SKIPPED)
_REQUIRED = ("sharpe_ci", "psr", "dsr", "pbo", "permutation",
             "newey_west", "replication", "subperiod")


def _trial_sharpes(trials):
    """Accept a TrialLog (has .sharpes) or a raw list/array of per-period Sharpes."""
    if trials is None:
        return None
    if hasattr(trials, "sharpes"):
        return list(trials.sharpes)
    return list(np.asarray(trials, dtype=float))


def certify(returns, trials=None, *, prices=None, positions=None,
            second_asset_returns=None, returns_matrix=None,
            recent_returns=None, periods_per_year: float = PERIODS_PER_YEAR_1H,
            fee: float = 0.001, slippage: float = 0.0005,
            thresholds: dict | None = None) -> dict:
    """
    Run the full battery on a claimed edge.

    Parameters
    ----------
    returns : per-period NET returns of the claimed edge (the thing on trial).
    trials  : a TrialLog OR a list/array of EVERY tried config's per-period
              Sharpe (incl. the winner). Required for the DSR — omit it and the
              run can never be CERTIFIED.
    prices, positions : enable the permutation test (timing-skill null).
    second_asset_returns : enable the replication check (a second asset).
    returns_matrix : T×N net-returns DataFrame of all configs → enables PBO.
    recent_returns : a recent held-out return slice → enables the sub-period check.
    fee, slippage : costs used by the permutation test's internal P&L.

    Returns
    -------
    dict: {"criteria": {name: {status, passed, detail, ...}}, "overall": bool,
           "status": CERTIFIED|REJECTED|INCOMPLETE, "n_trials": int}
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    C: dict = {}

    def crit(status, passed, detail, **extra):
        return {"status": status, "passed": passed, "detail": detail, **extra}

    # 1. Sharpe CI excludes zero (positive side)
    ci = sharpe_confidence_interval(r, periods_per_year)
    C["sharpe_ci"] = crit("PASS" if ci["ci_low_ann"] > 0 else "FAIL",
                          ci["ci_low_ann"] > 0,
                          f"Sharpe {ci['sharpe_ann']:+.2f} "
                          f"[{ci['ci_low_ann']:+.2f},{ci['ci_high_ann']:+.2f}]",
                          sharpe_ann=ci["sharpe_ann"], ci_low=ci["ci_low_ann"])

    # 2. PSR
    psr = probabilistic_sharpe_ratio(r)
    C["psr"] = crit("PASS" if psr >= thr["psr"] else "FAIL", psr >= thr["psr"],
                    f"PSR {psr:.1%} (bar {thr['psr']:.0%})", psr=psr)

    # 3. DSR — fail-closed without trials, and VACUOUS-closed with too few /
    # zero-spread trials. With n<2 or no Sharpe spread the luck benchmark SR₀
    # collapses to 0 (statistics.expected_max_sharpe) and the DSR silently
    # degenerates to plain PSR-vs-0 — i.e. the anti-data-mining gate switches
    # itself off. Logging only the winner must therefore NOT certify.
    sh = _trial_sharpes(trials)
    if sh is None or len(sh) == 0:
        C["dsr"] = crit("NO_TRIALS", False,
                        "no trial Sharpes supplied — cannot account for "
                        "multiplicity (fail-closed)")
        n_trials = 0
    else:
        d = deflated_sharpe_ratio(r, np.array(sh), periods_per_year)
        n_trials = d["n_trials"]
        if n_trials < 2 or d["var_trial_sr"] <= 0:
            C["dsr"] = crit("INSUFFICIENT_TRIALS", False,
                            f"only {n_trials} trial(s) / zero Sharpe spread → "
                            f"DSR collapses to PSR (luck SR₀ {d['sr0_ann']:+.2f}); "
                            f"log the WHOLE search for honest deflation")
        else:
            C["dsr"] = crit("PASS" if d["dsr"] >= thr["dsr"] else "FAIL",
                            d["dsr"] >= thr["dsr"],
                            f"DSR {d['dsr']:.1%} over {n_trials} trials "
                            f"(luck SR₀ {d['sr0_ann']:+.2f} ann)", dsr=d["dsr"])

    # 4. PBO (CSCV) — needs a T×N matrix with enough rows (T ≥ 2·n_blocks) AND
    # ≥2 configs. A too-short matrix makes PBO uncomputable (NaN); that is a
    # data-sufficiency SKIP, never a verdict-killing FAIL.
    n_blocks = 10
    rm = returns_matrix.dropna() if returns_matrix is not None else None
    if rm is not None and rm.shape[1] >= 2 and rm.shape[0] >= 2 * n_blocks:
        pbo = probability_of_backtest_overfitting(rm, n_blocks=n_blocks)
        if not np.isfinite(pbo["pbo"]):
            C["pbo"] = crit("SKIPPED", None, "PBO not computable (matrix too short)")
        else:
            ok = pbo["pbo"] < thr["pbo"]
            C["pbo"] = crit("PASS" if ok else "FAIL", ok,
                            f"PBO {pbo['pbo']:.0%} (bar <{thr['pbo']:.0%}), "
                            f"mean OOS rank {pbo['mean_oos_rank']:.0%}", pbo=pbo["pbo"])
    else:
        C["pbo"] = crit("SKIPPED", None,
                        f"no returns_matrix, <2 configs, or <{2*n_blocks} rows")

    # 5. Permutation test — needs prices + positions
    if prices is not None and positions is not None:
        pm = permutation_test(prices, positions, fee, slippage,
                              periods_per_year=periods_per_year)
        ok = pm["p_value"] <= thr["perm_p"]
        C["permutation"] = crit("PASS" if ok else "FAIL", ok,
                                f"p {pm['p_value']:.3f} (real Sharpe "
                                f"{pm['real_sharpe_ann']:+.2f} vs null "
                                f"{pm['null_mean']:+.2f})", p_value=pm["p_value"])
    else:
        C["permutation"] = crit("SKIPPED", None, "no prices/positions supplied")

    # 6. Newey-West HAC mean
    nw = newey_west_mean(r)
    ok = nw["p_one_sided"] <= thr["nw_p"]
    C["newey_west"] = crit("PASS" if ok else "FAIL", ok,
                           f"one-sided p {nw['p_one_sided']:.3f} (t {nw['t_stat']:+.2f})",
                           p_one_sided=nw["p_one_sided"])

    # 7. Replication on a second asset
    if second_asset_returns is not None:
        r2 = np.asarray(second_asset_returns, dtype=float)
        r2 = r2[np.isfinite(r2)]
        ci2 = sharpe_confidence_interval(r2, periods_per_year)
        nw2 = newey_west_mean(r2)
        ok = ci2["ci_low_ann"] > 0 and nw2["p_one_sided"] <= thr["nw_p"]
        C["replication"] = crit("PASS" if ok else "FAIL", ok,
                                f"2nd asset Sharpe {ci2['sharpe_ann']:+.2f} "
                                f"[{ci2['ci_low_ann']:+.2f},{ci2['ci_high_ann']:+.2f}], "
                                f"NW p {nw2['p_one_sided']:.3f}")
    else:
        C["replication"] = crit("SKIPPED", None, "no second_asset_returns supplied")

    # 8. Sub-period robustness on a recent held-out window
    if recent_returns is not None:
        rr = np.asarray(recent_returns, dtype=float)
        rr = rr[np.isfinite(rr)]
        ci3 = sharpe_confidence_interval(rr, periods_per_year)
        nw3 = newey_west_mean(rr)
        ok = ci3["sharpe_ann"] > 0 and nw3["p_one_sided"] <= thr["nw_p"]
        C["subperiod"] = crit("PASS" if ok else "FAIL", ok,
                              f"recent Sharpe {ci3['sharpe_ann']:+.2f}, "
                              f"NW p {nw3['p_one_sided']:.3f}")
    else:
        C["subperiod"] = crit("SKIPPED", None, "no recent_returns supplied")

    # ----- overall verdict (strict / fail-closed) -----
    # REJECTED  = a criterion was computed and FAILED (evidence against).
    # INCOMPLETE = no failures, but some required check could not be run
    #              (SKIPPED / NO_TRIALS / INSUFFICIENT_TRIALS) — not certifiable.
    # CERTIFIED  = every required criterion PASSED.
    statuses = [C[k]["status"] for k in _REQUIRED]
    if any(s == "FAIL" for s in statuses):
        status = "REJECTED"
    elif all(s == "PASS" for s in statuses):
        status = "CERTIFIED"
    else:
        status = "INCOMPLETE"

    return {"criteria": C, "overall": status == "CERTIFIED",
            "status": status, "n_trials": n_trials}


def format_certification(report: dict) -> str:
    """Human-readable rendering of certify()'s output."""
    mark = {"PASS": "✓", "FAIL": "✗", "SKIPPED": "–",
            "NO_TRIALS": "–", "INSUFFICIENT_TRIALS": "–"}
    L = ["=" * 64, f"  Certification: {report['status']}"
         f"  (trials charged: {report['n_trials']})", "=" * 64]
    for name, c in report["criteria"].items():
        L.append(f"  {mark.get(c['status'],'?')} {name:<13}{c['status']:<10} {c['detail']}")
    L.append("=" * 64)
    return "\n".join(L)
