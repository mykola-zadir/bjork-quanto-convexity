"""
backtest/statistics.py
----------------------
Statistical significance & overfitting diagnostics.

This module answers the question every backtest must survive:

        "Is this edge real, or did I data-mine it?"

Four complementary tools, each attacking the question from a different angle:

  1. SHARPE CONFIDENCE INTERVAL — a backtest Sharpe is an ESTIMATE with a
     standard error, not a fact. We report "Sharpe 0.8 ± 0.6" so a number
     indistinguishable from zero is never mistaken for an edge.

  2. PROBABILISTIC SHARPE RATIO (PSR) — P(true Sharpe > benchmark | data),
     correcting for the sample's skewness and fat tails (which inflate the
     reliability of naive Sharpe estimates on crypto).

  3. DEFLATED SHARPE RATIO (DSR) — PSR with the benchmark raised to the
     Sharpe you'd expect the BEST of N random strategies to show. Running a
     36-combo grid search and keeping the best is a multiple-testing exercise;
     DSR quantifies exactly how much of the winner is selection luck.

  4. PBO via CSCV — Probability of Backtest Overfitting through Combinatorially
     Symmetric Cross-Validation: how often does the in-sample-best config fall
     in the bottom half out-of-sample? PBO ≈ 0.5 ⇒ in-sample ranking is noise.

  5. PERMUTATION TEST — destroys the signal/return alignment while preserving
     the position series' exposure, turnover, and autocorrelation (circular
     rotation). If the real Sharpe doesn't beat this matched null, the timing
     contains no information.

References:
  Lo (2002), "The Statistics of Sharpe Ratios"
  Mertens (2002), comment on Lo — SE under non-normality
  Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier" (PSR)
  Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"
  Bailey, Borwein, López de Prado, Zhu (2015), "The Probability of Backtest
      Overfitting" (CSCV)

All Sharpe quantities here are computed and tested in PER-PERIOD units and
annualised only for display — the distributional results hold per-period.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm, skew as _skew, kurtosis as _kurtosis

from backtest.fast_engine import compute_pnl, PERIODS_PER_YEAR_1H

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Moments helper
# ---------------------------------------------------------------------------

def _moments(returns: np.ndarray) -> dict:
    """Mean, std (ddof=1), skewness, and NON-EXCESS kurtosis of a return series."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 4 or r.std(ddof=1) == 0:
        return {"T": T, "mean": 0.0, "std": 0.0, "skew": 0.0, "kurt": 3.0, "sr": 0.0}
    return {
        "T":    T,
        "mean": float(r.mean()),
        "std":  float(r.std(ddof=1)),
        "skew": float(_skew(r)),
        "kurt": float(_kurtosis(r, fisher=False)),  # non-excess: Normal = 3
        "sr":   float(r.mean() / r.std(ddof=1)),    # per-period Sharpe
    }


# ---------------------------------------------------------------------------
# 1. Sharpe standard error & confidence interval
# ---------------------------------------------------------------------------

def sharpe_confidence_interval(
    returns,
    periods_per_year: float = PERIODS_PER_YEAR_1H,
    confidence: float = 0.95,
) -> dict:
    """
    Sharpe ratio with its standard error and confidence interval.

    The estimated per-period Sharpe SR̂ has (Mertens 2002, asymptotic):

        SE(SR̂) = sqrt( (1 - γ₃·SR̂ + (γ₄-1)/4 · SR̂²) / (T-1) )

    where γ₃ is skewness and γ₄ non-excess kurtosis. Under normality this
    collapses to Lo's sqrt((1 + SR̂²/2)/(T-1)); fat tails (γ₄ > 3) and negative
    skew — both typical of crypto — WIDEN the interval, i.e. naive normal-theory
    error bars are too optimistic exactly where it hurts.

    Everything is annualised by sqrt(periods_per_year) for reporting.

    Returns dict: sharpe_ann, se_ann, ci_low_ann, ci_high_ann, significant
    (True iff the CI excludes zero), plus the per-period inputs.
    """
    m = _moments(np.asarray(returns, dtype=float))
    sr, T = m["sr"], m["T"]
    if T < 4 or m["std"] == 0:
        return {"sharpe_ann": 0.0, "se_ann": np.inf,
                "ci_low_ann": -np.inf, "ci_high_ann": np.inf,
                "significant": False, **m}

    var_sr = (1.0 - m["skew"] * sr + (m["kurt"] - 1.0) / 4.0 * sr ** 2) / (T - 1)
    se = float(np.sqrt(max(var_sr, 0.0)))
    z = norm.ppf(0.5 + confidence / 2.0)
    ann = np.sqrt(periods_per_year)

    lo, hi = (sr - z * se) * ann, (sr + z * se) * ann
    return {
        "sharpe_ann":  sr * ann,
        "se_ann":      se * ann,
        "ci_low_ann":  lo,
        "ci_high_ann": hi,
        "confidence":  confidence,
        "significant": bool(lo > 0.0 or hi < 0.0),
        **m,
    }


# ---------------------------------------------------------------------------
# 2. Probabilistic Sharpe Ratio
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """
    PSR = P(true Sharpe > sr_benchmark | observed returns)
        = Φ( (SR̂ - SR*) · sqrt(T-1) / sqrt(1 - γ₃SR̂ + (γ₄-1)/4·SR̂²) )

    sr_benchmark is in PER-PERIOD units. PSR ≥ 0.95 is the usual bar for
    "statistically convincing". Skew/kurtosis enter exactly as in the SE above,
    so a fat-tailed strategy needs a longer track record for the same PSR.
    """
    m = _moments(np.asarray(returns, dtype=float))
    sr, T = m["sr"], m["T"]
    if T < 4 or m["std"] == 0:
        return 0.5
    denom = np.sqrt(max(1.0 - m["skew"] * sr + (m["kurt"] - 1.0) / 4.0 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1) / denom))


# ---------------------------------------------------------------------------
# 3. Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

def expected_max_sharpe(n_trials: int, var_trial_sr: float) -> float:
    """
    E[max of n_trials independent zero-skill Sharpe estimates] (per-period):

        SR₀ = sqrt(V) · [ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)) ]

    where V is the cross-trial variance of the Sharpe estimates and γ is the
    Euler–Mascheroni constant. This is the Sharpe that pure selection luck
    hands you for free when you keep the best of N tries.
    """
    n = max(int(n_trials), 1)
    if n == 1 or var_trial_sr <= 0:
        return 0.0
    sd = np.sqrt(var_trial_sr)
    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sd * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    best_returns,
    trial_sharpes,
    periods_per_year: float = PERIODS_PER_YEAR_1H,
) -> dict:
    """
    DSR (Bailey & López de Prado 2014): the PSR of the BEST strategy, measured
    against the Sharpe that the best of N random trials would have shown.

        DSR = PSR(SR₀)   with   SR₀ = expected max noise Sharpe over N trials

    Parameters
    ----------
    best_returns : the selected (best) config's net per-period returns.
    trial_sharpes : per-period Sharpe of EVERY config tried in the search —
        the honest accounting of the multiple-testing burden. Their variance
        and count feed SR₀.

    Returns dict: dsr, psr_vs_zero, sr0_ann (the luck benchmark, annualised),
    n_trials. DSR ≥ 0.95 ⇒ the edge survives its own selection process.
    """
    trial_sharpes = np.asarray(trial_sharpes, dtype=float)
    trial_sharpes = trial_sharpes[np.isfinite(trial_sharpes)]
    n = len(trial_sharpes)
    var_sr = float(trial_sharpes.var(ddof=1)) if n > 1 else 0.0

    sr0 = expected_max_sharpe(n, var_sr)
    return {
        "dsr":          probabilistic_sharpe_ratio(best_returns, sr_benchmark=sr0),
        "psr_vs_zero":  probabilistic_sharpe_ratio(best_returns, sr_benchmark=0.0),
        "sr0_ann":      sr0 * np.sqrt(periods_per_year),
        "n_trials":     n,
        "var_trial_sr": var_sr,
    }


# ---------------------------------------------------------------------------
# 4. Probability of Backtest Overfitting (CSCV)
# ---------------------------------------------------------------------------

def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame,
    n_blocks: int = 10,
    max_subsets: int = 252,
    seed: int = 42,
) -> dict:
    """
    PBO via Combinatorially Symmetric Cross-Validation (Bailey et al. 2015).

    Input: a (T × N) matrix of net per-period returns, one column per config
    tried in the search. Procedure:

      1. Split the T rows into n_blocks contiguous blocks (preserving the
         time-ordering inside each block).
      2. For every way of choosing half the blocks as "train":
           a. rank configs by Sharpe on the train half, take the winner n*;
           b. find n*'s Sharpe RANK on the held-out half, as a relative rank
              ω ∈ (0,1); logit λ = ln(ω / (1-ω)).
      3. PBO = fraction of subsets with λ ≤ 0 — i.e. how often the in-sample
         champion lands in the BOTTOM half out-of-sample.

    Reading it: PBO ≈ 0.5 ⇒ in-sample selection carries zero information
    (pure overfitting); PBO ≲ 0.2 with positive median OOS rank ⇒ the ranking
    genuinely generalises.

    Implementation note: per-block sums and sums-of-squares are precomputed so
    each of the C(n_blocks, n_blocks/2) subsets is combined in O(N) — the full
    procedure is exact, not sampled, up to max_subsets (then uniformly sampled).
    """
    M = returns_matrix.to_numpy(dtype=float)
    T, N = M.shape
    if N < 2 or T < 2 * n_blocks:
        return {"pbo": np.nan, "logits": np.array([]), "n_subsets": 0,
                "mean_oos_rank": np.nan}

    edges = np.linspace(0, T, n_blocks + 1, dtype=int)
    s1 = np.empty((n_blocks, N))   # per-block sum of returns
    s2 = np.empty((n_blocks, N))   # per-block sum of squared returns
    cnt = np.empty(n_blocks)
    for b in range(n_blocks):
        seg = M[edges[b]:edges[b + 1]]
        s1[b] = seg.sum(axis=0)
        s2[b] = (seg ** 2).sum(axis=0)
        cnt[b] = seg.shape[0]

    def _sharpes(block_idx) -> np.ndarray:
        n_obs = cnt[block_idx].sum()
        mu = s1[block_idx].sum(axis=0) / n_obs
        var = s2[block_idx].sum(axis=0) / n_obs - mu ** 2
        sd = np.sqrt(np.maximum(var, 1e-300))
        return mu / sd

    all_subsets = list(combinations(range(n_blocks), n_blocks // 2))
    if len(all_subsets) > max_subsets:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(all_subsets), size=max_subsets, replace=False)
        all_subsets = [all_subsets[i] for i in sel]

    logits, oos_ranks = [], []
    full = set(range(n_blocks))
    for train in all_subsets:
        test = sorted(full - set(train))
        sr_is = _sharpes(list(train))
        sr_oos = _sharpes(test)
        star = int(np.argmax(sr_is))
        # relative OOS rank of the IS winner, in (0, 1)
        omega = (np.sum(sr_oos < sr_oos[star]) + 0.5) / N
        omega = min(max(omega, 1.0 / (2 * N)), 1.0 - 1.0 / (2 * N))
        logits.append(np.log(omega / (1.0 - omega)))
        oos_ranks.append(omega)

    logits = np.asarray(logits)
    return {
        "pbo":           float((logits <= 0).mean()),
        "logits":        logits,
        "n_subsets":     len(logits),
        "mean_oos_rank": float(np.mean(oos_ranks)),
    }


# ---------------------------------------------------------------------------
# 5. Permutation test (circular-rotation null)
# ---------------------------------------------------------------------------

def permutation_test(
    prices,
    target_positions,
    fee: float = 0.001,
    slippage: float = 0.0005,
    n_permutations: int = 1000,
    periods_per_year: float = PERIODS_PER_YEAR_1H,
    seed: int = 42,
) -> dict:
    """
    Does the TIMING of the positions carry information, beyond what their
    exposure and turnover would earn by luck?

    Null model: circularly rotate the target-position series by a random
    offset against the (unchanged) price series. Rotation preserves the
    position series exactly — same exposure, same number of trades, same
    turnover and therefore identical total costs, same autocorrelation — and
    destroys only the alignment with returns. That makes it a STRICTLY
    matched null: any Sharpe difference is attributable to timing skill.

    p_value = (1 + #{null Sharpe ≥ real Sharpe}) / (1 + n_permutations)

    p ≤ 0.05 ⇒ the timing beats ≥95% of luck-matched imposters.
    """
    if isinstance(prices, pd.Series):
        if isinstance(target_positions, pd.Series):
            target_positions = target_positions.reindex(prices.index).fillna(0.0)
        prices = prices.to_numpy(dtype=float)
    px = np.asarray(prices, dtype=float)
    tp = np.asarray(target_positions, dtype=float)
    T = len(px)

    def _net_sharpe(positions: np.ndarray) -> float:
        out = compute_pnl(px, positions, fee, slippage)
        net = out["net_returns"]
        sd = net.std(ddof=1)
        return float(net.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0

    real = _net_sharpe(tp)

    rng = np.random.default_rng(seed)
    # offsets far enough from 0 / T that the null isn't the real signal
    min_off = max(int(0.05 * T), 1)
    offsets = rng.integers(min_off, T - min_off, size=n_permutations)
    null = np.array([_net_sharpe(np.roll(tp, k)) for k in offsets])

    p = float((1 + np.sum(null >= real)) / (1 + n_permutations))
    return {
        "real_sharpe_ann": real,
        "p_value":         p,
        "null_mean":       float(null.mean()),
        "null_std":        float(null.std(ddof=1)),
        "null_q95":        float(np.quantile(null, 0.95)),
        "n_permutations":  n_permutations,
        "significant_5pct": bool(p <= 0.05),
    }


# ---------------------------------------------------------------------------
# 6. Newey-West (HAC) mean test — for OVERLAPPING-window series
# ---------------------------------------------------------------------------

def newey_west_mean(x, max_lag: int | None = None, overlap: int | None = None) -> dict:
    """
    Test whether the mean of an AUTOCORRELATED series differs from zero,
    with heteroskedasticity-and-autocorrelation-consistent (HAC) errors.

    WHY THIS EXISTS. Daily observations of a 30-day forward quantity (like
    the variance risk premium VRP_t = IV²_t − RV_{t→t+30d}) overlap by 29
    days, so consecutive observations share almost all their information.
    The effective sample is ~T/30, not T — a naive t-stat treats every day
    as independent and is overconfident by roughly √30 ≈ 5.5×. Overlapping
    h-period windows induce an MA(h−1) error structure (Hansen & Hodrick
    1980), so the long-run variance of the mean must sum the autocovariances:

        Var(x̄) = [ γ₀ + 2 Σ_{j=1..L} w_j γ_j ] / T,   w_j = 1 − j/(L+1)

    with Bartlett-kernel weights w_j guaranteeing positive semi-definiteness
    (Newey & West 1987). L must be ≥ h−1 to capture the overlap; we default
    to L = ⌈1.5·h⌉ when `overlap` is given, else the Newey-West automatic
    plug-in ⌊4(T/100)^{2/9}⌋.

    Returns dict: mean, se_hac, se_naive, t_stat, p_two_sided, p_one_sided
    (H1: mean > 0), n, max_lag. The se_naive/se_hac ratio shows how badly a
    naive test would have lied.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 8:
        return {"mean": np.nan, "se_hac": np.nan, "se_naive": np.nan,
                "t_stat": np.nan, "p_two_sided": np.nan, "p_one_sided": np.nan,
                "n": T, "max_lag": 0}

    if max_lag is None:
        max_lag = (int(np.ceil(1.5 * overlap)) if overlap
                   else int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))))
    max_lag = min(max_lag, T - 2)

    d = x - x.mean()
    gamma0 = float(d @ d) / T
    lrv = gamma0
    for j in range(1, max_lag + 1):
        gamma_j = float(d[j:] @ d[:-j]) / T
        lrv += 2.0 * (1.0 - j / (max_lag + 1.0)) * gamma_j
    lrv = max(lrv, 1e-300)

    se_hac = float(np.sqrt(lrv / T))
    se_naive = float(d.std(ddof=1) / np.sqrt(T))
    t = float(x.mean() / se_hac)
    return {
        "mean":        float(x.mean()),
        "se_hac":      se_hac,
        "se_naive":    se_naive,
        "t_stat":      t,
        "p_two_sided": float(2.0 * norm.sf(abs(t))),
        "p_one_sided": float(norm.sf(t)),       # H1: mean > 0
        "n":           T,
        "max_lag":     int(max_lag),
    }


# ---------------------------------------------------------------------------
# 7. Newey-West OLS — HAC standard errors for regression coefficients
# ---------------------------------------------------------------------------

def newey_west_ols(y, X, max_lag: int | None = None) -> dict:
    """
    OLS with Newey-West (HAC) standard errors — for testing whether a
    regressor carries information when the residuals are autocorrelated
    (e.g. anything involving volatility, where clustering is the norm).

        β̂ = (X'X)⁻¹X'y
        V̂ = (X'X)⁻¹ S (X'X)⁻¹,   S = Σ_t u_t u_t' + Σ_j w_j Σ_t (u_t u_{t-j}' + u_{t-j} u_t')

    with u_t = x_t·e_t (the score), Bartlett weights w_j = 1 − j/(L+1)
    (Newey & West 1987 — guarantees V̂ is positive semi-definite). Classical
    OLS SEs assume the scores are serially uncorrelated; vol clustering
    violates that and shrinks them spuriously.

    Parameters: y (T,), X (T,k) including the constant column.
    Returns dict: beta, se (HAC), se_ols (classical, for comparison),
    t_stat, p_two_sided, p_one_sided (H1: β > 0), n, max_lag.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[keep], X[keep]
    T, k = X.shape
    if max_lag is None:
        max_lag = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    max_lag = min(max_lag, T - 2)

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    u = X * e[:, None]                          # scores u_t = x_t e_t

    S = u.T @ u
    for j in range(1, max_lag + 1):
        gamma = u[j:].T @ u[:-j]
        S += (1.0 - j / (max_lag + 1.0)) * (gamma + gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    sigma2 = float(e @ e) / max(T - k, 1)
    se_ols = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    return {
        "beta": beta, "se": se, "se_ols": se_ols, "t_stat": t,
        "p_two_sided": 2.0 * norm.sf(np.abs(t)),
        "p_one_sided": norm.sf(t),              # H1: β > 0, per coefficient
        "n": T, "max_lag": int(max_lag),
    }


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------

def significance_report(
    best_returns,
    trial_sharpes,
    returns_matrix: pd.DataFrame | None = None,
    prices=None,
    target_positions=None,
    fee: float = 0.001,
    slippage: float = 0.0005,
    periods_per_year: float = PERIODS_PER_YEAR_1H,
) -> dict:
    """
    Run the full battery on a search result and return one dict. Sections are
    skipped gracefully when their inputs are not supplied.
    """
    rep: dict = {}
    rep["sharpe_ci"] = sharpe_confidence_interval(best_returns, periods_per_year)
    rep["dsr"] = deflated_sharpe_ratio(best_returns, trial_sharpes, periods_per_year)
    if returns_matrix is not None:
        rep["pbo"] = probability_of_backtest_overfitting(returns_matrix)
    if prices is not None and target_positions is not None:
        rep["permutation"] = permutation_test(
            prices, target_positions, fee, slippage,
            periods_per_year=periods_per_year)
    return rep


def format_significance_report(rep: dict) -> str:
    """Human-readable rendering of significance_report() output."""
    L = ["=" * 64, "  Statistical Significance Report", "=" * 64]

    ci = rep.get("sharpe_ci")
    if ci:
        L += [
            "  1. Sharpe with sampling error",
            f"     Sharpe (ann.)        : {ci['sharpe_ann']:+.2f} ± {ci['se_ann']:.2f}",
            f"     {ci['confidence']:.0%} CI             : "
            f"[{ci['ci_low_ann']:+.2f}, {ci['ci_high_ann']:+.2f}]",
            f"     skew / kurtosis      : {ci['skew']:+.2f} / {ci['kurt']:.1f}"
            f"   (Normal: 0 / 3 — fat tails widen the CI)",
            f"     CI excludes zero     : {'YES' if ci['significant'] else 'NO'}",
            "",
        ]
    d = rep.get("dsr")
    if d:
        L += [
            "  2. Multiple-testing (selection luck)",
            f"     configs tried (N)    : {d['n_trials']}",
            f"     luck benchmark SR₀   : {d['sr0_ann']:+.2f} (ann.) — what the best",
            "                            of N zero-skill trials would show",
            f"     PSR vs 0             : {d['psr_vs_zero']:.1%}",
            f"     Deflated Sharpe      : {d['dsr']:.1%}   (≥95% = survives selection)",
            "",
        ]
    p = rep.get("pbo")
    if p and np.isfinite(p.get("pbo", np.nan)):
        L += [
            "  3. Probability of Backtest Overfitting (CSCV)",
            f"     PBO                  : {p['pbo']:.1%}   (~50% = IS ranking is noise)",
            f"     mean OOS rank of IS winner : {p['mean_oos_rank']:.1%}"
            f"   (>50% = generalises)",
            f"     subsets evaluated    : {p['n_subsets']}",
            "",
        ]
    pm = rep.get("permutation")
    if pm:
        L += [
            "  4. Permutation test (timing skill, luck-matched null)",
            f"     real Sharpe          : {pm['real_sharpe_ann']:+.2f}",
            f"     null Sharpe          : {pm['null_mean']:+.2f} ± {pm['null_std']:.2f}"
            f"  (95th pct {pm['null_q95']:+.2f})",
            f"     p-value              : {pm['p_value']:.3f}"
            f"   ({'SIGNIFICANT' if pm['significant_5pct'] else 'not significant'} at 5%)",
            "",
        ]
    L.append("=" * 64)
    return "\n".join(L)
