# Quanto Convexity in Crypto Perpetuals — A Björk Identity Test

**Does the market obey a no-arbitrage identity you can derive from Itô's lemma?**

Binance lists two perpetual futures on the *same* BTC/USD index. One is margined in
USDT (**linear**); the other is margined and settled **in BTC itself** (**inverse**).
Because the inverse contract pays out in the asset it tracks, it is a **quanto** — and
change-of-numeraire (Björk, *Arbitrage Theory in Continuous Time*, ch. 4 / 13 / 26)
forces a specific relationship between the two contracts' funding rates:

```
fair funding spread (linear − inverse)  =  σ² · Δt
```

This is not a pattern discovered in data. It is a mathematical consequence of `1/S`
being convex — Itô's lemma contributes a mandatory `+σ² dt` drift term. So the market
either obeys it or it doesn't, and that is a measurable, falsifiable question.

**This repository measures it, and the market does not obey it.**

---

## Result

| | Observed spread | Theory (σ²Δt) | Shortfall | Newey–West p |
|---|---|---|---|---|
| **BTC** | +1.77 %/yr | +33.0 %/yr | −31.6 %/yr | 8.8 × 10⁻¹⁰² |
| **ETH** | +3.10 %/yr | +59.6 %/yr | −57.3 %/yr | 3.0 × 10⁻⁷³ |

The market prices the convexity **with the correct sign but at roughly one-twentieth
of the required magnitude.**

**Why** — and this is the interesting part — the mechanism is structural, not a market
inefficiency waiting to be traded. Both contracts' funding formulas anchor to the *same*
USD index, and ~43 % of inverse-perp prints sit clamped at the 1 bp base rate. The funding
channel is *mechanically incapable* of expressing a 33 %/yr differential. If the convexity
premium is priced anywhere, it must live in the mark-price basis — for which no public
history exists. That limitation is documented, not hidden.

### Is the residual tradeable? It was, and then it wasn't.

The pair (short linear + long inverse) isolates the spread exactly — see below. Holding
the theory-predicted side permanently, with zero parameters and zero turnover:

| | Full sample | 2021–2023 | 2024 → now |
|---|---|---|---|
| **BTC** | Sharpe **+3.26**, 95 % CI [+2.60, +3.92] | **+5.09** | **−1.28** |
| **ETH** | Sharpe **+4.22** — independent replication ✓ | strong | dead |

**Verdict: REJECTED.** The Sharpe confidence interval excludes zero and the effect
replicates on a second asset — but it fails the out-of-sample sub-period gate and the
Deflated Sharpe Ratio. The residual convexity compensation existed early and was
competed away as the market matured.

A deliberately conservative note on those Sharpe numbers: the modelled volatility is tiny
(max drawdown ≈ 1 %) because inter-perp mark-premium wobble is unmodelled. Treat the
ratio as an artifact of a smooth carry series, not a deployable expectation.

---

## The key derivation: why the funding spread *is* the P&L

The strategy's validity rests on one claim, so it is enforced as **test [0]** of the suite
rather than asserted in prose:

> With both legs marked and converted to USD at the same close, the price legs of
> {short \$1 linear, long \$1 inverse} cancel **exactly** — to floating-point precision.
> What remains is purely the funding spread minus costs.

```
r = +5 %  → linear +0.050000,  inverse (converted @ S₁) +0.050000,  difference 6.9e-17
r = −5 %  → linear −0.050000,  inverse (converted @ S₁) −0.050000,  difference −4.2e-17
r = +20 % → linear +0.200000,  inverse (converted @ S₁) +0.200000,  difference −2.8e-17
```

The `r²` convexity term appears only under *entry-price* conversion — which is precisely
the numeraire subtlety the theory is about. The pair is therefore a scalpel: it isolates
exactly the quantity the theorem prices.

**A second lesson, learned the hard way and kept in the record:** the first version traded
the *deviation* between observed and fair spread, expecting convergence. Every variant lost
money. Funding P&L is a **flow**, not a level — you cannot buy a "cheap" spread and profit
from re-rating; you only accrue what you sit on. Convergence logic on a carry instrument is
a category error. That is why the surviving rule has zero parameters.

---

## Methodology

Results are only as good as the discipline behind them:

- **Pre-registered.** The hypotheses, rule grid and pass/fail bars are frozen in the study
  docstring before any P&L was computed.
- **Strict causality.** A side decided at day *t* earns day *t+1*'s spread. The fair-value
  σ̂² uses returns through *t−1* only. Prefix-invariance is unit-tested.
- **Honest costs.** Every side change charges four legs at 7 bp (close two, open two).
- **Multiple-testing control.** The Deflated Sharpe Ratio is charged against **232
  cumulative trials** — every configuration ever tried across the parent research
  programme, not just this study's ten. `prior_trial_sharpes.json` is committed so the
  benchmark is auditable.
- **Full battery.** Mertens skew/kurtosis-aware Sharpe CI, Probabilistic and Deflated
  Sharpe, PBO via CSCV, Newey–West HAC, second-asset replication, out-of-sample sub-period.
- **Adversarially reviewed.** An independent review pass found two real defects — a cache
  selection flaw silently dropping ~140 days of history, and partial days entering the
  statistics as full days. Both are fixed and regression-tested; the verdicts were unchanged.

---

## Reproduce

Data is committed (≈8 MB), so the study runs offline and deterministically.

```bash
pip install -r requirements.txt
python run_tests.py                    # 16 tests, including the derivation itself
python verify_quanto_convexity.py      # → results/quanto_convexity.json + study_log.md
```

## Layout

```
strategy/quanto_convexity.py    the signal + P&L model (the derivation, in code)
verify_quanto_convexity.py      pre-registered study: identity test → trading → certify
backtest/statistics.py          Mertens CI, PSR, DSR, PBO/CSCV, Newey–West
backtest/certify.py             one-call certification gate (fail-closed)
utils/dapi_fetch.py             coin-margined (inverse) funding history
tests/                          16 tests, incl. exact contract arithmetic as test [0]
data/                           committed parquet caches — fully reproducible
```

---

## What I'd conclude

The tradeable answer is negative and the rejection is clean. The *scientific* answer is the
point: a no-arbitrage identity derived from first principles is violated by two orders of
magnitude, with an identifiable structural cause. Testing a theory and reporting that
reality disagrees — with the mechanism named — is a more honest result than another curve
that happens to fit.

Two continuations follow naturally: measure the same identity on smaller, less-arbitraged
coin-margined pairs, and begin archiving both contracts' mark-price premiums live, which is
where the theory says the premium must actually reside.

---

*Part of a broader systematic-research programme:
[crypto-quant-research](https://github.com/7nickyeet7z-ops/crypto-quant-research).*
