
## QUANTO CONVEXITY (Björk ch.4/13/26) — linear vs inverse perp funding spread

Identity: fair (f_lin − f_inv) = σ²Δt (change of numeraire, quanto corr = 1). Pair price legs cancel exactly (tested) → spread is the observable. Ledger before: 222.

## (I) Identity test — does the market price the quanto adjustment?

```
     prints pinned@1bp spread bp/d σ²/365 bp/d   H0A p   H0B p
BTC    6450        43%       0.485        9.04   0.000   0.000
ETH    6427        42%       0.851       16.32   0.000   0.000
```
- BTC: observed spread +1.77%/yr vs quanto theory +33.0%/yr → shortfall -31.6%/yr (NW p=8.8e-102).
- ETH: observed spread +3.10%/yr vs quanto theory +59.6%/yr → shortfall -57.3%/yr (NW p=3.0e-73).
- READING: H0B rejected ⇒ the funding channel does NOT carry the quanto premium (both funding formulas anchor the SAME USD index, so they are mechanically pinned together); if priced anywhere it must sit in the mark-price basis, for which no public history exists (documented limitation).

## (II) Trading — pair carry rules (frozen grid, 8 trials)

```
trial                 ann.ret  Sharpe  95% CI          MaxDD   NW p flips/yr
ETH|always|theoryside    +3.1%   +4.22 [+3.46,+4.98]   -1.6%  0.000      0.0
BTC|always|theoryside    +1.7%   +3.26 [+2.60,+3.92]   -1.2%  0.000      0.0
ETH|receive|sm20        -1.5%   -1.08 [-1.83,-0.34]  -21.2%  0.954     16.8
BTC|receive|sm20        -3.0%   -2.21 [-2.87,-1.56]  -20.8%  1.000     17.6
BTC|theory|db1e-4       -1.4%   -2.98 [-3.60,-2.35]   -9.1%  1.000      0.4
BTC|theory|db0          -1.5%   -2.99 [-3.59,-2.40]   -9.3%  1.000      0.8
ETH|receive|sm5         -6.3%   -3.28 [-3.96,-2.59]  -37.6%  1.000     36.6
ETH|theory|db0          -2.1%   -3.80 [-4.48,-3.13]  -12.6%  1.000      0.4
ETH|theory|db1e-4       -2.2%   -4.00 [-4.59,-3.41]  -12.8%  1.000      0.4
BTC|receive|sm5         -8.4%   -4.43 [-5.04,-3.81]  -39.7%  1.000     40.0
```
- BEST BTC BTC|always|theoryside: REJECTED (Sharpe-CI PASS, DSR FAIL vs 232, PBO PASS, repl(ETH twin) PASS, sub-period FAIL).
- DIAGNOSIS: 2021-23 +5.09 vs 2024→now -1.28; 0.0 flips/yr.
- DSR luck benchmark charged against 232 cumulative trials (222 prior + 10 from this study).
