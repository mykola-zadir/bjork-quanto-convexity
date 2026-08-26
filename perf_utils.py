"""
perf_utils.py
-------------
Minimal self-contained helpers for this standalone study: performance summary,
log formatting, cached-parquet loading, and the frozen prior-trial ledger that
sets the Deflated Sharpe luck benchmark. (In the parent research repo these
live in backtest/study_utils.py, backtest/optimize.py and utils/perp_fetch.py;
they are reproduced here so this repository runs on its own.)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.statistics import sharpe_confidence_interval, newey_west_mean

PPY = 365
ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "results" / "study_log.md"


def log(msg: str = ""):
    print(msg)
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def log_header(title: str):
    log("")
    log(f"## {title}")
    log("")


def perf(r, ppy: int = PPY):
    """Ann. return, Sharpe + Mertens CI, MaxDD, Newey-West p, per-period Sharpe."""
    r = pd.Series(r).dropna()
    if len(r) < 30:
        return None
    ci = sharpe_confidence_interval(r.values, ppy)
    nw = newey_west_mean(r.values)
    eq = (1 + r).cumprod()
    return {"ann_ret": float(r.mean() * ppy), "sharpe": ci["sharpe_ann"],
            "ci_lo": ci["ci_low_ann"], "ci_hi": ci["ci_high_ann"],
            "maxdd": float((eq / eq.cummax() - 1).min()),
            "nw_p": nw["p_one_sided"],
            "pp_sharpe": float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else 0.0,
            "n": len(r)}


def perf_row(key, m, w=30):
    return (f"{key:<{w}}{m['ann_ret']:>+9.1%}{m['sharpe']:>+8.2f}"
            f" [{m['ci_lo']:+.2f},{m['ci_hi']:+.2f}]{m['maxdd']:>+8.1%}{m['nw_p']:>7.3f}")


def _span(p: Path) -> int:
    m = re.search(r"_(\d{8})_(\d{8})\.parquet$", p.name)
    return (pd.Timestamp(m.group(2)) - pd.Timestamp(m.group(1))).days if m else -1


def load_cached_perp(symbol: str, interval: str):
    """Widest-span committed parquet for a symbol/interval (data/ dir)."""
    files = list((ROOT / "data").glob(f"PERP_{symbol.upper()}_{interval}_*.parquet"))
    return pd.read_parquet(max(files, key=_span)) if files else None


def load_prior_trials() -> dict:
    """The frozen cumulative trial ledger: per-period Sharpe of EVERY config
    tried across the parent research program. Feeding all of them to the
    Deflated Sharpe Ratio is what stops this study from mining its own result."""
    return json.loads((ROOT / "prior_trial_sharpes.json").read_text())
