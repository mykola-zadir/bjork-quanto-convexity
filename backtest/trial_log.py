"""
backtest/trial_log.py
---------------------
TrialLog — an auto-accumulating registry of EVERY backtested config's Sharpe.

WHY THIS EXISTS. The single most important guard against fooling yourself is
the Deflated Sharpe Ratio fed with the TRUE number of trials you ran (not just
the winner). [Deflated Sharpe Ratio (DSR): the probability the best of N tried
strategies is genuinely skilled rather than the luckiest draw — it deflates a
naive Sharpe by how many configs you searched.] Hand-counting trials is
error-prone (it is easy to forget a sweep, or double-count). A TrialLog makes
the accounting automatic: every config appends its per-period Sharpe as it is
tested, and .deflate() feeds the whole registry to the DSR with the best
trial's own return stream — so honest multiplicity is impossible to forget.

All Sharpes are stored in PER-PERIOD units (mean/std, ddof=1) because that is
exactly what deflated_sharpe_ratio() consumes; annualisation happens only at
the very end for display.
"""

import warnings

import numpy as np
import pandas as pd

from backtest.fast_engine import PERIODS_PER_YEAR_1H
from backtest.statistics import deflated_sharpe_ratio


class TrialLog:
    """Accumulating registry of trial Sharpes for honest DSR multiplicity."""

    def __init__(self, periods_per_year: float = PERIODS_PER_YEAR_1H):
        self.ppy = periods_per_year
        self._records: list[dict] = []     # {label, pp_sharpe, returns|None}

    # -- registration -------------------------------------------------------

    def add(self, returns=None, *, sharpe: float | None = None,
            label: str | None = None) -> float:
        """
        Register one trial. Provide EITHER its per-period `returns` (the Sharpe
        is computed and the stream kept, so this trial can later serve as the
        DSR's best_returns) OR a precomputed PER-PERIOD `sharpe`. Returns the
        per-period Sharpe that was stored.
        """
        if returns is not None:
            r = np.asarray(returns, dtype=float)
            r = r[np.isfinite(r)]
            sd = r.std(ddof=1) if r.size > 1 else 0.0
            pp = float(r.mean() / sd) if sd > 0 else 0.0
            kept = r
        elif sharpe is not None:
            pp, kept = float(sharpe), None
        else:
            raise ValueError("add() needs either `returns` or `sharpe`")
        self._records.append({"label": label or f"trial_{len(self._records)}",
                              "pp_sharpe": pp, "returns": kept})
        return pp

    def add_annualised(self, sharpe_ann: float, label: str | None = None) -> float:
        """
        Convenience: register a trial known only by its ANNUALISED Sharpe (it is
        converted to the per-period units the registry stores). Pass a genuinely
        annualised number here — feeding an already-per-period Sharpe would shrink
        it by √ppy and silently weaken the DSR, so we warn on an implausibly small
        value that looks like a per-period Sharpe passed by mistake.
        """
        if self.ppy > 1 and 0 < abs(sharpe_ann) < 0.05:
            warnings.warn(f"add_annualised got |Sharpe|={abs(sharpe_ann):.3g} — that "
                          f"looks PER-PERIOD, not annualised. Use add(sharpe=...) "
                          f"for per-period Sharpes.", stacklevel=2)
        return self.add(sharpe=sharpe_ann / np.sqrt(self.ppy), label=label)

    # -- views --------------------------------------------------------------

    @property
    def sharpes(self) -> list[float]:
        """Every trial's per-period Sharpe (the array DSR must be fed)."""
        return [r["pp_sharpe"] for r in self._records]

    @property
    def labels(self) -> list[str]:
        return [r["label"] for r in self._records]

    def __len__(self) -> int:
        return len(self._records)

    def best(self) -> dict:
        """
        The GLOBAL highest-Sharpe trial (the one selection would actually keep).
        Note it may lack a stored return stream — deflate() handles that case
        loudly rather than silently substituting a lower-Sharpe streamed trial.
        """
        if not self._records:
            raise ValueError("TrialLog is empty")
        return max(self._records, key=lambda r: r["pp_sharpe"])

    # -- the payoff ---------------------------------------------------------

    def deflate(self, best_returns=None, periods_per_year: float | None = None) -> dict:
        """
        Deflated Sharpe Ratio over ALL registered trials. best_returns defaults
        to the GLOBAL best trial's own stored stream. If that winning trial was
        logged without a return stream we RAISE — deflating a different (lower-
        Sharpe) trial's stream against a benchmark built from the real winner
        would be internally inconsistent. Add the winner with returns=, or pass
        best_returns explicitly.
        """
        ppy = periods_per_year or self.ppy
        if best_returns is None:
            b = self.best()
            if b["returns"] is None:
                raise ValueError(
                    f"highest-Sharpe trial '{b['label']}' was logged without a "
                    f"return stream; add it with returns= or pass best_returns "
                    f"explicitly (deflating a lower trial would be inconsistent).")
            best_returns = b["returns"]
        return deflated_sharpe_ratio(best_returns, np.array(self.sharpes), ppy)

    def summary(self) -> str:
        if not self._records:
            return "TrialLog: (empty)"
        s = np.array(self.sharpes)
        b = max(self._records, key=lambda r: r["pp_sharpe"])
        rt = np.sqrt(self.ppy)
        return (f"TrialLog: {len(self)} trials, "
                f"Sharpe(ann) range [{s.min()*rt:+.2f}, {s.max()*rt:+.2f}], "
                f"best='{b['label']}' ({b['pp_sharpe']*rt:+.2f}).")
