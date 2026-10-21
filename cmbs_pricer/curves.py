"""Rate and spread curves.

All curves are illustrative (dummy) and quoted as annual rates with quarterly
compounding, which matches the quarterly payment dates used throughout UK CMBS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FREQ = 4   # quarterly


@dataclass(frozen=True)
class ZeroCurve:
    """Zero-rate curve, linearly interpolated, flat extrapolated."""

    name: str
    tenors: tuple[float, ...]   # years
    rates: tuple[float, ...]    # decimal, quarterly compounded

    def zero(self, t) -> np.ndarray:
        return np.interp(np.asarray(t, dtype=float), self.tenors, self.rates)

    def df(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return (1.0 + self.zero(t) / FREQ) ** (-FREQ * t)

    def forwards(self, n_periods: int) -> np.ndarray:
        """Simple quarterly forward rates for periods 1..n (annualised)."""
        t = np.arange(n_periods + 1) / FREQ
        d = self.df(t)
        return (d[:-1] / d[1:] - 1.0) * FREQ

    def swap_rate(self, years: float) -> float:
        """Par rate of a quarterly fixed-for-floating swap of the given tenor."""
        n = max(1, int(round(years * FREQ)))
        d = self.df(np.arange(1, n + 1) / FREQ)
        return float((1.0 - d[-1]) / (d.sum() / FREQ))

    def shifted(self, bp: float) -> "ZeroCurve":
        return ZeroCurve(self.name, self.tenors, tuple(r + bp / 1e4 for r in self.rates))


# Illustrative SONIA OIS curve (inverted at the front, upward-sloping beyond 3y).
SONIA = ZeroCurve(
    "SONIA",
    (0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20),
    (0.0375, 0.0368, 0.0355, 0.0345, 0.0345, 0.0355, 0.0370, 0.0390, 0.0410, 0.0420),
)

# Illustrative nominal gilt curve (gilts trading cheap to SONIA swaps).
GILT = ZeroCurve(
    "Gilt",
    (0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20),

    (0.0380, 0.0375, 0.0365, 0.0362, 0.0368, 0.0385, 0.0405, 0.0430, 0.0460, 0.0475),
)


@dataclass(frozen=True)
class SpreadCurve:
    """Discount-margin curve by rating and WAL (bp over SONIA)."""

    tenors: tuple[float, ...]
    spreads_bp: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def spread_bp(self, rating: str, wal: float) -> float:
        return float(np.interp(wal, self.tenors, self.spreads_bp[rating]))

    def shifted(self, bp: float) -> "SpreadCurve":
        return SpreadCurve(
            self.tenors, {k: tuple(s + bp for s in v) for k, v in self.spreads_bp.items()}
        )


# Illustrative UK/European CMBS secondary spread levels.
CMBS_SPREADS = SpreadCurve(
    tenors=(1, 3, 5, 7, 10),
    spreads_bp={
        "AAA": (85, 100, 115, 130, 145),
        "AA": (140, 160, 180, 200, 220),
        "A": (190, 215, 240, 265, 290),
        "BBB": (280, 310, 340, 370, 400),
        "BB": (450, 490, 530, 570, 610),
        "B": (650, 700, 750, 800, 850),
    },
)
