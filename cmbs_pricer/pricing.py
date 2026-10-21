"""Bond maths for the CMBS notes: price, yield, WAL, durations."""
from __future__ import annotations

import numpy as np

from .curves import FREQ


def _bisect(f, lo: float, hi: float, tol: float = 1e-10, max_iter: int = 200) -> float:
    flo = f(lo)
    if np.sign(flo) == np.sign(f(hi)):
        return float("nan")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or hi - lo < tol:
            return mid
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi = mid

    return 0.5 * (lo + hi)


def pv_at_dm(cf: np.ndarray, fwd: np.ndarray, dm: float) -> float:
    """PV discounting at projected SONIA forwards + discount margin."""
    df = np.cumprod(1.0 / (1.0 + (fwd[: len(cf)] + dm) / FREQ))
    return float((cf * df).sum())


def pv_at_yield(cf: np.ndarray, y: float) -> float:
    k = np.arange(1, len(cf) + 1)
    return float((cf / (1.0 + y / FREQ) ** k).sum())


def dm_from_price(cf, fwd, pv_target) -> float:
    return _bisect(lambda m: pv_at_dm(cf, fwd, m) - pv_target, -0.05, 1.0)


def yield_from_price(cf, pv_target) -> float:
    return _bisect(lambda y: pv_at_yield(cf, y) - pv_target, -0.5, 3.0)


def wal(principal: np.ndarray) -> float:
    tot = principal.sum()
    if tot <= 0:
        return float("nan")
    t = np.arange(1, len(principal) + 1) / FREQ
    return float((t * principal).sum() / tot)


def modified_duration(cf: np.ndarray, y: float) -> float:
    k = np.arange(1, len(cf) + 1)
    pv = cf / (1.0 + y / FREQ) ** k
    if pv.sum() <= 0:
        return float("nan")
    macaulay = (k / FREQ * pv).sum() / pv.sum()
    return float(macaulay / (1.0 + y / FREQ))


def spread_duration(cf, fwd, dm, bump=1e-4) -> float:
    p0 = pv_at_dm(cf, fwd, dm)
    if p0 <= 0:
        return float("nan")
    up, dn = pv_at_dm(cf, fwd, dm + bump), pv_at_dm(cf, fwd, dm - bump)
    return float((dn - up) / (2 * p0 * bump))


def convexity(cf: np.ndarray, y: float, bump=1e-4) -> float:
    p0 = pv_at_yield(cf, y)
    if p0 <= 0:
        return float("nan")
    up, dn = pv_at_yield(cf, y + bump), pv_at_yield(cf, y - bump)
    return float((up + dn - 2 * p0) / (p0 * bump ** 2))
