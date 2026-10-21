"""Top-level entry point: pool -> cash flows -> waterfall -> tranche pricing."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import pricing as px
from .cashflows import PoolCashflows, Scenario, project_pool
from .curves import CMBS_SPREADS, GILT, SONIA, SpreadCurve, ZeroCurve
from .waterfall import Deal, WaterfallResult, run_waterfall

EFF_DUR_SHIFT_BP = 25.0

@dataclass
class DealResult:
    pool: pd.DataFrame
    scenario: Scenario
    pool_cf: PoolCashflows
    waterfall: WaterfallResult
    pricing: pd.DataFrame


def _run(pool, deal, scenario, sonia, gilt):
    pcf = project_pool(pool, scenario, sonia, gilt, hedge_fixed=deal.issuer_swap)
    return pcf, run_waterfall(pcf, deal)


def analyse(pool: pd.DataFrame, deal: Deal = Deal(), scenario: Scenario = Scenario(),
            spreads: SpreadCurve = CMBS_SPREADS, residual_yield: float = 0.12,
            sonia: ZeroCurve = SONIA, gilt: ZeroCurve = GILT,
            effective_duration: bool = True) -> DealResult:
    """Project the deal under `scenario` and price each class off `spreads`.

    Rated classes are priced at the discount margin read off the spread curve
    at the class's WAL; the residual is priced to `residual_yield`.
    """
    pcf, wf = _run(pool, deal, scenario, sonia, gilt)
    fwd = pcf.sonia_fwd
    cfs = wf.cashflows()

    shocked = {}
    if effective_duration:
        for sgn in (+1, -1):
            sc = replace(scenario, rate_shift_bp=scenario.rate_shift_bp + sgn * EFF_DUR_SHIFT_BP)
            p, w = _run(pool, deal, sc, sonia, gilt)
            shocked[sgn] = (p.sonia_fwd, w.cashflows())

    sizes = np.array([tr.size_pct for tr in deal.tranches])
    ce = 1.0 - np.cumsum(sizes) / sizes.sum()
    rows = []
    for j, tr in enumerate(deal.tranches):
        cf = cfs[j]
        face = wf.begin_balance[j, 0]
        prin = wf.principal[j]
        w = px.wal(prin)
        paid_q = np.nonzero(prin > 1.0)[0]

        if tr.rating == "NR":
            pv = px.pv_at_yield(cf, residual_yield)
            dm = px.dm_from_price(cf, fwd, pv)
            price_fn = lambda c, f: px.pv_at_yield(c, residual_yield)
        else:
            dm = spreads.spread_bp(tr.rating, w if np.isfinite(w) else 0.0) / 1e4
            pv = px.pv_at_dm(cf, fwd, dm)
            price_fn = lambda c, f, dm=dm: px.pv_at_dm(c, f, dm)

        y = px.yield_from_price(cf, pv)
        eff = np.nan
        if shocked and pv > 0:
            p_up = price_fn(shocked[+1][1][j], shocked[+1][0])
            p_dn = price_fn(shocked[-1][1][j], shocked[-1][0])
            eff = (p_dn - p_up) / (2 * pv * EFF_DUR_SHIFT_BP / 1e4)

        rows.append({
            "Class": tr.name,
            "Rating": tr.rating,
            "Size (GBP m)": face / 1e6,
            "Size %": tr.size_pct / sizes.sum(),
            "CE %": ce[j],
            "Coupon": f"S+{tr.margin_bp:.0f}" if tr.rating != "NR" else "Residual",
            "WAL (y)": w,
            "Window": f"Q{paid_q[0] + 1}-Q{paid_q[-1] + 1}" if len(paid_q) else "-",
            "DM (bp)": dm * 1e4,
            "Price": 100 * pv / face,

             "Yield %": 100 * y,
             "Mod Dur": px.modified_duration(cf, y) if np.isfinite(y) else np.nan,
             "Spread Dur": px.spread_duration(cf, fwd, dm),
             "Eff Dur": eff,
             "Convexity": px.convexity(cf, y) if np.isfinite(y) else np.nan,
             "Loss %": wf.writedown[j].sum() / face,
             "Unpaid Int (GBP m)": wf.interest_shortfall[j, -1] / 1e6,
        })

    return DealResult(pool, scenario, pcf, wf, pd.DataFrame(rows))
