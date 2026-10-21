"""Loan-level cash flow projection (quarterly, deterministic).

Each loan is projected as an "expected" balance: term defaults (CDR) and
voluntary prepayments (CPR) remove fractions of the balance each quarter.
Balloon (refinancing) risk is binary per loan: at maturity a loan refinances
only if its stressed LTV is at or below the refinance LTV threshold;
otherwise it goes into workout and is liquidated after the recovery lag.

No servicer advancing: defaulted loans stop paying interest immediately.

Fixed-rate loans can be hedged with an issuer-level swap (on by default):
the issuer pays the fixed coupon and receives SONIA + the loan's fixed margin
on the performing balance, so all collateral interest is SONIA-linked like
the notes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .curves import FREQ, GILT, SONIA, ZeroCurve
from .loans import SPENS_MARGIN, protection_status


@dataclass(frozen=True)
class Scenario:
    cpr: float = 0.10                 # annual CPR when a loan is open (no fee)
    protected_cpr_factor: float = 0.15 # share of CPR still prepaying through fees / make-whole
    cdr: float = 0.005                # annual term default rate
    value_decline: float = 0.0        # property value haircut applied from day one
    refi_ltv: float = 0.70            # max LTV at which a balloon can refinance
    liquidation_cost: float = 0.08    # costs as % of stressed value
    recovery_lag_q: int = 6           # quarters from default to liquidation
    rate_shift_bp: float = 0.0        # parallel shift to SONIA and gilt curves


@dataclass
class PoolCashflows:
    periods: np.ndarray       # 1..T
    begin_balance: np.ndarray # loans x T, performing balance at start of period
    interest: np.ndarray
    sched_principal: np.ndarray
    prepayment: np.ndarray
    balloon: np.ndarray
    default: np.ndarray       # balance moved to workout (term + balloon defaults)
    recovery: np.ndarray      # principal recovered (lagged)
    loss: np.ndarray          # realised loss (lagged)

    penalty: np.ndarray       # prepayment fees / make-whole received
    swap_net: np.ndarray      # issuer swap receipts (+) / payments (-), included in interest
    sonia_fwd: np.ndarray     # T
    state: np.ndarray         # loans x T protection state codes

    def summary(self) -> pd.DataFrame:
        s = lambda a: a.sum(axis=0)
        df = pd.DataFrame({
            "period": self.periods,
            "performing_balance": s(self.begin_balance),
            "interest": s(self.interest),
            "scheduled_principal": s(self.sched_principal),
            "prepayment": s(self.prepayment),
            "balloon": s(self.balloon),
            "default": s(self.default),
            "recovery": s(self.recovery),
            "loss": s(self.loss),
            "prepay_fees": s(self.penalty),
            "swap_net": s(self.swap_net),
            "sonia_fwd": self.sonia_fwd,
        })
        df["principal"] = df[["scheduled_principal", "prepayment", "balloon", "recovery"]].sum(axis=1)
        return df


def _make_whole_factors(pool: pd.DataFrame, n: int, sonia: ZeroCurve, gilt: ZeroCurve) -> np.ndarray:
    """Make-whole fee per GBP 1 prepaid at the end of each period (loans x T)."""
    out = np.zeros((len(pool), n))
    spens_curve = gilt.shifted(SPENS_MARGIN * 1e4)
    for i, row in enumerate(pool.itertuples()):
        if row.protection not in ("Spens", "Yield Maintenance"):
            continue
        curve = spens_curve if row.protection == "Spens" else sonia
        rem = int(row.rem_term_q)
        df_all = curve.df(np.arange(rem + 1) / FREQ)
        c = row.fixed_rate / FREQ
        for t in range(min(rem - 1, n)):
            # prepaid at end of period t+1; remaining coupons t+2..rem plus par at rem
            d = df_all[t + 2: rem + 1] / df_all[t + 1]
            out[i, t] = max(0.0, c * d.sum() + d[-1] - 1.0)
    return out


def project_pool(pool: pd.DataFrame, scenario: Scenario,
                 sonia: ZeroCurve = SONIA, gilt: ZeroCurve = GILT,
                 hedge_fixed: bool = True) -> PoolCashflows:
    sonia = sonia.shifted(scenario.rate_shift_bp)
    gilt = gilt.shifted(scenario.rate_shift_bp)

    n_loans = len(pool)
    rem = pool["rem_term_q"].to_numpy()
    T = int(rem.max()) + scenario.recovery_lag_q + 1
    fwd = np.maximum(sonia.forwards(T), 0.0)
    state = protection_status(pool, T)
    mw = _make_whole_factors(pool, T, sonia, gilt)

    is_fixed = (pool["rate_type"] == "Fixed").to_numpy()
    fixed_rate = pool["fixed_rate"].fillna(0).to_numpy()
    margin = pool["margin_bp"].fillna(0).to_numpy() / 1e4
    fixed_margin = pool["fixed_margin_bp"].fillna(0).to_numpy() / 1e4
    amort = pool["amort_pct"].to_numpy()
    sd2_end = pool["stepdown_2pct_end_q"].to_numpy()
    stressed_value = pool["property_value"].to_numpy() * (1 - scenario.value_decline)

    mdr = 1 - (1 - scenario.cdr) ** (1 / FREQ)
    smm_open = 1 - (1 - scenario.cpr) ** (1 / FREQ)
    smm_prot = 1 - (1 - scenario.cpr * scenario.protected_cpr_factor) ** (1 / FREQ)
    smm_by_state = np.array([0.0, smm_prot, 0.0, smm_prot, smm_open, 0.0])

    shape = (n_loans, T)
    arr = {k: np.zeros(shape) for k in
           ("begin", "interest", "swap", "sched", "prepay", "balloon", "default", "recovery", "loss",

"penalty")}

    bal = pool["balance"].to_numpy(dtype=float).copy()
    contractual = bal.copy() # balance of a surviving loan (amortisation only)
    lag = scenario.recovery_lag_q

    for t in range(T):
        alive = (t < rem) & (bal > 0)
        is_mat = t == rem - 1
        arr["begin"][:, t] = bal

        # Recovery rate per surviving loan, used for any default this period.
        rec_rate = np.where(contractual > 0,
                            np.minimum(1.0, stressed_value * (1 - scenario.liquidation_cost)
                                       / np.maximum(contractual, 1.0)), 1.0)

        dflt = np.where(alive, bal * mdr, 0.0)
        b = bal - dflt
        rate = np.where(is_fixed, fixed_rate, fwd[t] + margin)
        interest = np.where(alive, b * rate / FREQ, 0.0)
        if hedge_fixed:
            hedged = np.where(is_fixed, fwd[t] + fixed_margin, rate)
            arr["swap"][:, t] = np.where(alive, b * (hedged - rate) / FREQ, 0.0)
            interest = interest + arr["swap"][:, t]
        sched = np.where(alive & ~is_mat, np.minimum(b, b * amort / FREQ), 0.0)
        contractual = np.where(alive & ~is_mat, contractual * (1 - amort / FREQ), contractual)
        b = b - sched

        smm = np.where(alive & ~is_mat, smm_by_state[state[:, t]], 0.0)
        prepay = b * smm
        fee = np.select(
            [state[:, t] == 1, state[:, t] == 3],
            [mw[:, t], np.where(t + 1 <= sd2_end, 0.02, 0.01)],
            0.0,
        )
        arr["penalty"][:, t] = prepay * fee
        b = b - prepay

        # Balloon: refinance if stressed LTV passes, else workout.
        refi_ok = contractual / np.maximum(stressed_value, 1.0) <= scenario.refi_ltv
        balloon = np.where(is_mat & alive & refi_ok, b, 0.0)
        balloon_dflt = np.where(is_mat & alive & ~refi_ok, b, 0.0)
        b = b - balloon - balloon_dflt

        total_dflt = dflt + balloon_dflt
        if t + lag < T:
            arr["recovery"][:, t + lag] += total_dflt * rec_rate
            arr["loss"][:, t + lag] += total_dflt * (1 - rec_rate)

        arr["interest"][:, t] = interest
        arr["sched"][:, t] = sched
        arr["prepay"][:, t] = prepay
        arr["balloon"][:, t] = balloon
        arr["default"][:, t] = total_dflt
        bal = np.where(alive, b, bal)

    return PoolCashflows(
        periods=np.arange(1, T + 1),
        begin_balance=arr["begin"],
        interest=arr["interest"],
        sched_principal=arr["sched"],
        prepayment=arr["prepay"],
        balloon=arr["balloon"],
        default=arr["default"],
        recovery=arr["recovery"],
        loss=arr["loss"],
        penalty=arr["penalty"],
        swap_net=arr["swap"],
        sonia_fwd=fwd,
        state=state,
    )
