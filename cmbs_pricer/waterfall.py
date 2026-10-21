"""Note structure and priority of payments.

Revenue waterfall (each quarter):
  1. Senior fees (servicing on performing balance + fixed issuer/trustee costs)
  2. Note interest, Class A -> F, each at SONIA + margin; unpaid interest is
     deferred and carried forward (no interest-on-interest)
  3. Excess revenue cures the period's realised losses (paid as principal,
     the UK "principal deficiency ledger" mechanic)
  4. Remainder to Class R (residual / first-loss)

Principal waterfall:
  * "sequential": A until zero, then B, ... then R.
  * "pro_rata": all classes pro rata by balance until cumulative losses breach
    the sequential trigger, then switches permanently to sequential.

Uncured losses write classes down in reverse order (R first).
Prepayment fees go to the classes that receive principal that period, in
proportion to the principal they receive.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cashflows import PoolCashflows
from .curves import FREQ


@dataclass(frozen=True)
class Tranche:
    name: str
    rating: str           # "NR" for the residual
    size_pct: float       # % of pool at closing
    margin_bp: float      # coupon margin over SONIA (0 for residual)


@dataclass(frozen=True)
class Deal:
    tranches: tuple[Tranche, ...] = (
        Tranche("A", "AAA", 0.600, 95),
        Tranche("B", "AA", 0.090, 170),
        Tranche("C", "A", 0.090, 235),
        Tranche("D", "BBB", 0.080, 340),
        Tranche("E", "BB", 0.070, 550),
        Tranche("F", "B", 0.040, 750),
        Tranche("R", "NR", 0.030, 0),
    )
    servicing_fee_bp: float = 12.5
    fixed_costs_pa: float = 250_000.0
    issuer_swap: bool = True             # swap fixed-rate loans to SONIA
    principal_mode: str = "sequential"   # or "pro_rata"
    seq_trigger_cum_loss: float = 0.02   # pro rata -> sequential switch


@dataclass
class WaterfallResult:
    tranches: tuple[Tranche, ...]
    begin_balance: np.ndarray     # tranches x T
    interest_due: np.ndarray
    interest_paid: np.ndarray
    interest_shortfall: np.ndarray # cumulative deferred interest at period end
    principal: np.ndarray
    writedown: np.ndarray
    prepay_fees: np.ndarray
    residual_cash: np.ndarray     # excess revenue to R (T)
    fees: np.ndarray              # senior fees (T)
    loss_cured: np.ndarray        # (T)

    sequential_flag: np.ndarray     # (T) bool

    def cashflows(self) -> np.ndarray:
        """Total cash to each class (tranches x T)."""
        cf = self.interest_paid + self.principal + self.prepay_fees
        cf[-1] += self.residual_cash
        return cf

    def tranche_table(self, idx: int) -> pd.DataFrame:
        cf = self.cashflows()
        return pd.DataFrame({
            "period": np.arange(1, cf.shape[1] + 1),
            "begin_balance": self.begin_balance[idx],
            "interest_due": self.interest_due[idx],
            "interest_paid": self.interest_paid[idx],
            "deferred_interest": self.interest_shortfall[idx],
            "principal": self.principal[idx],
            "writedown": self.writedown[idx],
            "prepay_fees": self.prepay_fees[idx],
            "total_cash": cf[idx],
        })


def run_waterfall(pcf: PoolCashflows, deal: Deal) -> WaterfallResult:
    summ = pcf.summary()
    T = len(summ)
    k = len(deal.tranches)
    pool0 = summ["performing_balance"].iloc[0]

    sizes = np.array([tr.size_pct for tr in deal.tranches])
    bal = pool0 * sizes / sizes.sum()
    margins = np.array([tr.margin_bp for tr in deal.tranches]) / 1e4
    has_coupon = np.array([tr.rating != "NR" for tr in deal.tranches])

    out = {n: np.zeros((k, T)) for n in
           ("begin", "due", "paid", "short", "prin", "wd", "fees")}
    residual = np.zeros(T)
    fees_arr = np.zeros(T)
    cured_arr = np.zeros(T)
    seq_flag = np.zeros(T, dtype=bool)

    shortfall = np.zeros(k)
    cum_loss = 0.0
    sequential = deal.principal_mode == "sequential"

    for t in range(T):
        r = summ.iloc[t]
        out["begin"][:, t] = bal

        # --- Revenue ---------------------------------------------------------
        fees = r.performing_balance * deal.servicing_fee_bp / 1e4 / FREQ + deal.fixed_costs_pa / FREQ
        fees = fees if bal.sum() > 1 else 0.0
        avail = r.interest - fees
        fees_arr[t] = min(fees, r.interest)
        avail = max(avail, 0.0)

        due = np.where(has_coupon, bal * (r.sonia_fwd + margins) / FREQ, 0.0) + shortfall
        paid = np.zeros(k)
        for j in range(k):
            paid[j] = min(avail, due[j])
            avail -= paid[j]
        shortfall = due - paid
        out["due"][:, t] = due
        out["paid"][:, t] = paid
        out["short"][:, t] = shortfall

        cure = min(avail, r.loss)
        avail -= cure
        cured_arr[t] = cure
        residual[t] = avail

        # --- Principal ---------------------------------------------------------

        cum_loss += r.loss
        if not sequential and cum_loss / pool0 > deal.seq_trigger_cum_loss:
            sequential = True
        seq_flag[t] = sequential

        prin_avail = r.principal + cure
        prin = np.zeros(k)
        if sequential:
            for j in range(k):
                prin[j] = min(prin_avail, bal[j])
                prin_avail -= prin[j]
        else:
            share = bal / bal.sum() if bal.sum() > 0 else np.zeros(k)
            prin = np.minimum(prin_avail * share, bal)
        bal = bal - prin

        # --- Losses: reverse sequential write-down ----------------------------
        loss = r.loss - cure
        wd = np.zeros(k)
        for j in reversed(range(k)):
            wd[j] = min(loss, bal[j])
            loss -= wd[j]
        bal = bal - wd

        # --- Prepayment fees follow principal ---------------------------------
        if prin.sum() > 0:
            out["fees"][:, t] = r.prepay_fees * prin / prin.sum()
        else:
            residual[t] += r.prepay_fees

        out["prin"][:, t] = prin
        out["wd"][:, t] = wd

    return WaterfallResult(
        tranches=deal.tranches,
        begin_balance=out["begin"],
        interest_due=out["due"],
        interest_paid=out["paid"],
        interest_shortfall=out["short"],
        principal=out["prin"],
        writedown=out["wd"],
        prepay_fees=out["fees"],
        residual_cash=residual,
        fees=fees_arr,
        loss_cured=cured_arr,
        sequential_flag=seq_flag,
    )
