"""CLI: python -m cmbs_pricer [--cdr 0.01 --value-decline 0.2 ...]"""
import argparse

import pandas as pd

from . import Deal, Scenario, analyse, generate_loan_pool


def main() -> None:
    p = argparse.ArgumentParser(description="Price a dummy 75-loan UK CMBS deal.")
    p.add_argument("--loans", type=int, default=75)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cpr", type=float, default=Scenario.cpr)
    p.add_argument("--cdr", type=float, default=Scenario.cdr)
    p.add_argument("--value-decline", type=float, default=Scenario.value_decline)
    p.add_argument("--refi-ltv", type=float, default=Scenario.refi_ltv)
    p.add_argument("--rate-shift-bp", type=float, default=Scenario.rate_shift_bp)
    p.add_argument("--pro-rata", action="store_true", help="pro rata principal with sequential trigger")
    p.add_argument("--export-tape", metavar="CSV", help="write the loan tape to CSV")
    a = p.parse_args()

    pool = generate_loan_pool(a.loans, a.seed)
    if a.export_tape:
        pool.to_csv(a.export_tape, index=False)
    sc = Scenario(cpr=a.cpr, cdr=a.cdr, value_decline=a.value_decline,
                  refi_ltv=a.refi_ltv, rate_shift_bp=a.rate_shift_bp)
    deal = Deal(principal_mode="pro_rata" if a.pro_rata else "sequential")

    res = analyse(pool, deal, sc)

    print(f"Pool: {len(pool)} loans, GBP {pool.balance.sum() / 1e6:,.1f}m, "
          f"WA LTV {(pool.ltv * pool.balance).sum() / pool.balance.sum():.1%}, "
          f"pool loss {res.pool_cf.loss.sum() / pool.balance.sum():.2%}")
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", "{:,.2f}".format):
        print(res.pricing.to_string(index=False))


if __name__ == "__main__":
    main()
