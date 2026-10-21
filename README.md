# UK CMBS Pricer

A UK commercial mortgage-backed securities (CMBS) deal analyser. It takes a pool of **75 dummy UK
commercial real estate loans**, projects their cash flows quarter by quarter, runs them through a note
waterfall, and prices each class from a rating and WAL spread curve. For each class it gives price,
yield, discount margin, WAL, modified/spread/effective duration and losses.

All loans, curves and spreads are **fictitious** and generated for illustration.

```
Loan tape (75 UK loans) ──► Loan cash flows ──► Waterfall ──► Tranche pricing
 property, LTV, ICR,         CPR / CDR / balloon   fees → interest A..F   DM from spread curve @ WAL
 SONIA or fixed + swap       call protection       PDL cure → residual    price, yield, durations
 lockout / Spens / YM ...    recoveries (lagged)   sequential / pro rata stress grid
```

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py            # interactive app
python -m cmbs_pricer           # base-case pricing table in the terminal
python -m cmbs_pricer --cdr 0.03 --value-decline 0.4 --refi-ltv 0.65   # stress
python -m pytest                # tests
```

## Base case output

| Class | Rating | Size | CE | Coupon | WAL | DM | Price | Mod Dur | Eff Dur |
|---|---|---|---|---|---|---|---|---|---|
| A | AAA | £1,195m | 40% | S+95 | 2.6y | 97bp | 100.04 | 2.41 | 0.04 |
| B | AA | £179m | 31% | S+170 | 4.2y | 172bp | 99.96 | 3.73 | 0.02 |
| C | A | £179m | 22% | S+235 | 4.7y | 236bp | 99.98 | 4.05 | 0.01 |
| D | BBB | £159m | 14% | S+340 | 4.9y | 339bp | 100.06 | 4.14 | 0.01 |
| E | BB | £139m | 7% | S+550 | 5.9y | 548bp | 100.10 | 4.54 | 0.02 |
| F | B | £80m | 3% | S+750 | 7.4y | 807bp | 91.42 | 5.01 | −2.21 |
| R | NR | £60m | 0% | residual | 8.7y | 826bp | 108.37 | 3.65 | −1.51 |

The notes are SONIA floaters. Their *effective* duration (bump the curve and re-run the whole model) is
about zero, even though their yield-based modified duration is 2–5 years. The risk is in spread
duration. Class F prices below par because of excess-spread compression, explained under Waterfall
below.

## The collateral: 75 UK loans (`cmbs_pricer/loans.py`)

| Field | How it's generated |
|---|---|
| Property type | Office, Logistics, Retail, Build-to-Rent, Student Accommodation, Hotel, Healthcare,
Mixed Use |

| Location | 19 UK cities across London, South East, Midlands, North West, Yorkshire, Scotland, Wales,
etc. |
| Balance / value | Lognormal around £22m (London × 1.5), LTV ~ N(58%, 8%) clipped to 40–75% |
| NOI / ICR / debt yield | Value × sector cap rate with noise |
| Rate | 65% floating SONIA + margin (3–5y terms); 35% fixed = swap + margin (5/7/10y terms) |
| Amortisation | 60% interest-only, the rest 1–2% p.a. |
| Call protection | See below |

The seed is fixed, so the pool is reproducible. Change `--seed` or the sidebar to generate a different
one. A copy of the tape is in `data/uk_loan_tape.csv`.

### Call protection, UK flavour

| Type | Mechanic | Used on |
|---|---|---|
| **Lockout** | No voluntary prepayment | Most loans, first 0–8 quarters from origination |
| **Spens** | UK make-whole: PV of remaining coupons at **gilts + 50bp** minus par | Fixed |
| **Yield maintenance** | Make-whole at SONIA swaps flat | Fixed |
| **Defeasance** | Borrower substitutes gilts, so the pool never sees a prepayment (rare in the UK,
standard in US conduits) | Fixed |
| **Step-down** | 2% fee in loan year 1, 1% in year 2 | Floating |
| **Open period** | Last 1–4 quarters, prepayable at par | All |

Prepayment speed depends on the protection state. It's zero in lockout or defeasance, `CPR × factor`
when a fee or make-whole applies, and full CPR when the loan is open. Fees received go to the classes
being paid principal that quarter.

## Cash flows (`cmbs_pricer/cashflows.py`)

- Quarterly, deterministic, loan by loan.
- **Term defaults**: the CDR removes a fraction of the balance each quarter.
- **Balloon risk**: at maturity a loan refinances only if its LTV after the value decline is at or
below the refinance LTV. Otherwise it goes to workout.
- **Recovery**: `min(1, stressed value × (1 − liquidation costs) / loan balance)`, received after the
recovery lag.
- **Issuer swap**: fixed-rate loans are swapped to SONIA + their fixed margin, so the collateral
matches the floating notes. You can switch this off to see the basis risk.
- No servicer advancing, so loans in workout stop paying interest.

## Waterfall (`cmbs_pricer/waterfall.py`)

1. Senior fees: 12.5bp servicing plus £250k p.a. fixed costs.
2. Interest to Class A → F at SONIA + margin. Unpaid interest is deferred.
3. Excess revenue cures the quarter's realised losses (the principal deficiency ledger, PDL).
4. Whatever is left goes to Class R, the residual / first-loss piece.

Principal is paid **sequentially** or **pro rata**. Pro rata switches to sequential if cumulative
losses exceed a trigger. Losses not cured in step 3 write classes down from the bottom (R first).

**Excess-spread compression.** Under sequential pay, once Class A has amortised, the remaining junior
notes carry coupons of S+550 to S+750. The few long-dated loans left in the pool pay about S+300. Class
F therefore ends up short of interest even with no losses. This is a real structural feature of
sequential deals, and one reason European CMBS 2.0 uses pro rata allocation with sequential triggers.
Switch to pro rata in the app to compare.

## Pricing (`cmbs_pricer/pricing.py`, `model.py`)

- Rated classes are discounted at **SONIA forwards + DM**. The DM is read off the rating spread curve
at the class WAL.
- Class R is priced to a target yield (12% by default).
- **Yield** is the quarterly-compounded IRR at that price.
- **Modified duration and convexity** are measured off that yield.
- **Spread duration**: bump the DM by ±1bp.
- **Effective duration**: shift the SONIA and gilt curves by ±25bp, re-run loans, swap and waterfall,
and reprice at the same DM.

## The app (`app.py`)

The sidebar sets the scenario (CPR, CDR, value decline, refinance LTV, liquidation costs, lag, rate
shift) and the structure (sequential or pro rata, issuer swap, spread shift, residual yield). There are
six tabs:

1. **Tranche pricing**: the pricing table, note balances over time, and the spread curve with each
class priced on it.
2. **Collateral**: breakdowns by property type, region, LTV, maturity year and call protection.
3. **Loan tape**: the full tape, downloadable as CSV.
4. **Cash flows**: pool cash flows by quarter, plus per-class tables.
5. **Call protection**: a heatmap of each loan's protection state by quarter, and prepayment fees
received.
6. **Stress grid**: loss, price or WAL for a chosen class across value decline × CDR.

## Simplifications and next steps

- Defaults are expected-value (fractional), not Monte Carlo. A loan-level stochastic default model
would be next.
- No liquidity facility, appraisal reductions, Class X, VRR risk-retention strip, or loan extension
options.
- The issuer swap notional follows the performing balance, a balance-guaranteed swap with no swap
counterparty risk modelled.
- The Spens and YM make-whole ignore amortisation in the remaining schedule.
- **Real data**: for UK and European CMBS the loan-level source is the ESMA securitisation templates
via European DataWarehouse or SecRep. SEC EDGAR (Reg AB II asset-level data) covers US conduits only.
