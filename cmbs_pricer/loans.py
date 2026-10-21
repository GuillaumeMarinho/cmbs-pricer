"""Dummy UK commercial real estate loan pool.

Generates a reproducible tape of UK CRE loans with the fields a CMBS model
needs: balances, values, coupons (fixed or SONIA floating), amortisation,
remaining term and loan-level call protection.

Call protection conventions used (UK market flavour):
  * Lockout          - no voluntary prepayment allowed.
  * Spens            - UK make-whole: PV of remaining coupons at gilts + spens
                       margin, minus par. Typical on fixed-rate loans.
  * Yield maint.     - make-whole at SONIA swaps flat.
  * Defeasance       - borrower substitutes gilts; the pool never sees a
                       prepayment (rare in the UK, common in US conduit).
  * Step-down        - percentage fee that steps down (e.g. 2% / 1% / 0%).
                       Typical on floating-rate loans.
  * Open period      - final quarters before maturity, prepayable at par.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .curves import SONIA

PROPERTY_TYPES = {
    # type: (pool weight, cap rate, floating margin bp, fixed margin bp)
    "Office": (0.20, 0.0675, 350, 275),
    "Logistics": (0.22, 0.0525, 285, 225),
    "Retail": (0.10, 0.0750, 400, 310),
    "Build-to-Rent": (0.12, 0.0475, 270, 215),
    "Student Accommodation": (0.10, 0.0550, 300, 235),
    "Hotel": (0.08, 0.0775, 425, 325),
    "Healthcare": (0.06, 0.0625, 330, 260),
    "Mixed Use": (0.12, 0.0625, 325, 255),
}

LOCATIONS = [
    # (city, region, pool weight)
    ("London - City", "London", 0.12),
    ("London - West End", "London", 0.10),
    ("London - Docklands", "London", 0.05),
    ("London - Outer", "London", 0.06),
    ("Reading", "South East", 0.04),
    ("Guildford", "South East", 0.03),
    ("Milton Keynes", "South East", 0.04),
    ("Cambridge", "East of England", 0.04),
    ("Bristol", "South West", 0.05),
    ("Birmingham", "West Midlands", 0.07),
    ("Nottingham", "East Midlands", 0.04),
    ("Manchester", "North West", 0.09),
    ("Liverpool", "North West", 0.04),
    ("Leeds", "Yorkshire", 0.05),
    ("Sheffield", "Yorkshire", 0.02),
    ("Newcastle", "North East", 0.03),
    ("Edinburgh", "Scotland", 0.05),
    ("Glasgow", "Scotland", 0.04),
    ("Cardiff", "Wales", 0.04),
]

_NAME_A = ["Thames", "Albion", "Mercia", "Pennine", "Caledonia", "Severn", "Wessex",
            "Northgate", "Kingsway", "Crown", "Harbour", "Meridian", "Oakfield",
            "Riverside", "Victoria", "Regent", "Cathedral", "Exchange", "Granite", "Beacon"]
_NAME_B = {
    "Office": ["House", "Court", "Plaza", "Tower"],
    "Logistics": ["Distribution Park", "Logistics Hub", "Trade Park", "Industrial Estate"],
    "Retail": ["Retail Park", "Shopping Centre", "Parade"],
    "Build-to-Rent": ["Residences", "Living", "Quarter"],
    "Student Accommodation": ["Student Village", "Halls", "Campus Living"],
    "Hotel": ["Hotel", "Hotel & Spa", "Suites"],
    "Healthcare": ["Care Home", "Medical Centre", "Healthcare Portfolio"],
    "Mixed Use": ["Yard", "Square", "Wharf"],
}

SPENS_MARGIN = 0.0050   # gilts + 50bp


def generate_loan_pool(n_loans: int = 75, seed: int = 7) -> pd.DataFrame:
    """Return a dummy UK CRE loan tape (one row per loan)."""
    rng = np.random.default_rng(seed)

    types = list(PROPERTY_TYPES)
    w_type = np.array([PROPERTY_TYPES[t][0] for t in types])
    ptype = rng.choice(types, size=n_loans, p=w_type / w_type.sum())

    w_loc = np.array([l[2] for l in LOCATIONS])
    loc_idx = rng.choice(len(LOCATIONS), size=n_loans, p=w_loc / w_loc.sum())
    city = np.array([LOCATIONS[i][0] for i in loc_idx])
    region = np.array([LOCATIONS[i][1] for i in loc_idx])

    # Balances: lognormal around ~GBP 25m, London loans larger.
    london = region == "London"
    bal = rng.lognormal(mean=np.log(22e6), sigma=0.6, size=n_loans) * np.where(london, 1.5, 1.0)
    bal = np.clip(bal, 5e6, 150e6).round(-5)

    ltv = np.clip(rng.normal(0.58, 0.08, n_loans), 0.40, 0.75)
    value = (bal / ltv).round(-5)
    ltv = bal / value

    cap = np.array([PROPERTY_TYPES[t][1] for t in ptype]) * rng.normal(1.0, 0.08, n_loans)
    noi = (value * cap).round(-3)

    is_fixed = rng.random(n_loans) < 0.35
    orig_term_q = np.where(is_fixed, rng.choice([20, 28, 40], n_loans, p=[0.4, 0.4, 0.2]),
                           rng.choice([12, 20], n_loans, p=[0.3, 0.7]))
    seasoning_q = rng.integers(0, 9, n_loans)
    rem_term_q = np.maximum(orig_term_q - seasoning_q, 4)

    # Pricing: margin by property type, adjusted for leverage.

    lev_adj = (ltv - 0.58) * 600 # +/- ~50bp across the LTV range
    float_margin = np.array([PROPERTY_TYPES[t][2] for t in ptype]) + lev_adj + rng.normal(0, 20,
n_loans)
    fixed_margin = np.array([PROPERTY_TYPES[t][3] for t in ptype]) + lev_adj + rng.normal(0, 15,
n_loans)
    float_margin = np.round(float_margin / 5) * 5
    swap = np.array([SONIA.swap_rate(q / 4) for q in orig_term_q])
    fixed_rate = np.round((swap + fixed_margin / 1e4) * 1e4) / 1e4

    margin_bp = np.where(is_fixed, np.nan, float_margin)
    coupon_fixed = np.where(is_fixed, fixed_rate, np.nan)
    current_rate = np.where(is_fixed, fixed_rate, SONIA.zero(0.25) + float_margin / 1e4)

    amortising = rng.random(n_loans) < 0.40
    amort_pct = np.where(amortising, rng.choice([0.01, 0.02], n_loans), 0.0)

    # --- Call protection -------------------------------------------------------
    protection = np.where(
        is_fixed,
        rng.choice(["Spens", "Yield Maintenance", "Defeasance"], n_loans, p=[0.5, 0.3, 0.2]),
        "Step-down",
    )
    lockout_orig = np.where(is_fixed, rng.choice([4, 8], n_loans), rng.choice([0, 2, 4], n_loans))
    lockout_q = np.maximum(lockout_orig - seasoning_q, 0)
    open_q = np.where(is_fixed, rng.choice([1, 2], n_loans), rng.choice([1, 2, 4], n_loans))
    open_q = np.minimum(open_q, rem_term_q)

    # Step-down fees: 2% in loan year 1, 1% in year 2, then open (measured from origination).
    sd1_end_q = np.where(protection == "Step-down", np.maximum(4 - seasoning_q, 0), 0)
    sd2_end_q = np.where(protection == "Step-down", np.maximum(8 - seasoning_q, 0), 0)

    names = []
    for i in range(n_loans):
        a = _NAME_A[rng.integers(len(_NAME_A))]
        b = _NAME_B[ptype[i]][rng.integers(len(_NAME_B[ptype[i]]))]
        names.append(f"{a} {b}, {city[i].split(' - ')[0]}")

    annual_interest = bal * current_rate
    df = pd.DataFrame({
        "loan_id": "",
        "property_name": names,
        "property_type": ptype,
        "city": city,
        "region": region,
        "balance": bal,
        "property_value": value,
        "ltv": ltv,
        "noi": noi,
        "icr": noi / annual_interest,
        "debt_yield": noi / bal,
        "rate_type": np.where(is_fixed, "Fixed", "Floating"),
        "margin_bp": margin_bp,
        "fixed_rate": coupon_fixed,
        "fixed_margin_bp": np.where(is_fixed, np.round(fixed_margin), np.nan),
        "current_rate": current_rate,
        "amort_pct": amort_pct,
        "orig_term_q": orig_term_q,
        "seasoning_q": seasoning_q,
        "rem_term_q": rem_term_q,
        "protection": protection,
        "lockout_q": lockout_q,
        "open_q": open_q,
        "stepdown_2pct_end_q": sd1_end_q,
        "stepdown_1pct_end_q": sd2_end_q,
    })
    df = df.sort_values("balance", ascending=False, ignore_index=True)
    df["loan_id"] = [f"UK{i + 1:03d}" for i in range(n_loans)]
    return df


def protection_status(pool: pd.DataFrame, n_periods: int) -> np.ndarray:
    """Matrix (loans x periods) of protection state codes, period index 0 = Q1.

    0 = lockout, 1 = make-whole (Spens / YM), 2 = defeasance-only,
    3 = step-down fee, 4 = open, 5 = matured.
    """
    q = np.arange(1, n_periods + 1)[None, :]
    rem = pool["rem_term_q"].to_numpy()[:, None]
    lock = pool["lockout_q"].to_numpy()[:, None]
    open_start = rem - pool["open_q"].to_numpy()[:, None]
    prot = pool["protection"].to_numpy()[:, None]
    sd_end = pool["stepdown_1pct_end_q"].to_numpy()[:, None]

    state = np.full((len(pool), n_periods), 4)
    make_whole = np.isin(prot, ["Spens", "Yield Maintenance"])
    state = np.where(make_whole & (q <= open_start), 1, state)
    state = np.where((prot == "Defeasance") & (q <= open_start), 2, state)
    state = np.where((prot == "Step-down") & (q <= sd_end) & (q <= open_start), 3, state)
    state = np.where(q <= lock, 0, state)
    state = np.where(q > rem, 5, state)
    return state
