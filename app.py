"""Streamlit front-end for the UK CMBS pricer. Run: streamlit run app.py"""
import base64
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cmbs_pricer import CMBS_SPREADS, Deal, Scenario, analyse, generate_loan_pool
from cmbs_pricer.loans import protection_status

st.set_page_config(page_title="UK CMBS Pricer", layout="wide", initial_sidebar_state="collapsed")

# --- palette: deep forest green surfaces, yellow accent, cream ink -----------
BG, CARD, CARD_2, LINE = "#10261a", "#183523", "#1f4029", "#2f5a3b"
INK, INK_MUTED, ACCENT = "#f2ead3", "#a9b8a0", "#f2c230"
# Two single-hue ramps validated on the dark card surface (brighter = more senior):
# green for investment grade (A-D), yellow for sub-IG and residual (E, F, R).

GREEN = ["#d9efc2", "#a9d68e", "#78b965", "#4f9a4a"]
YELLOW = ["#ffd966", "#e0ae1f", "#b8860b"]
TRANCHE = GREEN + YELLOW
SEQ = ["#1d3f2a", "#2f5f3a", "#4f9a4a", "#78b965", "#a9d68e", "#d9efc2"]
GBPm = lambda x: f"£{x / 1e6:,.1f}m"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Manrope:wght@300;400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"], .stApp {{ background: {BG}; color: {INK};
  font-family: 'Manrope', system-ui, sans-serif; }}
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none;
}}
.block-container {{ max-width: 1320px; padding-top: 1.2rem; padding-bottom: 4rem; }}
p, label, span, div {{ font-family: 'Manrope', system-ui, sans-serif; }}
.serif {{ font-family: 'Instrument Serif', Georgia, serif; font-weight: 400; }}
.accent {{ color: {ACCENT}; }}

/* nav */
.nav {{ display: flex; align-items: center; justify-content: space-between; padding: 8px 4px 0; }}
.logo {{ font-size: 44px; color: {ACCENT}; letter-spacing: -1px; font-weight: 400; line-height: 1; }}
.logo sup {{ font-size: 26px; margin-left: 2px; }}
.links a {{ color: {ACCENT}; text-decoration: none; font-size: 19px; margin: 0 22px; }}
.links a:hover {{ color: {INK}; }}
.pill {{ display: inline-block; border-radius: 999px; padding: 12px 28px; font-size: 18px;
  text-decoration: none !important; transition: filter .15s; }}
.pill:hover {{ filter: brightness(1.12); }}
.pill.soft {{ background: #2c5537; color: {INK} !important; }}
.pill.solid {{ background: {ACCENT}; color: {BG} !important; font-weight: 500; }}

/* hero */
.hero {{ position: relative; text-align: center; padding: 110px 0 70px; overflow: hidden; }}
.hero svg {{ position: absolute; inset: 0; width: 100%; height: 100%; z-index: 0; }}
.hero > div {{ position: relative; z-index: 1; }}
.hero h1 {{ font-size: clamp(48px, 8vw, 118px); line-height: 0.98; font-weight: 400; color: {INK};
  margin: 0 0 56px; letter-spacing: -2px; font-family: 'Manrope', sans-serif; }}
.hero h1 .serif {{ font-size: 1.12em; letter-spacing: -1px; }}
.hero p.sub {{ color: {INK_MUTED}; font-size: 18px; max-width: 640px; margin: -28px auto 40px; }}

/* section headings */
.kicker {{ font-family: 'Instrument Serif', Georgia, serif; color: {ACCENT}; font-size: 26px; margin:
18px 0 8px; }}
.h2 {{ font-size: 64px; line-height: 0.95; color: {INK}; margin: 0 0 18px; font-weight: 400; letter-
spacing: -1px; }}
.h2 .serif {{ font-size: 1.08em; }}
.lede {{ color: {INK}; opacity: .85; font-size: 17px; line-height: 1.5; }}
.section {{ border-top: 1px solid {LINE}; margin-top: 64px; padding-top: 36px; }}

/* cards */
.kpi {{ background: {CARD}; border-radius: 14px; padding: 22px 26px; height: 100%; }}
.kpi .label {{ font-size: 20px; color: {INK}; }}
.kpi .num {{ font-size: clamp(36px, 3.6vw, 56px); font-weight: 400; letter-spacing: -1px; margin: 14px
0 22px;
  line-height: 1; white-space: nowrap; }}
.kpi .l1 {{ font-size: 16px; color: {INK}; opacity: .9; }}
.kpi .l2 {{ font-size: 16px; color: {ACCENT}; margin-top: 4px; }}
.st-key-controls {{ border: 1.5px solid {ACCENT}; border-radius: 14px; padding: 16px 16px 6px; }}
[class*="st-key-card"] {{ background: {CARD}; border-radius: 14px; padding: 22px 24px 12px; }}
.cardhead {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }}
.cardhead .t {{ font-size: 26px; color: {INK}; }}
.cardhead .note {{ display: flex; align-items: center; gap: 18px; color: {INK}; font-size: 16px; text-
align: left; }}
.cardhead .big {{ font-size: 46px; color: {ACCENT}; letter-spacing: -1px; white-space: nowrap; }}

/* streamlit widgets */
[data-testid="stExpander"] details {{ background: {CARD_2}; border: none; border-radius: 10px; margin-
bottom: 10px; }}
[data-testid="stExpander"] summary p {{ font-size: 19px; color: {INK}; }}
[data-testid="stWidgetLabel"] p {{ color: {INK_MUTED}; }}
.stDownloadButton button, .stButton button {{ border-radius: 999px; background: {ACCENT}; color: {BG};
  border: none; padding: 8px 22px; font-weight: 500; }}
[data-testid="stAlert"] {{ background: #2a3a1a; border-radius: 12px; }}

[data-testid="stAlert"] p {{ color: {ACCENT}; }}
.foot {{ color: {INK_MUTED}; font-size: 14px; text-align: center; margin-top: 70px; }}
"""


def html(s: str):
    """Render raw HTML (single line so markdown never turns it into a code block)."""
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def style(fig, height=340, **kw):
    layout = dict(height=height, margin=dict(l=10, r=10, t=48, b=10), paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Manrope, sans-serif", color=INK,
size=13),
                  title_font=dict(family="Manrope, sans-serif", size=19, color=INK),
legend=dict(orientation="h", y=-0.28),
                  bargap=0.18, barcornerradius=4, hoverlabel=dict(namelength=-1, bgcolor=CARD_2,
font_color=INK))
    fig.update_layout(**{**layout, **kw})
    fig.update_layout(title_x=0, title_xanchor="left")
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont=dict(color=INK_MUTED), automargin=True,
                     title_font=dict(color=INK_MUTED))
    fig.update_yaxes(gridcolor="rgba(242,234,211,0.07)", zeroline=False,
tickfont=dict(color=INK_MUTED),
                     automargin=True, title_font=dict(color=INK_MUTED))
    return fig


def chart(fig, **kw):
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False}, **kw)


def heading(anchor, kicker, title_html, lede=""):
    html(f'<div class="section" id="{anchor}"><div class="kicker">{kicker}</div>'
         f'<div class="h2">{title_html}</div>'
         + (f'<div class="lede">{lede}</div>' if lede else "") + "</div>")


@st.cache_data
def load_pool(n, seed):
    return generate_loan_pool(n, seed)


@st.cache_data
def run(n, seed, deal, scenario, spread_shift, residual_yield):
    return analyse(load_pool(n, seed), deal, scenario, CMBS_SPREADS.shifted(spread_shift),
residual_yield)


html(f"<style>{CSS}</style>")

# ------------------------------------------------------------------ nav ------
tape_b64 = base64.b64encode(load_pool(75, 7).to_csv(index=False).encode()).decode()
html(f"""
<div class="nav">
  <div class="logo">ukcmbs<sup>≀</sup></div>
  <div class="links">
    <a href="#report">report</a><a href="#pricing">pricing</a><a href="#collateral">collateral</a>
    <a href="#cashflows">cash flows</a><a href="#protection">protection</a><a href="#stress">stress</a>
  </div>
  <a class="pill soft" download="uk_cmbs_loan_tape.csv" href="data:text/csv;base64,{tape_b64}">Download
tape</a>
</div>
""")

# ----------------------------------------------------------------- hero ------
html(f"""
<div class="hero">
  <svg viewBox="0 0 1300 520" preserveAspectRatio="none" aria-hidden="true">
    <path d="M0 330 C 320 330, 560 380, 760 520" stroke="{LINE}" fill="none" stroke-width="1.2"/>
    <path d="M1300 330 C 980 330, 820 400, 700 520" stroke="{LINE}" fill="none" stroke-width="1.2"/>
    <path d="M0 450 C 260 450, 380 470, 470 520" stroke="{LINE}" fill="none" stroke-width="1.2"/>

    <path d="M1300 450 C 1040 450, 920 470, 840 520" stroke="{LINE}" fill="none" stroke-width="1.2"/>
    <path d="M150 334 C 200 336, 230 340, 270 346" stroke="{ACCENT}" fill="none" stroke-width="3"/>
  </svg>
  <div>
    <h1><span class="serif accent">From</span> loan tape <br/> to <span class="serif">priced</span>
         <span class="serif accent">tranche.</span></h1>
    <p class="sub">75 dummy UK commercial real estate loans, a full note waterfall, and every class
priced
        off a SONIA spread curve. All data is fictitious.</p>
    <a class="pill solid" href="#report">Price the deal ↓</a>
  </div>
</div>
""")

# --------------------------------------------------------- report section ----
html('<div id="report"></div>')
left, right = st.columns([1, 2.35], gap="medium")

with left:
    html("""<div class="h2" style="font-size:60px;margin-top:6px"><span class="serif">Deal</span>
<br/>report</div>
            <div class="lede">Every number on this page re-runs live from the assumptions below:
            loans → cash flows → waterfall → pricing.</div>
            <div class="kicker">Assumptions</div>""")
    with st.container(key="controls"):
        with st.expander("Credit & prepayment", expanded=True):
            cpr = st.slider("CPR when open (%)", 0.0, 50.0, 10.0, 1.0) / 100
            prot_factor = st.slider("Prepay through fees / make-whole (% of CPR)", 0, 100, 15, 5) / 100
            cdr = st.slider("Term CDR (%)", 0.0, 10.0, 0.5, 0.25) / 100
            value_decline = st.slider("Property value decline (%)", 0, 60, 0, 5) / 100
            refi_ltv = st.slider("Max refinance LTV at maturity (%)", 50, 85, 70, 1) / 100
        with st.expander("Workout & rates"):
            liq_cost = st.slider("Liquidation costs (% of value)", 0, 20, 8, 1) / 100
            lag = st.slider("Recovery lag (quarters)", 1, 12, 6)
            rate_shift = st.slider("SONIA / gilt parallel shift (bp)", -200, 300, 0, 25)
        with st.expander("Structure"):
            mode = st.radio("Principal allocation", ["sequential", "pro_rata"], horizontal=True)
            trigger = st.slider("Sequential trigger: cumulative loss (%)", 0.0, 10.0, 2.0, 0.5,
                                disabled=mode == "sequential") / 100
            swap = st.toggle("Issuer swap on fixed-rate loans", True)
        with st.expander("Pricing"):
            spread_shift = st.slider("Spread curve shift (bp)", -100, 300, 0, 10)
            resid_y = st.slider("Residual (Class R) target yield (%)", 5.0, 25.0, 12.0, 0.5) / 100
        with st.expander("Loan pool"):
            n_loans = st.number_input("Number of loans", 10, 200, 75, step=5)
            seed = st.number_input("Random seed", 0, 9999, 7)

scenario = Scenario(cpr, prot_factor, cdr, value_decline, refi_ltv, liq_cost, lag, rate_shift)
deal = Deal(principal_mode=mode, seq_trigger_cum_loss=trigger, issuer_swap=swap)
pool = load_pool(n_loans, seed)
res = run(n_loans, seed, deal, scenario, spread_shift, resid_y)
summ = res.pool_cf.summary()
wf = res.waterfall
pr = res.pricing.set_index("Class")
w = pool.balance / pool.balance.sum()
pool_loss = summ.loss.sum() / pool.balance.sum()

with right:
    a, f = pr.loc["A"], pr.loc["F"]
    unpaid = res.pricing["Unpaid Int (GBP m)"].sum()
    cards = [
        ("Pool balance", f"£{pool.balance.sum() / 1e9:,.2f}bn", ACCENT,
         f"{len(pool)} loans · WA LTV {(pool.ltv * w).sum():.1%}",
         f"WA ICR {(pool.icr * w).sum():.2f}x · fixed {w[pool.rate_type == 'Fixed'].sum():.0%}"),
        ("Class A price", f"{a['Price']:.2f}", INK,
         f"DM {a['DM (bp)']:.0f}bp · WAL {a['WAL (y)']:.1f}y",
         f"Spread dur {a['Spread Dur']:.2f} · eff dur {a['Eff Dur']:.2f}"),
        ("Pool loss", f"{pool_loss:.2%}", ACCENT,
         f"CDR {cdr:.2%} · values −{value_decline:.0%}",
         f"Unpaid note interest {GBPm(unpaid * 1e6)}"),
    ]
    cols = st.columns(3, gap="small")

    for col, (label, num, colr, l1, l2) in zip(cols, cards):
        with col:
            html(f'<div class="kpi"><div class="label">{label}</div>'
                  f'<div class="num" style="color:{colr}">{num}</div>'
                  f'<div class="l1">{l1}</div><div class="l2">{l2}</div></div>')

    html('<div style="height:14px"></div>')
    with st.container(key="card_main"):
        gap_pts = f["Price"] - 100
        html(f"""<div class="cardhead"><div class="t">Pool cash flow by year</div>
                 <div class="note"><span>Class F trades<br/>{'below' if gap_pts < 0 else 'above'} par
by</span>
                 <span class="big">{gap_pts:+.1f}pts</span></div></div>""")
        yr = summ.assign(year=2026 + (summ.period - 1) // 4 + 1).groupby("year")[["principal",
"interest", "loss"]].sum()
        yr = yr[(yr.principal + yr.interest + yr.loss) > 1]
        x = yr.index.astype(str)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=(yr.principal + yr.interest) / 1e6, name="Total collections",
                              marker_color="rgba(242,234,211,0.08)",
                              hovertemplate="%{x}: £%{y:,.1f}m<extra>Total</extra>"))
        spline = dict(shape="spline", smoothing=0.8)
        fig.add_trace(go.Scatter(x=x, y=yr.principal / 1e6, name="Principal", mode="lines",
line=dict(color=INK, width=2.5, **spline),
                                  hovertemplate="%{x}: £%{y:,.1f}m<extra>Principal</extra>"))
        fig.add_trace(go.Scatter(x=x, y=yr.interest / 1e6, name="Interest", mode="lines",
line=dict(color=ACCENT, width=2, **spline),
                                  hovertemplate="%{x}: £%{y:,.1f}m<extra>Interest</extra>"))
        if yr.loss.sum() > 0:
            fig.add_trace(go.Scatter(x=x, y=yr.loss / 1e6, name="Realised loss", mode="lines",
                                      line=dict(color=YELLOW[2], width=2, dash="dot", **spline),
                                      hovertemplate="%{x}: £%{y:,.1f}m<extra>Loss</extra>"))
        fig.update_yaxes(rangemode="tozero", ticksuffix="m", tickprefix="£")
        chart(style(fig, height=600, bargap=0.12, margin=dict(l=8, r=8, t=10, b=8), hovermode="x unified",
                               legend=dict(orientation="h", y=1.08, x=0), xaxis_title=None),
width="stretch")

# ---------------------------------------------------------- pricing ----------
heading("pricing", "Tranche pricing", 'Every <span class="serif accent">class,</span> priced.',
         "Rated classes are discounted at SONIA forwards plus the discount margin read off the rating spread curve "
         "at each class's WAL; Class R is priced to its target yield. Eff Dur bumps SONIA/gilts ±25bp and re-runs "
         "the full model, so floaters sit near zero; Spread Dur bumps the DM.")
fmt = {"Size (GBP m)": "{:,.1f}", "Size %": "{:.1%}", "CE %": "{:.1%}", "WAL (y)": "{:.2f}",
       "DM (bp)": "{:.0f}", "Price": "{:.3f}", "Yield %": "{:.2f}", "Mod Dur": "{:.2f}",
       "Spread Dur": "{:.2f}", "Eff Dur": "{:.2f}", "Convexity": "{:.1f}", "Loss %": "{:.1%}",
       "Unpaid Int (GBP m)": "{:.2f}"}
st.dataframe(res.pricing.style.format(fmt, na_rep="–"), hide_index=True, width="stretch")
if unpaid > 0.05:
    st.warning("Some classes end with unpaid interest: once senior notes have amortised, the remaining junior "
                "coupons can exceed the interest the residual loans pay (excess-spread compression), and loans in "
                "workout pay no interest (no servicer advancing).")

c1, c2 = st.columns(2, gap="small")
with c1, st.container(key="card_bal"):
    fig = go.Figure()
    for j, tr in enumerate(wf.tranches):
        fig.add_trace(go.Scatter(x=summ.period, y=wf.begin_balance[j] / 1e6, name=f"Class {tr.name}",
stackgroup="b",
                                 line=dict(width=1, color=CARD), fillcolor=TRANCHE[j],
                                 hovertemplate="Q%{x}: £%{y:,.1f}m"))
    chart(style(fig, title="Note balances (£m)", xaxis_title="Quarter"), width="stretch")
with c2, st.container(key="card_curve"):
    fig = go.Figure()
    curve_x = np.linspace(1, 10, 40)
    sc = CMBS_SPREADS.shifted(spread_shift)
    for j, tr in enumerate(t for t in wf.tranches if t.rating != "NR"):
        fig.add_trace(go.Scatter(x=curve_x, y=[sc.spread_bp(tr.rating, x) for x in curve_x],

name=tr.rating,
                                  line=dict(width=2, color=TRANCHE[j]), hovertemplate="%{x:.1f}y: %{y:.0f}bp"))
        row = res.pricing.iloc[j]
        if np.isfinite(row["WAL (y)"]):
             fig.add_trace(go.Scatter(x=[row["WAL (y)"]], y=[row["DM (bp)"]], mode="markers",
showlegend=False,
                                      marker=dict(size=11, color=TRANCHE[j], line=dict(width=2,
color=CARD)),
                                      name=f"Class {tr.name}",
                                      hovertemplate=f"Class {tr.name}: WAL %{{x:.2f}}y, DM %{{y:.0f}}bp"))
    chart(style(fig, title="Spread curve & priced classes (bp over SONIA)", xaxis_title="WAL (years)"),
                     width="stretch")

# ------------------------------------------------------- collateral ----------
heading("collateral", "The collateral", f'<span class="serif accent">{len(pool)}</span> UK loans.',
        "Offices, logistics, build-to-rent, student housing, hotels and more, from London to Glasgow. "
        "Floating loans pay SONIA + margin; fixed loans are swapped to SONIA at issuer level.")
p2 = pool.assign(
    ltv_band=pd.cut(pool.ltv, [0, .5, .55, .6, .65, .7, 1], labels=["<50%", "50-55%", "55-60%", "60-65%", "65-70%", ">70%"]),
    maturity_year=(2026 + np.ceil(pool.rem_term_q / 4)).astype(int).astype(str),
)


def strat(col, title, order=None):
    g = p2.groupby(col, observed=True).agg(balance=("balance", "sum"), loans=("loan_id", "count"),
                                           ltv=("ltv", lambda s: np.average(s, weights=p2.loc[s.index,
"balance"])))
    g = g.reindex(order) if order else g.sort_values("balance")
    fig = go.Figure(go.Bar(x=g.balance / 1e6, y=g.index.astype(str), orientation="h",
marker_color=GREEN[2],
                           customdata=np.c_[g.loans, g.ltv * 100],
                           hovertemplate="%{y}: £%{x:,.1f}m · %{customdata[0]} loans · WA LTV %{customdata[1]:.1f}%<extra></extra>"))
    return style(fig, height=320, title=title, xaxis_title="£m", yaxis_type="category")


c1, c2 = st.columns(2, gap="small")
with c1, st.container(key="card_type"):
    chart(strat("property_type", "By property type"), width="stretch")
with c2, st.container(key="card_region"):
    chart(strat("region", "By region"), width="stretch")
c1, c2, c3 = st.columns(3, gap="small")
with c1, st.container(key="card_ltv"):
    chart(strat("ltv_band", "By LTV", ["<50%", "50-55%", "55-60%", "60-65%", "65-70%", ">70%"]),
width="stretch")
with c2, st.container(key="card_mat"):
    chart(strat("maturity_year", "By maturity year", sorted(p2.maturity_year.unique())),
width="stretch")
with c3, st.container(key="card_prot"):
    chart(strat("protection", "By call protection"), width="stretch")

with st.expander("Loan tape", expanded=False):
    st.dataframe(pool.style.format({
        "balance": "£{:,.0f}", "property_value": "£{:,.0f}", "noi": "£{:,.0f}", "ltv": "{:.1%}",
        "icr": "{:.2f}x", "debt_yield": "{:.1%}", "margin_bp": "{:.0f}", "fixed_rate": "{:.2%}",
        "fixed_margin_bp": "{:.0f}", "current_rate": "{:.2%}", "amort_pct": "{:.0%}"}, na_rep="–"),
        hide_index=True, width="stretch", height=480)
    st.download_button("Download loan tape (CSV)", pool.to_csv(index=False), "uk_cmbs_loan_tape.csv",
"text/csv")

# ------------------------------------------------------- cash flows ----------
heading("cashflows", "Cash flows", 'Where the <span class="serif accent">money</span> goes.')
with st.container(key="card_cf"):
    fig = go.Figure()
    # alternate green / yellow so stacked neighbours never share a hue
    for col, name, colr in [("interest", "Interest (post-swap)", GREEN[1]),
                            ("scheduled_principal", "Scheduled amortisation", YELLOW[1]),
                            ("prepayment", "Prepayment", GREEN[3]), ("balloon", "Balloon", YELLOW[0]),
                            ("recovery", "Recovery", GREEN[0])]:

        fig.add_trace(go.Bar(x=summ.period, y=summ[col] / 1e6, name=name, marker_color=colr,
                             marker_line=dict(width=1, color=CARD),
                             hovertemplate=f"{name} Q%{{x}}: £%{{y:,.1f}}m<extra></extra>"))
    fig.add_trace(go.Bar(x=summ.period, y=-summ.loss / 1e6, name="Realised loss",
marker_color=YELLOW[2],
                         hovertemplate="Loss Q%{x}: £%{y:,.1f}m<extra></extra>"))
    chart(style(fig, height=400, barmode="relative", title="Pool cash flows by quarter (£m)",
                          xaxis_title="Quarter"), width="stretch")

c1, c2 = st.columns([1, 3])
cls = c1.selectbox("Class cash flows", [t.name for t in wf.tranches])
j = [t.name for t in wf.tranches].index(cls)
st.dataframe((wf.tranche_table(j).set_index("period") / 1e6).style.format("{:,.2f}"), width="stretch",
height=340)
with st.expander("Pool-level detail (incl. issuer swap net and prepayment fees), £m"):
    st.dataframe((summ.set_index("period").drop(columns="sonia_fwd") / 1e6).style.format("{:,.2f}"),
                 width="stretch", height=300)

# --------------------------------------------------- call protection ---------
heading("protection", "Call protection", 'Lockout, <span class="serif accent">Spens</span> & step-downs.',
         "Each row is a loan, each column a quarter. Lockout blocks prepayment; Spens / yield maintenance charge a "
         "make-whole (gilts + 50bp / SONIA swaps); defeasance means the pool never sees the prepayment; step-down "
         "fees are 2% then 1%; open is prepayable at par.")
n_q = int(pool.rem_term_q.max())
state = protection_status(pool, n_q)
labels = ["Lockout", "Make-whole", "Defeasance", "Step-down fee", "Open", "Matured"]
# protected states in yellows (deeper = harder to prepay), defeasance / open in greens
colors = [YELLOW[2], YELLOW[1], GREEN[3], YELLOW[0], GREEN[1], "rgba(242,234,211,0.05)"]
cs = []
for i, c in enumerate(colors):
    cs += [[i / 6, c], [(i + 1) / 6, c]]
order = np.lexsort((pool.rem_term_q, pool.protection))
c1, c2 = st.columns([2.2, 1], gap="small")
with c1, st.container(key="card_heat"):
    fig = go.Figure(go.Heatmap(
         z=state[order], x=np.arange(1, n_q + 1), y=pool.loan_id.to_numpy()[order], colorscale=cs,
zmin=-0.5, zmax=5.5,
         xgap=1, ygap=1, customdata=np.array(labels, dtype=object)[state[order]],
         hovertemplate="%{y} · Q%{x}: %{customdata}<extra></extra>",
         colorbar=dict(tickvals=list(range(6)), ticktext=labels, outlinewidth=0,
tickfont=dict(color=INK))))
    chart(style(fig, height=640, title="Protection state by loan and quarter", xaxis_title="Quarter",
                           yaxis=dict(showticklabels=False, title="Loans (grouped by protection type)")), width="stretch")
with c2, st.container(key="card_fees"):
    fig = go.Figure(go.Bar(x=summ.period, y=summ.prepay_fees / 1e6, marker_color=ACCENT,
                            hovertemplate="Q%{x}: £%{y:,.2f}m<extra></extra>"))
    html(f'<div class="kpi" style="padding:0 0 10px"><div class="label">Fees & make-whole received</div>'
          f'<div class="num accent">{GBPm(summ.prepay_fees.sum())}</div>'
          f'<div class="l1">Paid to the classes receiving principal that quarter</div></div>')
    chart(style(fig, height=420, title="By quarter (£m)", xaxis_title="Quarter"), width="stretch")

# -------------------------------------------------------- stress grid --------
heading("stress", "Stress grid", 'How much can it <span class="serif accent">take?</span>',
        "Re-runs the full model for each combination of value decline and term CDR, holding every other "
        "assumption fixed.")
c1, c2 = st.columns([1, 2])
cls = c1.selectbox("Class", [t.name for t in wf.tranches], index=3)
metric = c2.radio("Metric", ["Loss %", "Price", "WAL (y)"], horizontal=True)
vds = [0, 10, 20, 30, 40, 50]
cdrs = [0.0, 0.5, 1.0, 2.0, 4.0]


@st.cache_data
def grid(n, seed, deal, scenario, spread_shift, resid_y, cls, metric):
    z = np.zeros((len(cdrs), len(vds)))
    for a_, c in enumerate(cdrs):

        for b_, v in enumerate(vds):
             sc = replace(scenario, cdr=c / 100, value_decline=v / 100)
             r = analyse(load_pool(n, seed), deal, sc, CMBS_SPREADS.shifted(spread_shift), resid_y,
                         effective_duration=False)
             z[a_, b_] = r.pricing.set_index("Class").loc[cls, metric]
    return z


with st.spinner("Running 30 scenarios…"):
    z = grid(n_loans, seed, deal, scenario, spread_shift, resid_y, cls, metric)
ztxt = np.vectorize(lambda v: f"{v:.1%}" if metric == "Loss %" else f"{v:.2f}")(z)
scale = SEQ if metric == "Loss %" else SEQ[::-1]
with st.container(key="card_grid"):
    fig = go.Figure(go.Heatmap(z=z, x=[f"{v}%" for v in vds], y=[f"{c}%" for c in cdrs],
colorscale=scale,
                               text=ztxt, texttemplate="%{text}", xgap=3, ygap=3, showscale=False,
                               textfont=dict(size=14),
                               hovertemplate="Value decline %{x}, CDR %{y}: %{text}<extra></extra>"))
    chart(style(fig, height=380, title=f"Class {cls} – {metric}", xaxis_title="Property value decline",
                          yaxis_title="Term CDR", xaxis_type="category", yaxis_type="category"),
width="stretch")

html('<div class="foot">ukcmbs≀ · illustrative model · all loans, curves and spreads are fictitious</div>')
