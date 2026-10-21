import numpy as np
import pytest

from cmbs_pricer import Deal, Scenario, analyse, generate_loan_pool, project_pool
from cmbs_pricer.loans import protection_status


@pytest.fixture(scope="module")
def pool():
    return generate_loan_pool(75, seed=7)


def test_pool_shape(pool):
    assert len(pool) == 75
    assert pool.loan_id.is_unique
    assert pool.ltv.between(0.39, 0.76).all()
    assert set(pool.rate_type) == {"Fixed", "Floating"}


def test_no_prepayment_in_lockout_or_defeasance(pool):
    pcf = project_pool(pool, Scenario(cpr=0.5))
    blocked = np.isin(pcf.state, [0, 2])
    assert np.allclose(pcf.prepayment[blocked], 0.0)


def test_make_whole_fees_only_when_protected(pool):
    pcf = project_pool(pool, Scenario(cpr=0.3))
    open_or_locked = np.isin(pcf.state, [0, 2, 4, 5])
    assert np.allclose(pcf.penalty[open_or_locked], 0.0)
    assert pcf.penalty.sum() > 0


@pytest.mark.parametrize("sc", [
    Scenario(),
    Scenario(cdr=0.03, value_decline=0.45, refi_ltv=0.6),
    Scenario(cpr=0.4, rate_shift_bp=200),
])
@pytest.mark.parametrize("mode", ["sequential", "pro_rata"])
def test_collateral_reconciles_to_notes(pool, sc, mode):
    pcf = project_pool(pool, sc)
    s = pcf.summary()
    collateral = pool.balance.sum()
    # every pound of collateral is either repaid, recovered or lost
    assert np.isclose(s.principal.sum() + s.loss.sum(), collateral)
    res = analyse(pool, Deal(principal_mode=mode), sc, effective_duration=False)
    wf = res.waterfall
    assert np.isclose(wf.principal.sum() + wf.writedown.sum(), collateral)
    assert np.isclose(wf.writedown.sum() + wf.loss_cured.sum(), s.loss.sum())


def test_losses_hit_junior_first(pool):
    res = analyse(pool, scenario=Scenario(cdr=0.03, value_decline=0.45, refi_ltv=0.6),
                  effective_duration=False)
    loss = res.pricing.set_index("Class")["Loss %"]
    assert loss["R"] >= loss["F"] >= loss["E"] >= loss["A"]
    assert loss["A"] == 0

def test_base_case_seniors_near_par(pool):
    p = analyse(pool).pricing.set_index("Class")
    for c in "ABCDE":
        assert 99 < p.loc[c, "Price"] < 101
        assert abs(p.loc[c, "Eff Dur"]) < 0.5 # floaters: little rate duration
        assert p.loc[c, "Spread Dur"] > 1
