"""UK CMBS pricer: dummy loan pool -> cash flows -> waterfall -> tranche pricing."""
from .cashflows import Scenario, project_pool
from .curves import CMBS_SPREADS, GILT, SONIA, SpreadCurve, ZeroCurve
from .loans import generate_loan_pool
from .model import DealResult, analyse
from .waterfall import Deal, Tranche, run_waterfall

__all__ = [
    "Scenario", "project_pool", "CMBS_SPREADS", "GILT", "SONIA", "SpreadCurve",
    "ZeroCurve", "generate_loan_pool", "DealResult", "analyse", "Deal", "Tranche",
    "run_waterfall",
]
