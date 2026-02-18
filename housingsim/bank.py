"""
Bank underwriting module.

Applies Swedish-style credit policy (LTV cap, DSTI limit, amortization rules)
to buy intents and returns qualified buyers.
"""

from __future__ import annotations

import numpy as np

from .config import CreditPolicyConfig, TaxPolicyConfig
from .households import HouseholdArrays, _monthly_mortgage_payment
from .state import RegionStock


def qualify_buyers(
    hh: HouseholdArrays,
    buy_intents: dict,
    t_str: str,
    mortgage_rate: float,
    credit_policy: CreditPolicyConfig,
    region_stocks: list[RegionStock],
    tenure_idx: int,
    rng: np.random.Generator,
    tax_policy: TaxPolicyConfig | None = None,
) -> dict:
    """
    Run bank underwriting on buy intents.

    Parameters
    ----------
    buy_intents : dict with keys hh_idx, target_region, target_quality, weight
    tenure_idx  : 1=brf, 2=smallhouse
    t_str       : current time string YYYY-MM for policy lookup
    tax_policy  : optional property tax; monthly tax added to stressed payment

    Returns
    -------
    dict with same structure as buy_intents but filtered to qualified buyers,
    plus 'transaction_value' and 'mortgage_balance' arrays.
    """
    idx = buy_intents["hh_idx"]
    if len(idx) == 0:
        return _empty_qualified()

    target_region = buy_intents["target_region"]
    target_quality = buy_intents["target_quality"]
    weight = buy_intents["weight"]
    target_payment = buy_intents["target_payment"]

    ltv_cap = credit_policy.ltv_cap_at(t_str)
    stress_rate = mortgage_rate + credit_policy.stress_rate_buffer
    max_dsti = credit_policy.max_dsti
    tax_rate = tax_policy.tax_rate_at(t_str) if tax_policy else 0.0

    # Get property values for each intent
    property_values = np.array([
        region_stocks[target_region[i]].effective_price(tenure_idx, target_quality[i])
        for i in range(len(idx))
    ], dtype=np.float64)

    # Required downpayment
    min_downpayment = property_values * (1 - ltv_cap)

    # Monthly property tax (included in DSTI stress test)
    monthly_tax = property_values * tax_rate / 12

    # Stressed monthly payment (mortgage + property tax)
    stressed_mortgage = np.array([
        _monthly_mortgage_payment(
            property_values[i], stress_rate, ltv=ltv_cap,
            amort_rate=credit_policy.amort_rate(ltv_cap)
        )
        for i in range(len(idx))
    ], dtype=np.float64)
    stressed_payment = stressed_mortgage + monthly_tax

    income = hh.income_monthly[idx].astype(np.float64)
    wealth = hh.wealth_liquid[idx].astype(np.float64)

    # Feasibility checks
    has_downpayment = wealth >= min_downpayment
    dsti_ok = (stressed_payment / np.maximum(income, 1.0)) <= max_dsti

    qualified_mask = has_downpayment & dsti_ok

    if not qualified_mask.any():
        return _empty_qualified()

    q = qualified_mask
    actual_ltv = np.minimum(
        (property_values[q] - np.minimum(wealth[q], property_values[q])) / property_values[q],
        ltv_cap,
    )
    actual_ltv = np.clip(actual_ltv, 0.0, ltv_cap)

    actual_mortgage = property_values[q] * actual_ltv
    actual_amort_rate = np.array([
        credit_policy.amort_rate(ltv) for ltv in actual_ltv
    ])
    actual_mortgage_payment = np.array([
        _monthly_mortgage_payment(
            property_values[q][i], mortgage_rate, ltv=actual_ltv[i],
            amort_rate=actual_amort_rate[i]
        )
        for i in range(q.sum())
    ], dtype=np.float32)

    # Total payment includes actual mortgage + property tax
    actual_payment = actual_mortgage_payment + (property_values[q] * tax_rate / 12).astype(np.float32)

    return {
        "hh_idx": idx[q],
        "target_region": target_region[q],
        "target_quality": target_quality[q],
        "target_payment": actual_payment,
        "weight": weight[q],
        "transaction_value": property_values[q],
        "mortgage_balance": actual_mortgage,
        "shut_out_count": float((~qualified_mask * weight).sum()),
        "shut_out_no_deposit": float((~has_downpayment * weight).sum()),
        "shut_out_dsti": float(((~dsti_ok) & has_downpayment).astype(float) @ weight),
    }


def _empty_qualified() -> dict:
    return {
        "hh_idx": np.array([], dtype=int),
        "target_region": np.array([], dtype=int),
        "target_quality": np.array([], dtype=int),
        "target_payment": np.array([], dtype=np.float32),
        "weight": np.array([], dtype=np.float64),
        "transaction_value": np.array([], dtype=np.float64),
        "mortgage_balance": np.array([], dtype=np.float64),
        "shut_out_count": 0.0,
        "shut_out_no_deposit": 0.0,
        "shut_out_dsti": 0.0,
    }
