"""
Household agents stored as struct-of-arrays for performance.

All arrays have length N (number of synthetic households).
Indices:
  regions:  0=STHLM, 1=GBG, 2=MALMO, 3=REST
  tenures:  0=rent_regulated, 1=buy_bostadsratt, 2=buy_smallhouse
  quality:  0..4  (represents tiers 1..5)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import N_REGIONS, N_QUALITY, REGIONS


# ---------------------------------------------------------------------------
# Synthetic population calibration parameters
# ---------------------------------------------------------------------------

# Regional population shares (roughly Swedish reality)
REGION_POP_SHARES = np.array([0.25, 0.14, 0.09, 0.52])

# Initial tenure distribution per region (rent, brf, smallhouse)
INITIAL_TENURE_DIST = {
    0: [0.35, 0.45, 0.20],   # STHLM: more brf
    1: [0.38, 0.35, 0.27],   # GBG
    2: [0.40, 0.30, 0.30],   # MALMO
    3: [0.30, 0.25, 0.45],   # REST: more smallhouse
}

# Monthly income distribution params (log-normal, SEK)
# Median ~35k SEK/month, std roughly 15k
INCOME_LOG_MEAN = np.log(35_000)
INCOME_LOG_STD = 0.45

# Wealth: log-normal, highly skewed; median ~150k SEK liquid
WEALTH_LOG_MEAN = np.log(150_000)
WEALTH_LOG_STD = 1.2


# ---------------------------------------------------------------------------
# Household arrays dataclass
# ---------------------------------------------------------------------------

@dataclass
class HouseholdArrays:
    """
    Struct-of-arrays representation of all household agents.
    N = number of agents. weights sum to approx. total real households.
    """

    N: int

    # Identity & demography
    ids: np.ndarray          # int64 (N,)
    weights: np.ndarray      # float64 (N,) — each represents this many real hh
    region: np.ndarray       # int8  (N,) — 0..3
    age_head: np.ndarray     # float32 (N,)
    hh_size: np.ndarray      # int8 (N,)

    # Finances
    income_monthly: np.ndarray   # float32 (N,)
    wealth_liquid: np.ndarray    # float32 (N,)

    # Housing status
    tenure: np.ndarray           # int8 (N,) — 0=rent,1=brf,2=small
    quality: np.ndarray          # int8 (N,) — 0..4
    housing_cost: np.ndarray     # float32 (N,) — monthly housing payment
    mortgage_balance: np.ndarray # float32 (N,) — outstanding mortgage (owners)
    home_value: np.ndarray       # float32 (N,) — estimated current value

    # Rental queue
    rent_queue_score: np.ndarray  # float32 (N,)

    # Preferences (all in [0,1])
    pref_region: np.ndarray       # float32 (N, N_REGIONS) — preference weight
    pref_space: np.ndarray        # float32 (N,)
    pref_stability: np.ndarray    # float32 (N,)
    moving_cost_sensitivity: np.ndarray  # float32 (N,)

    # Move propensity (updated each step by life events)
    move_propensity: np.ndarray   # float32 (N,) baseline monthly move prob


def generate_households(n: int, rng: np.random.Generator) -> HouseholdArrays:
    """Generate N synthetic households with calibrated Swedish-like distributions."""

    # Regions: sample proportional to population shares
    region = rng.choice(N_REGIONS, size=n, p=REGION_POP_SHARES).astype(np.int8)

    # Age: uniform 22-80
    age_head = rng.uniform(22, 80, size=n).astype(np.float32)

    # Household size: 1-5 (weighted)
    hh_size = rng.choice([1, 2, 3, 4, 5], size=n,
                         p=[0.30, 0.35, 0.18, 0.12, 0.05]).astype(np.int8)

    # Income: log-normal
    income_monthly = rng.lognormal(INCOME_LOG_MEAN, INCOME_LOG_STD, size=n).astype(np.float32)
    # Scale income with region (STHLM highest)
    region_income_mult = np.array([1.25, 1.10, 1.00, 0.92])
    income_monthly = income_monthly * region_income_mult[region]

    # Wealth: log-normal; owners have more wealth (assigned post-tenure)
    wealth_liquid = rng.lognormal(WEALTH_LOG_MEAN, WEALTH_LOG_STD, size=n).astype(np.float32)

    # Tenure: sample from region-specific distribution
    tenure = np.empty(n, dtype=np.int8)
    for r in range(N_REGIONS):
        mask = region == r
        cnt = mask.sum()
        if cnt > 0:
            tenure[mask] = rng.choice(3, size=cnt, p=INITIAL_TENURE_DIST[r])

    # Quality: roughly bell-shaped, slightly correlated with income
    # Higher income → higher quality
    income_percentile = (income_monthly.argsort().argsort() / n)  # 0..1
    raw_quality = rng.normal(0.5 + 0.8 * (income_percentile - 0.5), 0.3, size=n)
    quality = np.clip((raw_quality * 5).astype(int), 0, 4).astype(np.int8)

    # Preferences
    pref_region = np.zeros((n, N_REGIONS), dtype=np.float32)
    for i in range(n):
        # Strong preference for own region, some for adjacent
        base = rng.dirichlet(np.ones(N_REGIONS) * 0.5)
        base[region[i]] += 2.0
        base /= base.sum()
        pref_region[i] = base

    pref_space = np.clip(
        rng.normal(0.5, 0.2, size=n) + 0.1 * hh_size, 0.1, 1.0
    ).astype(np.float32)

    pref_stability = rng.beta(2, 2, size=n).astype(np.float32)
    # Older households prefer stability
    pref_stability = np.clip(pref_stability + 0.01 * (age_head - 40), 0, 1).astype(np.float32)

    moving_cost_sensitivity = rng.beta(2, 3, size=n).astype(np.float32)

    # Move propensity: base monthly probability of considering a move
    # Younger → higher; renters → higher
    base_propensity = 0.015 + 0.005 * (age_head < 35) - 0.003 * (tenure > 0)
    move_propensity = np.clip(
        rng.normal(base_propensity, 0.005, size=n), 0.002, 0.15
    ).astype(np.float32)

    # Rental queue score: renters who have been in queue longer → higher score
    rent_queue_score = np.zeros(n, dtype=np.float32)
    renter_mask = tenure == 0
    # Renters start with a score reflecting time already in system (5-20 years proxy)
    rent_queue_score[renter_mask] = rng.gamma(3, 4, size=renter_mask.sum()).astype(np.float32)

    # Housing cost and mortgage
    housing_cost = np.zeros(n, dtype=np.float32)
    mortgage_balance = np.zeros(n, dtype=np.float32)
    home_value = np.zeros(n, dtype=np.float32)

    # Renters: ~25-35% of income
    renter_mask = tenure == 0
    housing_cost[renter_mask] = (
        income_monthly[renter_mask] * rng.uniform(0.20, 0.35, size=renter_mask.sum())
    ).astype(np.float32)

    # Owners: set rough home values and mortgage costs
    owner_mask = tenure > 0
    # Home value = 4-8x annual income, with regional multiplier
    region_price_mult = np.array([2.0, 1.4, 1.1, 0.9])
    hv_mult = rng.uniform(4, 8, size=owner_mask.sum()) * region_price_mult[region[owner_mask]] / 2
    home_value[owner_mask] = (income_monthly[owner_mask] * 12 * hv_mult).astype(np.float32)

    # LTV roughly 40-75% for existing owners
    ltv = rng.uniform(0.30, 0.75, size=owner_mask.sum())
    mortgage_balance[owner_mask] = (home_value[owner_mask] * ltv).astype(np.float32)

    # Monthly mortgage cost: interest + amortization
    annual_rate = 0.035  # approximate
    monthly_cost = mortgage_balance[owner_mask] * (annual_rate / 12 + 0.02 / 12)
    housing_cost[owner_mask] = monthly_cost.astype(np.float32)

    # Weights: each household represents approx. 4.5M real households / n
    total_real_households = 4_500_000
    weights = np.full(n, total_real_households / n, dtype=np.float64)

    return HouseholdArrays(
        N=n,
        ids=np.arange(n, dtype=np.int64),
        weights=weights,
        region=region,
        age_head=age_head,
        hh_size=hh_size,
        income_monthly=income_monthly,
        wealth_liquid=wealth_liquid,
        tenure=tenure,
        quality=quality,
        housing_cost=housing_cost,
        mortgage_balance=mortgage_balance,
        home_value=home_value,
        rent_queue_score=rent_queue_score,
        pref_region=pref_region,
        pref_space=pref_space,
        pref_stability=pref_stability,
        moving_cost_sensitivity=moving_cost_sensitivity,
        move_propensity=move_propensity,
    )


# ---------------------------------------------------------------------------
# Life events
# ---------------------------------------------------------------------------

def apply_life_events(
    hh: HouseholdArrays,
    rng: np.random.Generator,
    income_growth_monthly: float,
) -> None:
    """
    Apply stochastic life events and global income growth in-place.

    Events:
    - Income growth (deterministic global factor + noise)
    - Job change (income shock)
    - New child (household size +1, move propensity boost)
    - Separation (size -1, income -30%, move propensity spike)
    - Aging (age +1/12)
    - Queue score accumulation for renters
    """
    n = hh.N

    # Global income growth
    hh.income_monthly *= (1 + income_growth_monthly)

    # Job change: ~1% of households per month get an income shock
    job_change = rng.random(n) < 0.01
    if job_change.any():
        shocks = rng.normal(0, 0.10, size=job_change.sum())
        hh.income_monthly[job_change] *= (1 + shocks)
        hh.income_monthly[job_change] = np.maximum(hh.income_monthly[job_change], 8_000)
        hh.move_propensity[job_change] = np.minimum(
            hh.move_propensity[job_change] * 1.5, 0.15
        )

    # New child: ~0.4% per month
    new_child = rng.random(n) < 0.004
    if new_child.any():
        hh.hh_size[new_child] = np.minimum(hh.hh_size[new_child] + 1, 6).astype(np.int8)
        hh.pref_space[new_child] = np.minimum(hh.pref_space[new_child] + 0.10, 1.0)
        hh.move_propensity[new_child] = np.minimum(
            hh.move_propensity[new_child] * 1.8, 0.15
        )

    # Separation: ~0.3% per month
    separation = rng.random(n) < 0.003
    if separation.any():
        hh.hh_size[separation] = np.maximum(hh.hh_size[separation] - 1, 1).astype(np.int8)
        hh.income_monthly[separation] *= rng.uniform(0.65, 0.85, size=separation.sum())
        hh.move_propensity[separation] = np.minimum(
            hh.move_propensity[separation] * 2.5, 0.15
        )

    # Age everyone
    hh.age_head = (hh.age_head + 1 / 12).astype(np.float32)

    # Queue score accumulates for renters
    renter_mask = hh.tenure == 0
    hh.rent_queue_score[renter_mask] += rng.exponential(0.08, size=renter_mask.sum())

    # Decay move propensity back toward baseline (~0.5% reversion per month)
    hh.move_propensity = np.clip(
        hh.move_propensity * 0.98 + 0.015 * 0.02, 0.002, 0.15
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Decision phase
# ---------------------------------------------------------------------------

TENURE_STABILITY = np.array([0.3, 0.7, 0.8])   # stability score per tenure
TENURE_SPACE_MULT = np.array([0.6, 0.8, 1.0])   # space score per tenure


def compute_utilities(
    hh: HouseholdArrays,
    target_region: int,
    target_tenure: int,
    target_quality: int,
    target_payment: float,
    mortgage_rate: float,
) -> np.ndarray:
    """
    Compute utility for all households considering a specific (region, tenure, quality)
    option.  Returns shape (N,) utility scores.
    """
    n = hh.N
    income = hh.income_monthly
    TARGET_BURDEN = 0.30

    # Cost burden penalty
    burden = target_payment / np.maximum(income, 1.0)
    cost_penalty = 2.0 * np.maximum(0, burden - TARGET_BURDEN)

    # Space value: preference × quality fraction
    space_val = hh.pref_space * (target_quality + 1) / N_QUALITY

    # Stability value
    stab_val = hh.pref_stability * TENURE_STABILITY[target_tenure]

    # Regional preference
    region_val = hh.pref_region[:, target_region]

    # Moving cost (only penalises if moving = different region or tenure)
    is_move = (hh.region != target_region) | (hh.tenure != target_tenure)
    move_penalty = hh.moving_cost_sensitivity * 0.3 * is_move

    # Risk penalty: rises with high debt / low wealth
    owner = target_tenure > 0
    if owner:
        debt_proxy = target_payment * 12 * 20  # rough loan amount
        risk_penalty = 0.3 * np.maximum(0, debt_proxy / np.maximum(income * 12, 1) - 4)
    else:
        risk_penalty = np.zeros(n, dtype=np.float32)

    utility = (
        region_val
        + space_val
        + stab_val
        - cost_penalty
        - move_penalty
        - risk_penalty
    )
    return utility.astype(np.float32)


def choose_actions(
    hh: HouseholdArrays,
    rng: np.random.Generator,
    mortgage_rate: float,
    region_stocks: list,  # list[RegionStock]
    credit_policy: "CreditPolicyConfig",  # type ignore
) -> dict:
    """
    For each household, decide an action: stay / attempt_rent / attempt_buy_brf / attempt_buy_small.

    Returns dict with keys:
      'stay': boolean mask
      'rent_applicants': structured array of (hh_idx, target_region, quality, weight)
      'buy_brf': structured array of (hh_idx, target_region, quality, weight, price)
      'buy_small': structured array of (hh_idx, target_region, quality, weight, price)
    """
    n = hh.N

    # Step 1: decide whether to even consider moving
    consider_move = rng.random(n) < hh.move_propensity

    # Step 2: for movers, find best option across all regions & quality tiers
    best_action = np.zeros(n, dtype=np.int8)   # 0=stay, 1=rent, 2=brf, 3=small
    best_region = hh.region.copy()
    best_quality = hh.quality.copy()
    best_payment = np.zeros(n, dtype=np.float32)
    best_utility = np.full(n, -np.inf, dtype=np.float32)

    # First compute "stay" utility (current situation)
    stay_util = compute_utilities(
        hh,
        target_region=0,  # placeholder; overridden below for stay option
        target_tenure=0,
        target_quality=0,
        target_payment=0.0,
        mortgage_rate=mortgage_rate,
    )
    # For stay, use actual current payment and region/tenure
    stay_burden = hh.housing_cost / np.maximum(hh.income_monthly, 1.0)
    stay_cost_penalty = 2.0 * np.maximum(0, stay_burden - 0.30)
    # Simplified stay utility
    stay_util = (
        hh.pref_region[np.arange(n), hh.region]
        + hh.pref_space * (hh.quality + 1) / N_QUALITY
        + hh.pref_stability * TENURE_STABILITY[hh.tenure]
        - stay_cost_penalty
    )
    best_utility = stay_util.copy()

    # Evaluate move options only for households considering a move
    movers = np.where(consider_move)[0]
    if len(movers) > 0:
        for r in range(N_REGIONS):
            for q in range(N_QUALITY):
                stock = region_stocks[r]

                # --- Rent option ---
                # Regulated rent: approx. income fraction
                # Rent cost: roughly proportional to rent_index and quality
                rent_cost = 5_000 * (1 + 0.3 * q) * stock.rent_index
                rent_util = compute_utilities(
                    hh,
                    target_region=r,
                    target_tenure=0,
                    target_quality=q,
                    target_payment=rent_cost,
                    mortgage_rate=mortgage_rate,
                )
                improve_rent = (
                    consider_move
                    & (rent_util > best_utility)
                )
                best_utility[improve_rent] = rent_util[improve_rent]
                best_action[improve_rent] = 1
                best_region[improve_rent] = r
                best_quality[improve_rent] = q
                best_payment[improve_rent] = rent_cost

                # --- Buy BRF ---
                brf_price = stock.effective_price(1, q)
                monthly_pay_brf = _monthly_mortgage_payment(
                    brf_price, mortgage_rate, ltv=min(credit_policy.ltv_cap_at("9999-99"), 0.85)
                )
                brf_util = compute_utilities(
                    hh,
                    target_region=r,
                    target_tenure=1,
                    target_quality=q,
                    target_payment=monthly_pay_brf,
                    mortgage_rate=mortgage_rate,
                )
                improve_brf = consider_move & (brf_util > best_utility)
                best_utility[improve_brf] = brf_util[improve_brf]
                best_action[improve_brf] = 2
                best_region[improve_brf] = r
                best_quality[improve_brf] = q
                best_payment[improve_brf] = monthly_pay_brf

                # --- Buy Small House ---
                sh_price = stock.effective_price(2, q)
                monthly_pay_sh = _monthly_mortgage_payment(
                    sh_price, mortgage_rate, ltv=min(credit_policy.ltv_cap_at("9999-99"), 0.85)
                )
                sh_util = compute_utilities(
                    hh,
                    target_region=r,
                    target_tenure=2,
                    target_quality=q,
                    target_payment=monthly_pay_sh,
                    mortgage_rate=mortgage_rate,
                )
                improve_sh = consider_move & (sh_util > best_utility)
                best_utility[improve_sh] = sh_util[improve_sh]
                best_action[improve_sh] = 3
                best_region[improve_sh] = r
                best_quality[improve_sh] = q
                best_payment[improve_sh] = monthly_pay_sh

    # Build output arrays
    stay_mask = best_action == 0
    rent_mask = best_action == 1
    brf_mask = best_action == 2
    small_mask = best_action == 3

    def _to_intent(mask, tenure_idx):
        idx = np.where(mask)[0]
        return {
            "hh_idx": idx,
            "target_region": best_region[idx],
            "target_quality": best_quality[idx],
            "target_payment": best_payment[idx],
            "weight": hh.weights[idx],
        }

    return {
        "stay": stay_mask,
        "rent_applicants": _to_intent(rent_mask, 0),
        "buy_brf": _to_intent(brf_mask, 1),
        "buy_small": _to_intent(small_mask, 2),
    }


def _monthly_mortgage_payment(
    property_value: float,
    annual_rate: float,
    ltv: float,
    amort_rate: float = 0.02,
    loan_years: int = 30,
) -> float:
    """
    Simple monthly mortgage payment (interest + amortization).
    Uses annuity formula for interest, linear amortization.
    """
    principal = property_value * ltv
    if principal <= 0:
        return 0.0
    monthly_rate = annual_rate / 12
    if monthly_rate > 0:
        n_months = loan_years * 12
        payment = principal * monthly_rate / (1 - (1 + monthly_rate) ** (-n_months))
    else:
        payment = principal / (loan_years * 12)
    # Add monthly amortization fraction on top (Swedish amortization rules)
    payment += principal * amort_rate / 12
    return float(payment)


def apply_outcomes(
    hh: HouseholdArrays,
    rental_granted: dict,
    tx_brf: dict,
    tx_small: dict,
) -> None:
    """Update household state based on market outcomes."""

    for outcome, new_tenure in [
        (rental_granted, 0),
        (tx_brf, 1),
        (tx_small, 2),
    ]:
        idx = outcome.get("hh_idx", np.array([], dtype=int))
        if len(idx) == 0:
            continue

        hh.tenure[idx] = new_tenure
        hh.region[idx] = outcome["target_region"]
        hh.quality[idx] = outcome["target_quality"]
        hh.housing_cost[idx] = outcome["target_payment"]

        if new_tenure > 0:
            # Owners: update home value and mortgage balance
            hh.home_value[idx] = outcome.get("transaction_value", hh.home_value[idx])
            hh.mortgage_balance[idx] = outcome.get("mortgage_balance", hh.mortgage_balance[idx])
            # Reset queue score (they've left the rental queue)
            hh.rent_queue_score[idx] = 0.0
        else:
            # New renters: reset queue score to small baseline
            hh.rent_queue_score[idx] = 0.5
            hh.mortgage_balance[idx] = 0.0
            hh.home_value[idx] = 0.0

        # Reduce move propensity after a successful move
        hh.move_propensity[idx] = np.maximum(hh.move_propensity[idx] * 0.3, 0.002)
