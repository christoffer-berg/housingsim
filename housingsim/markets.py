"""
Market allocation modules.

OwnershipMarket: matches qualified buyers to available supply,
                 then updates price indices.

RentalMarket:    probabilistic queue allocation, updates queue pressure.
"""

from __future__ import annotations

import numpy as np

from .state import RegionStock


# ---------------------------------------------------------------------------
# Ownership market
# ---------------------------------------------------------------------------

def match_and_transact(
    qualified: dict,
    region_stocks: list[RegionStock],
    tenure_idx: int,
    t_str: str,
    price_update_kappa: float,
    price_noise_std: float,
    rng: np.random.Generator,
) -> dict:
    """
    Match qualified buyers against available supply and execute transactions.

    For each region:
      - Supply = turnover_listings + new completions (already added to stock)
      - Demand = weighted count of qualified buyers in that region
      - Execute min(demand, supply) transactions
      - Update price index based on excess demand/supply

    Returns dict of executed transactions (same structure as qualified, but
    filtered to those that actually transacted).
    """
    idx = qualified["hh_idx"]
    if len(idx) == 0:
        return _empty_tx()

    target_region = qualified["target_region"]
    target_quality = qualified["target_quality"]
    weight = qualified["weight"]
    tx_value = qualified.get("transaction_value", np.zeros(len(idx)))
    mortgage = qualified.get("mortgage_balance", np.zeros(len(idx)))
    target_payment = qualified["target_payment"]

    transacted = np.zeros(len(idx), dtype=bool)

    for r in range(len(region_stocks)):
        stock = region_stocks[r]
        region_mask = target_region == r
        if not region_mask.any():
            # No buyers in this region; still update price (supply > demand)
            supply = stock.monthly_vacancies(tenure_idx)
            stock.update_ownership_price(
                tenure_idx,
                demand_weighted=0.0,
                supply_available=supply,
                kappa=price_update_kappa,
                noise_std=price_noise_std,
                rng=rng,
            )
            continue

        # Demand in this region (weighted)
        demand_weighted = float(weight[region_mask].sum())

        # Supply: monthly listings (turnover)
        supply = stock.monthly_vacancies(tenure_idx)

        # Execute transactions: allocate min(demand, supply)
        # Sort buyers by quality preference (simple: random allocation)
        region_buyers = np.where(region_mask)[0]
        if supply >= demand_weighted:
            # Everyone gets served
            transacted[region_buyers] = True
        else:
            # Partial fill: randomly select buyers proportional to weight
            fill_rate = supply / max(demand_weighted, 1.0)
            fill_rate = min(fill_rate, 1.0)
            filled = rng.random(len(region_buyers)) < fill_rate
            transacted[region_buyers[filled]] = True

        # Price update based on total region demand vs supply (across all quality tiers)
        stock.update_ownership_price(
            tenure_idx,
            demand_weighted=demand_weighted,
            supply_available=supply,
            kappa=price_update_kappa,
            noise_std=price_noise_std,
            rng=rng,
        )

    t = transacted
    return {
        "hh_idx": idx[t],
        "target_region": target_region[t],
        "target_quality": target_quality[t],
        "target_payment": target_payment[t],
        "weight": weight[t],
        "transaction_value": tx_value[t],
        "mortgage_balance": mortgage[t],
        "volume_weighted": float(weight[t].sum()),
        "volume_count": int(t.sum()),
    }


def _empty_tx() -> dict:
    return {
        "hh_idx": np.array([], dtype=int),
        "target_region": np.array([], dtype=int),
        "target_quality": np.array([], dtype=int),
        "target_payment": np.array([], dtype=np.float32),
        "weight": np.array([], dtype=np.float64),
        "transaction_value": np.array([], dtype=np.float64),
        "mortgage_balance": np.array([], dtype=np.float64),
        "volume_weighted": 0.0,
        "volume_count": 0,
    }


# ---------------------------------------------------------------------------
# Rental market
# ---------------------------------------------------------------------------

def allocate_rentals(
    rent_applicants: dict,
    region_stocks: list[RegionStock],
    rng: np.random.Generator,
) -> dict:
    """
    Probabilistic queue-based rental allocation.

    Allocation probability per applicant is:
      p_alloc = base_rate * (queue_score / mean_queue_score) / queue_pressure

    Returns dict of granted allocations.
    """
    idx = rent_applicants["hh_idx"]
    if len(idx) == 0:
        return _empty_rental()

    target_region = rent_applicants["target_region"]
    target_quality = rent_applicants["target_quality"]
    weight = rent_applicants["weight"]
    target_payment = rent_applicants["target_payment"]

    granted = np.zeros(len(idx), dtype=bool)

    for r in range(len(region_stocks)):
        stock = region_stocks[r]
        region_mask = target_region == r
        if not region_mask.any():
            # No applicants; queue pressure decays slightly
            stock.update_queue_pressure(
                applicants_weighted=0.0,
                vacancies=stock.monthly_vacancies(0),
            )
            continue

        applicants_weighted = float(weight[region_mask].sum())
        vacancies = stock.monthly_vacancies(0)

        # Update queue pressure
        stock.update_queue_pressure(applicants_weighted, vacancies)

        # Allocation probability: base = vacancies/applicants, boosted by queue score
        # We don't have per-agent queue scores here (only the stock arrays have
        # the household data), so we compute a region-level fill rate and then
        # scale per-applicant probability by a random draw weighted by queue score.
        # (queue_score is stored in HouseholdArrays; we receive it via applicants dict
        # if populated by choose_actions.)
        base_fill = min(1.0, vacancies / max(applicants_weighted, 1.0))

        region_buyers_local = np.where(region_mask)[0]
        queue_scores = rent_applicants.get("queue_score", np.ones(len(idx)))
        q_scores = queue_scores[region_buyers_local]
        mean_q = max(float(q_scores.mean()), 0.01)
        # Individual probability: base * (own_score / mean)
        indiv_prob = np.clip(base_fill * (q_scores / mean_q), 0.0, 1.0)

        filled = rng.random(len(region_buyers_local)) < indiv_prob
        granted[region_buyers_local[filled]] = True

    g = granted
    return {
        "hh_idx": idx[g],
        "target_region": target_region[g],
        "target_quality": target_quality[g],
        "target_payment": target_payment[g],
        "weight": weight[g],
        "volume_weighted": float(weight[g].sum()),
    }


def _empty_rental() -> dict:
    return {
        "hh_idx": np.array([], dtype=int),
        "target_region": np.array([], dtype=int),
        "target_quality": np.array([], dtype=int),
        "target_payment": np.array([], dtype=np.float32),
        "weight": np.array([], dtype=np.float64),
        "volume_weighted": 0.0,
    }
