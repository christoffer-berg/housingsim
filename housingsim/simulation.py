"""
Main simulation engine.

Ties together all modules into a monthly loop.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import ScenarioConfig, months_between
from .state import RegionStock, build_initial_stocks
from .households import (
    HouseholdArrays,
    generate_households,
    apply_life_events,
    choose_actions,
    apply_outcomes,
)
from .bank import qualify_buyers
from .markets import match_and_transact, allocate_rentals
from .developers import ConstructionPipeline
from .metrics import MetricsRecorder
from .calibration import CalibrationData


def run_scenario(
    cfg: ScenarioConfig,
    seed_override: int | None = None,
    verbose: bool = True,
    calibration: CalibrationData | None = None,
) -> MetricsRecorder:
    """
    Run a single scenario with one random seed.

    Parameters
    ----------
    calibration : CalibrationData, optional
        When provided, uses real-world data (Riksbank/SCB) to initialise
        prices, population shares, income multipliers, and the rate path.
        Falls back to synthetic defaults where data is missing.

    Returns a MetricsRecorder containing all monthly KPIs.
    """
    seed = seed_override if seed_override is not None else cfg.random_seed
    rng = np.random.default_rng(seed)

    # --- Initialise state with real or synthetic data ---
    pop_shares = calibration.population_shares if calibration else None
    inc_mults = calibration.income_multipliers if calibration else None

    hh = generate_households(
        cfg.n_households, rng,
        population_shares=pop_shares,
        income_multipliers=inc_mults,
    )
    region_stocks = build_initial_stocks()

    if calibration is not None:
        calibration.apply_to_stocks(region_stocks)

        # Patch the rate path in credit-policy config if real data available
        real_path = calibration.rate_path_for(cfg.time.start, cfg.time.end)
        if real_path:
            from .config import TimedValue
            cfg = cfg.model_copy(deep=True)
            cfg.macro.policy_rate_path = [
                TimedValue(**{"from": d["from"], "value": d["value"]})
                for d in real_path
            ]

        # Patch construction permits if real data available
        real_permits = calibration.construction_permits
        if real_permits:
            cfg = cfg.model_copy(deep=True)
            cfg.construction_policy.base_permits.update(real_permits)

    pipeline = ConstructionPipeline()
    recorder = MetricsRecorder()

    months = months_between(cfg.time.start, cfg.time.end)
    n_months = len(months)

    if verbose:
        print(f"\n=== {cfg.scenario_name} | seed={seed} | {n_months} months ===")

    t0 = time.perf_counter()

    for step, t_str in enumerate(months):
        # ------------------------------------------------------------------
        # 1. Macro update
        # ------------------------------------------------------------------
        mortgage_rate = cfg.macro.mortgage_rate_at(t_str)
        income_growth_monthly = (1 + cfg.macro.income_growth_annual) ** (1 / 12) - 1

        # ------------------------------------------------------------------
        # 2. Life events
        # ------------------------------------------------------------------
        apply_life_events(hh, rng, income_growth_monthly)

        # ------------------------------------------------------------------
        # 3. Decision phase
        # ------------------------------------------------------------------
        choices = choose_actions(
            hh,
            rng=rng,
            mortgage_rate=mortgage_rate,
            region_stocks=region_stocks,
            credit_policy=cfg.credit_policy,
        )

        # Attach queue scores to rent applicants for allocation
        rent_app = choices["rent_applicants"]
        if len(rent_app["hh_idx"]) > 0:
            rent_app["queue_score"] = hh.rent_queue_score[rent_app["hh_idx"]]

        # ------------------------------------------------------------------
        # 4a. Rental allocation
        # ------------------------------------------------------------------
        rental_granted = allocate_rentals(
            choices["rent_applicants"],
            region_stocks=region_stocks,
            rng=rng,
        )

        # ------------------------------------------------------------------
        # 4b. Bank qualification
        # ------------------------------------------------------------------
        qualified_brf = qualify_buyers(
            hh=hh,
            buy_intents=choices["buy_brf"],
            t_str=t_str,
            mortgage_rate=mortgage_rate,
            credit_policy=cfg.credit_policy,
            region_stocks=region_stocks,
            tenure_idx=1,
            rng=rng,
        )

        qualified_small = qualify_buyers(
            hh=hh,
            buy_intents=choices["buy_small"],
            t_str=t_str,
            mortgage_rate=mortgage_rate,
            credit_policy=cfg.credit_policy,
            region_stocks=region_stocks,
            tenure_idx=2,
            rng=rng,
        )

        # ------------------------------------------------------------------
        # 4c. Ownership market matching + price update
        # ------------------------------------------------------------------
        tx_brf = match_and_transact(
            qualified=qualified_brf,
            region_stocks=region_stocks,
            tenure_idx=1,
            t_str=t_str,
            price_update_kappa=cfg.price_update_kappa,
            price_noise_std=cfg.price_noise_std,
            rng=rng,
        )

        tx_small = match_and_transact(
            qualified=qualified_small,
            region_stocks=region_stocks,
            tenure_idx=2,
            t_str=t_str,
            price_update_kappa=cfg.price_update_kappa,
            price_noise_std=cfg.price_noise_std,
            rng=rng,
        )

        # ------------------------------------------------------------------
        # 5. Apply outcomes to households
        # ------------------------------------------------------------------
        apply_outcomes(hh, rental_granted, tx_brf, tx_small)

        # ------------------------------------------------------------------
        # 6. Update mortgage balances for owners (amortization)
        # ------------------------------------------------------------------
        _amortize_mortgages(hh, mortgage_rate, cfg.credit_policy)

        # ------------------------------------------------------------------
        # 7. Construction: decide starts → advance pipeline
        # ------------------------------------------------------------------
        pipeline.decide_starts(t_str, region_stocks, cfg.construction_policy, rng)
        pipeline.advance_and_complete(region_stocks)

        # ------------------------------------------------------------------
        # 8. Record KPIs
        # ------------------------------------------------------------------
        recorder.record(
            t_str=t_str,
            hh=hh,
            region_stocks=region_stocks,
            tx_brf=tx_brf,
            tx_small=tx_small,
            rental_granted=rental_granted,
            pipeline=pipeline,
            macro_rate=mortgage_rate,
            shut_out_brf=qualified_brf.get("shut_out_count", 0.0),
            shut_out_small=qualified_small.get("shut_out_count", 0.0),
        )

        if verbose and (step % 12 == 0 or step == n_months - 1):
            elapsed = time.perf_counter() - t0
            print(
                f"  {t_str}  rate={mortgage_rate:.3f}"
                f"  ownership={recorder._records[-1]['homeownership_rate']:.3f}"
                f"  burden={recorder._records[-1]['median_housing_cost_burden']:.3f}"
                f"  [{elapsed:.1f}s]"
            )

    return recorder


def run_batch(
    cfg: ScenarioConfig,
    output_dir: str = "runs",
    verbose: bool = True,
    calibration: CalibrationData | None = None,
) -> list[MetricsRecorder]:
    """
    Run cfg.n_runs independent seeds; save parquet + metadata.
    Returns list of recorders (one per run).
    """
    import pandas as pd

    out_path = Path(output_dir) / cfg.scenario_name
    out_path.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_path / "config.json", "w") as f:
        json.dump(cfg.model_dump(), f, indent=2)

    recorders = []
    seeds = [cfg.random_seed + i for i in range(cfg.n_runs)]

    for run_idx, seed in enumerate(seeds):
        if verbose:
            print(f"\nRun {run_idx + 1}/{cfg.n_runs} (seed={seed})")
        recorder = run_scenario(
            cfg, seed_override=seed, verbose=verbose, calibration=calibration
        )
        fname = out_path / f"run_{run_idx:03d}.parquet"
        recorder.save_parquet(str(fname))
        recorders.append(recorder)

    # Also save a combined mean across runs
    if len(recorders) > 1:
        dfs = [r.to_dataframe() for r in recorders]
        numeric_cols = dfs[0].select_dtypes(include="number").columns.tolist()
        combined = dfs[0][["month"]].copy()
        for col in numeric_cols:
            vals = np.stack([df[col].values for df in dfs])
            combined[col + "_mean"] = vals.mean(axis=0)
            combined[col + "_std"] = vals.std(axis=0)
        combined.to_parquet(str(out_path / "combined.parquet"), index=False)
        if verbose:
            print(f"\n  Combined stats → {out_path / 'combined.parquet'}")

    return recorders


def _amortize_mortgages(
    hh: HouseholdArrays,
    mortgage_rate: float,
    credit_policy,
) -> None:
    """
    Update mortgage balances and housing costs for owners.
    Monthly amortization + interest adjustment.
    """
    owner_mask = (hh.tenure > 0) & (hh.mortgage_balance > 0)
    if not owner_mask.any():
        return

    mb = hh.mortgage_balance[owner_mask]
    hv = np.maximum(hh.home_value[owner_mask], 1.0)
    ltv = np.clip(mb / hv, 0.0, 1.0)

    amort_rates = np.array([credit_policy.amort_rate(l) for l in ltv])
    monthly_amort = mb * amort_rates / 12

    # Interest payment
    monthly_interest = mb * (mortgage_rate / 12)

    # New balance
    new_mb = np.maximum(mb - monthly_amort, 0.0)
    hh.mortgage_balance[owner_mask] = new_mb.astype(np.float32)

    # Update housing cost (interest + amortization + rough HOA fee for BRF)
    hoa_fee = np.where(hh.tenure[owner_mask] == 1, 2_500.0, 0.0)  # BRF monthly fee
    hh.housing_cost[owner_mask] = (
        monthly_interest + monthly_amort + hoa_fee
    ).astype(np.float32)
