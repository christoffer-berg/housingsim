"""Pydantic models for scenario configuration and policy modules."""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Regions and tenure types
# ---------------------------------------------------------------------------

REGIONS = ["STHLM", "GBG", "MALMO", "REST"]
REGION_IDX = {r: i for i, r in enumerate(REGIONS)}
N_REGIONS = len(REGIONS)

TENURES = ["rent_regulated", "buy_bostadsratt", "buy_smallhouse"]
TENURE_IDX = {t: i for i, t in enumerate(TENURES)}
N_TENURES = len(TENURES)

N_QUALITY = 5  # quality tiers 1..5


# ---------------------------------------------------------------------------
# Time-stepped policy primitives (piecewise-constant paths)
# ---------------------------------------------------------------------------

class TimedValue(BaseModel):
    """A value that takes effect from a given YYYY-MM date string."""
    from_: str = Field(..., alias="from")
    value: float

    model_config = {"populate_by_name": True}


class TimedRegionValue(BaseModel):
    """A region-specific value that takes effect from a given YYYY-MM date."""
    from_: str = Field(..., alias="from")
    region: str
    value: float

    model_config = {"populate_by_name": True}


def resolve_value(path: list[TimedValue], t_str: str, default: float) -> float:
    """Return the most-recent value whose from_ <= t_str, else default."""
    result = default
    for tv in path:
        if tv.from_ <= t_str:
            result = tv.value
    return result


def resolve_region_values(
    path: list[TimedRegionValue],
    t_str: str,
    defaults: dict[str, float],
) -> dict[str, float]:
    """Return per-region dict of the most-recent applicable value."""
    result = dict(defaults)
    for tv in path:
        if tv.from_ <= t_str:
            result[tv.region] = tv.value
    return result


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class MacroConfig(BaseModel):
    base_policy_rate: float = 0.0175
    policy_rate_path: list[TimedValue] = Field(default_factory=list)
    mortgage_spread: float = 0.015  # spread over policy rate
    income_growth_annual: float = 0.025
    migration_growth_annual: float = 0.005

    def rate_at(self, t_str: str) -> float:
        return resolve_value(self.policy_rate_path, t_str, self.base_policy_rate)

    def mortgage_rate_at(self, t_str: str) -> float:
        return self.rate_at(t_str) + self.mortgage_spread


class CreditPolicyConfig(BaseModel):
    ltv_cap: list[TimedValue] = Field(
        default_factory=lambda: [TimedValue(**{"from": "2000-01", "value": 0.85})]
    )
    max_dsti: float = 0.40  # max debt service to income (stressed)
    stress_rate_buffer: float = 0.03  # add to current rate for stress test
    # amortization: simplified as annual fraction of outstanding debt
    amort_above_50_ltv: float = 0.02   # 2 %/yr when LTV > 50 %
    amort_above_70_ltv: float = 0.01   # extra 1 %/yr when LTV > 70 %

    def ltv_cap_at(self, t_str: str) -> float:
        return resolve_value(self.ltv_cap, t_str, 0.85)

    def amort_rate(self, ltv: float) -> float:
        rate = 0.0
        if ltv > 0.50:
            rate += self.amort_above_50_ltv
        if ltv > 0.70:
            rate += self.amort_above_70_ltv
        return rate


class TaxPolicyConfig(BaseModel):
    """
    Property (real-estate) tax levied annually as a fraction of property value.

    Sweden replaced progressive fastighetsskatt with a capped fastighetsavgift
    in 2008. A new policy might re-introduce a rate-based tax.

    property_tax_rate: annual fraction of market value (e.g. 0.01 = 1 %)
    exemption_years:   new builds exempt for this many years after completion
    """
    property_tax_rate: list[TimedValue] = Field(
        default_factory=lambda: [TimedValue(**{"from": "2000-01", "value": 0.0})]
    )
    brf_fee_annual: float = 30_000.0  # SEK/yr baseline HOA fee for BRF (replaces hard-coded)
    exemption_years: int = 0

    def tax_rate_at(self, t_str: str) -> float:
        return resolve_value(self.property_tax_rate, t_str, 0.0)

    def monthly_tax(self, property_value: float, t_str: str) -> float:
        return property_value * self.tax_rate_at(t_str) / 12


class ConstructionPolicyConfig(BaseModel):
    approval_delay_months: int = 10
    build_time_months: int = 18
    # baseline permits per month per region (units)
    base_permits: dict[str, int] = Field(
        default_factory=lambda: {
            "STHLM": 400,
            "GBG": 200,
            "MALMO": 150,
            "REST": 300,
        }
    )
    permits_multiplier: list[TimedRegionValue] = Field(default_factory=list)
    # split of new builds across tenure types
    tenure_split: dict[str, float] = Field(
        default_factory=lambda: {
            "rent_regulated": 0.30,
            "buy_bostadsratt": 0.55,
            "buy_smallhouse": 0.15,
        }
    )
    supply_elasticity: float = 0.20  # extra permits per 10 % price appreciation

    def permits_at(self, t_str: str, region: str) -> int:
        base = self.base_permits.get(region, 200)
        multipliers = resolve_region_values(
            self.permits_multiplier, t_str, {r: 1.0 for r in REGIONS}
        )
        return max(0, int(base * multipliers.get(region, 1.0)))


# ---------------------------------------------------------------------------
# Top-level scenario config
# ---------------------------------------------------------------------------

class TimeRange(BaseModel):
    start: str = "2024-01"
    end: str = "2030-12"


class ScenarioConfig(BaseModel):
    scenario_name: str = "baseline"
    time: TimeRange = Field(default_factory=TimeRange)
    n_households: int = 10_000
    random_seed: int = 42
    n_runs: int = 1

    macro: MacroConfig = Field(default_factory=MacroConfig)
    credit_policy: CreditPolicyConfig = Field(default_factory=CreditPolicyConfig)
    tax_policy: TaxPolicyConfig = Field(default_factory=TaxPolicyConfig)
    construction_policy: ConstructionPolicyConfig = Field(
        default_factory=ConstructionPolicyConfig
    )

    # Market price update sensitivity
    price_update_kappa: float = 0.10  # governs speed of price adjustment
    price_noise_std: float = 0.005    # monthly noise std on price index

    @model_validator(mode="after")
    def validate_time(self) -> "ScenarioConfig":
        if self.time.start >= self.time.end:
            raise ValueError("time.start must be before time.end")
        return self

    @classmethod
    def from_yaml(cls, path: str) -> "ScenarioConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


def months_between(start: str, end: str) -> list[str]:
    """Return list of YYYY-MM strings from start to end inclusive."""
    from datetime import date

    y, m = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    result = []
    while (y, m) <= (ey, em):
        result.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return result
