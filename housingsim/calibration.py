"""
Calibration module.

Reads cached real-world data from data/ and converts it into:
  1. A calibrated ScenarioConfig macro rate path (Riksbank repo rate)
  2. Calibrated RegionStock initial conditions (prices, stock sizes)
  3. Calibrated household income/population weights

Falls back to synthetic defaults wherever data is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .config import REGIONS, TimedValue

if TYPE_CHECKING:
    from .config import ScenarioConfig, MacroConfig
    from .state import RegionStock

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Synthetic fallbacks (used when real data is unavailable)
# ---------------------------------------------------------------------------

# Base prices (SEK) at mid quality tier per region for each tenure type
SYNTHETIC_BRF_PRICES: dict[str, float] = {
    "STHLM": 4_500_000,
    "GBG":   2_800_000,
    "MALMO": 2_000_000,
    "REST":  1_600_000,
}
SYNTHETIC_SMALL_PRICES: dict[str, float] = {
    "STHLM": 6_000_000,
    "GBG":   3_800_000,
    "MALMO": 2_800_000,
    "REST":  2_200_000,
}
SYNTHETIC_INCOME_MONTHLY: dict[str, float] = {
    "STHLM": 42_000,
    "GBG":   37_000,
    "MALMO": 34_000,
    "REST":  31_000,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load(name: str) -> pd.DataFrame | None:
    p = DATA_DIR / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


def _latest_year(df: pd.DataFrame) -> int:
    return int(df["year"].max())


def _region_latest(df: pd.DataFrame, value_col: str) -> dict[str, float]:
    """Return {region: value} for the most recent year in df."""
    latest = _latest_year(df)
    row = df[df["year"] == latest].groupby("region")[value_col].mean()
    result: dict[str, float] = {}
    for r in REGIONS:
        if r in row.index:
            result[r] = float(row[r])
    return result


# ---------------------------------------------------------------------------
# 1. Macro rate path from Riksbank data
# ---------------------------------------------------------------------------

def calibrate_rate_path(start: str, end: str) -> list[dict]:
    """
    Return a piecewise rate path for the scenario config macro section.
    Covers [start, end]. Uses real Riksbank data where available, otherwise
    returns a single synthetic value.
    """
    df = _load("riksbank_repo_rate")
    if df is None or df.empty:
        print("  [calibration] No repo rate data; using synthetic rate path.")
        return [{"from": start, "value": 0.0225}]

    path = []
    df = df.copy()
    df["month"] = df["date"].str[:7]
    monthly = df.groupby("month")["rate"].last().reset_index()
    monthly = monthly[
        (monthly["month"] >= start) & (monthly["month"] <= end)
    ]

    prev = None
    for _, row in monthly.iterrows():
        r = round(float(row["rate"]), 6)
        if r != prev:
            path.append({"from": row["month"], "value": r})
            prev = r

    if not path:
        return [{"from": start, "value": 0.0225}]

    print(f"  [calibration] Loaded {len(path)} rate change points from Riksbank data.")
    return path


# ---------------------------------------------------------------------------
# 2. Initial price calibration for RegionStock
# ---------------------------------------------------------------------------

def calibrate_prices(
    region_stocks: list["RegionStock"],
    reference_year: int | None = None,
) -> None:
    """
    Set initial price_index values in region_stocks using real SCB price data.
    Modifies stocks in-place.

    Price index is normalised so that the national mean = 1.0.
    """
    brf_df = _load("scb_brf_prices")
    sh_df = _load("scb_house_prices")

    # --- BRF prices ---
    brf_prices = _get_prices(brf_df, "price_sek", reference_year, SYNTHETIC_BRF_PRICES)
    sh_prices = _get_prices(sh_df, "price_sek_per_m2", reference_year, SYNTHETIC_SMALL_PRICES)

    # Normalise both to national mean = 1.0
    brf_mean = float(np.mean(list(brf_prices.values())))
    sh_mean = float(np.mean(list(sh_prices.values())))

    quality_shape = np.array([0.7, 0.85, 1.0, 1.20, 1.45])

    for stock in region_stocks:
        region = REGIONS[stock.region_idx]

        brf_idx = brf_prices.get(region, brf_mean) / max(brf_mean, 1.0)
        sh_idx = sh_prices.get(region, sh_mean) / max(sh_mean, 1.0)

        stock.price_index[1] = brf_idx * quality_shape
        stock.price_index[2] = sh_idx * 1.05 * quality_shape  # small house premium

    source = "real SCB" if (brf_df is not None or sh_df is not None) else "synthetic"
    print(f"  [calibration] Price indices set from {source} data.")


def _get_prices(
    df: pd.DataFrame | None,
    value_col: str,
    reference_year: int | None,
    fallback: dict[str, float],
) -> dict[str, float]:
    if df is None or df.empty or value_col not in df.columns:
        return dict(fallback)

    yr = reference_year or _latest_year(df)
    sub = df[df["year"] == yr]
    if sub.empty:
        sub = df[df["year"] == _latest_year(df)]

    result = dict(fallback)
    for region in REGIONS:
        row = sub[sub["region"] == region]
        if not row.empty:
            result[region] = float(row[value_col].mean())
    return result


# ---------------------------------------------------------------------------
# 3. Regional population shares for household generation
# ---------------------------------------------------------------------------

def calibrate_population_shares(reference_year: int | None = None) -> np.ndarray:
    """
    Return array of shape (4,) with population shares for [STHLM, GBG, MALMO, REST].
    Falls back to synthetic shares if data unavailable.
    """
    SYNTHETIC = np.array([0.25, 0.14, 0.09, 0.52])

    df = _load("scb_population")
    if df is None or df.empty:
        return SYNTHETIC

    yr = reference_year or _latest_year(df)
    sub = df[df["year"] == yr]
    if sub.empty:
        return SYNTHETIC

    # Total Swedish population ~ 10.5M; REST = total - (STHLM + GBG + MALMO)
    total_sweden = 10_500_000
    pops = {}
    for region in ["STHLM", "GBG", "MALMO"]:
        row = sub[sub["region"] == region]
        pops[region] = float(row["population"].sum()) if not row.empty else 0.0

    pops["REST"] = max(0.0, total_sweden - sum(pops.values()))
    total = sum(pops.values())
    if total <= 0:
        return SYNTHETIC

    shares = np.array([pops[r] / total for r in REGIONS])
    print(
        f"  [calibration] Population shares from SCB {yr}: "
        + " ".join(f"{r}={s:.2f}" for r, s in zip(REGIONS, shares))
    )
    return shares


# ---------------------------------------------------------------------------
# 4. Regional income calibration
# ---------------------------------------------------------------------------

def calibrate_income_multipliers(reference_year: int | None = None) -> dict[str, float]:
    """
    Return per-region income multipliers relative to national mean.
    E.g. {"STHLM": 1.18, "GBG": 1.05, "MALMO": 0.98, "REST": 0.90}
    """
    SYNTHETIC = {"STHLM": 1.25, "GBG": 1.10, "MALMO": 1.00, "REST": 0.92}

    df = _load("scb_income")
    if df is None or df.empty:
        return SYNTHETIC

    yr = reference_year or _latest_year(df)
    sub = df[df["year"] == yr]
    if sub.empty:
        return SYNTHETIC

    incomes = {}
    for region in REGIONS:
        row = sub[sub["region"] == region]
        if not row.empty:
            incomes[region] = float(row["median_income_annual_sek"].mean())

    if not incomes:
        return SYNTHETIC

    mean_income = float(np.mean(list(incomes.values())))
    multipliers = {r: incomes.get(r, mean_income) / mean_income for r in REGIONS}
    print(
        f"  [calibration] Income multipliers from SCB {yr}: "
        + " ".join(f"{r}={v:.2f}" for r, v in multipliers.items())
    )
    return multipliers


# ---------------------------------------------------------------------------
# 5. Construction baseline calibration
# ---------------------------------------------------------------------------

def calibrate_construction_permits(reference_year: int | None = None) -> dict[str, int]:
    """
    Return per-region monthly construction starts baseline.
    Uses SCB data if available, else synthetic defaults.
    """
    SYNTHETIC = {"STHLM": 400, "GBG": 200, "MALMO": 150, "REST": 300}

    df = _load("scb_construction")
    if df is None or df.empty:
        return SYNTHETIC

    yr = reference_year or _latest_year(df)
    sub = df[(df["year"] == yr) & (df["region"] != "ALL")]
    if sub.empty:
        return SYNTHETIC

    # Quarterly → monthly
    result: dict[str, int] = {}
    for region in ["STHLM", "GBG", "MALMO"]:
        row = sub[sub["region"] == region]
        if not row.empty:
            annual = float(row["starts"].sum())
            result[region] = max(1, int(annual / 12))

    # REST: derive from total minus known regions
    all_row = df[(df["year"] == yr) & (df["region"] == "ALL")]
    if not all_row.empty and result:
        total_annual = float(all_row["starts"].sum())
        known = sum(v * 12 for v in result.values())
        result["REST"] = max(1, int((total_annual - known) / 12))
    else:
        result["REST"] = SYNTHETIC["REST"]

    for r in REGIONS:
        if r not in result:
            result[r] = SYNTHETIC[r]

    print(
        f"  [calibration] Construction permits from SCB {yr}: "
        + " ".join(f"{r}={result[r]}/mo" for r in REGIONS)
    )
    return result


# ---------------------------------------------------------------------------
# Master calibration entry point
# ---------------------------------------------------------------------------

class CalibrationData:
    """
    Container for all calibrated parameters derived from real data.
    Passed into run_scenario() to initialise the simulation with real-world values.
    """

    def __init__(self, reference_year: int | None = None):
        self.reference_year = reference_year
        self.rate_path: list[dict] = []
        self.population_shares: np.ndarray = np.array([0.25, 0.14, 0.09, 0.52])
        self.income_multipliers: dict[str, float] = {}
        self.construction_permits: dict[str, int] = {}
        self._prices_loaded = False

    @classmethod
    def load(cls, reference_year: int | None = None) -> "CalibrationData":
        """Load all calibration data from disk cache."""
        cal = cls(reference_year=reference_year)
        cal.population_shares = calibrate_population_shares(reference_year)
        cal.income_multipliers = calibrate_income_multipliers(reference_year)
        cal.construction_permits = calibrate_construction_permits(reference_year)
        cal._prices_loaded = (
            (_load("scb_brf_prices") is not None)
            or (_load("scb_house_prices") is not None)
        )
        return cal

    def apply_to_stocks(self, region_stocks: list["RegionStock"]) -> None:
        calibrate_prices(region_stocks, self.reference_year)

    def rate_path_for(self, start: str, end: str) -> list[dict]:
        return calibrate_rate_path(start, end)

    @property
    def is_real_data(self) -> bool:
        return (
            (_load("riksbank_repo_rate") is not None)
            or self._prices_loaded
        )
