"""
Fetch real Swedish housing-market data from public APIs.

Sources
-------
1. Riksbanken SWEA API  — repo rate (policy rate) history
   https://api.riksbank.se/swea/v1/

2. SCB (Statistics Sweden) JSON-stat API
   https://api.scb.se/OV0104/v1/doris/sv/ssd/
   Tables used:
     - BO/BO0501/BO0501A/FastprisSHRegionAr   House prices by region/year
     - BO/BO0501/BO0501C/FastprisBRFAr        BRF (condo) prices by region/year
     - BO/BO0101/BO0101B/LghNyKv              New apartments (construction starts)
     - BE/BE0101/BE0101A/BefolkningNy         Population by county
     - HE/HE0110/HE0110A/Taxering1            Household income by county

All responses are cached as parquet in data/ so subsequent runs are instant.
If an API call fails the function returns None and the caller falls back to
synthetic defaults.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import requests

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RIKSBANK_BASE = "https://api.riksbank.se/swea/v1"
SCB_BASE = "https://api.scb.se/OV0104/v1/doris/sv/ssd"

# Map SCB county codes → simulation regions
# SE0180 = Stockholm, SE1480 = Gothenburg, SE1280 = Malmö
# Everything else → REST
COUNTY_TO_REGION: dict[str, str] = {
    "01": "STHLM",   # Stockholm county
    "14": "GBG",     # Västra Götaland
    "12": "MALMO",   # Skåne
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, timeout: int = 20) -> Any | None:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [WARN] GET {url} failed: {e}")
        return None


def _post_scb(table_path: str, query: dict, timeout: int = 30) -> Any | None:
    url = f"{SCB_BASE}/{table_path}"
    try:
        r = requests.post(url, json=query, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [WARN] SCB POST {url} failed: {e}")
        return None


def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}.parquet"


def _load_cache(name: str) -> pd.DataFrame | None:
    p = _cache_path(name)
    if p.exists():
        return pd.read_parquet(p)
    return None


def _save_cache(df: pd.DataFrame, name: str) -> None:
    df.to_parquet(_cache_path(name), index=False)


# ---------------------------------------------------------------------------
# 1. Riksbanken: repo rate history
# ---------------------------------------------------------------------------

def fetch_riksbank_repo_rate(force: bool = False) -> pd.DataFrame | None:
    """
    Fetch Riksbanken's policy (repo) rate history.

    Returns DataFrame with columns: date (str YYYY-MM-DD), rate (float, fraction)
    """
    name = "riksbank_repo_rate"
    if not force:
        cached = _load_cache(name)
        if cached is not None:
            print(f"  [cache] {name}")
            return cached

    print("  Fetching Riksbanken repo rate …")
    # SWEA series: SECBREPOEFF = effective repo rate (%)
    url = f"{RIKSBANK_BASE}/Observations/SECBREPOEFF/1994-01-01"
    data = _get(url)
    if data is None:
        return None

    records = []
    for obs in data:
        try:
            date = obs.get("date") or obs.get("Period") or obs.get("period")
            val = obs.get("value") or obs.get("Value")
            if date and val is not None:
                records.append({"date": str(date)[:10], "rate": float(val) / 100.0})
        except (ValueError, TypeError):
            continue

    if not records:
        print("  [WARN] No repo rate observations parsed")
        return None

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    _save_cache(df, name)
    print(f"  Saved {len(df)} repo rate observations → data/{name}.parquet")
    return df


def repo_rate_to_scenario_path(df: pd.DataFrame, start: str, end: str) -> list[dict]:
    """
    Convert a repo-rate DataFrame to a scenario-compatible piecewise path.
    Returns list of {from: "YYYY-MM", value: float} dicts.
    """
    # Resample to monthly (last observation in each month)
    df = df.copy()
    df["month"] = df["date"].str[:7]
    monthly = df.groupby("month")["rate"].last().reset_index()
    monthly = monthly[
        (monthly["month"] >= start) & (monthly["month"] <= end)
    ]

    # Emit a new entry only when the rate changes
    path = []
    prev = None
    for _, row in monthly.iterrows():
        if row["rate"] != prev:
            path.append({"from": row["month"], "value": round(float(row["rate"]), 6)})
            prev = row["rate"]
    return path


# ---------------------------------------------------------------------------
# 2. SCB: house price index by region
# ---------------------------------------------------------------------------

def fetch_scb_house_prices(force: bool = False) -> pd.DataFrame | None:
    """
    Fetch SCB Fastighetsprisstatistik — average price per m² for one/two-dwelling
    buildings by county and year.

    Returns DataFrame: region (STHLM/GBG/MALMO/REST), year (int), price_sek (float)
    """
    name = "scb_house_prices"
    if not force:
        cached = _load_cache(name)
        if cached is not None:
            print(f"  [cache] {name}")
            return cached

    print("  Fetching SCB house prices by region …")
    # Table: FastprisSHRegionAr — smallhouse prices by county and year
    query = {
        "query": [
            {
                "code": "Region",
                "selection": {"filter": "item", "values": ["01", "12", "14"]},
            },
            {
                "code": "ContentsCode",
                "selection": {"filter": "item", "values": ["BO0501D5"]},  # kr/m²
            },
            {
                "code": "Tid",
                "selection": {"filter": "all", "values": ["*"]},
            },
        ],
        "response": {"format": "JSON"},
    }
    data = _post_scb("BO/BO0501/BO0501A/FastprisSHRegionAr", query)
    if data is None or "data" not in data:
        return None

    records = []
    for row in data["data"]:
        try:
            county = row["key"][0]
            year = int(row["key"][-1])
            val = row["values"][0]
            if val in (".", "..", ""):
                continue
            region = COUNTY_TO_REGION.get(county, "REST")
            records.append({"region": region, "year": year, "price_sek_per_m2": float(val)})
        except (ValueError, IndexError, KeyError):
            continue

    if not records:
        print("  [WARN] No house price data parsed from SCB")
        return None

    df = pd.DataFrame(records)
    # Add REST as mean of non-metro
    rest_years = df.groupby("year")["price_sek_per_m2"].mean().reset_index()
    rest_years["region"] = "REST"
    rest_years["price_sek_per_m2"] *= 0.65   # REST is cheaper than average
    df = pd.concat([df, rest_years], ignore_index=True)

    _save_cache(df, name)
    print(f"  Saved SCB house prices → data/{name}.parquet  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# 3. SCB: BRF (condo/bostadsrätt) price index
# ---------------------------------------------------------------------------

def fetch_scb_brf_prices(force: bool = False) -> pd.DataFrame | None:
    """
    Fetch SCB BRF (bostadsrätt) average prices by county and year.

    Returns DataFrame: region, year, price_sek (float, avg transaction price)
    """
    name = "scb_brf_prices"
    if not force:
        cached = _load_cache(name)
        if cached is not None:
            print(f"  [cache] {name}")
            return cached

    print("  Fetching SCB BRF prices …")
    query = {
        "query": [
            {
                "code": "Region",
                "selection": {"filter": "item", "values": ["01", "12", "14"]},
            },
            {
                "code": "ContentsCode",
                "selection": {"filter": "item", "values": ["BO0501H1"]},  # avg purchase price
            },
            {
                "code": "Tid",
                "selection": {"filter": "all", "values": ["*"]},
            },
        ],
        "response": {"format": "JSON"},
    }
    data = _post_scb("BO/BO0501/BO0501C/FastprisBRFAr", query)
    if data is None or "data" not in data:
        return None

    records = []
    for row in data["data"]:
        try:
            county = row["key"][0]
            year = int(row["key"][-1])
            val = row["values"][0]
            if val in (".", "..", ""):
                continue
            region = COUNTY_TO_REGION.get(county, "REST")
            records.append({"region": region, "year": year, "price_sek": float(val)})
        except (ValueError, IndexError, KeyError):
            continue

    if not records:
        print("  [WARN] No BRF price data parsed from SCB")
        return None

    df = pd.DataFrame(records)
    # REST proxy
    rest = df.groupby("year")["price_sek"].mean().reset_index()
    rest["region"] = "REST"
    rest["price_sek"] *= 0.55
    df = pd.concat([df, rest], ignore_index=True)

    _save_cache(df, name)
    print(f"  Saved SCB BRF prices → data/{name}.parquet  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# 4. SCB: construction starts (new apartments)
# ---------------------------------------------------------------------------

def fetch_scb_construction(force: bool = False) -> pd.DataFrame | None:
    """
    Fetch SCB quarterly construction starts (new apartments) by county.

    Returns DataFrame: region, year, quarter, starts (int)
    """
    name = "scb_construction"
    if not force:
        cached = _load_cache(name)
        if cached is not None:
            print(f"  [cache] {name}")
            return cached

    print("  Fetching SCB construction starts …")
    query = {
        "query": [
            {
                "code": "Region",
                "selection": {"filter": "item", "values": ["01", "12", "14", "00"]},
                # 00 = whole country (used to derive REST)
            },
            {
                "code": "Hustyp",
                "selection": {"filter": "item", "values": ["FLERBO"]},  # multi-dwelling
            },
            {
                "code": "ContentsCode",
                "selection": {"filter": "item", "values": ["BO0101AY"]},  # started dwellings
            },
            {
                "code": "Tid",
                "selection": {"filter": "all", "values": ["*"]},
            },
        ],
        "response": {"format": "JSON"},
    }
    data = _post_scb("BO/BO0101/BO0101B/LghNyKv", query)
    if data is None or "data" not in data:
        return None

    records = []
    for row in data["data"]:
        try:
            county = row["key"][0]
            period = row["key"][-1]   # e.g. "2023K1"
            val = row["values"][0]
            if val in (".", "..", ""):
                continue
            year = int(period[:4])
            quarter = int(period[-1])
            region = COUNTY_TO_REGION.get(county, "REST" if county != "00" else "ALL")
            records.append({
                "region": region, "year": year, "quarter": quarter,
                "starts": int(float(val))
            })
        except (ValueError, IndexError, KeyError):
            continue

    if not records:
        print("  [WARN] No construction data parsed from SCB")
        return None

    df = pd.DataFrame(records)
    _save_cache(df, name)
    print(f"  Saved SCB construction → data/{name}.parquet  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# 5. SCB: population by county
# ---------------------------------------------------------------------------

def fetch_scb_population(force: bool = False) -> pd.DataFrame | None:
    """
    Fetch SCB population by county and year.

    Returns DataFrame: region, year, population (int)
    """
    name = "scb_population"
    if not force:
        cached = _load_cache(name)
        if cached is not None:
            print(f"  [cache] {name}")
            return cached

    print("  Fetching SCB population by county …")
    query = {
        "query": [
            {
                "code": "Region",
                "selection": {"filter": "item", "values": ["01", "12", "14"]},
            },
            {
                "code": "ContentsCode",
                "selection": {"filter": "item", "values": ["BE0101N1"]},  # total pop
            },
            {
                "code": "Tid",
                "selection": {"filter": "all", "values": ["*"]},
            },
        ],
        "response": {"format": "JSON"},
    }
    data = _post_scb("BE/BE0101/BE0101A/BefolkningNy", query)
    if data is None or "data" not in data:
        return None

    records = []
    for row in data["data"]:
        try:
            county = row["key"][0]
            year = int(row["key"][-1])
            val = row["values"][0]
            if val in (".", "..", ""):
                continue
            region = COUNTY_TO_REGION.get(county, "REST")
            records.append({"region": region, "year": year, "population": int(float(val))})
        except (ValueError, IndexError, KeyError):
            continue

    if not records:
        print("  [WARN] No population data parsed from SCB")
        return None

    df = pd.DataFrame(records)
    _save_cache(df, name)
    print(f"  Saved SCB population → data/{name}.parquet  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# 6. SCB: household income by county
# ---------------------------------------------------------------------------

def fetch_scb_income(force: bool = False) -> pd.DataFrame | None:
    """
    Fetch SCB median household income by county and year.

    Returns DataFrame: region, year, median_income_sek (float, annual)
    """
    name = "scb_income"
    if not force:
        cached = _load_cache(name)
        if cached is not None:
            print(f"  [cache] {name}")
            return cached

    print("  Fetching SCB household income …")
    query = {
        "query": [
            {
                "code": "Region",
                "selection": {"filter": "item", "values": ["01", "12", "14"]},
            },
            {
                "code": "ContentsCode",
                "selection": {
                    "filter": "item",
                    "values": ["HE0110J9"],  # median disposable income
                },
            },
            {
                "code": "Tid",
                "selection": {"filter": "all", "values": ["*"]},
            },
        ],
        "response": {"format": "JSON"},
    }
    data = _post_scb("HE/HE0110/HE0110A/TabVXRegionAr", query)
    if data is None or "data" not in data:
        return None

    records = []
    for row in data["data"]:
        try:
            county = row["key"][0]
            year = int(row["key"][-1])
            val = row["values"][0]
            if val in (".", "..", ""):
                continue
            region = COUNTY_TO_REGION.get(county, "REST")
            records.append({
                "region": region, "year": year,
                "median_income_annual_sek": float(val) * 1000  # SCB reports in kkr
            })
        except (ValueError, IndexError, KeyError):
            continue

    if not records:
        print("  [WARN] No income data parsed from SCB")
        return None

    df = pd.DataFrame(records)
    # REST proxy: 85% of national average of the three regions
    rest = df.groupby("year")["median_income_annual_sek"].mean().reset_index()
    rest["region"] = "REST"
    rest["median_income_annual_sek"] *= 0.87
    df = pd.concat([df, rest], ignore_index=True)

    _save_cache(df, name)
    print(f"  Saved SCB income → data/{name}.parquet  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Master fetch function
# ---------------------------------------------------------------------------

def fetch_all(force: bool = False) -> dict[str, pd.DataFrame | None]:
    """
    Fetch all data sources. Returns dict keyed by source name.
    Safe to call repeatedly; uses disk cache unless force=True.
    """
    print("\nFetching real-world data …")
    results = {}
    results["repo_rate"] = fetch_riksbank_repo_rate(force=force)
    results["house_prices"] = fetch_scb_house_prices(force=force)
    results["brf_prices"] = fetch_scb_brf_prices(force=force)
    results["construction"] = fetch_scb_construction(force=force)
    results["population"] = fetch_scb_population(force=force)
    results["income"] = fetch_scb_income(force=force)

    n_ok = sum(v is not None for v in results.values())
    print(f"\nFetched {n_ok}/{len(results)} sources successfully.")
    if n_ok < len(results):
        failed = [k for k, v in results.items() if v is None]
        print(f"  Falling back to synthetic data for: {', '.join(failed)}")
    return results
