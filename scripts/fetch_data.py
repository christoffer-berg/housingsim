"""
Fetch real Swedish housing-market data and cache to data/*.parquet.

Usage:
  python scripts/fetch_data.py              # fetch all, use cache if fresh
  python scripts/fetch_data.py --force      # re-fetch all from APIs
  python scripts/fetch_data.py --source riksbank scb_prices   # specific sources

Sources
-------
  riksbank       Riksbanken SWEA API — policy rate history
  scb_prices     SCB Fastighetsprisstatistik — BRF + smallhouse prices by region
  scb_const      SCB construction starts by county
  scb_pop        SCB population by county
  scb_income     SCB household income by county

All data is cached under data/ as Parquet. The simulation reads these files
automatically when run with --real-data (or when CalibrationData.load() is called).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from housingsim.data_fetch import (
    fetch_riksbank_repo_rate,
    fetch_scb_house_prices,
    fetch_scb_brf_prices,
    fetch_scb_construction,
    fetch_scb_population,
    fetch_scb_income,
)

SOURCE_MAP = {
    "riksbank":   fetch_riksbank_repo_rate,
    "scb_prices": [fetch_scb_house_prices, fetch_scb_brf_prices],
    "scb_const":  fetch_scb_construction,
    "scb_pop":    fetch_scb_population,
    "scb_income": fetch_scb_income,
}

ALL_SOURCES = list(SOURCE_MAP.keys())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch real Swedish data for housingsim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore disk cache and re-fetch from APIs",
    )
    parser.add_argument(
        "--source", nargs="+", choices=ALL_SOURCES + ["all"], default=["all"],
        help="Which data sources to fetch (default: all)",
    )
    args = parser.parse_args()

    sources = ALL_SOURCES if "all" in args.source else args.source

    print(f"Fetching sources: {', '.join(sources)}")
    print(f"Cache dir: {Path(__file__).parent.parent / 'data'}/\n")

    n_ok = 0
    n_total = 0

    for src in sources:
        fetchers = SOURCE_MAP[src]
        if not isinstance(fetchers, list):
            fetchers = [fetchers]
        for fn in fetchers:
            print(f"→ {fn.__name__}")
            n_total += 1
            result = fn(force=args.force)
            if result is not None:
                n_ok += 1
                print(f"  OK  ({len(result)} rows)\n")
            else:
                print(f"  FAILED — will use synthetic fallback\n")

    print(f"\nDone: {n_ok}/{n_total} sources fetched successfully.")
    if n_ok < n_total:
        print("Partial data is fine; the simulation falls back to synthetic values.")
    else:
        print("Run the simulation with --real-data to use this data:")
        print("  python scripts/run_simulation.py scenarios/baseline.yaml --real-data")


if __name__ == "__main__":
    main()
