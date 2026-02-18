"""
Metrics recording module.

Records per-month KPIs to a list of dicts, which can be converted to
a pandas DataFrame at the end of a run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import REGIONS, TENURES
from .households import HouseholdArrays
from .state import RegionStock
from .developers import ConstructionPipeline


class MetricsRecorder:
    """
    Accumulates monthly snapshots of KPIs.
    """

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def record(
        self,
        t_str: str,
        hh: HouseholdArrays,
        region_stocks: list[RegionStock],
        tx_brf: dict,
        tx_small: dict,
        rental_granted: dict,
        pipeline: ConstructionPipeline,
        macro_rate: float,
        shut_out_brf: float = 0.0,
        shut_out_small: float = 0.0,
        property_tax_rate: float = 0.0,
    ) -> None:

        row: dict[str, Any] = {
            "month": t_str,
            "mortgage_rate": macro_rate,
            "property_tax_rate": property_tax_rate,
        }

        total_weighted = float(hh.weights.sum())

        # --- Global KPIs ---
        owners = hh.tenure > 0
        row["homeownership_rate"] = float((hh.weights[owners]).sum() / total_weighted)

        # Median housing cost burden
        burden = hh.housing_cost / np.maximum(hh.income_monthly, 1.0)
        sorted_burden = np.sort(burden)
        cum_weight = np.cumsum(hh.weights[np.argsort(burden)])
        median_idx = np.searchsorted(cum_weight, total_weighted / 2)
        median_idx = min(median_idx, len(sorted_burden) - 1)
        row["median_housing_cost_burden"] = float(sorted_burden[median_idx])

        # Share shut out of buying
        total_want_buy = shut_out_brf + shut_out_small + float(
            tx_brf.get("volume_weighted", 0) + tx_small.get("volume_weighted", 0)
        )
        if total_want_buy > 0:
            row["share_shut_out_buying"] = (shut_out_brf + shut_out_small) / total_want_buy
        else:
            row["share_shut_out_buying"] = 0.0

        # Mobility rate (households that moved this month)
        moved = (
            len(tx_brf.get("hh_idx", []))
            + len(tx_small.get("hh_idx", []))
            + len(rental_granted.get("hh_idx", []))
        )
        row["mobility_rate"] = moved / hh.N

        # Transaction volumes
        row["tx_volume_brf"] = float(tx_brf.get("volume_weighted", 0))
        row["tx_volume_small"] = float(tx_small.get("volume_weighted", 0))
        row["rental_granted_volume"] = float(rental_granted.get("volume_weighted", 0))

        # --- Per-region KPIs ---
        for r_idx, region in enumerate(REGIONS):
            stock = region_stocks[r_idx]
            prefix = f"r_{region.lower()}"

            # Price indices (BRF and smallhouse at mid quality tier = 2)
            row[f"{prefix}_price_idx_brf"] = float(stock.price_index[1, 2])
            row[f"{prefix}_price_idx_small"] = float(stock.price_index[2, 2])
            row[f"{prefix}_rent_index"] = float(stock.rent_index)
            row[f"{prefix}_queue_pressure"] = float(stock.queue_pressure)

            # Construction
            row[f"{prefix}_starts"] = float(pipeline.starts_this_month.get(region, 0))
            row[f"{prefix}_completions"] = float(pipeline.completions_this_month.get(region, 0))
            row[f"{prefix}_pipeline_total"] = float(
                pipeline.pipeline_count_by_region().get(region, 0)
            )

            # Stock sizes
            row[f"{prefix}_stock_rent"] = float(stock.units[0].sum())
            row[f"{prefix}_stock_brf"] = float(stock.units[1].sum())
            row[f"{prefix}_stock_small"] = float(stock.units[2].sum())

            # Regional homeownership
            reg_mask = hh.region == r_idx
            if reg_mask.any():
                reg_owners = reg_mask & owners
                row[f"{prefix}_homeownership_rate"] = float(
                    hh.weights[reg_owners].sum() / hh.weights[reg_mask].sum()
                )
                reg_burden = burden[reg_mask]
                row[f"{prefix}_median_burden"] = float(np.median(reg_burden))

                # Effective BRF price per region (mid quality)
                mid_q = 2
                row[f"{prefix}_brf_price_abs"] = float(stock.effective_price(1, mid_q))
                row[f"{prefix}_small_price_abs"] = float(stock.effective_price(2, mid_q))

        self._records.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    def save_parquet(self, path: str) -> None:
        df = self.to_dataframe()
        df.to_parquet(path, index=False)
        print(f"  Saved metrics → {path}")
