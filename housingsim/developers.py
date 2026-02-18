"""
Developer agents and construction pipeline.

Developers decide on new starts based on price signals and policy.
The pipeline counts down to completion and then adds units to the stock.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ConstructionPolicyConfig, N_REGIONS, REGIONS, TENURES
from .state import RegionStock


# ---------------------------------------------------------------------------
# Construction pipeline entry
# ---------------------------------------------------------------------------

@dataclass
class PipelineEntry:
    region_idx: int
    tenure_idx: int
    quality_idx: int
    units: float             # weighted real units
    months_remaining: int    # counts down; 0 → complete this month


# ---------------------------------------------------------------------------
# Developer / pipeline state
# ---------------------------------------------------------------------------

class ConstructionPipeline:
    """
    Manages the construction pipeline across all regions.
    """

    def __init__(self) -> None:
        self.entries: list[PipelineEntry] = []
        # Track stats
        self.starts_this_month: dict[str, float] = {r: 0.0 for r in REGIONS}
        self.completions_this_month: dict[str, float] = {r: 0.0 for r in REGIONS}

    def decide_starts(
        self,
        t_str: str,
        region_stocks: list[RegionStock],
        policy: ConstructionPolicyConfig,
        rng: np.random.Generator,
    ) -> None:
        """
        Developers decide how many units to start this month.

        Base permits from policy, scaled by:
        - supply_elasticity * (price appreciation over 12m, approx. by comparing
          current price to a smoothed baseline — simplified here as a fixed proxy)
        - Random noise
        """
        for r_idx, stock in enumerate(region_stocks):
            region = REGIONS[r_idx]
            base_permits = policy.permits_at(t_str, region)

            # Simple price signal: if BRF price index is high relative to 1.0,
            # developers build more
            avg_brf_price = float(stock.price_index[1].mean())
            price_boost = policy.supply_elasticity * (avg_brf_price - 1.0)
            adjusted = base_permits * (1.0 + price_boost)

            # Small random variation ±10%
            adjusted *= rng.uniform(0.90, 1.10)
            adjusted = max(0.0, adjusted)

            # Split by tenure
            total_delay = policy.approval_delay_months + policy.build_time_months
            for t_idx, tenure in enumerate(TENURES):
                split = policy.tenure_split.get(tenure, 0.33)
                units = adjusted * split

                # Quality tier split (mostly middle tiers for new builds)
                quality_dist = np.array([0.05, 0.15, 0.45, 0.25, 0.10])
                for q_idx in range(5):
                    q_units = units * quality_dist[q_idx]
                    if q_units < 1:
                        continue
                    self.entries.append(
                        PipelineEntry(
                            region_idx=r_idx,
                            tenure_idx=t_idx,
                            quality_idx=q_idx,
                            units=q_units,
                            months_remaining=total_delay,
                        )
                    )

            self.starts_this_month[region] = adjusted

    def advance_and_complete(
        self,
        region_stocks: list[RegionStock],
    ) -> None:
        """
        Decrement pipeline counters; add completed units to stock.
        """
        completions: dict[str, float] = {r: 0.0 for r in REGIONS}

        remaining = []
        for entry in self.entries:
            entry.months_remaining -= 1
            if entry.months_remaining <= 0:
                # Add to stock
                region_stocks[entry.region_idx].units[entry.tenure_idx, entry.quality_idx] += entry.units
                completions[REGIONS[entry.region_idx]] += entry.units
            else:
                remaining.append(entry)

        self.entries = remaining
        self.completions_this_month = completions

    def pipeline_count_by_region(self) -> dict[str, float]:
        counts: dict[str, float] = {r: 0.0 for r in REGIONS}
        for entry in self.entries:
            counts[REGIONS[entry.region_idx]] += entry.units
        return counts
