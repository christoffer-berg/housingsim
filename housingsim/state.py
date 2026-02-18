"""RegionStock and housing-stock state management."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from .config import N_REGIONS, N_QUALITY, REGIONS, TENURES


# ---------------------------------------------------------------------------
# Region housing stock
# ---------------------------------------------------------------------------

@dataclass
class RegionStock:
    """
    Housing stock state for one region.

    Arrays are indexed by quality tier 0..N_QUALITY-1 (tiers 1..5).
    Tenure axes: 0=rent_regulated, 1=buy_bostadsratt, 2=buy_smallhouse.
    """

    region_idx: int

    # Stock counts per (tenure, quality): shape (3, N_QUALITY)
    units: np.ndarray = field(default_factory=lambda: np.zeros((3, N_QUALITY)))

    # Price indices per (tenure, quality): shape (3, N_QUALITY)
    # Ownership only; rentals use regulated rent path
    price_index: np.ndarray = field(
        default_factory=lambda: np.ones((3, N_QUALITY))
    )

    # Regulated rent level index (scalar, grows slowly)
    rent_index: float = 1.0

    # Queue pressure scalar: >1 means excess demand in rental market
    queue_pressure: float = 1.0

    # Monthly turnover rates per tenure (fraction of stock that lists)
    turnover_rate: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.018, 0.014])  # rent, brf, small
    )

    @property
    def name(self) -> str:
        return REGIONS[self.region_idx]

    def total_units(self, tenure_idx: int) -> float:
        return float(self.units[tenure_idx].sum())

    def monthly_vacancies(self, tenure_idx: int) -> float:
        """Approximate new vacancies/listings appearing this month."""
        return float(
            (self.units[tenure_idx] * self.turnover_rate[tenure_idx]).sum()
        )

    def effective_price(self, tenure_idx: int, quality_idx: int) -> float:
        """Return price_index * quality_multiplier for a given (tenure, quality)."""
        quality_mult = 0.6 + 0.2 * quality_idx  # q=0 → 0.6, q=4 → 1.4
        return float(self.price_index[tenure_idx, quality_idx] * quality_mult)

    def update_ownership_price(
        self,
        tenure_idx: int,
        demand_weighted: float,
        supply_available: float,
        kappa: float,
        noise_std: float,
        rng: np.random.Generator,
    ) -> None:
        """Update price index for an ownership tenure via log-linear rule."""
        stock = max(self.units[tenure_idx].sum(), 1.0)
        excess = (demand_weighted - supply_available) / stock
        noise = rng.normal(0, noise_std)
        factor = float(np.exp(kappa * excess + noise))
        # Clamp monthly change to ±15 %
        factor = float(np.clip(factor, 0.85, 1.15))
        self.price_index[tenure_idx] *= factor

    def update_queue_pressure(
        self,
        applicants_weighted: float,
        vacancies: float,
        alpha: float = 0.15,
    ) -> None:
        """
        Update rental queue pressure as a weighted average.
        alpha = speed of adjustment toward new signal.
        """
        if vacancies < 1.0:
            new_signal = 5.0
        else:
            new_signal = min(5.0, applicants_weighted / vacancies)
        self.queue_pressure = (1 - alpha) * self.queue_pressure + alpha * new_signal


def build_initial_stocks(cfg_stocks: dict | None = None) -> list[RegionStock]:
    """
    Create initial RegionStock objects with calibrated Swedish-like numbers.

    Regional stock sizes are rough approximations (weighted households 10k → scaled).
    """
    # Base housing stocks (thousands of units); roughly calibrated to Sweden
    # STHLM ~1.1M, GBG ~0.55M, MALMO ~0.37M, REST ~2.0M
    base_stocks_k = {
        "STHLM": {"rent_regulated": 220, "buy_bostadsratt": 280, "buy_smallhouse": 120},
        "GBG":   {"rent_regulated": 110, "buy_bostadsratt": 120, "buy_smallhouse":  80},
        "MALMO": {"rent_regulated":  75, "buy_bostadsratt":  80, "buy_smallhouse":  55},
        "REST":  {"rent_regulated": 350, "buy_bostadsratt": 300, "buy_smallhouse": 550},
    }

    # Quality tier distribution: roughly bell-curve around tier 3
    quality_dist = np.array([0.10, 0.20, 0.40, 0.20, 0.10])

    # Initial price index per region (STHLM highest)
    base_price_index = {
        "STHLM": 1.40,
        "GBG":   1.10,
        "MALMO": 0.90,
        "REST":  0.80,
    }

    stocks = []
    for r_idx, region in enumerate(REGIONS):
        s = RegionStock(region_idx=r_idx)
        for t_idx, tenure in enumerate(TENURES):
            total = base_stocks_k[region][tenure] * 1000
            s.units[t_idx] = total * quality_dist

        # Set price indices (ownership only; rent stays at 1.0)
        pi = base_price_index[region]
        # brf (t=1) and smallhouse (t=2); smallhouse slightly higher
        s.price_index[1] = pi * np.array([0.7, 0.85, 1.0, 1.20, 1.45])
        s.price_index[2] = pi * 1.05 * np.array([0.7, 0.85, 1.0, 1.20, 1.45])

        stocks.append(s)

    return stocks
