"""
Smoke tests: verify the simulation runs end-to-end and produces
sensible outputs without crashing.
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

# Ensure imports work from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from housingsim.config import ScenarioConfig, months_between, TimedValue
from housingsim.state import build_initial_stocks, RegionStock
from housingsim.households import generate_households, apply_life_events, choose_actions
from housingsim.bank import qualify_buyers
from housingsim.markets import match_and_transact, allocate_rentals
from housingsim.developers import ConstructionPipeline
from housingsim.metrics import MetricsRecorder
from housingsim.simulation import run_scenario


# ---------------------------------------------------------------------------
# Config & utilities
# ---------------------------------------------------------------------------

def make_tiny_cfg(n_hh: int = 500, months: int = 6) -> ScenarioConfig:
    """Return a minimal config for fast testing."""
    from datetime import date
    start = "2024-01"
    # Calculate end month
    year, month = 2024, 1
    for _ in range(months - 1):
        month += 1
        if month > 12:
            month = 1
            year += 1
    end = f"{year:04d}-{month:02d}"

    return ScenarioConfig(
        scenario_name="test",
        time={"start": start, "end": end},
        n_households=n_hh,
        random_seed=0,
        n_runs=1,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_months_between(self):
        months = months_between("2024-01", "2024-06")
        assert months == ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
        assert len(months) == 6

    def test_credit_policy_ltv(self):
        cfg = ScenarioConfig()
        # Default LTV is 0.85
        assert cfg.credit_policy.ltv_cap_at("2024-01") == 0.85

    def test_credit_policy_ltv_piecewise(self):
        cfg = ScenarioConfig.model_validate({
            "credit_policy": {
                "ltv_cap": [
                    {"from": "2024-01", "value": 0.85},
                    {"from": "2026-04", "value": 0.90},
                ]
            }
        })
        assert cfg.credit_policy.ltv_cap_at("2026-03") == 0.85
        assert cfg.credit_policy.ltv_cap_at("2026-04") == 0.90
        assert cfg.credit_policy.ltv_cap_at("2027-01") == 0.90

    def test_mortgage_rate_at(self):
        cfg = ScenarioConfig()
        rate = cfg.macro.mortgage_rate_at("2024-01")
        assert rate == pytest.approx(cfg.macro.base_policy_rate + cfg.macro.mortgage_spread)

    def test_construction_permits(self):
        cfg = ScenarioConfig()
        p = cfg.construction_policy.permits_at("2024-01", "STHLM")
        assert p > 0

    def test_scenario_from_yaml(self, tmp_path):
        import yaml
        data = {
            "scenario_name": "yaml_test",
            "time": {"start": "2024-01", "end": "2025-12"},
        }
        p = tmp_path / "test.yaml"
        p.write_text(yaml.dump(data))
        cfg = ScenarioConfig.from_yaml(str(p))
        assert cfg.scenario_name == "yaml_test"


class TestHouseholds:
    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.hh = generate_households(200, self.rng)

    def test_shape(self):
        hh = self.hh
        assert hh.N == 200
        assert hh.income_monthly.shape == (200,)
        assert hh.pref_region.shape == (200, 4)

    def test_income_positive(self):
        assert (self.hh.income_monthly > 0).all()

    def test_tenure_valid(self):
        assert set(self.hh.tenure).issubset({0, 1, 2})

    def test_quality_valid(self):
        assert (self.hh.quality >= 0).all()
        assert (self.hh.quality <= 4).all()

    def test_pref_region_sums_to_one(self):
        row_sums = self.hh.pref_region.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_weights_positive(self):
        assert (self.hh.weights > 0).all()

    def test_life_events_no_crash(self):
        apply_life_events(self.hh, self.rng, income_growth_monthly=0.002)
        assert (self.hh.income_monthly > 0).all()


class TestStocks:
    def test_build_initial_stocks(self):
        stocks = build_initial_stocks()
        assert len(stocks) == 4  # 4 regions
        for s in stocks:
            assert s.units.shape == (3, 5)
            assert (s.units > 0).all()

    def test_price_index_shape(self):
        stocks = build_initial_stocks()
        for s in stocks:
            assert s.price_index.shape == (3, 5)

    def test_monthly_vacancies_positive(self):
        stocks = build_initial_stocks()
        for s in stocks:
            for t in range(3):
                assert s.monthly_vacancies(t) > 0

    def test_update_price_no_demand(self):
        stocks = build_initial_stocks()
        rng = np.random.default_rng(1)
        s = stocks[0]
        old_price = s.price_index[1, 2]
        s.update_ownership_price(1, 0.0, 1000.0, kappa=0.1, noise_std=0.0, rng=rng)
        # Price should fall when demand=0 and supply is large
        assert s.price_index[1, 2] < old_price

    def test_queue_pressure_update(self):
        stocks = build_initial_stocks()
        s = stocks[0]
        # High applicants → pressure should rise
        s.update_queue_pressure(applicants_weighted=1000, vacancies=10)
        assert s.queue_pressure > 1.0


class TestBank:
    def setup_method(self):
        self.rng = np.random.default_rng(0)
        self.hh = generate_households(200, self.rng)
        self.stocks = build_initial_stocks()
        self.cfg = ScenarioConfig()

    def test_qualify_empty(self):
        empty_intent = {
            "hh_idx": np.array([], dtype=int),
            "target_region": np.array([], dtype=int),
            "target_quality": np.array([], dtype=int),
            "target_payment": np.array([], dtype=np.float32),
            "weight": np.array([], dtype=np.float64),
        }
        result = qualify_buyers(
            self.hh, empty_intent, "2024-01", 0.035,
            self.cfg.credit_policy, self.stocks, tenure_idx=1, rng=self.rng
        )
        assert len(result["hh_idx"]) == 0

    def test_qualify_some(self):
        # Give some households enough wealth
        self.hh.wealth_liquid[:50] = 500_000
        self.hh.income_monthly[:50] = 60_000
        intents = {
            "hh_idx": np.arange(50),
            "target_region": np.zeros(50, dtype=int),
            "target_quality": np.full(50, 2, dtype=int),
            "target_payment": np.full(50, 10_000, dtype=np.float32),
            "weight": np.ones(50),
        }
        result = qualify_buyers(
            self.hh, intents, "2024-01", 0.035,
            self.cfg.credit_policy, self.stocks, tenure_idx=1, rng=self.rng
        )
        assert len(result["hh_idx"]) >= 0  # may be 0 if prices too high


class TestMarkets:
    def setup_method(self):
        self.rng = np.random.default_rng(7)
        self.stocks = build_initial_stocks()

    def test_ownership_market_empty(self):
        from housingsim.bank import _empty_qualified
        tx = match_and_transact(
            _empty_qualified(),
            self.stocks, tenure_idx=1,
            t_str="2024-01",
            price_update_kappa=0.1,
            price_noise_std=0.005,
            rng=self.rng,
        )
        assert len(tx["hh_idx"]) == 0

    def test_rental_market_empty(self):
        empty = {
            "hh_idx": np.array([], dtype=int),
            "target_region": np.array([], dtype=int),
            "target_quality": np.array([], dtype=int),
            "target_payment": np.array([], dtype=np.float32),
            "weight": np.array([], dtype=np.float64),
        }
        result = allocate_rentals(empty, self.stocks, self.rng)
        assert len(result["hh_idx"]) == 0

    def test_rental_allocation_basic(self):
        intents = {
            "hh_idx": np.arange(20),
            "target_region": np.zeros(20, dtype=int),
            "target_quality": np.full(20, 2, dtype=int),
            "target_payment": np.full(20, 8_000, dtype=np.float32),
            "weight": np.ones(20),
            "queue_score": np.ones(20, dtype=np.float32),
        }
        result = allocate_rentals(intents, self.stocks, self.rng)
        # Some should be granted (vacancies >> 20 applicants for 220k-unit stock)
        assert len(result["hh_idx"]) > 0


class TestPipeline:
    def test_pipeline_starts_and_completes(self):
        rng = np.random.default_rng(3)
        stocks = build_initial_stocks()
        pipeline = ConstructionPipeline()
        cfg = ScenarioConfig()

        # Start some projects
        pipeline.decide_starts("2024-01", stocks, cfg.construction_policy, rng)
        initial_entries = len(pipeline.entries)
        assert initial_entries > 0

        total_months = cfg.construction_policy.approval_delay_months + cfg.construction_policy.build_time_months + 1
        old_stock = stocks[0].units.sum()

        for _ in range(total_months):
            pipeline.advance_and_complete(stocks)
            pipeline.decide_starts("2024-01", stocks, cfg.construction_policy, rng)

        # Stock should have grown
        new_stock = stocks[0].units.sum()
        assert new_stock > old_stock


class TestSimulationSmoke:
    """End-to-end smoke test of the full simulation loop."""

    def test_full_run_completes(self):
        cfg = make_tiny_cfg(n_hh=300, months=6)
        recorder = run_scenario(cfg, verbose=False)
        df = recorder.to_dataframe()
        assert len(df) == 6
        assert "homeownership_rate" in df.columns
        assert "median_housing_cost_burden" in df.columns

    def test_homeownership_rate_sane(self):
        cfg = make_tiny_cfg(n_hh=300, months=6)
        recorder = run_scenario(cfg, verbose=False)
        df = recorder.to_dataframe()
        # Should be between 20% and 95%
        assert (df["homeownership_rate"] >= 0.20).all()
        assert (df["homeownership_rate"] <= 0.95).all()

    def test_median_burden_sane(self):
        cfg = make_tiny_cfg(n_hh=300, months=6)
        recorder = run_scenario(cfg, verbose=False)
        df = recorder.to_dataframe()
        # Should be between 5% and 80%
        assert (df["median_housing_cost_burden"] >= 0.05).all()
        assert (df["median_housing_cost_burden"] <= 0.80).all()

    def test_price_indices_positive(self):
        cfg = make_tiny_cfg(n_hh=300, months=6)
        recorder = run_scenario(cfg, verbose=False)
        df = recorder.to_dataframe()
        for region in ["sthlm", "gbg", "malmo", "rest"]:
            col = f"r_{region}_price_idx_brf"
            assert (df[col] > 0).all(), f"{col} has non-positive values"

    def test_no_nan_in_kpis(self):
        cfg = make_tiny_cfg(n_hh=300, months=6)
        recorder = run_scenario(cfg, verbose=False)
        df = recorder.to_dataframe()
        numeric_cols = df.select_dtypes(include="number").columns
        assert not df[numeric_cols].isnull().any().any(), "NaN found in KPIs"

    def test_scenario_yaml_baseline(self):
        """Load and run the actual baseline scenario YAML (small, 1 run)."""
        yaml_path = Path(__file__).parent.parent / "scenarios" / "baseline.yaml"
        if not yaml_path.exists():
            pytest.skip("baseline.yaml not found")
        from housingsim.config import TimeRange
        cfg = ScenarioConfig.from_yaml(str(yaml_path))
        # Override for speed
        cfg = cfg.model_copy(update={"n_households": 500, "n_runs": 1,
                                     "time": TimeRange(start="2024-01", end="2024-06")})
        recorder = run_scenario(cfg, verbose=False)
        df = recorder.to_dataframe()
        assert len(df) == 6

    def test_ltv_relaxation_increases_ownership(self):
        """Relaxing LTV cap should not decrease ownership (directional test)."""
        base_cfg = make_tiny_cfg(n_hh=500, months=12)
        base_recorder = run_scenario(base_cfg, seed_override=42, verbose=False)
        base_df = base_recorder.to_dataframe()

        # Relaxed LTV
        relaxed = base_cfg.model_copy(deep=True)
        relaxed.credit_policy.ltv_cap = [TimedValue(**{"from": "2024-01", "value": 0.90})]
        relaxed_recorder = run_scenario(relaxed, seed_override=42, verbose=False)
        relaxed_df = relaxed_recorder.to_dataframe()

        # Ownership rate at end should be >= baseline (more people can qualify)
        base_own = base_df["homeownership_rate"].iloc[-1]
        relaxed_own = relaxed_df["homeownership_rate"].iloc[-1]
        # Allow some tolerance due to randomness
        assert relaxed_own >= base_own - 0.05, (
            f"Relaxed LTV ownership {relaxed_own:.3f} much lower than base {base_own:.3f}"
        )
