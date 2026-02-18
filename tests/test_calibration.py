"""
Tests for the calibration module and data fetch helpers.

These tests work completely offline — they verify:
  1. Calibration functions return sensible fallback values when no data is cached
  2. CalibrationData integrates correctly with the simulation
  3. Data parsing logic handles empty/malformed API responses gracefully
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from housingsim.calibration import (
    CalibrationData,
    calibrate_population_shares,
    calibrate_income_multipliers,
    calibrate_construction_permits,
    calibrate_rate_path,
)
from housingsim.config import REGIONS, ScenarioConfig, TimeRange
from housingsim.state import build_initial_stocks
from housingsim.simulation import run_scenario


# ---------------------------------------------------------------------------
# CalibrationData with no cached data (pure fallback)
# ---------------------------------------------------------------------------

class TestCalibrationFallbacks:
    """All tests here rely on no data being in data/ — just checks fallbacks."""

    def test_population_shares_fallback(self, monkeypatch):
        # Monkeypatch _load to return None (no cached data)
        import housingsim.calibration as cal_mod
        monkeypatch.setattr(cal_mod, "_load", lambda name: None)
        shares = calibrate_population_shares()
        assert len(shares) == 4
        assert abs(shares.sum() - 1.0) < 1e-6
        assert (shares > 0).all()

    def test_income_multipliers_fallback(self, monkeypatch):
        import housingsim.calibration as cal_mod
        monkeypatch.setattr(cal_mod, "_load", lambda name: None)
        mults = calibrate_income_multipliers()
        assert set(mults.keys()) == set(REGIONS)
        for v in mults.values():
            assert v > 0

    def test_construction_permits_fallback(self, monkeypatch):
        import housingsim.calibration as cal_mod
        monkeypatch.setattr(cal_mod, "_load", lambda name: None)
        permits = calibrate_construction_permits()
        assert set(permits.keys()) == set(REGIONS)
        for v in permits.values():
            assert v > 0

    def test_rate_path_fallback(self, monkeypatch):
        import housingsim.calibration as cal_mod
        monkeypatch.setattr(cal_mod, "_load", lambda name: None)
        path = calibrate_rate_path("2024-01", "2025-12")
        assert isinstance(path, list)
        assert len(path) >= 1
        assert "from" in path[0]
        assert "value" in path[0]

    def test_calibration_data_load_no_cache(self, monkeypatch):
        import housingsim.calibration as cal_mod
        monkeypatch.setattr(cal_mod, "_load", lambda name: None)
        cal = CalibrationData.load()
        assert len(cal.population_shares) == 4
        assert isinstance(cal.income_multipliers, dict)
        assert isinstance(cal.construction_permits, dict)
        assert not cal.is_real_data


# ---------------------------------------------------------------------------
# CalibrationData with synthetic cached data
# ---------------------------------------------------------------------------

class TestCalibrationWithSyntheticCache:
    """Inject synthetic DataFrames to simulate having fetched real data."""

    @pytest.fixture
    def mock_repo_rate_df(self):
        rows = [{"date": f"2020-{m:02d}-01", "rate": 0.0175 + 0.001 * m}
                for m in range(1, 13)]
        rows += [{"date": f"2021-{m:02d}-01", "rate": 0.0100} for m in range(1, 13)]
        return pd.DataFrame(rows)

    @pytest.fixture
    def mock_prices_df(self):
        return pd.DataFrame([
            {"region": "STHLM", "year": 2022, "price_sek": 4_500_000},
            {"region": "GBG",   "year": 2022, "price_sek": 2_800_000},
            {"region": "MALMO", "year": 2022, "price_sek": 2_000_000},
            {"region": "REST",  "year": 2022, "price_sek": 1_600_000},
        ])

    @pytest.fixture
    def mock_population_df(self):
        return pd.DataFrame([
            {"region": "STHLM", "year": 2022, "population": 2_400_000},
            {"region": "GBG",   "year": 2022, "population": 1_100_000},
            {"region": "MALMO", "year": 2022, "population": 750_000},
            {"region": "REST",  "year": 2022, "population": 6_250_000},
        ])

    @pytest.fixture
    def mock_income_df(self):
        return pd.DataFrame([
            {"region": "STHLM", "year": 2022, "median_income_annual_sek": 520_000},
            {"region": "GBG",   "year": 2022, "median_income_annual_sek": 460_000},
            {"region": "MALMO", "year": 2022, "median_income_annual_sek": 420_000},
            {"region": "REST",  "year": 2022, "median_income_annual_sek": 390_000},
        ])

    @pytest.fixture
    def mock_construction_df(self):
        return pd.DataFrame([
            {"region": "STHLM", "year": 2022, "quarter": 1, "starts": 1200},
            {"region": "STHLM", "year": 2022, "quarter": 2, "starts": 1400},
            {"region": "STHLM", "year": 2022, "quarter": 3, "starts": 1300},
            {"region": "STHLM", "year": 2022, "quarter": 4, "starts": 1100},
            {"region": "GBG",   "year": 2022, "quarter": 1, "starts": 600},
            {"region": "GBG",   "year": 2022, "quarter": 2, "starts": 700},
            {"region": "GBG",   "year": 2022, "quarter": 3, "starts": 650},
            {"region": "GBG",   "year": 2022, "quarter": 4, "starts": 550},
            {"region": "MALMO", "year": 2022, "quarter": 1, "starts": 400},
            {"region": "MALMO", "year": 2022, "quarter": 2, "starts": 450},
            {"region": "MALMO", "year": 2022, "quarter": 3, "starts": 420},
            {"region": "MALMO", "year": 2022, "quarter": 4, "starts": 380},
            {"region": "ALL",   "year": 2022, "quarter": 1, "starts": 5000},
            {"region": "ALL",   "year": 2022, "quarter": 2, "starts": 5500},
            {"region": "ALL",   "year": 2022, "quarter": 3, "starts": 5200},
            {"region": "ALL",   "year": 2022, "quarter": 4, "starts": 4800},
        ])

    def _make_loader(self, data_map):
        def _load(name):
            return data_map.get(name)
        return _load

    def test_population_shares_from_data(self, monkeypatch, mock_population_df):
        import housingsim.calibration as cal_mod
        monkeypatch.setattr(cal_mod, "_load",
                            self._make_loader({"scb_population": mock_population_df}))
        shares = calibrate_population_shares(reference_year=2022)
        assert len(shares) == 4
        assert abs(shares.sum() - 1.0) < 1e-6
        # STHLM should be smallest of the three known counties (not REST)
        assert shares[0] < shares[3]  # STHLM < REST

    def test_income_multipliers_from_data(self, monkeypatch, mock_income_df):
        import housingsim.calibration as cal_mod
        monkeypatch.setattr(cal_mod, "_load",
                            self._make_loader({"scb_income": mock_income_df}))
        mults = calibrate_income_multipliers(reference_year=2022)
        assert mults["STHLM"] > mults["REST"]
        assert mults["STHLM"] > 1.0

    def test_construction_permits_from_data(self, monkeypatch, mock_construction_df):
        import housingsim.calibration as cal_mod
        monkeypatch.setattr(cal_mod, "_load",
                            self._make_loader({"scb_construction": mock_construction_df}))
        permits = calibrate_construction_permits(reference_year=2022)
        assert set(REGIONS).issubset(set(permits.keys()))
        # STHLM should have more starts than MALMO
        assert permits["STHLM"] > permits["MALMO"]
        # Monthly should be ~(annual/12); mock STHLM = 5000 starts/yr → ~416/mo
        assert 450 > permits["STHLM"] > 350

    def test_rate_path_from_data(self, monkeypatch, mock_repo_rate_df):
        import housingsim.calibration as cal_mod
        monkeypatch.setattr(cal_mod, "_load",
                            self._make_loader({"riksbank_repo_rate": mock_repo_rate_df}))
        path = calibrate_rate_path("2020-01", "2021-12")
        assert len(path) >= 1
        # First entry should be from the start or first available
        assert all("from" in p and "value" in p for p in path)

    def test_price_calibration_applied_to_stocks(
        self, monkeypatch, mock_prices_df
    ):
        import housingsim.calibration as cal_mod
        monkeypatch.setattr(
            cal_mod, "_load",
            self._make_loader({
                "scb_brf_prices": mock_prices_df,
                "scb_house_prices": mock_prices_df.rename(
                    columns={"price_sek": "price_sek_per_m2"}
                ),
            })
        )
        stocks = build_initial_stocks()
        from housingsim.calibration import calibrate_prices
        calibrate_prices(stocks, reference_year=2022)
        # STHLM should have higher price index than REST for BRF
        sthlm_brf = stocks[0].price_index[1, 2]  # STHLM, BRF, mid quality
        rest_brf = stocks[3].price_index[1, 2]    # REST, BRF, mid quality
        assert sthlm_brf > rest_brf

    def test_simulation_with_calibration(self, monkeypatch, mock_population_df,
                                          mock_income_df, mock_prices_df):
        """End-to-end: CalibrationData applied to a short simulation run."""
        import housingsim.calibration as cal_mod
        monkeypatch.setattr(
            cal_mod, "_load",
            self._make_loader({
                "scb_population": mock_population_df,
                "scb_income": mock_income_df,
                "scb_brf_prices": mock_prices_df,
                "scb_house_prices": mock_prices_df.rename(
                    columns={"price_sek": "price_sek_per_m2"}
                ),
            })
        )
        cal = CalibrationData.load(reference_year=2022)
        assert cal.is_real_data

        cfg = ScenarioConfig(
            scenario_name="cal_test",
            time=TimeRange(start="2024-01", end="2024-04"),
            n_households=200,
            random_seed=0,
        )
        recorder = run_scenario(cfg, verbose=False, calibration=cal)
        df = recorder.to_dataframe()
        assert len(df) == 4
        assert not df.select_dtypes(include="number").isnull().any().any()


# ---------------------------------------------------------------------------
# Data fetch helpers (offline/mocked)
# ---------------------------------------------------------------------------

class TestDataFetchParsing:
    """Test the parsing logic in data_fetch without making real HTTP calls."""

    def test_repo_rate_to_path_empty(self):
        from housingsim.data_fetch import repo_rate_to_scenario_path
        df = pd.DataFrame(columns=["date", "rate"])
        path = repo_rate_to_scenario_path(df, "2024-01", "2024-06")
        assert path == []

    def test_repo_rate_to_path_single_change(self):
        from housingsim.data_fetch import repo_rate_to_scenario_path
        rows = [
            {"date": "2024-01-15", "rate": 0.02},
            {"date": "2024-02-15", "rate": 0.02},
            {"date": "2024-03-15", "rate": 0.025},
            {"date": "2024-04-15", "rate": 0.025},
        ]
        df = pd.DataFrame(rows)
        path = repo_rate_to_scenario_path(df, "2024-01", "2024-06")
        # Should emit 2 entries: initial 0.02, then 0.025 when it changes
        assert len(path) == 2
        assert path[0]["value"] == pytest.approx(0.02)
        assert path[1]["value"] == pytest.approx(0.025)
