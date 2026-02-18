# Swedish Housing Market Simulation

An agent-based simulation model of the Swedish housing market. Households make monthly tenure decisions under configurable policy scenarios (LTV/amortization rules, construction permitting speed, interest-rate paths).

## Quick start

```bash
pip install -e ".[dev]"

# Run baseline scenario (10k households, 3 seeds, 2024–2030)
python scripts/run_simulation.py scenarios/baseline.yaml

# Run multiple scenarios for comparison
python scripts/run_simulation.py scenarios/baseline.yaml scenarios/ltv_90_from_2026_04.yaml

# Run all scenarios
python scripts/run_simulation.py scenarios/

# Plot results
python scripts/plot_results.py runs/baseline/ --output plots/baseline.png
python scripts/plot_results.py runs/baseline/ runs/ltv_90_from_2026_04/ --output plots/comparison.png

# Run tests
pytest
```

## Architecture

```
housingsim/
├── config.py        Pydantic scenario/policy models + piecewise-constant time paths
├── state.py         RegionStock – housing units, price indices, queue pressure
├── households.py    Struct-of-arrays household agents + utility-based decisions
├── bank.py          Credit policy: LTV cap, DSTI stress test, amortization rules
├── markets.py       Ownership market (price update) + rental queue allocation
├── developers.py    Construction pipeline with approval and build delays
├── metrics.py       KPI recorder → DataFrame/Parquet
└── simulation.py    Monthly simulation loop + batch runner

scenarios/           YAML scenario configs
scripts/
├── run_simulation.py   CLI entry point
└── plot_results.py     Multi-scenario plotting
tests/
└── test_smoke.py       31 unit + integration tests
runs/                   Output directory (created at runtime)
```

## Simulation loop (monthly)

1. **Macro update** – mortgage rate from scenario path, income growth
2. **Life events** – job changes, new children, separations (stochastic)
3. **Decisions** – each household scores all (region, tenure, quality) options via a utility function and picks the best move vs. staying
4. **Rental allocation** – probabilistic queue allocation; queue scores accumulate for existing renters
5. **Bank qualification** – LTV cap, stressed DSTI check, amortization rules
6. **Ownership market** – match qualified buyers to turnover supply; update price index via log-linear excess-demand rule
7. **Apply outcomes** – update household tenure, region, mortgage balance
8. **Amortize mortgages** – update mortgage balances and housing costs
9. **Construction** – developers start projects; pipeline advances; completions add to stock
10. **Record KPIs**

## Regions and tenure types

| Region | Description |
|--------|-------------|
| STHLM  | Stockholm metropolitan area |
| GBG    | Gothenburg |
| MALMO  | Malmö |
| REST   | Rest of Sweden |

Tenure types: `rent_regulated`, `buy_bostadsratt`, `buy_smallhouse`

Quality tiers: 1–5 (low to high)

## Scenario format

```yaml
scenario_name: "my_scenario"
time:
  start: "2024-01"
  end: "2030-12"
n_households: 10000
random_seed: 42
n_runs: 3

macro:
  policy_rate_path:
    - {from: "2024-01", value: 0.0225}
    - {from: "2026-01", value: 0.0150}

credit_policy:
  ltv_cap:
    - {from: "2024-01", value: 0.85}
    - {from: "2026-04", value: 0.90}

construction_policy:
  approval_delay_months: 10
  permits_multiplier:
    - {from: "2027-01", region: "STHLM", value: 1.40}
```

## KPIs recorded

Per month (global):
- `homeownership_rate`
- `median_housing_cost_burden`
- `share_shut_out_buying`
- `mobility_rate`
- `mortgage_rate`

Per month, per region (`r_{region}_`):
- `price_idx_brf`, `price_idx_small`  – price indices (base = initial)
- `brf_price_abs`, `small_price_abs` – absolute price at mid quality
- `queue_pressure`
- `starts`, `completions`, `pipeline_total`
- `stock_rent`, `stock_brf`, `stock_small`
- `homeownership_rate`, `median_burden`

## Performance

With N=10,000 households and a 7-year run (84 months), a single seed completes in approximately 15–30 seconds on a modern CPU. Scale to N=50,000 by switching to fully vectorised numpy operations (the struct-of-arrays design supports this without structural changes).
