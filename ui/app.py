"""
HousingSim Web UI – Streamlit application.

Launch with:
    streamlit run ui/app.py
    # or from repo root:
    pip install -e ".[ui]" && streamlit run ui/app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

# ── Resolve parent dir so housingsim package is importable ────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from housingsim.config import ScenarioConfig  # noqa: E402
from housingsim.simulation import run_scenario  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HousingSim – Swedish Housing Market Simulator",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Load preset scenarios from scenarios/ directory
# ─────────────────────────────────────────────────────────────────────────────
PRESETS: dict[str, dict] = {}
for _yaml_file in sorted((ROOT / "scenarios").glob("*.yaml")):
    with open(_yaml_file) as _f:
        PRESETS[_yaml_file.stem] = yaml.safe_load(_f)

PRESET_NAMES = list(PRESETS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "results_scenario" not in st.session_state:
    st.session_state.results_scenario = None

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
REGIONS_LOWER = ["sthlm", "gbg", "malmo", "rest"]
REGION_LABELS = {
    "sthlm": "Stockholm",
    "gbg": "Gothenburg",
    "malmo": "Malmö",
    "rest": "Rest of Sweden",
}
REGION_COLORS = {
    "sthlm": "#e74c3c",
    "gbg": "#3498db",
    "malmo": "#2ecc71",
    "rest": "#f39c12",
}


def _nested_get(d: dict, *keys, default):
    """Safely navigate a nested dict, returning default if any key is missing."""
    v = d
    for k in keys:
        if not isinstance(v, dict):
            return default
        v = v.get(k)
        if v is None:
            return default
    return v


def _ltv_from_preset(p: dict) -> float:
    ltv_list = _nested_get(p, "credit_policy", "ltv_cap", default=[])
    if isinstance(ltv_list, list) and ltv_list:
        return float(ltv_list[0].get("value", 0.85))
    return 0.85


def _ptax_from_preset(p: dict) -> float:
    pt_list = _nested_get(p, "tax_policy", "property_tax_rate", default=[])
    if isinstance(pt_list, list) and pt_list:
        return float(pt_list[0].get("value", 0.0))
    return 0.0


def _rate_from_preset(p: dict) -> float:
    rate_list = _nested_get(p, "macro", "policy_rate_path", default=[])
    if isinstance(rate_list, list) and rate_list:
        return float(rate_list[0].get("value", 0.0225))
    return float(_nested_get(p, "macro", "base_policy_rate", default=0.0225))


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – scenario builder
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("HousingSim")
    st.caption("Swedish Housing Market Simulator")
    st.divider()

    # Preset selector
    st.subheader("Load preset")
    preset_choice = st.selectbox(
        "Start from a pre-built scenario",
        PRESET_NAMES,
        index=0,
        help="Pick a pre-built policy scenario as a starting point, then adjust below.",
    )
    p = PRESETS.get(preset_choice, {})

    st.divider()

    # Simulation settings
    st.subheader("Simulation settings")
    scenario_name = st.text_input(
        "Scenario name",
        value=_nested_get(p, "scenario_name", default=preset_choice),
    )
    col_a, col_b = st.columns(2)
    n_households = col_a.number_input(
        "Households",
        min_value=500, max_value=50_000, step=500,
        value=int(_nested_get(p, "n_households", default=5_000)),
        help="Number of synthetic household agents. 5,000–10,000 is fast; 50,000 is thorough.",
    )
    n_runs = col_b.number_input(
        "Seeds (runs)",
        min_value=1, max_value=5,
        value=int(_nested_get(p, "n_runs", default=1)),
        help="Independent Monte Carlo runs. More seeds = uncertainty bands in results.",
    )
    random_seed = st.number_input(
        "Random seed", min_value=0, max_value=9999,
        value=int(_nested_get(p, "random_seed", default=42)),
    )

    st.divider()

    # Macro
    st.subheader("Macro parameters")
    base_rate = st.slider(
        "Riksbank policy rate",
        0.0, 0.10, _rate_from_preset(p), 0.0025,
        format="%.2f",
        help="Base interest rate set by Sveriges Riksbank. Mortgage rate = policy rate + spread.",
    )
    mortgage_spread = st.slider(
        "Mortgage spread (bank margin)",
        0.005, 0.030,
        float(_nested_get(p, "macro", "mortgage_spread", default=0.015)),
        0.005,
        format="%.3f",
        help="Banks add this spread on top of the Riksbank rate. Swedish banks typically charge 1–2%.",
    )
    income_growth = st.slider(
        "Annual income growth",
        0.0, 0.06,
        float(_nested_get(p, "macro", "income_growth_annual", default=0.025)),
        0.005,
        format="%.3f",
        help="Year-on-year gross income growth across all households.",
    )

    st.divider()

    # Credit policy
    st.subheader("Credit policy (Finansinspektionen)")
    ltv_cap = st.slider(
        "LTV cap (loan-to-value)",
        0.50, 0.95, _ltv_from_preset(p), 0.05,
        format="%.2f",
        help="Maximum fraction of property value that may be borrowed. FI currently sets this at 85%.",
    )
    max_dsti = st.slider(
        "Max DSTI (debt-service / income)",
        0.20, 0.60,
        float(_nested_get(p, "credit_policy", "max_dsti", default=0.40)),
        0.05,
        format="%.2f",
        help="Maximum share of gross monthly income a household may spend on mortgage service.",
    )

    st.divider()

    # Tax policy
    st.subheader("Tax policy")
    property_tax = st.slider(
        "Annual property tax rate",
        0.0, 0.02, _ptax_from_preset(p), 0.001,
        format="%.3f",
        help="Tax levied annually as % of market value. Sweden abolished progressive fastighetsskatt in 2008.",
    )
    exemption_years = st.number_input(
        "New-build tax exemption (years)",
        0, 10,
        int(_nested_get(p, "tax_policy", "exemption_years", default=0)),
        help="New constructions pay no property tax for this many years after completion.",
    )

    st.divider()

    # Construction policy
    st.subheader("Construction policy")
    st.caption("Monthly permits granted per region")
    cp = _nested_get(p, "construction_policy", "base_permits", default={})
    col_s, col_g = st.columns(2)
    col_m, col_r = st.columns(2)
    permits_sthlm = col_s.number_input("Stockholm", 0, 2000, int(cp.get("STHLM", 400)), 50)
    permits_gbg = col_g.number_input("Gothenburg", 0, 1000, int(cp.get("GBG", 200)), 50)
    permits_malmo = col_m.number_input("Malmö", 0, 1000, int(cp.get("MALMO", 150)), 50)
    permits_rest = col_r.number_input("Rest of SE", 0, 2000, int(cp.get("REST", 300)), 50)

    approval_delay = st.slider(
        "Approval delay (months)", 1, 36,
        int(_nested_get(p, "construction_policy", "approval_delay_months", default=10)),
        help="Months between permit application and granted approval.",
    )
    build_time = st.slider(
        "Build time (months)", 6, 48,
        int(_nested_get(p, "construction_policy", "build_time_months", default=18)),
        help="Months from permit approval to completed units entering the market.",
    )

    st.divider()

    # Assemble config dict from sidebar widgets
    cfg_dict = {
        "scenario_name": scenario_name,
        "n_households": int(n_households),
        "n_runs": int(n_runs),
        "random_seed": int(random_seed),
        "time": {"start": "2024-01", "end": "2030-12"},
        "macro": {
            "base_policy_rate": float(base_rate),
            "mortgage_spread": float(mortgage_spread),
            "income_growth_annual": float(income_growth),
            "migration_growth_annual": 0.005,
            "policy_rate_path": [{"from": "2024-01", "value": float(base_rate)}],
        },
        "credit_policy": {
            "ltv_cap": [{"from": "2024-01", "value": float(ltv_cap)}],
            "max_dsti": float(max_dsti),
            "stress_rate_buffer": 0.03,
            "amort_above_50_ltv": 0.02,
            "amort_above_70_ltv": 0.01,
        },
        "tax_policy": {
            "property_tax_rate": [{"from": "2024-01", "value": float(property_tax)}],
            "brf_fee_annual": 30_000.0,
            "exemption_years": int(exemption_years),
        },
        "construction_policy": {
            "approval_delay_months": int(approval_delay),
            "build_time_months": int(build_time),
            "base_permits": {
                "STHLM": int(permits_sthlm),
                "GBG": int(permits_gbg),
                "MALMO": int(permits_malmo),
                "REST": int(permits_rest),
            },
            "tenure_split": {
                "rent_regulated": 0.30,
                "buy_bostadsratt": 0.55,
                "buy_smallhouse": 0.15,
            },
            "supply_elasticity": 0.20,
        },
        "price_update_kappa": 0.10,
        "price_noise_std": 0.005,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────
st.title("Swedish Housing Market Simulator")

tab_assume, tab_run, tab_results = st.tabs(
    ["Assumptions", "Run simulation", "Results & download"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Assumptions
# ═══════════════════════════════════════════════════════════════════════════════
with tab_assume:
    mortgage_rate_eff = base_rate + mortgage_spread
    total_permits = permits_sthlm + permits_gbg + permits_malmo + permits_rest
    years_to_market = (approval_delay + build_time) / 12

    st.subheader("What this simulation assumes")
    st.caption(
        "Every slider on the left changes a policy assumption. "
        "This tab explains in plain language what each setting means and how it affects the model."
    )

    # ── Three columns of assumption cards ────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Mortgage conditions")
        st.info(
            f"The **Riksbank policy rate** is **{base_rate:.2%}**. "
            f"Banks add a **{mortgage_spread:.1%} spread**, so households "
            f"pay an effective **mortgage rate of {mortgage_rate_eff:.2%}**. "
            f"The model uses a stress-test rate that is 3 pp higher to check affordability."
        )

        st.markdown("#### Borrowing limits")
        st.info(
            f"Buyers may borrow at most **{ltv_cap:.0%} of the property value** (LTV cap). "
            f"Monthly debt-service payments cannot exceed **{max_dsti:.0%} of gross income** "
            f"under the stress-test rate. Households above LTV 70% must amortise 3%/year; "
            f"above 50% LTV, 2%/year."
        )

    with col2:
        st.markdown("#### Housing supply")
        st.info(
            f"**{total_permits:,} permits/month** are issued nationally "
            f"(Stockholm {permits_sthlm}, Gothenburg {permits_gbg}, "
            f"Malmö {permits_malmo}, Rest {permits_rest}). "
            f"New units take **{years_to_market:.1f} years** to reach the market "
            f"({approval_delay} months approval + {build_time} months construction). "
            f"Of new builds: **30% rentals, 55% BRF co-ops, 15% small houses**."
        )

        st.markdown("#### Simulation scope")
        st.info(
            f"The model runs **{int(n_households):,} synthetic Swedish households** "
            f"for **84 months (2024–2030)**, with **{int(n_runs)} independent seed{'s' if int(n_runs) > 1 else ''}**. "
            f"Households make monthly tenure/location decisions based on utility, "
            f"credit constraints, and local market conditions. "
            f"Four regions: Stockholm, Gothenburg, Malmö, Rest of Sweden."
        )

    with col3:
        st.markdown("#### Tax policy")
        if property_tax == 0.0:
            st.info(
                "**No property tax** is applied – matching Sweden's current system. "
                "Sweden abolished the progressive fastighetsskatt in 2008. "
                "BRF owners still pay HOA fees (~30,000 SEK/year baseline)."
            )
        else:
            monthly_cost_4msek = 4_000_000 * property_tax / 12
            st.info(
                f"A **{property_tax:.1%} annual property tax** is levied on market value. "
                f"For a 4 MSEK home that is roughly **{monthly_cost_4msek:,.0f} SEK/month** extra. "
                + (
                    f"New builds are **exempt for {int(exemption_years)} years**."
                    if exemption_years
                    else "No new-build exemption."
                )
            )

        st.markdown("#### What the model tracks")
        st.info(
            "Monthly snapshots of: homeownership rate, median housing-cost burden "
            "(cost/income), share of would-be buyers shut out by credit rules, "
            "BRF and small-house price indices, rental queue pressure, and "
            "construction starts/completions – all broken down by region."
        )

    # ── Key parameters table ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Key parameters at a glance")
    params_table = pd.DataFrame([
        {
            "Category": "Macro",
            "Parameter": "Policy rate",
            "Value": f"{base_rate:.2%}",
            "Plain-English effect": "Drives mortgage cost; higher rate → fewer buyers qualify",
        },
        {
            "Category": "Macro",
            "Parameter": "Mortgage spread",
            "Value": f"{mortgage_spread:.1%}",
            "Plain-English effect": "Bank margin on top of Riksbank rate",
        },
        {
            "Category": "Macro",
            "Parameter": "Income growth",
            "Value": f"{income_growth:.1%}/yr",
            "Plain-English effect": "Rising incomes improve affordability over time",
        },
        {
            "Category": "Credit",
            "Parameter": "LTV cap",
            "Value": f"{ltv_cap:.0%}",
            "Plain-English effect": "Higher cap → less equity required → more buyers qualify",
        },
        {
            "Category": "Credit",
            "Parameter": "Max DSTI",
            "Value": f"{max_dsti:.0%}",
            "Plain-English effect": "Tighter limit → fewer households can afford to buy",
        },
        {
            "Category": "Tax",
            "Parameter": "Property tax",
            "Value": f"{property_tax:.1%}/yr",
            "Plain-English effect": "Raises ongoing ownership cost → dampens demand",
        },
        {
            "Category": "Supply",
            "Parameter": "Monthly permits (national)",
            "Value": f"{total_permits:,} units",
            "Plain-English effect": "More permits → less upward price pressure over time",
        },
        {
            "Category": "Supply",
            "Parameter": "Time to market",
            "Value": f"{years_to_market:.1f} yrs",
            "Plain-English effect": "Long lags mean supply response is slow even after policy changes",
        },
    ])
    st.dataframe(params_table, use_container_width=True, hide_index=True)

    # ── Exact YAML ───────────────────────────────────────────────────────────
    with st.expander("Show exact YAML config (for reproducibility)"):
        st.code(yaml.dump(cfg_dict, sort_keys=False, allow_unicode=True), language="yaml")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Run
# ═══════════════════════════════════════════════════════════════════════════════
with tab_run:
    st.subheader("Run your scenario")
    st.markdown(
        f"**Scenario:** `{scenario_name}` &nbsp;·&nbsp; "
        f"**{int(n_households):,} households** &nbsp;·&nbsp; "
        f"**84 months** (2024–2030) &nbsp;·&nbsp; "
        f"**{int(n_runs)} seed{'s' if int(n_runs) > 1 else ''}**"
    )

    # Estimated runtime hint
    secs_per_run = max(5, int(n_households) // 500)  # rough heuristic
    total_secs = secs_per_run * int(n_runs)
    st.caption(f"Expected runtime: roughly {total_secs}–{total_secs * 2} seconds.")

    if st.button("Run Simulation", type="primary", use_container_width=True):
        try:
            cfg = ScenarioConfig.model_validate(cfg_dict)
        except Exception as e:
            st.error(f"Invalid configuration: {e}")
            st.stop()

        progress = st.progress(0, text="Starting…")
        all_dfs: list[pd.DataFrame] = []

        for run_i in range(cfg.n_runs):
            seed = cfg.random_seed + run_i
            progress.progress(
                run_i / cfg.n_runs,
                text=f"Running seed {run_i + 1}/{cfg.n_runs} (seed={seed})…",
            )
            try:
                recorder = run_scenario(cfg, seed_override=seed, verbose=False)
            except Exception as e:
                st.error(f"Simulation failed on run {run_i + 1}: {e}")
                raise
            df = recorder.to_dataframe()
            df["run"] = run_i
            df["seed"] = seed
            all_dfs.append(df)

        progress.progress(1.0, text="Complete!")
        combined = pd.concat(all_dfs, ignore_index=True)
        st.session_state.results = combined
        st.session_state.results_scenario = scenario_name
        st.success(
            f"Done! {len(all_dfs)} run(s) complete. "
            "Switch to the **Results & download** tab to explore the output."
        )

    # ── Summary of last run ───────────────────────────────────────────────────
    if st.session_state.results is not None:
        st.divider()
        st.caption(
            f"Last completed: **{st.session_state.results_scenario}** "
            f"({len(st.session_state.results['run'].unique())} seed(s))"
        )
        res = st.session_state.results
        df0 = res[res["run"] == 0].sort_values("month")
        start_row, end_row = df0.iloc[0], df0.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Homeownership (end)",
            f"{end_row['homeownership_rate']:.1%}",
            delta=f"{end_row['homeownership_rate'] - start_row['homeownership_rate']:+.1%} vs start",
        )
        col2.metric(
            "Cost burden · median (end)",
            f"{end_row['median_housing_cost_burden']:.1%}",
            delta=f"{end_row['median_housing_cost_burden'] - start_row['median_housing_cost_burden']:+.1%} vs start",
        )
        col3.metric(
            "Buyers shut out (end)",
            f"{end_row['share_shut_out_buying']:.1%}",
        )
        col4.metric(
            "Mortgage rate (end)",
            f"{end_row['mortgage_rate']:.2%}",
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Results & download
# ═══════════════════════════════════════════════════════════════════════════════
with tab_results:
    if st.session_state.results is None:
        st.info(
            "No results yet. "
            "Configure your scenario in the sidebar and click **Run Simulation** in the Run tab."
        )
        st.stop()

    res = st.session_state.results
    runs = sorted(res["run"].unique())
    multi_run = len(runs) > 1

    # ── Helper: add a time-series trace with optional uncertainty band ────────
    def add_trace(fig, df_all, col_name, label, color, row=1, col=1, dash="solid"):
        if not multi_run:
            df0 = df_all[df_all["run"] == 0].sort_values("month")
            fig.add_trace(
                go.Scatter(
                    x=df0["month"], y=df0[col_name],
                    name=label, line=dict(color=color, dash=dash),
                ),
                row=row, col=col,
            )
        else:
            grp = df_all.groupby("month")[col_name]
            months = grp.mean().index.tolist()
            mu = grp.mean().values
            sg = grp.std().fillna(0).values
            # Uncertainty band
            fig.add_trace(
                go.Scatter(
                    x=months + months[::-1],
                    y=list(mu + sg) + list((mu - sg)[::-1]),
                    fill="toself", fillcolor=color,
                    opacity=0.12, line=dict(width=0),
                    showlegend=False, name=f"{label} ±1σ",
                ),
                row=row, col=col,
            )
            # Mean line
            fig.add_trace(
                go.Scatter(
                    x=months, y=mu, name=label,
                    line=dict(color=color, dash=dash),
                ),
                row=row, col=col,
            )

    # ── Chart 1: National KPIs ────────────────────────────────────────────────
    st.subheader("National KPIs")
    fig_nat = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Homeownership rate",
            "Median housing-cost burden (cost/income)",
            "Share of would-be buyers shut out",
            "Mortgage rate",
        ],
        vertical_spacing=0.15,
    )

    kpi_specs = [
        (1, 1, "homeownership_rate", "Homeownership", "#2c3e50"),
        (1, 2, "median_housing_cost_burden", "Cost burden", "#c0392b"),
        (2, 1, "share_shut_out_buying", "Shut out", "#8e44ad"),
        (2, 2, "mortgage_rate", "Mortgage rate", "#16a085"),
    ]
    for r, c, col_name, label, color in kpi_specs:
        if col_name in res.columns:
            add_trace(fig_nat, res, col_name, label, color, row=r, col=c)

    fig_nat.update_yaxes(tickformat=".1%", row=1, col=1)
    fig_nat.update_yaxes(tickformat=".1%", row=1, col=2)
    fig_nat.update_yaxes(tickformat=".1%", row=2, col=1)
    fig_nat.update_yaxes(tickformat=".2%", row=2, col=2)
    fig_nat.update_layout(height=520, showlegend=False, margin=dict(t=50))
    st.plotly_chart(fig_nat, use_container_width=True)

    # ── Chart 2: BRF price indices by region ─────────────────────────────────
    st.subheader("BRF cooperative price indices by region")
    fig_prices = go.Figure()
    for region in REGIONS_LOWER:
        col_name = f"r_{region}_price_idx_brf"
        if col_name in res.columns:
            add_trace(fig_prices, res, col_name, REGION_LABELS[region], REGION_COLORS[region])
    fig_prices.add_hline(
        y=1.0, line_dash="dot", line_color="#aaa",
        annotation_text="Baseline (Jan 2024 = 1.0)",
        annotation_position="bottom right",
    )
    fig_prices.update_layout(
        height=360,
        yaxis_title="Price index (Jan 2024 = 1.0)",
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_prices, use_container_width=True)

    # ── Chart 3: Rental queue pressure + homeownership ───────────────────────
    st.subheader("Rental market pressure & regional homeownership")
    fig_region = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Rental queue pressure (>1 = excess demand)", "Regional homeownership rate"],
    )
    for region in REGIONS_LOWER:
        color = REGION_COLORS[region]
        label = REGION_LABELS[region]
        q_col = f"r_{region}_queue_pressure"
        own_col = f"r_{region}_homeownership_rate"
        if q_col in res.columns:
            add_trace(fig_region, res, q_col, label, color, row=1, col=1)
        if own_col in res.columns:
            add_trace(fig_region, res, own_col, label, color, row=1, col=2, dash="solid")

    fig_region.add_hline(
        y=1.0, line_dash="dot", line_color="#aaa", row=1, col=1,
    )
    fig_region.update_yaxes(tickformat=".1%", row=1, col=2)
    fig_region.update_layout(
        height=380,
        legend=dict(orientation="h", y=-0.2),
        showlegend=True,
    )
    st.plotly_chart(fig_region, use_container_width=True)

    # ── Chart 4: Construction ─────────────────────────────────────────────────
    st.subheader("Construction activity")
    fig_build = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Monthly construction starts", "Pipeline (units under construction)"],
    )
    for region in REGIONS_LOWER:
        color = REGION_COLORS[region]
        label = REGION_LABELS[region]
        s_col = f"r_{region}_starts"
        pipe_col = f"r_{region}_pipeline_total"
        if s_col in res.columns:
            add_trace(fig_build, res, s_col, label, color, row=1, col=1)
        if pipe_col in res.columns:
            add_trace(fig_build, res, pipe_col, label, color, row=1, col=2)
    fig_build.update_layout(
        height=360,
        legend=dict(orientation="h", y=-0.2),
        showlegend=True,
    )
    st.plotly_chart(fig_build, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Downloads
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Download results")
    st.caption(
        "Download the full monthly time-series for all KPIs and all seeds. "
        "Use these to build your own analyses, visualisations, or reports."
    )

    scenario_slug = (st.session_state.results_scenario or "results").replace(" ", "_")

    col_dl1, col_dl2, col_dl3 = st.columns(3)

    # CSV
    csv_bytes = res.to_csv(index=False).encode()
    col_dl1.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"{scenario_slug}_monthly_kpis.csv",
        mime="text/csv",
        use_container_width=True,
        help="Flat CSV with one row per month per seed. Good for pandas, R, Excel.",
    )

    # Excel (multi-sheet)
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        # Sheet 1: full monthly data
        res.to_excel(writer, index=False, sheet_name="Monthly KPIs")

        # Sheet 2: summary
        df0 = res[res["run"] == 0].sort_values("month")
        summary_rows = []
        kpi_meta = {
            "homeownership_rate": "Homeownership rate",
            "median_housing_cost_burden": "Median cost burden",
            "share_shut_out_buying": "Share shut out of buying",
            "mortgage_rate": "Mortgage rate",
        }
        for col_name, display_name in kpi_meta.items():
            if col_name in df0.columns:
                summary_rows.append({
                    "KPI": display_name,
                    "Start (2024-01)": df0.iloc[0][col_name],
                    "End (2030-12)": df0.iloc[-1][col_name],
                    "Min over period": df0[col_name].min(),
                    "Max over period": df0[col_name].max(),
                    "Mean over period": df0[col_name].mean(),
                })
        pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="Summary")

        # Sheet 3: config
        flat_cfg = {
            "scenario_name": scenario_name,
            "n_households": int(n_households),
            "n_runs": int(n_runs),
            "random_seed": int(random_seed),
            "policy_rate": base_rate,
            "mortgage_spread": mortgage_spread,
            "mortgage_rate_effective": mortgage_rate_eff,
            "income_growth_annual": income_growth,
            "ltv_cap": ltv_cap,
            "max_dsti": max_dsti,
            "property_tax_rate": property_tax,
            "exemption_years": int(exemption_years),
            "permits_stockholm": int(permits_sthlm),
            "permits_gothenburg": int(permits_gbg),
            "permits_malmo": int(permits_malmo),
            "permits_rest": int(permits_rest),
            "approval_delay_months": int(approval_delay),
            "build_time_months": int(build_time),
        }
        pd.DataFrame([flat_cfg]).T.reset_index().rename(
            columns={"index": "Parameter", 0: "Value"}
        ).to_excel(writer, index=False, sheet_name="Config")

    excel_buf.seek(0)
    col_dl2.download_button(
        label="Download Excel",
        data=excel_buf.getvalue(),
        file_name=f"{scenario_slug}_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        help="Three sheets: Monthly KPIs, Summary stats, and the exact config used.",
    )

    # Parquet
    parquet_buf = io.BytesIO()
    res.to_parquet(parquet_buf, index=False)
    parquet_buf.seek(0)
    col_dl3.download_button(
        label="Download Parquet",
        data=parquet_buf.getvalue(),
        file_name=f"{scenario_slug}_monthly_kpis.parquet",
        mime="application/octet-stream",
        use_container_width=True,
        help="Parquet format — compact, fast to load with pandas or any data tool.",
    )

    # Data preview
    with st.expander("Preview data (first 10 rows)"):
        preview_cols = [
            "month", "run", "seed",
            "homeownership_rate", "median_housing_cost_burden",
            "share_shut_out_buying", "mortgage_rate",
            "r_sthlm_price_idx_brf", "r_gbg_price_idx_brf",
            "r_malmo_price_idx_brf", "r_rest_price_idx_brf",
        ]
        display_cols = [c for c in preview_cols if c in res.columns]
        st.dataframe(
            res[display_cols].head(10).style.format({
                c: "{:.3f}" for c in display_cols if c not in ("month", "run", "seed")
            }),
            use_container_width=True,
        )
        st.caption(f"Full dataset: {len(res):,} rows × {len(res.columns)} columns")
