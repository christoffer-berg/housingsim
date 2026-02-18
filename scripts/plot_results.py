"""
Plot simulation results from one or more scenario runs.

Usage:
  python scripts/plot_results.py runs/baseline/run_000.parquet
  python scripts/plot_results.py runs/baseline/ runs/ltv_90_from_2026_04/
  python scripts/plot_results.py runs/baseline/ --output plots/baseline.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


REGIONS = ["STHLM", "GBG", "MALMO", "REST"]
REGION_COLORS = {"sthlm": "#1f77b4", "gbg": "#ff7f0e", "malmo": "#2ca02c", "rest": "#9467bd"}
SCENARIO_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]


def load_run(path: Path) -> pd.DataFrame:
    """Load a single parquet run file."""
    return pd.read_parquet(path)


def load_scenario(path: Path) -> tuple[str, list[pd.DataFrame]]:
    """Load all run_*.parquet files from a scenario directory."""
    dfs = []
    for f in sorted(path.glob("run_*.parquet")):
        dfs.append(load_run(f))
    scenario_name = path.name
    return scenario_name, dfs


def scenario_band(dfs: list[pd.DataFrame], col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, lower 1-sigma, upper 1-sigma across runs."""
    arr = np.stack([df[col].values for df in dfs if col in df.columns])
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return mean, mean - std, mean + std


def date_ticks(months: list[str]) -> tuple[list[int], list[str]]:
    """Return tick positions and labels for Jan of each year."""
    ticks_pos = []
    ticks_lbl = []
    for i, m in enumerate(months):
        if m.endswith("-01"):
            ticks_pos.append(i)
            ticks_lbl.append(m[:4])
    return ticks_pos, ticks_lbl


def plot_scenarios(
    scenarios: list[tuple[str, list[pd.DataFrame]]],
    output: str | None = None,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle("Swedish Housing Market Simulation", fontsize=14, fontweight="bold")

    months = scenarios[0][1][0]["month"].tolist()
    ticks_pos, ticks_lbl = date_ticks(months)

    def _plot(ax, col, title, ylabel, pct=False, invert=False):
        for s_idx, (name, dfs) in enumerate(scenarios):
            color = SCENARIO_COLORS[s_idx % len(SCENARIO_COLORS)]
            if not dfs:
                continue
            mean, lo, hi = scenario_band(dfs, col)
            ax.plot(mean, label=name, color=color, linewidth=1.8)
            ax.fill_between(range(len(mean)), lo, hi, alpha=0.15, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(ticks_pos)
        ax.set_xticklabels(ticks_lbl, rotation=45, fontsize=8)
        if pct:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    def _plot_regions(ax, col_template, title, ylabel):
        for s_idx, (name, dfs) in enumerate(scenarios):
            if not dfs:
                continue
            linestyles = ["-", "--", "-.", ":"]
            for r_idx, region in enumerate(REGIONS):
                col = col_template.format(r=region.lower())
                if col not in dfs[0].columns:
                    continue
                mean, lo, hi = scenario_band(dfs, col)
                color = REGION_COLORS[region.lower()]
                ls = linestyles[s_idx % len(linestyles)]
                lbl = f"{region}" if s_idx == 0 else None
                ax.plot(mean, label=lbl, color=color, linestyle=ls, linewidth=1.5)
                if s_idx == 0:
                    ax.fill_between(range(len(mean)), lo, hi, alpha=0.10, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(ticks_pos)
        ax.set_xticklabels(ticks_lbl, rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    ax = axes.flat

    # Panel 1: BRF price index by region (first scenario baseline)
    _plot_regions(ax[0], "r_{r}_price_idx_brf", "BRF Price Index by Region", "Index")

    # Panel 2: Small house price index by region
    _plot_regions(ax[1], "r_{r}_price_idx_small", "Small House Price Index by Region", "Index")

    # Panel 3: Homeownership rate (global)
    _plot(ax[2], "homeownership_rate", "Homeownership Rate", "Rate", pct=True)

    # Panel 4: Median housing cost burden
    _plot(ax[3], "median_housing_cost_burden", "Median Housing Cost Burden", "Cost / Income", pct=True)

    # Panel 5: Share shut out of buying
    _plot(ax[4], "share_shut_out_buying", "Share Shut Out of Buying", "Share", pct=True)

    # Panel 6: Rental queue pressure (STHLM highlight)
    _plot(ax[5], "r_sthlm_queue_pressure", "STHLM Rental Queue Pressure", "Pressure Index")

    # Panel 7: Construction completions (all regions stacked) — just STHLM for simplicity
    _plot(ax[6], "r_sthlm_completions", "STHLM Monthly Completions", "Units")

    # Panel 8: Mortgage rate path
    _plot(ax[7], "mortgage_rate", "Mortgage Rate", "Rate", pct=True)

    # Panel 9: Mobility rate
    _plot(ax[8], "mobility_rate", "Monthly Mobility Rate", "Rate (fraction of agents)")

    plt.tight_layout()

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved plot → {output}")
    else:
        plt.show()


def main() -> None:
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(description="Plot simulation results")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to parquet files or scenario directories",
    )
    parser.add_argument("--output", default=None, help="Save plot to this PNG path")
    args = parser.parse_args()

    scenarios: list[tuple[str, list[pd.DataFrame]]] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            name, dfs = load_scenario(p)
            if dfs:
                scenarios.append((name, dfs))
            else:
                print(f"No run_*.parquet files found in {p}")
        elif p.suffix == ".parquet":
            scenarios.append((p.stem, [load_run(p)]))
        else:
            print(f"Skipping {p}")

    if not scenarios:
        print("No data to plot.")
        sys.exit(1)

    plot_scenarios(scenarios, output=args.output)


if __name__ == "__main__":
    main()
