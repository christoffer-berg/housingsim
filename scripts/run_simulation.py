"""
CLI entry point for running housing market simulations.

Usage:
  python scripts/run_simulation.py scenarios/baseline.yaml
  python scripts/run_simulation.py scenarios/baseline.yaml --runs 5 --output runs/
  python scripts/run_simulation.py scenarios/baseline.yaml scenarios/ltv_90_from_2026_04.yaml

Run all scenarios in a directory:
  python scripts/run_simulation.py scenarios/ --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Swedish housing market simulation runner"
    )
    parser.add_argument(
        "scenarios",
        nargs="+",
        help="Path(s) to scenario YAML file(s) or a directory of YAML files",
    )
    parser.add_argument(
        "--output",
        default="runs",
        help="Output directory for results (default: runs/)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override n_runs in scenario config",
    )
    parser.add_argument(
        "--households",
        type=int,
        default=None,
        help="Override n_households in scenario config",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--all",
        dest="run_all",
        action="store_true",
        help="Run all .yaml files found in specified paths",
    )
    return parser.parse_args()


def collect_yaml_paths(raw_paths: list[str], run_all: bool) -> list[Path]:
    paths = []
    for raw in raw_paths:
        p = Path(raw)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.yaml")))
        elif p.suffix in {".yaml", ".yml"}:
            paths.append(p)
        else:
            print(f"Warning: skipping {p} (not a .yaml file or directory)")
    return paths


def main() -> None:
    # Add repo root to path so imports work when run directly
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))

    from housingsim.config import ScenarioConfig
    from housingsim.simulation import run_batch

    args = parse_args()
    yaml_paths = collect_yaml_paths(args.scenarios, args.run_all)

    if not yaml_paths:
        print("No scenario files found.")
        sys.exit(1)

    print(f"Found {len(yaml_paths)} scenario(s):")
    for p in yaml_paths:
        print(f"  - {p}")

    for path in yaml_paths:
        print(f"\n{'=' * 60}")
        print(f"Loading scenario: {path}")
        cfg = ScenarioConfig.from_yaml(str(path))

        if args.runs is not None:
            cfg = cfg.model_copy(update={"n_runs": args.runs})
        if args.households is not None:
            cfg = cfg.model_copy(update={"n_households": args.households})

        recorders = run_batch(
            cfg,
            output_dir=args.output,
            verbose=not args.quiet,
        )
        print(f"Completed: {cfg.scenario_name} ({len(recorders)} run(s))")
        df = recorders[0].to_dataframe()
        print(f"  Months simulated: {len(df)}")
        print(f"  Final homeownership rate: {df['homeownership_rate'].iloc[-1]:.3f}")
        print(f"  Final median burden: {df['median_housing_cost_burden'].iloc[-1]:.3f}")
        brf_col = "r_sthlm_price_idx_brf"
        if brf_col in df.columns:
            print(f"  STHLM BRF price index (end): {df[brf_col].iloc[-1]:.3f}")

    print(f"\nAll results saved to: {args.output}/")


if __name__ == "__main__":
    main()
