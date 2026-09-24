#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib>=3.9",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "seaborn>=0.13",
# ]
# ///
"""Create every energy/force RMSE plot from existing metric CSV files."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from metric_sources import SOURCES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument('--source', choices=SOURCES, action='append', dest='sources')
    parser.add_argument('--results-dir', type=Path, default=HERE / 'results')
    parser.add_argument('--plots-dir', type=Path, default=HERE / 'plots')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = (
        ("Figure 2", HERE / "figure_2.py"),
        ("Figure SI 2", HERE / "figure_SI_2.py"),
        ("Figure SI 3", HERE / "figure_SI_3.py"),
        ("Figure SI 4", HERE / "figure_SI_4.py"),
    )
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    failures: list[str] = []

    sources = args.sources or [source for source in SOURCES
        if (args.results_dir / source / 'rmse_per_system.csv').is_file()]
    if not sources:
        raise SystemExit('No per-source RMSE results found; run generate_metrics.py --metric energy-force first.')
    for source in dict.fromkeys(sources):
        for label, script in stages:
            command = [sys.executable, "-u", str(script), '--source', source,
                       '--results-dir', str(args.results_dir.resolve()), '--plots-dir', str(args.plots_dir.resolve())]
            print(f"\n=== {label} ===", flush=True)
            print(f"$ {shlex.join(command)}", flush=True)
            if args.dry_run:
                continue
            result = subprocess.run(command, cwd=HERE, env=env, check=False)
            if result.returncode:
                failures.append(f"{label} ({source}, exit {result.returncode})")

    if failures:
        raise SystemExit("Failed plot stages: " + ", ".join(failures))
    print(f"\nAll energy/force plots are in {args.plots_dir.resolve()}")


if __name__ == "__main__":
    main()
