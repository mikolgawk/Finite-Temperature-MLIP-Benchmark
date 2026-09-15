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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = (
        ("Figure 2", HERE / "figure_2.py"),
        ("Aggregate RMSEs by system type", HERE / "compute_mean_rmses_by_system_type.py"),
        ("Figure SI 2", HERE / "figure_SI_2.py"),
        ("Figure SI 3", HERE / "figure_SI_3.py"),
        ("Figure SI 4", HERE / "figure_SI_4.py"),
    )
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    failures: list[str] = []

    for label, script in stages:
        command = [sys.executable, "-u", str(script)]
        print(f"\n=== {label} ===", flush=True)
        print(f"$ {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=HERE, env=env, check=False)
        if result.returncode:
            failures.append(f"{label} (exit {result.returncode})")

    if failures:
        raise SystemExit("Failed plot stages: " + ", ".join(failures))
    print(f"\nAll energy/force plots are in {HERE / 'plots'}")


if __name__ == "__main__":
    main()
