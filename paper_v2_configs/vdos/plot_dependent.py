#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["adjustText>=1.3", "matplotlib>=3.9", "numpy>=1.26", "pandas>=2.2", "seaborn>=0.13"]
# ///
"""Replot combined Pareto fronts and observable correlations with updated VDOS.

RDF, pressure, RMSE and timing inputs retain their existing benchmark definitions.
Every generated PDF and merged table stays inside ``vdos`` by default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PARETO = HERE.parent / "pareto_plots"
MODEL_FILE = "vdos_model_mean_ev_normalized_same_simulation_length.csv"
SOURCES = ("mlip-trajs-torchsim-eager", "mlip-trajs-torchsim-accelerated")
ACCELERATED_COMPARISON_SOURCES = ("mlip-trajs-torchsim-accelerated", "mlip-trajs-ase-accelerated")


def commands(source: str, results_dir: Path, plots_dir: Path) -> list[list[str]]:
    results = results_dir / source
    plots = plots_dir / source
    vdos_file = results / MODEL_FILE
    output: list[list[str]] = []
    for name in ("figure_7", "figure_SI_13"):
        command = [
            sys.executable, "-u", str(PARETO / f"{name}.py"),
            "--source", source, "--vdos-metrics-file", str(vdos_file),
            "--output-file", str(plots / f"{name}.pdf"),
            "--output-csv", str(results / f"{name}_metrics.csv"),
        ]
        if name == "figure_7":
            command.extend(("--frontier-csv", str(results / "figure_7_frontier.csv")))
        output.append(command)
    for name in ("figure_SI_11", "figure_SI_12"):
        script = HERE / f"{name}.py"
        output.append([
            sys.executable, "-u", str(script),
            "--source", source, "--vdos-file", str(vdos_file),
            "--output-plot", str(plots / f"{name}.pdf"),
            "--output-csv", str(results / f"{name}_metrics.csv"),
            "--output-json", str(results / f"{name}_correlation.json"),
        ])
    # The TorchSim accelerated stage owns the comparison, so eager/accelerated
    # invocations cannot produce duplicate overlays. Missing ASE data simply
    # leaves SI18 out; the standalone wrapper always requires both inputs.
    if source == ACCELERATED_COMPARISON_SOURCES[0] and all(
        (results_dir / runner / MODEL_FILE).is_file()
        for runner in ACCELERATED_COMPARISON_SOURCES
    ):
        output.append([
            sys.executable, "-u", str(HERE / "figure_SI_18.py"),
            "--results-dir", str(results_dir), "--plots-dir", str(plots_dir),
        ])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", choices=SOURCES, dest="sources")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--plots-dir", type=Path, default=HERE / "plots")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    plots_dir = args.plots_dir.resolve()
    sources = args.sources or [
        source for source in SOURCES if (results_dir / source / MODEL_FILE).is_file()
    ]
    if not sources and not args.dry_run:
        raise SystemExit(f"No corrected VDOS summaries found in {results_dir}")
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    for source in dict.fromkeys(sources or SOURCES):
        for command in commands(source, results_dir, plots_dir):
            print(f"$ {shlex.join(command)}", flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=HERE, env=env, check=True)


if __name__ == "__main__":
    main()
