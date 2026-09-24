#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "adjustText>=1.3",
#   "matplotlib>=3.9",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "seaborn>=0.13",
# ]
# ///
"""Create every VDOS plot from existing, source-specific metric tables."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE.parent
SOURCES = (
    "mlip-trajs-ase",
    "mlip-trajs-torchsim-eager",
    "mlip-trajs-ase-accelerated",
    "mlip-trajs-torchsim-accelerated",
)
MODEL_MEANS = "vdos_model_mean_ev_normalized_same_simulation_length.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", choices=SOURCES, dest="sources")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--plots-dir", type=Path, default=HERE / "plots")
    parser.add_argument(
        "--pressure-file",
        type=Path,
        default=None,
    )
    parser.add_argument("--f1-file", type=Path, default=CONFIG_DIR / "data/matbench-scores/f1-scores.csv")
    parser.add_argument("--ksrme-file", type=Path, default=CONFIG_DIR / "data/matbench-scores/ksrme-scores.csv")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    return parser.parse_args()


def run_stage(label: str, command: list[str], dry_run: bool) -> bool:
    print(f"\n=== {label} ===", flush=True)
    print(f"$ {shlex.join(command)}", flush=True)
    if dry_run:
        return True
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    return subprocess.run(command, cwd=HERE, env=env, check=False).returncode == 0


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    plots_dir = args.plots_dir.resolve()
    sources = args.sources or [
        source for source in SOURCES if (results_dir / source / MODEL_MEANS).is_file()
    ]
    if not sources and not args.dry_run:
        raise SystemExit(f"No source-specific VDOS results found in {results_dir}")
    if not sources:
        sources = list(SOURCES)

    failures: list[str] = []
    for source in dict.fromkeys(sources):
        source_results = results_dir / source
        pressure_file = args.pressure_file or CONFIG_DIR / "pressures/results" / source / "model_pressure_error_metric.csv"
        source_plots = plots_dir / source
        common = ["--source", source]
        stages = [
            (
                "VDOS model-error overview",
                [
                    sys.executable, "-u", str(HERE / "plot_vdos_results.py"),
                    "--model-means-file", str(source_results / MODEL_MEANS),
                    "--output-file", str(source_plots / "plot_vdos_model_errors.pdf"),
                    "--title", f"Matched-length VDOS error: {source}",
                ],
            ),
            (
                "Figure 5",
                [sys.executable, "-u", str(HERE / "figure_5.py"), *common,
                 "--output-file", str(source_plots / "plot_vdos_panel_combined.pdf")],
            ),
            (
                "Figures SI 7, 8, and 9",
                [
                    sys.executable, "-u", str(HERE / "figure_SI_7_8_9.py"), *common,
                    "--pressure-file", str(pressure_file.resolve()),
                    "--f1-file", str(args.f1_file.resolve()),
                    "--ksrme-file", str(args.ksrme_file.resolve()),
                    "--rdf-output-file", str(source_plots / "plot_rdf_correlations_1x3.pdf"),
                    "--pressure-output-file", str(source_plots / "plot_pressure_histogram_correlations_1x3_same_length.pdf"),
                    "--vdos-output-file", str(source_plots / "plot_vdos_correlations_1x3_same_length.pdf"),
                ],
            ),
            (
                "Figure SI 16",
                [sys.executable, "-u", str(HERE / "figure_SI_16.py"), *common,
                 "--output-file", str(source_plots / "plot_SI_pareto_vdos_time_same_length.pdf")],
            ),
        ]
        for label, command in stages:
            if not run_stage(f"{label} ({source})", command, args.dry_run):
                failures.append(f"{label} ({source})")

    if failures:
        raise SystemExit("Failed plot stages: " + ", ".join(failures))
    print(f"\nAll VDOS plots are in {plots_dir}")


if __name__ == "__main__":
    main()
