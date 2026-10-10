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
"""Create VDOS plots, including correlations excluding hydrogen and Pt+H2O."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from system_filters import add_molecular_crystal_option
CONFIG_DIR = HERE.parent
SOURCES = (
    "mlip-trajs-ase",
    "mlip-trajs-torchsim-eager",
    "mlip-trajs-ase-accelerated",
    "mlip-trajs-torchsim-accelerated",
)
MODEL_MEANS = "vdos_model_mean_ev_normalized_same_simulation_length.csv"
NO_HYDROGEN_FORCE_SUFFIX = "_no_hydrogen_force_rmse"
NO_HYDROGEN_PT_WATER_SUFFIX = "_no_hydrogen_no_pt_water"


def without_hydrogen_force_rmse(
    command: list[str], *, exclude_pt_water: bool = False,
) -> list[str]:
    """Exclude hydrogen and optionally Pt+H2O using distinct output paths."""
    variant = command.copy()
    suffix = NO_HYDROGEN_PT_WATER_SUFFIX if exclude_pt_water else NO_HYDROGEN_FORCE_SUFFIX
    for index, argument in enumerate(variant[:-1]):
        if argument.endswith("output-file"):
            output = Path(variant[index + 1])
            variant[index + 1] = str(output.with_stem(output.stem + suffix))
    variant.append("--exclude-hydrogen")
    if exclude_pt_water:
        variant.append("--exclude-pt-water")
    return variant


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
    add_molecular_crystal_option(parser)
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
                    "--output-file", str(source_plots / "plot_vdos_results.pdf"),
                    "--title", f"Matched-length VDOS error: {source}",
                ],
            ),
            (
                "Figure 5",
                [sys.executable, "-u", str(HERE / "figure_5.py"), *common,
                 "--output-file", str(source_plots / "figure_5.pdf")],
            ),
            (
                "Figure 6 and its SI companion",
                [
                    sys.executable, "-u", str(HERE / "figure_6.py"), *common,
                    "--vdos-model-means-file", str(source_results / MODEL_MEANS),
                    "--pressure-file", str(pressure_file.resolve()),
                    "--f1-file", str(args.f1_file.resolve()),
                    "--ksrme-file", str(args.ksrme_file.resolve()),
                    "--output-file", str(source_plots / "figure_6.pdf"),
                    "--si-output-file", str(source_plots / "figure_6_all_labels.pdf"),
                ],
            ),
            (
                "Figures SI 7, 8, and 9",
                [
                    sys.executable, "-u", str(HERE / "figure_SI_7_8_9.py"), *common,
                    "--vdos-model-means-file", str(source_results / MODEL_MEANS),
                    "--pressure-file", str(pressure_file.resolve()),
                    "--f1-file", str(args.f1_file.resolve()),
                    "--ksrme-file", str(args.ksrme_file.resolve()),
                    "--rdf-output-file", str(source_plots / "figure_SI_7_8_9_rdf.pdf"),
                    "--pressure-output-file", str(source_plots / "figure_SI_7_8_9_pressure.pdf"),
                    "--vdos-output-file", str(source_plots / "figure_SI_7_8_9_vdos.pdf"),
                ],
            ),
            (
                "Figure SI 16",
                [sys.executable, "-u", str(HERE / "figure_SI_16.py"), *common,
                 "--output-file", str(source_plots / "figure_SI_16.pdf")],
            ),
        ]
        correlation_stages = [
            (label + label_suffix, without_hydrogen_force_rmse(command, exclude_pt_water=exclude_pt_water))
            for label_suffix, exclude_pt_water in (
                (" without hydrogen", False),
                (" without hydrogen and Pt+H2O", True),
            )
            for label, command in stages
            if Path(command[2]).name in {"figure_6.py", "figure_SI_7_8_9.py"}
        ]
        stages.extend(correlation_stages)
        for label, command in stages:
            if args.include_molecular_crystals and Path(command[2]).name in {'figure_5.py', 'figure_SI_16.py'}:
                command.append("--include-molecular-crystals")
            if not run_stage(f"{label} ({source})", command, args.dry_run):
                failures.append(f"{label} ({source})")

    if failures:
        raise SystemExit("Failed plot stages: " + ", ".join(failures))
    print(f"\nAll VDOS plots are in {plots_dir}")


if __name__ == "__main__":
    main()
