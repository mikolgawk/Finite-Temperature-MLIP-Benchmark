#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib>=3.9",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scipy>=1.13",
#   "seaborn>=0.13",
# ]
# ///
"""Create every pressure plot from existing pressure metrics."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE.parent
DATASETS = (
    "ase:md_eager",
    "torchsim:md_eager",
    "ase:md_accelerated",
    "torchsim:md_accelerated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASETS,
        help="backend:mode to plot (repeatable; default: every available dataset)",
    )
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--plots-dir", type=Path, default=HERE / "plots")
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    return parser.parse_args()


def has_pressure_data(path: Path) -> bool:
    return any(path.glob("*_same-simulation-length_pressure_per_frame.csv"))


def run_stage(label: str, command: list[str], dry_run: bool) -> bool:
    print(f"\n=== {label} ===", flush=True)
    print(f"$ {shlex.join(command)}", flush=True)
    if dry_run:
        return True
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    return subprocess.run(command, cwd=HERE, env=env, check=False).returncode == 0


def filtered_ranking(
    source: Path, destination: Path, *, backend: str, mode: str
) -> Path:
    """Select one provenance when an aggregate ranking CSV contains many."""
    if not source.is_file():
        return source
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if "backend" not in fieldnames and "mode" not in fieldnames:
        return source
    selected = [
        row for row in rows
        if ("backend" not in fieldnames or row["backend"].lower() == backend.lower())
        and ("mode" not in fieldnames or row["mode"].lower() == mode.lower())
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    return destination


def main() -> None:
    args = parse_args()
    if args.bins < 2:
        raise SystemExit("--bins must be at least 2")
    results_dir = args.results_dir.resolve()
    plots_dir = args.plots_dir.resolve()
    datasets = args.dataset or [
        dataset
        for dataset in DATASETS
        if has_pressure_data(results_dir.joinpath(*dataset.split(":")))
    ]
    if not datasets and not args.dry_run:
        raise SystemExit(f"No pressure datasets found in {results_dir}")
    if not datasets:
        datasets = list(DATASETS)

    mae_ranking = results_dir / "model_mean_pressure_comparison.csv"
    error_ranking = results_dir / "model_pressure_error_metric.csv"
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="pressure-plot-inputs-") as temp_dir:
        temp_root = Path(temp_dir)
        for dataset in dict.fromkeys(datasets):
            backend, mode = dataset.split(":")
            pressures_dir = results_dir / backend / mode
            output_dir = plots_dir / backend / mode
            source = f"mlip-trajs-{backend}" + ("-accelerated" if mode == "md_accelerated" else ("-eager" if backend == "torchsim" else ""))
            reference_args = (
                ["--reference-file", str(args.reference_file.resolve())]
                if args.reference_file else []
            )
            base = ["--pressures-dir", str(pressures_dir), *reference_args]
            dataset_temp = temp_root / backend / mode
            selected_mae = filtered_ranking(
                mae_ranking, dataset_temp / mae_ranking.name, backend=backend, mode=mode
            )
            selected_error = filtered_ranking(
                error_ranking, dataset_temp / error_ranking.name, backend=backend, mode=mode
            )
            stages = [
            (
                "Figure 4",
                [sys.executable, "-u", str(HERE / "figure_4.py"), *base,
                 "--ranking-file", str(selected_mae), "--bins", str(args.bins),
                 "--output-file", str(output_dir / "plot_pressure_panel_combined_pressure_mae.pdf")],
            ),
            (
                "Figure SI 4",
                [sys.executable, "-u", str(HERE / "figure_SI_4.py"), *base,
                 "--output-file", str(output_dir / "plot_pressure_violin_distributions.pdf")],
            ),
            (
                "Figure SI 6",
                [sys.executable, "-u", str(HERE / "figure_SI_6.py"), *base,
                 "--ranking-file", str(selected_error), "--bins", str(args.bins),
                 "--output-file", str(output_dir / "plot_pressure_panel_combined_pressure_errors.pdf")],
            ),
            (
                "Figure SI 15",
                [sys.executable, "-u", str(HERE / "figure_SI_15.py"),
                 "--timings-dir", str(CONFIG_DIR / "data" / source),
                 "--pressure-scores-file", str(selected_error),
                 "--output-file", str(output_dir / "plot_SI_pareto_pressure_time_same_length.pdf")],
            ),
            ]
            for label, command in stages:
                if not run_stage(f"{label} ({dataset})", command, args.dry_run):
                    failures.append(f"{label} ({dataset})")

    if failures:
        raise SystemExit("Failed plot stages: " + ", ".join(failures))
    print(f"\nAll pressure plots are in {plots_dir}")


if __name__ == "__main__":
    main()
