#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "ase>=3.26",
#   "h5py>=3.11",
#   "mdtraj>=1.10",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scipy>=1.13",
# ]
# ///
"""Generate the V2 energy/force, pressure, RDF, and VDOS metrics.

Run from anywhere in the repository with::

    uv run --script paper_v2_configs/generate_metrics.py

The model-specific energy/force and pressure evaluations must already have
produced their per-model CSV files. The RDF and VDOS stages consume the MD
trajectories under ``paper_v2_configs/data``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MetricStage:
    name: str
    description: str
    command: tuple[str, ...]
    working_directory: Path


def metric_stages() -> tuple[MetricStage, ...]:
    """Return the metric stages in the order required by the Pareto plots."""
    energy_force_dir = SCRIPT_DIR / "e_f_rmses"
    pressure_dir = SCRIPT_DIR / "pressures"
    rdf_dir = SCRIPT_DIR / "rdfs"
    vdos_dir = SCRIPT_DIR / "vdos"

    return (
        MetricStage(
            name="energy-force",
            description="Aggregate energy and force RMSEs",
            command=(
                sys.executable,
                "-u",
                str(energy_force_dir / "compute_mean_rmses_by_system_type.py"),
            ),
            working_directory=energy_force_dir,
        ),
        MetricStage(
            name="pressure",
            description="Compute pressure error and similarity metrics",
            command=(
                sys.executable,
                "-u",
                str(pressure_dir / "get_model_pressure_errors.py"),
            ),
            working_directory=pressure_dir,
        ),
        MetricStage(
            name="rdf",
            description="Compute matched-length RDF metrics",
            command=(
                sys.executable,
                "-u",
                str(rdf_dir / "run_rdf_pipeline.py"),
                "--compute-only",
            ),
            working_directory=rdf_dir,
        ),
        MetricStage(
            name="vdos",
            description="Compute matched-length VDOS metrics",
            command=(
                sys.executable,
                "-u",
                str(vdos_dir / "run_vdos_pipeline.py"),
                "--compute-only",
            ),
            working_directory=vdos_dir,
        ),
    )


def parse_args() -> argparse.Namespace:
    stage_names = tuple(stage.name for stage in metric_stages())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric",
        action="append",
        choices=stage_names,
        dest="metrics",
        help="metric to generate (repeatable; default: all four)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the selected commands without running them",
    )
    return parser.parse_args()


def run_stage(stage: MetricStage, *, dry_run: bool) -> None:
    print(f"\n=== {stage.description} ===", flush=True)
    print(f"$ {shlex.join(stage.command)}", flush=True)
    if dry_run:
        return

    environment = os.environ.copy()
    subprocess.run(
        stage.command,
        cwd=stage.working_directory,
        env=environment,
        check=True,
    )


def main() -> None:
    args = parse_args()
    selected = set(args.metrics) if args.metrics else None
    stages = [
        stage
        for stage in metric_stages()
        if selected is None or stage.name in selected
    ]

    for stage in stages:
        run_stage(stage, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\nDry run complete: {len(stages)} metric stage(s) selected.")
    else:
        print(f"\nGenerated {len(stages)} metric set(s).")


if __name__ == "__main__":
    main()
