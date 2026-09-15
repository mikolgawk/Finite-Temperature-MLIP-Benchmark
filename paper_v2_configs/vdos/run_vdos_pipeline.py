#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "ase>=3.26",
#   "h5py>=3.11",
#   "matplotlib>=3.9",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scipy>=1.13",
# ]
# ///
"""Compute and plot VDOS results for the four MD trajectory sources.

Run with::

    uv run run_vdos_pipeline.py

The four supported cases are baseline/accelerated MD crossed with ASE/TorchSim.
Only sources containing ``nvt_*.h5`` files are selected by default. Results and
plots are kept separate per source, following ``../rdfs/run_rdf_pipeline.py``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
COMPUTE_SCRIPT = SCRIPT_DIR / "get_normalized_VDOS.py"
PLOT_SCRIPT = SCRIPT_DIR / "plot_vdos_results.py"
MODEL_OUTPUT = "vdos_model_mean_ev_normalized_same_simulation_length.csv"

SOURCES = (
    "mlip-trajs-ase",
    "mlip-trajs-torchsim-eager",
    "mlip-trajs-ase-accelerated",
    "mlip-trajs-torchsim-accelerated",
)


def available_sources(data_dir: Path) -> list[str]:
    return [source for source in SOURCES if any((data_dir / source).glob("*/nvt_*.h5"))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot matched-length VDOS results for ASE/TorchSim MD."
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=SOURCES,
        dest="sources",
        help="trajectory source to process (repeatable; default: every available source)",
    )
    parser.add_argument("--system", action="append", dest="systems", help="system filter (repeatable)")
    parser.add_argument("--model", action="append", dest="models", help="model filter (repeatable)")
    parser.add_argument("--exclude-system-type", action="append", dest="excluded_system_types", help="exclude this system type from computation (repeatable)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="trajectory data root")
    parser.add_argument("--metadata", type=Path, help="reference md_metadata.json")
    parser.add_argument("--results-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--plots-dir", type=Path, default=SCRIPT_DIR / "plots")
    parser.add_argument("--e-min", type=float, help="minimum energy in eV used for scoring")
    parser.add_argument("--e-max", type=float, help="maximum energy in eV used for scoring")
    parser.add_argument("--pad-factor", type=int, default=1, help="FFT zero-padding factor")
    parser.add_argument("--chunk-size", type=int, default=32, help="Cartesian signals per FFT chunk")
    parser.add_argument(
        "--numerical-mlip-velocities",
        action="store_true",
        help="differentiate MLIP positions instead of using stored velocities",
    )
    parser.add_argument("--overwrite", action="store_true", help="recompute existing spectra")
    parser.add_argument("--plots-only", action="store_true", help="skip VDOS computation")
    parser.add_argument("--compute-only", action="store_true", help="skip plot creation")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    args = parser.parse_args()
    if args.plots_only and args.compute_only:
        parser.error("--plots-only and --compute-only cannot be used together")
    if args.e_min is not None and args.e_max is not None and args.e_min >= args.e_max:
        parser.error("--e-min must be smaller than --e-max")
    if args.pad_factor < 1:
        parser.error("--pad-factor must be positive")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if args.metadata is not None and not args.metadata.is_file():
        parser.error(f"metadata file does not exist: {args.metadata}")
    return args


def run_stage(label: str, command: list[str], *, dry_run: bool) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(f"$ {shlex.join(command)}", flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vdos")
    subprocess.run(command, cwd=SCRIPT_DIR, env=env, check=True)


def computation_command(args: argparse.Namespace, sources: list[str]) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(COMPUTE_SCRIPT),
        "--data-dir",
        str(args.data_dir),
        "--results-dir",
        str(args.results_dir),
        "--pad-factor",
        str(args.pad_factor),
        "--chunk-size",
        str(args.chunk_size),
    ]
    for source in sources:
        command.extend(("--source", source))
    for system in args.systems or ():
        command.extend(("--system", system))
    for model in args.models or ():
        command.extend(("--model", model))
    for system_type in args.excluded_system_types or ():
        command.extend(("--exclude-system-type", system_type))
    if args.metadata is not None:
        command.extend(("--metadata", str(args.metadata)))
    if args.e_min is not None:
        command.extend(("--e-min", str(args.e_min)))
    if args.e_max is not None:
        command.extend(("--e-max", str(args.e_max)))
    if args.numerical_mlip_velocities:
        command.append("--numerical-mlip-velocities")
    if args.overwrite:
        command.append("--overwrite")
    if args.dry_run:
        command.append("--dry-run")
    return command


def plot_command(source: str, results_dir: Path, plots_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(PLOT_SCRIPT),
        "--model-means-file",
        str(results_dir / source / MODEL_OUTPUT),
        "--output-file",
        str(plots_dir / source / "plot_vdos_model_errors.pdf"),
        "--title",
        f"Matched-length VDOS error: {source}",
    ]


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    args.plots_dir = args.plots_dir.resolve()

    sources = list(dict.fromkeys(args.sources or available_sources(args.data_dir)))
    if not sources:
        raise SystemExit(
            f"No nvt_*.h5 trajectories found under {args.data_dir}; "
            "select a source explicitly with --source."
        )
    print(f"Trajectory sources: {', '.join(sources)}")

    if not args.plots_only:
        run_stage(
            "Compute reference and MLIP VDOS",
            computation_command(args, sources),
            dry_run=args.dry_run,
        )
    if args.compute_only:
        return

    for source in sources:
        model_file = args.results_dir / source / MODEL_OUTPUT
        if not args.dry_run and not model_file.is_file():
            raise FileNotFoundError(f"VDOS model means were not produced for {source}: {model_file}")
        run_stage(
            f"Create VDOS model-error plot ({source})",
            plot_command(source, args.results_dir, args.plots_dir),
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
