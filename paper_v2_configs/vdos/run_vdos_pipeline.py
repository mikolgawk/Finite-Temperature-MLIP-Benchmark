#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "adjustText>=1.3",
#   "ase>=3.26",
#   "h5py>=3.11",
#   "matplotlib>=3.9",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scipy>=1.13",
#   "seaborn>=0.13",
# ]
# ///
"""Compute corrected VDOS spectra and produce the standard VDOS plots.

Run with::

    uv run --script paper_v2_configs/vdos/run_vdos_pipeline.py

Available TorchSim eager/accelerated sources are selected by default. Explicit
ASE sources are also supported. All corrected spectra, metrics, and figures
are kept under vdos, leaving the original analysis in vdos_legacy untouched.
"""

# Standalone scripts add the shared v2 helper directory before importing it.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from system_filters import add_molecular_crystal_option, excluded_system_types
DATA_DIR = SCRIPT_DIR.parent / "data"
COMPUTE_SCRIPT = SCRIPT_DIR / "get_normalized_VDOS.py"
PLOT_SCRIPT = SCRIPT_DIR / "plot_all.py"
MODEL_OUTPUT = "vdos_model_mean_ev_normalized_same_simulation_length.csv"

SOURCES = (
    "mlip-trajs-ase",
    "mlip-trajs-torchsim-eager",
    "mlip-trajs-ase-accelerated",
    "mlip-trajs-torchsim-accelerated",
)
DEFAULT_SOURCES = ("mlip-trajs-torchsim-eager", "mlip-trajs-torchsim-accelerated")


def available_sources(data_dir: Path) -> list[str]:
    return [source for source in DEFAULT_SOURCES if any((data_dir / source).glob("*/nvt_*.h5"))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot corrected matched-length VDOS results for ASE/TorchSim MD."
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=SOURCES,
        dest="sources",
        help="trajectory source to process (repeatable; default: available TorchSim eager/accelerated sources)",
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
    parser.add_argument(
        "--estimator", choices=("welch", "periodogram"), default="welch",
        help="spectral estimator (default: Welch averaging)",
    )
    parser.add_argument(
        "--weighting", choices=("mass", "velocity"), default="mass",
        help="sum mass-weighted or unweighted velocity powers (default: mass)",
    )
    parser.add_argument(
        "--window-fs", type=float, default=2000.0,
        help="requested Welch segment duration in fs; shortened for each matched pair to provide --min-segments (default: 2000)",
    )
    parser.add_argument(
        "--min-segments", type=int, default=4,
        help="minimum number of Welch segments with 50%% overlap (default: 4)",
    )
    parser.add_argument("--pad-factor", type=int, default=4, help="FFT zero-padding factor (default: 4; interpolates frequency bins only)")
    parser.add_argument("--chunk-size", type=int, default=32, help="Cartesian signals per FFT chunk")
    parser.add_argument(
        "--numerical-mlip-velocities",
        action="store_true",
        help="Compatibility option: corrected analysis always differentiates unwrapped positions for both trajectories.",
    )
    parser.add_argument("--overwrite", action="store_true", help="recompute existing spectra")
    parser.add_argument("--plots-only", action="store_true", help="skip VDOS computation")
    parser.add_argument("--compute-only", action="store_true", help="skip plot creation")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    add_molecular_crystal_option(parser)
    args = parser.parse_args()
    args.excluded_system_types = excluded_system_types(args)
    if args.plots_only and args.compute_only:
        parser.error("--plots-only and --compute-only cannot be used together")
    if args.e_min is not None and args.e_max is not None and args.e_min >= args.e_max:
        parser.error("--e-min must be smaller than --e-max")
    if args.pad_factor < 1:
        parser.error("--pad-factor must be positive")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if not math.isfinite(args.window_fs) or args.window_fs <= 0:
        parser.error("--window-fs must be finite and positive")
    if args.min_segments < 2:
        parser.error("--min-segments must be at least 2")
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
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vdos-updated")
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
        "--estimator",
        args.estimator,
        "--weighting",
        args.weighting,
        "--window-fs",
        str(args.window_fs),
        "--min-segments",
        str(args.min_segments),
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
    if args.include_molecular_crystals:
        command.append("--include-molecular-crystals")
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


def plot_command(
    sources: list[str], results_dir: Path, plots_dir: Path, *,
    include_molecular_crystals: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(PLOT_SCRIPT),
        "--results-dir", str(results_dir),
        "--plots-dir", str(plots_dir),
    ]
    for source in sources:
        command.extend(("--source", source))
    if include_molecular_crystals:
        command.append("--include-molecular-crystals")
    return command


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
        "Create corrected VDOS figures and correlations",
        plot_command(
            sources, args.results_dir, args.plots_dir,
            include_molecular_crystals=args.include_molecular_crystals,
        ),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
