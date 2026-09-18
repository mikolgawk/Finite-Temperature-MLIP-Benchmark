#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "adjustText>=1.3",
#   "ase>=3.26",
#   "h5py>=3.11",
#   "matplotlib>=3.9",
#   "mdtraj>=1.10",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "seaborn>=0.13",
# ]
# ///
"""Compute RDF results and create the RDF figures in one command.

Run with::

    uv run run_rdf_pipeline.py

Results and plots are kept separate for each trajectory source. Use
``--source`` to process only selected sources, or ``--plots-only`` to recreate
figures from existing RDF results without recomputing the RDFs.
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

RDF_SCRIPT = SCRIPT_DIR / "get-rdf-and-results-by-system-type-same-simulation-length.py"
FIGURE_3_SCRIPT = SCRIPT_DIR / "fig_3.py"
FIGURE_SI_14_SCRIPT = SCRIPT_DIR / "figure_SI_14.py"

SOURCES = (
    "mlip-trajs-ase",
    "mlip-trajs-torchsim-eager",
    "mlip-trajs-ase-accelerated",
    "mlip-trajs-torchsim-accelerated",
)


def available_sources() -> list[str]:
    """Return trajectory sources that contain at least one HDF5 trajectory."""
    return [
        source
        for source in SOURCES
        if any((DATA_DIR / source).glob("*/nvt_*.h5"))
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute reference/MLIP RDFs, score them, and create the RDF figures."
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=SOURCES,
        dest="sources",
        help="trajectory source to process (repeatable; default: every available source)",
    )
    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help="process only this system (repeatable)",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="process only this model (repeatable)",
    )
    parser.add_argument(
        "--exclude-system-type",
        action="append",
        dest="excluded_system_types",
        help="exclude this system type from computation (repeatable)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=SCRIPT_DIR / "results",
        help="root output directory for RDF CSV files (default: %(default)s)",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=SCRIPT_DIR / "plots",
        help="root output directory for figures (default: %(default)s)",
    )
    parser.add_argument(
        "--timings-dir",
        type=Path,
        help=(
            "directory containing per-system md_timing_<model>.csv files; only "
            "valid when processing one source (default: data/<source>)"
        ),
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="use existing RDF results and skip RDF computation",
    )
    parser.add_argument(
        "--compute-only",
        action="store_true",
        help="compute RDF results without creating figures",
    )
    parser.add_argument(
        "--skip-figure-3",
        action="store_true",
        help="do not create the combined RDF panel",
    )
    parser.add_argument(
        "--skip-figure-si-14",
        action="store_true",
        help="do not create the RDF-error/time Pareto figure",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands without running them",
    )
    args = parser.parse_args()

    if args.plots_only and args.compute_only:
        parser.error("--plots-only and --compute-only cannot be used together")
    if args.compute_only and (args.skip_figure_3 or args.skip_figure_si_14):
        parser.error("figure skip options are redundant with --compute-only")
    if args.timings_dir is not None and not args.timings_dir.is_dir():
        parser.error(f"timings directory does not exist: {args.timings_dir}")

    return args


def run_stage(label: str, command: list[str], *, dry_run: bool) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(f"$ {shlex.join(command)}", flush=True)
    if dry_run:
        return

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(command, cwd=SCRIPT_DIR, env=env, check=True)


def computation_command(args: argparse.Namespace, sources: list[str]) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(RDF_SCRIPT),
        "--results-dir",
        str(args.results_dir),
    ]
    for source in sources:
        command.extend(("--source", source))
    for system in args.systems or ():
        command.extend(("--system", system))
    for model in args.models or ():
        command.extend(("--model", model))
    for system_type in args.excluded_system_types or ():
        command.extend(("--exclude-system-type", system_type))
    return command


def figure_3_command(source_results: Path, output_file: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(FIGURE_3_SCRIPT),
        "--overall-rdf-file",
        str(source_results / "rdf_similarity_scores_same_simulation_length.csv"),
        "--rdf-csv-dir",
        str(source_results),
        "--rdf-save-dir",
        str(source_results / "rdf_same_simulation_length_saved"),
        "--output-file",
        str(output_file),
    ]


def figure_si_14_command(
    source_results: Path,
    timings_dir: Path,
    output_file: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(FIGURE_SI_14_SCRIPT),
        "--timings-dir",
        str(timings_dir),
        "--rdf-scores-file",
        str(source_results / "rdf_similarity_scores_same_simulation_length.csv"),
        "--output-file",
        str(output_file),
    ]


def main() -> None:
    args = parse_args()
    sources = list(dict.fromkeys(args.sources or available_sources()))
    if not sources:
        raise SystemExit(
            f"No nvt_*.h5 trajectories found under {DATA_DIR}; "
            "select a future source explicitly with --source."
        )
    if args.timings_dir is not None and len(sources) != 1:
        raise SystemExit("--timings-dir can only be used when exactly one --source is selected")

    args.results_dir = args.results_dir.resolve()
    args.plots_dir = args.plots_dir.resolve()

    print(f"Trajectory sources: {', '.join(sources)}")
    if not args.plots_only:
        run_stage(
            "Compute reference and MLIP RDFs",
            computation_command(args, sources),
            dry_run=args.dry_run,
        )

    if args.compute_only:
        return

    for source in sources:
        source_results = args.results_dir / source
        scores_file = source_results / "rdf_similarity_scores_same_simulation_length.csv"
        if not args.dry_run and not scores_file.is_file():
            raise FileNotFoundError(
                f"RDF scores were not produced for {source}: {scores_file}"
            )

        source_plots = args.plots_dir / source
        if not args.skip_figure_3:
            run_stage(
                f"Create figure 3 ({source})",
                figure_3_command(
                    source_results,
                    source_plots / "plot_rdf_panel_combined.pdf",
                ),
                dry_run=args.dry_run,
            )

        timings_dir = (
            args.timings_dir.resolve()
            if args.timings_dir is not None
            else DATA_DIR / source
        )
        if not args.skip_figure_si_14 and not args.dry_run and not timings_dir.is_dir():
            print(
                f"\n[WARN] Skipping figure SI 14 ({source}): timing directory "
                f"does not exist: {timings_dir}",
                flush=True,
            )
        elif not args.skip_figure_si_14:
            run_stage(
                f"Create figure SI 14 ({source})",
                figure_si_14_command(
                    source_results,
                    timings_dir,
                    source_plots / "plot_SI_pareto_rdf_time_same_length.pdf",
                ),
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
