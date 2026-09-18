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
"""Create every RDF plot from existing, source-specific RDF results."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCES = (
    "mlip-trajs-ase",
    "mlip-trajs-torchsim-eager",
    "mlip-trajs-ase-accelerated",
    "mlip-trajs-torchsim-accelerated",
)
SCORES = "rdf_similarity_scores_same_simulation_length.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", choices=SOURCES, dest="sources")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--plots-dir", type=Path, default=HERE / "plots")
    parser.add_argument(
        "--timings-dir",
        type=Path,
        help="Override timing directory for figure SI 14 (requires one source).",
    )
    parser.add_argument("--skip-figure-si-14", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    sources = args.sources or [
        source for source in SOURCES if (results_dir / source / SCORES).is_file()
    ]
    if not sources and not args.dry_run:
        raise SystemExit(f"No source-specific RDF results found in {results_dir}")
    if not sources:
        sources = list(SOURCES)

    command = [
        sys.executable,
        "-u",
        str(HERE / "run_rdf_pipeline.py"),
        "--plots-only",
        "--results-dir",
        str(results_dir),
        "--plots-dir",
        str(args.plots_dir.resolve()),
    ]
    for source in dict.fromkeys(sources):
        command.extend(("--source", source))
    if args.timings_dir:
        command.extend(("--timings-dir", str(args.timings_dir.resolve())))
    if args.skip_figure_si_14:
        command.append("--skip-figure-si-14")
    if args.dry_run:
        command.append("--dry-run")

    print("=== Create all RDF plots ===", flush=True)
    print(f"$ {shlex.join(command)}", flush=True)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(command, cwd=HERE, env=env, check=True)


if __name__ == "__main__":
    main()
