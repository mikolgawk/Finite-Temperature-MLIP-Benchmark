#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib>=3.9",
#   "numpy>=1.26",
#   "pandas>=2.2",
# ]
# ///
"""Compare v2 eager/accelerated VDOS errors with v1 VDOS results."""

from __future__ import annotations

import argparse
from pathlib import Path

from comparison_utils import SOURCE_DIRS, VDOS_SPEC, print_summary, run_comparison


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LEGACY_DIR = Path(
    "/home/mjgawkowski/phd_mlip_matbench_benchmarks/scripts/vdos-scripts-final-copy/results"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        choices=SOURCE_DIRS,
        dest="sources",
        help="v2 execution source to compare (repeatable; default: eager and accelerated)",
    )
    parser.add_argument(
        "--v2-results-root",
        type=Path,
        default=SCRIPT_DIR.parent / "vdos" / "results",
        help="v2 per-source VDOS results (default: refined vdos/results; select vdos_legacy/results explicitly for the original v2 analysis)",
    )
    parser.add_argument(
        "--legacy-results-dir",
        type=Path,
        default=DEFAULT_LEGACY_DIR,
        help="v1 VDOS results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "vdos",
        help="destination for comparison CSVs, manifest, and plot",
    )
    parser.add_argument(
        "--exclude-system",
        action="append",
        dest="excluded_systems",
        help="exact system name to exclude before aggregation (repeatable)",
    )
    parser.add_argument("--no-plot", action="store_true", help="do not create the PDF summary plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairwise, by_model, _ = run_comparison(
        VDOS_SPEC,
        v2_results_root=args.v2_results_root,
        legacy_results_dir=args.legacy_results_dir,
        output_dir=args.output_dir,
        sources=list(dict.fromkeys(args.sources or SOURCE_DIRS)),
        make_plot=not args.no_plot,
        excluded_systems=set(args.excluded_systems or ()),
    )
    print_summary(VDOS_SPEC, pairwise, by_model)
    print(f"\nWrote VDOS comparison outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
