#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["adjustText>=1.3", "matplotlib>=3.9", "numpy>=1.26", "pandas>=2.2", "seaborn>=0.13"]
# ///
"""Compare accelerated TorchSim and ASE using refined VDOS for both runners.

RDF, pressure, timings, source substitutions and styling use the original
Figure 7 helpers, with matching successful timing workloads for shared models.
Both VDOS inputs are required under this directory's results
root: missing refined results never fall back to the original VDOS analysis.
ASE UMA-S/M turbo configurations are excluded from this figure only.
"""

# ruff: noqa: E402 -- shared Pareto helpers live in the sibling directory

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
# Keep the standalone vdos directory ahead of shared helpers: inserting
# the sibling first would make subsequent imports such as plot_all resolve to
# its legacy Pareto pipeline during unittest discovery or embedded use.
PARETO_DIR = str(HERE.parent / "pareto_plots")
if PARETO_DIR not in sys.path:
    sys.path.append(PARETO_DIR)
import figure_7 as pareto
from _figure_data import SI18_EXCLUDED_MODELS
from system_filters import add_molecular_crystal_option


SOURCES = ("mlip-trajs-torchsim-accelerated", "mlip-trajs-ase-accelerated")
OUTPUT_GROUP = "torchsim_vs_ase_accelerated"
MODEL_FILE = "vdos_model_mean_ev_normalized_same_simulation_length.csv"
FONT_SIZE = 8
LABEL_DISTANCE_POINTS = 24


def refined_inputs(results_dir: Path) -> dict[str, Path]:
    """Require both independently computed refined summaries before loading."""
    paths = {source: results_dir / source / MODEL_FILE for source in SOURCES}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Figure SI18 requires refined VDOS metrics for both accelerated runners; "
            "compute the missing source first: " + ", ".join(map(str, missing))
        )
    return paths


def load_comparison(
    results_dir: Path, *, include_molecular_crystals: bool = False,
) -> pd.DataFrame:
    """Use runner-specific errors and paired timing workloads for shared models."""
    vdos_files = refined_inputs(results_dir)
    frames = []
    timings_dirs = {}
    for source in SOURCES:
        timings, rdf, _, pressure_mode = pareto.source_input_paths(source)
        timings_dirs[source] = timings
        frames.append(pareto.load_and_merge(
            timings_dir=timings,
            rdf_metrics_file=rdf,
            vdos_metrics_file=vdos_files[source],
            combined_metrics_file=None,
            pressure_metrics_file=pareto.pressure_path(source),
            pressure_backend=pareto.source_pressure_backend(source),
            pressure_mode=pressure_mode,
            pressure_scale_gpa=10.0,
            clip_pressure_error=True,
            dataset_label=source,
            include_molecular_crystals=include_molecular_crystals,
        ))
    merged = pareto.exclude_model_configurations(
        pd.concat(frames, ignore_index=True), SI18_EXCLUDED_MODELS,
    )
    merged = pareto.apply_matched_runner_timings(
        merged, timings_dirs=timings_dirs,
        include_molecular_crystals=include_molecular_crystals,
    )
    return pareto.mark_pareto_frontier(merged)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--plots-dir", type=Path, default=HERE / "plots")
    parser.add_argument("--layout", choices=("overlay", "panels"), default="overlay")
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--frontier-csv", type=Path)
    parser.add_argument("--data-only", action="store_true", help="export metrics without creating a figure")
    add_molecular_crystal_option(parser)
    args = parser.parse_args(argv)
    results = args.results_dir.resolve()
    plots = args.plots_dir.resolve()
    merged = load_comparison(results, include_molecular_crystals=args.include_molecular_crystals)
    metrics_path = args.output_csv or results / OUTPUT_GROUP / "figure_SI_18_metrics.csv"
    frontier_path = args.frontier_csv or metrics_path.with_name("figure_SI_18_frontier.csv")
    output_path = args.output_file or plots / OUTPUT_GROUP / "figure_SI_18.pdf"
    frontier = merged.loc[merged["is_pareto"]].sort_values(["dataset", "mean_time_per_step_ms"])
    for path, table in ((metrics_path, merged), (frontier_path, frontier)):
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        print(f"Saved: {path}")
    print(frontier[["dataset", "model", "mean_time_per_step_ms", "Combined Error [%]"]].to_string(index=False))
    if not args.data_only:
        pareto.plot_pareto(
            merged, output_path, layout=args.layout, font_size=FONT_SIZE,
            label_distance_points=LABEL_DISTANCE_POINTS,
        )


if __name__ == "__main__":
    main()
