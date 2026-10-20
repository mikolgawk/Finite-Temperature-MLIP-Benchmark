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
"""Compare accelerated TorchSim and ASE Pareto fronts on the same axes.

Use each runner's own RDF, refined VDOS and pressure metrics, with independent
frontiers. Shared configurations use identical successful timing workloads in
both runners. The TorchSim paper cohort retains its selected eager substitutions.
ASE UMA-S/M turbo configurations are excluded from this figure only.
Use --data-only to export the metrics/frontier CSVs without generating a figure.
"""

from figure_7 import run_pareto_figure
from _figure_data import SI18_EXCLUDED_MODELS


SOURCES = ("mlip-trajs-torchsim-accelerated", "mlip-trajs-ase-accelerated")
# The overlay is 4/3 as wide as the other single-panel Pareto figures. Eight
# points gives the same apparent text size as their six points at equal width.
FONT_SIZE = 8


def main(argv: list[str] | None = None) -> None:
    run_pareto_figure(
        figure_name="figure_SI_18", allowed_sources=SOURCES,
        default_sources=SOURCES, required_sources=SOURCES, argv=argv,
        font_size=FONT_SIZE, label_distance_points=24,
        matched_runner_timings=True,
        excluded_models_by_source=SI18_EXCLUDED_MODELS,
    )


if __name__ == "__main__":
    main()
