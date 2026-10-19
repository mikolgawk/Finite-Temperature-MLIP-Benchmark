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

Use each runner's own RDF, VDOS, pressure and timing metrics, with independent
frontiers. The TorchSim paper cohort retains its selected eager substitutions.
Use --data-only to export the metrics/frontier CSVs without generating a figure.
"""

from figure_7 import run_pareto_figure


SOURCES = ("mlip-trajs-torchsim-accelerated", "mlip-trajs-ase-accelerated")
# The overlay is 4/3 as wide as the other single-panel Pareto figures. Eight
# points gives the same apparent text size as their six points at equal width.
FONT_SIZE = 8


def main(argv: list[str] | None = None) -> None:
    run_pareto_figure(
        figure_name="figure_SI_18", allowed_sources=SOURCES,
        default_sources=SOURCES, required_sources=SOURCES, argv=argv,
        font_size=FONT_SIZE,
    )


if __name__ == "__main__":
    main()
