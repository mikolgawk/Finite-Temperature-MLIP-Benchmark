#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "adjustText>=1.3",
#   "h5py>=3.11",
#   "matplotlib>=3.9",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "seaborn>=0.13",
#   "vesin>=0.3",
# ]
# ///
"""Create every timing, Pareto, and correlation plot from existing metrics."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE.parent
SOURCES = (
    "mlip-trajs-ase",
    "mlip-trajs-torchsim-eager",
    "mlip-trajs-ase-accelerated",
    "mlip-trajs-torchsim-accelerated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", choices=SOURCES, dest="sources")
    parser.add_argument("--plots-dir", type=Path, default=HERE / "plots")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument(
        "--pressure-file",
        type=Path,
        default=CONFIG_DIR / "pressures/results/model_pressure_error_metric.csv",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    return parser.parse_args()


def available_sources() -> list[str]:
    return [
        source for source in SOURCES
        if (CONFIG_DIR / "rdfs/results" / source / "rdf_similarity_scores_same_simulation_length.csv").is_file()
        and (CONFIG_DIR / "vdos/results" / source / "vdos_model_mean_ev_normalized_same_simulation_length.csv").is_file()
    ]


def run_stage(label: str, command: list[str], dry_run: bool) -> bool:
    print(f"\n=== {label} ===", flush=True)
    print(f"$ {shlex.join(command)}", flush=True)
    if dry_run:
        return True
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    return subprocess.run(command, cwd=HERE, env=env, check=False).returncode == 0


def main() -> None:
    args = parse_args()
    plots_dir = args.plots_dir.resolve()
    results_dir = args.results_dir.resolve()
    sources = args.sources or available_sources()
    if not sources and not args.dry_run:
        raise SystemExit("No sources have both RDF and VDOS result tables")
    if not sources:
        sources = list(SOURCES)

    data_dir = CONFIG_DIR / "data"
    timing_plots = plots_dir / "timings"
    timing_results = results_dir / "timings"
    global_stages = [
        (
            "Model timings",
            [
                sys.executable, "-u", str(HERE / "plot-model-timings.py"),
                "--timings-dir", str(data_dir / "mlip-trajs-torchsim-eager"),
                "--accelerated-timings-dir", str(data_dir / "mlip-trajs-torchsim-accelerated"),
                "--eager-output-png", str(timing_plots / "model_timings_eager.png"),
                "--eager-output-pdf", str(timing_plots / "model_timings_eager.pdf"),
                "--accelerated-output-png", str(timing_plots / "model_timings_accelerated.png"),
                "--accelerated-output-pdf", str(timing_plots / "model_timings_accelerated.pdf"),
                "--summary-csv", str(timing_results / "model_timings_summary.csv"),
                "--observations-csv", str(timing_results / "model_timings_observations.csv"),
            ],
        ),
    ]
    for x_axis in ("atoms", "edges", "edge-density"):
        slug = x_axis.replace("-", "_")
        global_stages.append(
            (
                f"Timings versus {x_axis}",
                [
                    sys.executable, "-u", str(HERE / "plot-model-timings-vs-system-size.py"),
                    "--timings-dir", str(data_dir / "mlip-trajs-torchsim-eager"),
                    "--accelerated-timings-dir", str(data_dir / "mlip-trajs-torchsim-accelerated"),
                    "--x-axis", x_axis,
                    "--eager-output-png", str(timing_plots / f"model_timings_vs_{slug}_eager.png"),
                    "--eager-output-pdf", str(timing_plots / f"model_timings_vs_{slug}_eager.pdf"),
                    "--accelerated-output-png", str(timing_plots / f"model_timings_vs_{slug}_accelerated.png"),
                    "--accelerated-output-pdf", str(timing_plots / f"model_timings_vs_{slug}_accelerated.pdf"),
                    "--results-csv", str(timing_results / f"model_timings_vs_{slug}_observations.csv"),
                ],
            )
        )
    failures: list[str] = []
    for label, command in global_stages:
        if not run_stage(label, command, args.dry_run):
            failures.append(label)

    for source in dict.fromkeys(sources):
        source_plots = plots_dir / source
        source_results = results_dir / source
        provenance = [
            "--pressure-backend", "torchsim" if "torchsim" in source else "ase",
            "--pressure-mode", "md_accelerated" if source.endswith("-accelerated") else "md_eager",
        ]
        common = ["--source", source, "--pressure-file", str(args.pressure_file.resolve()), *provenance]
        stages = [
            (
                "Figure 7",
                [sys.executable, "-u", str(HERE / "figure_7.py"), "--source", source,
                 "--pressure-metrics-file", str(args.pressure_file.resolve()), *provenance,
                 "--output-file", str(source_plots / "figure_7.pdf"),
                 "--output-csv", str(source_results / "figure_7_metrics.csv")],
            ),
            (
                "Figure SI 10",
                [sys.executable, "-u", str(HERE / "figure_SI_10.py"), *common,
                 "--output-plot", str(source_plots / "figure_SI_10.pdf"),
                 "--output-csv", str(source_results / "figure_SI_10.csv"),
                 "--output-json", str(source_results / "figure_SI_10.json")],
            ),
            (
                "Figure SI 11",
                [sys.executable, "-u", str(HERE / "figure_SI_11.py"), *common,
                 "--output-plot", str(source_plots / "figure_SI_11.pdf"),
                 "--output-csv", str(source_results / "figure_SI_11.csv"),
                 "--output-json", str(source_results / "figure_SI_11.json")],
            ),
            (
                "Figure SI 12",
                [sys.executable, "-u", str(HERE / "figure_SI_12.py"), "--source", source,
                 "--output-plot", str(source_plots / "figure_SI_12.pdf"),
                 "--output-csv", str(source_results / "figure_SI_12.csv"),
                 "--output-json", str(source_results / "figure_SI_12.json")],
            ),
            (
                "Figure SI 13",
                [sys.executable, "-u", str(HERE / "figure_SI_13.py"), "--source", source,
                 "--pressure-metrics-file", str(args.pressure_file.resolve()), *provenance,
                 "--output-file", str(source_plots / "figure_SI_13.pdf"),
                 "--output-csv", str(source_results / "figure_SI_13.csv")],
            ),
        ]
        for label, command in stages:
            if not run_stage(f"{label} ({source})", command, args.dry_run):
                failures.append(f"{label} ({source})")

    if failures:
        raise SystemExit("Failed plot stages: " + ", ".join(failures))
    print(f"\nAll Pareto plots are in {plots_dir}")


if __name__ == "__main__":
    main()
