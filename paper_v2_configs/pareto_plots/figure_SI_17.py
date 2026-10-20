#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "numpy>=1.26", "pandas>=2.2"]
# ///
"""Compare native accelerated TorchSim and ASE MD timings on matched systems.

Only models with a terminal success or failure for every benchmark system in
both runners are included. Each uses the same successful systems and matching
MD workloads for its timing average; failed systems do not contribute timings.
Molecular crystals and registered physical failures are excluded by default.
The paper cohort's borrowed eager runs are not used in this runner comparison.
PET timing names are matched to TorchScript variants when their engine metadata
confirms TorchScript inference, including ASE filenames without that suffix.

Panel (a) shows mean MD time per step; panel (b) shows ASE/TorchSim mean time.
Ratios above one indicate faster TorchSim MD. Use --data-only to export the
comparison tables without creating a figure.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE.parent
sys.path.insert(0, str(CONFIG_DIR))
from md_success import load_failed_md_runs, registered_md_failure_reason
from model_display_names import MODEL_DISPLAY_NAMES, display_model_name, normalize_display_key
from system_filters import add_molecular_crystal_option, include_system

TORCHSIM_SOURCE = "mlip-trajs-torchsim-accelerated"
ASE_SOURCE = "mlip-trajs-ase-accelerated"
COMPARISON_DIR = "torchsim_vs_ase_accelerated"
DEFAULT_OUTPUT_FILE = HERE / "plots" / COMPARISON_DIR / "figure_SI_17.pdf"
DEFAULT_OUTPUT_CSV = HERE / "results" / COMPARISON_DIR / "figure_SI_17.csv"
DEFAULT_PAIRED_CSV = HERE / "results" / COMPARISON_DIR / "figure_SI_17_per_system.csv"
DEFAULT_METADATA_FILE = CONFIG_DIR / "data" / "ref-trajs" / "md_metadata.json"
FONT_SIZE = 6
FIGURE_SIZE = (3.53 * 3, 3.53 * 1.5)
OBSERVATION_COLUMNS = [
    "model", "system", "source", "raw_model", "n_steps", "seconds_per_step",
    "elapsed_seconds", "time_step_fs", "record_interval", "engine",
    "timing_file", "trajectory_file",
]
SUMMARY_COLUMNS = [
    "model", "model_display", "n_common_systems", "torchsim_mean_ms_per_step",
    "ase_mean_ms_per_step", "torchsim_speedup",
]


def timing_model_key(name: object, engine: object = "") -> str:
    """Match PET's runner naming difference using explicit execution metadata."""
    key = normalize_display_key(normalize_display_key(name).removesuffix("-ase"))
    if key in {"pet-oam-xl", "pet-omat-xl"} and "torchscript" in str(engine).lower():
        return f"{key}-torchscript"
    return key


def load_native_timings(
    directory: Path, source: str, include_molecular_crystals: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Read physical timing files without substituting any paper-cohort runs."""
    if not directory.is_dir():
        raise FileNotFoundError(f"MD timing directory does not exist: {directory}")
    records, skipped = [], []
    failures = load_failed_md_runs()
    for path in sorted(directory.glob("*/md_timing_*.csv")):
        system = path.parent.name
        if not include_system(system, include_molecular_crystals):
            continue
        raw_model = path.stem.removeprefix("md_timing_")
        trajectory = path.with_name(f"nvt_{raw_model}.h5")
        reason = registered_md_failure_reason(trajectory) or failures.get((source, system, raw_model))
        if reason:
            skipped.append(f"{path}: {reason}")
            continue
        if not trajectory.is_file():
            skipped.append(f"{path}: matching trajectory missing")
            continue
        try:
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 1:
                raise ValueError("expected one completed timing record")
            row = rows[0]
            # ASE files may add -ase to the filename while keeping the calculator
            # column identical to the production model identifier.
            calculator = str(row.get("calculator", "")).strip().lower().removesuffix("-ase")
            if calculator != raw_model.lower().removesuffix("-ase"):
                raise ValueError("calculator does not match timing filename")
            if row.get("system") != system:
                raise ValueError("system does not match timing directory")
            steps = float(row.get("n_steps") or 0)
            elapsed = float(row.get("elapsed_seconds") or 0)
            seconds = float(row.get("seconds_per_step") or 0)
            if not np.isfinite([steps, elapsed, seconds]).all() or min(steps, elapsed, seconds) <= 0:
                raise ValueError("timing values must be finite and positive")
            if not steps.is_integer():
                raise ValueError("n_steps must be an integer")
            if not np.isclose(seconds, elapsed / steps, rtol=0.02, atol=1e-9):
                raise ValueError("seconds_per_step disagrees with elapsed_seconds/n_steps")
            metadata = {name: float(row.get(name) or "nan")
                        for name in ("time_step_fs", "record_interval")}
            if any(np.isinf(value) or value <= 0 for value in metadata.values()):
                raise ValueError("time_step_fs and record_interval must be finite and positive when provided")
        except (OSError, csv.Error, ValueError, TypeError) as exc:
            skipped.append(f"{path}: {exc}")
            continue
        records.append({
            "model": timing_model_key(raw_model, row.get("engine", "")),
            "system": system, "source": source,
            "raw_model": raw_model, "n_steps": int(steps), "seconds_per_step": seconds,
            "elapsed_seconds": elapsed, **metadata, "engine": row.get("engine", ""),
            "timing_file": str(path), "trajectory_file": str(trajectory),
        })
    frame = pd.DataFrame(records, columns=OBSERVATION_COLUMNS)
    duplicates = frame.loc[frame.duplicated(["model", "system"], keep=False)]
    if not duplicates.empty:
        raise ValueError(f"Ambiguous timing files in {source}:\n"
                         + duplicates[["model", "system", "timing_file"]].to_string(index=False))
    return frame, skipped


def benchmark_systems(metadata_file: Path, include_molecular_crystals: bool = False) -> set[str]:
    """Read expected coverage from the reference benchmark, never partial outputs."""
    metadata = json.loads(metadata_file.read_text())
    systems = {system for system in metadata if include_system(system, include_molecular_crystals)}
    if not systems:
        raise ValueError(f"No benchmark systems selected from {metadata_file}")
    return systems


def finished_model_keys(
    directory: Path, source: str, observations: pd.DataFrame,
    expected_systems: set[str],
) -> set[str]:
    """Distinguish finished campaigns with failures from unfinished campaigns.

    Runners write an empty timing CSV after an exception. Together with valid
    completed timings and registered physical failures, these are terminal
    outcomes. Partial trajectories alone and malformed timing records are not.
    Failed PET filenames inherit execution aliases from that model's valid runs.
    """
    aliases: dict[str, set[str]] = {}
    valid = set()
    for row in observations.itertuples():
        aliases.setdefault(timing_model_key(row.raw_model), set()).add(row.model)
        valid.add((row.raw_model, row.system))
    failures = load_failed_md_runs()
    terminal: dict[str, set[str]] = {}
    for path in directory.glob("*/md_timing_*.csv"):
        system = path.parent.name
        if system not in expected_systems:
            continue
        raw = path.stem.removeprefix("md_timing_")
        trajectory = path.with_name(f"nvt_{raw}.h5")
        completed = (
            (raw, system) in valid or path.stat().st_size == 0
            or registered_md_failure_reason(trajectory)
            or (source, system, raw) in failures
        )
        if not completed:
            continue
        key = timing_model_key(raw)
        for model in aliases.get(key, {key}):
            terminal.setdefault(model, set()).add(system)
    # A manually audited failure may remain after its original outputs are removed.
    for (failed_source, system, raw), _ in failures.items():
        if failed_source == source and system in expected_systems:
            key = timing_model_key(raw)
            for model in aliases.get(key, {key}):
                terminal.setdefault(model, set()).add(system)
    return {model for model, systems in terminal.items() if expected_systems <= systems}


def compare_timings(torchsim: pd.DataFrame, ase: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair identical model/system workloads and average systems equally."""
    paired = torchsim.merge(ase, on=["model", "system"], how="inner",
                            suffixes=("_torchsim", "_ase"), validate="one_to_one")
    if paired.empty:
        for column in ("torchsim_ms_per_step", "ase_ms_per_step"):
            paired[column] = pd.Series(dtype=float)
        return paired, pd.DataFrame(columns=SUMMARY_COLUMNS)
    matched = paired["n_steps_torchsim"].eq(paired["n_steps_ase"])
    for column in ("time_step_fs", "record_interval"):
        left, right = paired[f"{column}_torchsim"], paired[f"{column}_ase"]
        known = np.isfinite(left) & np.isfinite(right)
        matched &= ~known | np.isclose(left, right, rtol=1e-8, atol=1e-10)
    paired = paired.loc[matched].copy()
    paired["torchsim_ms_per_step"] = paired["seconds_per_step_torchsim"] * 1000.
    paired["ase_ms_per_step"] = paired["seconds_per_step_ase"] * 1000.
    paired = paired.sort_values(["model", "system"]).reset_index(drop=True)
    if paired.empty:
        return paired, pd.DataFrame(columns=SUMMARY_COLUMNS)
    summary = paired.groupby("model", as_index=False).agg(
        n_common_systems=("system", "nunique"),
        torchsim_mean_ms_per_step=("torchsim_ms_per_step", "mean"),
        ase_mean_ms_per_step=("ase_ms_per_step", "mean"),
    )
    summary["torchsim_speedup"] = summary["ase_mean_ms_per_step"] / summary["torchsim_mean_ms_per_step"]
    summary["model_display"] = summary["model"].map(display_model_name)
    order = {model: index for index, model in enumerate(MODEL_DISPLAY_NAMES)}
    summary["_order"] = summary["model"].map(order).fillna(len(order))
    summary = summary.sort_values(["_order", "model"]).reset_index(drop=True)
    return paired, summary[SUMMARY_COLUMNS]


def comparison_available(data_dir: Path, include_molecular_crystals: bool = False) -> bool:
    """Skip S17 when no finished models have paired timing data."""
    folders = [data_dir / source for source in (TORCHSIM_SOURCE, ASE_SOURCE)]
    if not all(folder.is_dir() for folder in folders):
        return False
    torchsim, _ = load_native_timings(folders[0], TORCHSIM_SOURCE, include_molecular_crystals)
    ase, _ = load_native_timings(folders[1], ASE_SOURCE, include_molecular_crystals)
    _, summary = compare_timings(torchsim, ase)
    expected = benchmark_systems(DEFAULT_METADATA_FILE, include_molecular_crystals)
    finished = (
        finished_model_keys(folders[0], TORCHSIM_SOURCE, torchsim, expected)
        & finished_model_keys(folders[1], ASE_SOURCE, ase, expected)
    )
    return summary["model"].isin(finished).any()


def plot_comparison(summary: pd.DataFrame, output: Path) -> None:
    """Draw matched step times and the TorchSim speedup without plot titles."""
    import matplotlib.pyplot as plt

    if summary.empty:
        raise ValueError("No matched timing data to plot")
    style = {"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
             "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
             "legend.fontsize": FONT_SIZE, "axes.grid": True, "grid.linewidth": 0.5}
    with plt.rc_context(style):
        height = max(FIGURE_SIZE[1], 0.32 * len(summary) + 1.4)
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(FIGURE_SIZE[0], height), layout="constrained")
        positions = np.arange(len(summary))
        axes[0].barh(positions - 0.17, summary.torchsim_mean_ms_per_step, height=0.32,
                     color="#4C72B0", label="TorchSim", edgecolor="black", linewidth=0.5, zorder=3)
        axes[0].barh(positions + 0.17, summary.ase_mean_ms_per_step, height=0.32,
                     color="#DD8452", label="ASE", edgecolor="black", linewidth=0.5, zorder=3)
        axes[0].set_xscale("log")
        axes[0].set_xlabel("Mean MD time per step [ms]")
        axes[0].set_yticks(positions, [f"{row.model_display} (n={row.n_common_systems})"
                                     for row in summary.itertuples()])
        axes[0].set_ylabel("Model (matched systems)")
        axes[0].invert_yaxis()
        axes[0].legend(loc="best")
        axes[1].barh(positions, summary.torchsim_speedup, height=0.5, color="#4C72B0",
                     edgecolor="black", linewidth=0.5, zorder=3)
        axes[1].axvline(1., color="black", linestyle="--", linewidth=0.8)
        axes[1].set_xlabel("ASE / TorchSim mean step time")
        axes[1].set_xlim(0, max(1., float(summary.torchsim_speedup.max())) * 1.25)
        for position, speedup in zip(positions, summary.torchsim_speedup):
            axes[1].annotate(f"{speedup:.2f}×", (speedup, position), xytext=(3, 0),
                             textcoords="offset points", va="center", fontsize=FONT_SIZE,
                             bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.1})
        for label, ax in zip(("(a)", "(b)"), axes):
            ax.text(0, 1.02, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=FONT_SIZE)
            ax.grid(axis="x")
            ax.grid(axis="y", visible=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torchsim-dir", type=Path, default=CONFIG_DIR / "data" / TORCHSIM_SOURCE)
    parser.add_argument("--ase-dir", type=Path, default=CONFIG_DIR / "data" / ASE_SOURCE)
    parser.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA_FILE,
                        help="Reference MD metadata defining the full benchmark system set.")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--paired-csv", type=Path, default=DEFAULT_PAIRED_CSV)
    parser.add_argument("--data-only", action="store_true", help="Export timing tables without generating a figure.")
    add_molecular_crystal_option(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torchsim, torchsim_skipped = load_native_timings(args.torchsim_dir, TORCHSIM_SOURCE, args.include_molecular_crystals)
    ase, ase_skipped = load_native_timings(args.ase_dir, ASE_SOURCE, args.include_molecular_crystals)
    paired, summary = compare_timings(torchsim, ase)
    expected = benchmark_systems(args.metadata_file, args.include_molecular_crystals)
    finished = (
        finished_model_keys(args.torchsim_dir, TORCHSIM_SOURCE, torchsim, expected)
        & finished_model_keys(args.ase_dir, ASE_SOURCE, ase, expected)
    )
    unfinished = sorted(set(summary["model"]) - finished)
    paired = paired.loc[paired["model"].isin(finished)].reset_index(drop=True)
    summary = summary.loc[summary["model"].isin(finished)].reset_index(drop=True)
    if unfinished:
        print("Excluded unfinished model configurations: " + ", ".join(unfinished))
    if summary.empty:
        raise SystemExit("No shared models have finished the benchmark with matching successful timings in both runners.")
    candidate_pairs = len(torchsim.loc[torchsim.model.isin(finished)].merge(
        ase.loc[ase.model.isin(finished)], on=["model", "system"], how="inner", validate="one_to_one"))
    print(f"Finished benchmark coverage requires {len(expected)} systems per runner, allowing terminal failures.")
    print(f"Matched {len(summary)} models on {len(paired)} model/system pairs; "
          f"excluded {candidate_pairs - len(paired)} pairs with different MD workloads.")
    print(f"Skipped {len(torchsim_skipped)} TorchSim and {len(ase_skipped)} ASE timing records.")
    print(summary.to_string(index=False))
    for frame, path in ((summary, args.output_csv), (paired, args.paired_csv)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"Saved {path}")
    if not args.data_only:
        plot_comparison(summary, args.output_file)
        print(f"Saved {args.output_file}")


if __name__ == "__main__":
    main()
