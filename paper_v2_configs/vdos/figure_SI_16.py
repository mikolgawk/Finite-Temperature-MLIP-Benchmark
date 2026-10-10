#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib>=3.9",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "seaborn>=0.13",
# ]
# ///
"""Plot Pareto front for VDOS error vs mean MD step time.

Objectives:
- minimize VDOS error [%]
- minimize mean MD time per step [ms]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys

PAPER_V2_CONFIG_DIR = Path(__file__).resolve().parents[1]
if str(PAPER_V2_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_V2_CONFIG_DIR))
from model_display_names import MODEL_DISPLAY_NAMES, display_model_name
from correlation_labels import inward_label_offset, position_model_labels
from pareto_plot_style import FIGURE_SIZE, MODEL_LABEL_STYLE, apply_pareto_style
from md_success import registered_md_failure_reason
from metric_sources import cohort_timing_files
from system_filters import add_molecular_crystal_option, include_system

FONT_SIZE = 6

plt.rcParams.update(
    {
        "lines.markersize": 4,
        "lines.linewidth": 1.5,
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "figure.titlesize": FONT_SIZE,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
    }
)

palette = sns.color_palette("deep")

CALCULATOR_DISPLAY_NAMES = MODEL_DISPLAY_NAMES


def normalize_model_name(name: str) -> str:
    key = str(name).strip().lower()
    key = key.replace("-force-only", "").replace("-stress", "")
    if key.endswith("-eager"):
        key = key.removesuffix("-eager")
    return {"nequip-oam-l": "nequip"}.get(key, key)


def display_name(model: str) -> str:
    return display_model_name(model)


TIER_1 = ["chgnet", "mace-mp-0", "mace-mp-0-compile", "grace-mp"]
TIER_2 = ["mace-mpa-0", "mace-mpa-0-compile", "orb-v2"]
TIER_3 = [
    "mattersim-v1-5m",
    "grace-oam",
    "orb-v3-omat",
    "orb-v3-direct-omat",
    "esen-30m-oam",
    "nequip",
    "eq-v2-m-omat",
    "pet-oam-xl",
    "pet-omat-xl",
    "grace-oam-compiled",
    "mattersim-v1-5m-compile",
    "pet-oam-xl-torchscript",
    "pet-omat-xl-torchscript",
]
TIER_4 = [
    "mace-mh-omat",
    "mace-mh-omat-compile",
    "uma-s-omat",
    "uma-s-omat-compile",
    "uma-s-omat-turbo",
    "uma-m-omat",
    "uma-m-omat-compile",
    "uma-m-omat-turbo",
]

TIER_COLORS = {
    "Tier 1": palette[2],
    "Tier 2": palette[1],
    "Tier 3": palette[3],
    "Tier 4": palette[0],
    "Other": "#757575",
}

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE = "mlip-trajs-torchsim-eager"
SOURCES = (
    "mlip-trajs-ase",
    "mlip-trajs-torchsim-eager",
    "mlip-trajs-ase-accelerated",
    "mlip-trajs-torchsim-accelerated",
)
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "plots" / "figure_SI_16.pdf"


def source_input_paths(source: str) -> tuple[Path, Path]:
    """Return timing and VDOS summary paths produced for *source*."""
    return (
        CONFIG_DIR / "data" / source,
        SCRIPT_DIR
        / "results"
        / source
        / "vdos_model_mean_ev_normalized_same_simulation_length.csv",
    )


def load_model_avg_timings(timings_dir: Path, include_molecular_crystals: bool = False) -> pd.DataFrame:
    """Return mean MD step time for every model with valid timing CSVs."""
    if not timings_dir.is_dir():
        raise FileNotFoundError(f"Timing directory does not exist: {timings_dir}")
    model_timings: dict[str, list[float]] = {}
    paths = cohort_timing_files(timings_dir)
    all_systems = {path.parent for path in paths if include_system(path.parent.name, include_molecular_crystals)}
    for system_dir in sorted(all_systems):
        for csv_path in paths:
            if csv_path.parent != system_dir:
                continue
            if registered_md_failure_reason(csv_path):
                continue
            model_name = normalize_model_name(csv_path.stem.removeprefix("md_timing_"))
            try:
                df = pd.read_csv(csv_path)
                sps = float(df["seconds_per_step"].iloc[0])
                if not np.isfinite(sps) or sps <= 0.0:
                    continue
                model_timings.setdefault(model_name, []).append(sps)
            except Exception:
                pass
    rows = [
        {"model": m, "mean_time_per_step_ms": float(np.mean(vals)) * 1000}
        for m, vals in model_timings.items()
    ]
    if not rows:
        raise ValueError(f"No valid md_timing_*.csv files found under {timings_dir}")
    return pd.DataFrame(rows)


def _pick_first_existing(columns: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _as_percent(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values.dropna()
    if finite.empty:
        return values
    if float(finite.max()) <= 1.5:
        return values * 100.0
    return values


def _metric_as_percent(series: pd.Series, column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if "percent" in column.lower() or "[%]" in column:
        return values
    return _as_percent(values)


def load_and_merge(timings_dir: Path, vdos_file: Path, include_molecular_crystals: bool = False) -> pd.DataFrame:
    timings_df = load_model_avg_timings(timings_dir, include_molecular_crystals)
    if not vdos_file.is_file():
        raise FileNotFoundError(f"VDOS model summary file not found: {vdos_file}")
    vdos_df = pd.read_csv(vdos_file)

    model_col = _pick_first_existing(
        set(vdos_df.columns),
        ["model", "mlip_model", "Calculator", "calculator"],
    )
    if model_col is None:
        raise ValueError(
            f"VDOS scores file {vdos_file} needs one of: model, mlip_model, Calculator, calculator."
        )

    vdos_error_col = _pick_first_existing(
        set(vdos_df.columns),
        [
            "vdos_error_percent",
            "VDOS Error [%]",
            "final_mean_vdos_error",
            "mean_vdos_error",
            "vdos_error",
        ],
    )
    vdos_similarity_col = _pick_first_existing(
        set(vdos_df.columns),
        [
            "vdos_similarity_percent",
            "VDOS Similarity [%]",
            "final_mean_vdos_similarity",
            "mean_vdos_similarity",
        ],
    )
    if vdos_error_col is None and vdos_similarity_col is None:
        raise ValueError(
            "Missing VDOS metric column in VDOS scores file. Expected an error column "
            "such as final_mean_vdos_error or a similarity column such as final_mean_vdos_similarity."
        )

    metric_col = vdos_error_col if vdos_error_col is not None else vdos_similarity_col
    vdos_small = vdos_df[[model_col, metric_col]].copy()
    vdos_small["model"] = vdos_small[model_col].map(normalize_model_name)
    if vdos_error_col is not None:
        vdos_small["VDOS Error [%]"] = _metric_as_percent(
            vdos_small[vdos_error_col], vdos_error_col
        )
    else:
        vdos_small["VDOS Error [%]"] = 100.0 - _metric_as_percent(
            vdos_small[vdos_similarity_col], vdos_similarity_col
        )

    vdos_small = (
        vdos_small[["model", "VDOS Error [%]"]]
        .groupby("model", as_index=False)
        .mean(numeric_only=True)
    )
    merged = pd.merge(
        timings_df,
        vdos_small,
        on="model",
        how="inner",
    )

    merged["mean_time_per_step_ms"] = pd.to_numeric(
        merged["mean_time_per_step_ms"], errors="coerce"
    )
    merged["VDOS Error [%]"] = pd.to_numeric(merged["VDOS Error [%]"], errors="coerce")
    merged = merged.dropna(subset=["mean_time_per_step_ms", "VDOS Error [%]"]).copy()

    if merged.empty:
        raise ValueError("No overlapping models between timings and VDOS scores files.")

    return merged


def is_pareto_optimal(df: pd.DataFrame) -> np.ndarray:
    """Return mask for non-dominated points for objectives (min time, min VDOS error)."""
    times = df["mean_time_per_step_ms"].to_numpy()
    vdos_errors = df["VDOS Error [%]"].to_numpy()
    n = len(df)

    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_or_equal_time = times[j] <= times[i]
            better_or_equal_vdos_error = vdos_errors[j] <= vdos_errors[i]
            strictly_better_one = (times[j] < times[i]) or (
                vdos_errors[j] < vdos_errors[i]
            )
            if (
                better_or_equal_time
                and better_or_equal_vdos_error
                and strictly_better_one
            ):
                pareto[i] = False
                break

    return pareto


def model_tier(model_name: str) -> str:
    model_name = normalize_model_name(model_name)
    if model_name in TIER_1:
        return "Tier 1"
    if model_name in TIER_2:
        return "Tier 2"
    if model_name in TIER_3:
        return "Tier 3"
    if model_name in TIER_4:
        return "Tier 4"
    return "Other"


def plot_pareto(df: pd.DataFrame, output_file: Path) -> None:
    pareto_mask = is_pareto_optimal(df)

    all_df = df.copy()
    all_df["tier"] = all_df["model"].map(model_tier)
    pareto_df = df[pareto_mask].copy().sort_values("mean_time_per_step_ms")
    pareto_df["tier"] = pareto_df["model"].map(model_tier)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for tier_name in ["Tier 1", "Tier 2", "Tier 3", "Tier 4", "Other"]:
        tier_df = all_df[all_df["tier"] == tier_name]
        if tier_df.empty:
            continue
        ax.scatter(
            tier_df["mean_time_per_step_ms"],
            tier_df["VDOS Error [%]"],
            color=TIER_COLORS[tier_name],
            alpha=0.9,
            s=26,
            label=tier_name,
            zorder=2,
        )

    ax.scatter(
        pareto_df["mean_time_per_step_ms"],
        pareto_df["VDOS Error [%]"],
        facecolors="none",
        edgecolors="black",
        alpha=0.95,
        s=58,
        linewidths=1.0,
        label="Pareto-optimal",
        zorder=4,
    )

    ax.plot(
        pareto_df["mean_time_per_step_ms"],
        pareto_df["VDOS Error [%]"],
        color="black",
        linewidth=1.2,
        alpha=0.9,
        zorder=3,
    )

    x_vals = all_df["mean_time_per_step_ms"].to_numpy()
    y_vals = all_df["VDOS Error [%]"].to_numpy()
    apply_pareto_style(ax, x_vals, y_vals, ylabel="VDOS error [%]")
    for xi, yi, model in zip(x_vals, y_vals, all_df["model"]):
        offset, horizontal, vertical = inward_label_offset(ax, float(xi))
        label = ax.annotate(
            display_name(str(model)),
            xy=(xi, yi),
            xytext=offset,
            textcoords="offset points",
            ha=horizontal,
            va=vertical,
            **MODEL_LABEL_STYLE,
        )
        label._model_point_label = True

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    position_model_labels(fig, [ax], max_offset_points=6, max_distance_points=6)
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved: {output_file}")
    print(f"Models plotted: {len(all_df)}")
    print(f"Pareto-optimal models: {len(pareto_df)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Pareto front of VDOS error vs mean MD step time."
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default=DEFAULT_SOURCE,
        help="Trajectory source used to locate generated timings and VDOS metrics.",
    )
    parser.add_argument(
        "--timings-dir",
        type=Path,
        default=None,
        help="Override the directory containing per-system md_timing_*.csv files.",
    )
    parser.add_argument(
        "--vdos-scores-file",
        type=Path,
        default=None,
        help="Override the generated CSV with model-level VDOS metrics.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output plot path.",
    )
    add_molecular_crystal_option(parser)
    args = parser.parse_args()

    default_timings, default_vdos = source_input_paths(args.source)
    merged = load_and_merge(
        args.timings_dir or default_timings,
        args.vdos_scores_file or default_vdos,
        args.include_molecular_crystals,
    )
    plot_pareto(merged, args.output_file)


if __name__ == "__main__":
    main()
