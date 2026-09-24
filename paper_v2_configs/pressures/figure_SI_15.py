#!/usr/bin/env python3
"""Plot Pareto front for pressure error vs force evaluation time per atom.

Objective:
- minimize pressure error
- minimize mean force evaluation time per atom [ms]

The script supports pressure score CSVs from either:
1) pressure similarity outputs (percent-based columns), or
2) pressure MAE outputs (GPa-based columns).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys

PAPER_V2_CONFIG_DIR = Path(__file__).resolve().parents[1]
if str(PAPER_V2_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_V2_CONFIG_DIR))
from model_display_names import MODEL_DISPLAY_NAMES, display_model_name


FONT_SIZE = 6
LEGEND_FONT_SIZE = 6

plt.rcParams.update({
    "lines.markersize": 4,
    "lines.linewidth": 1.5,
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "grid.alpha": 1.0,
})

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
TIER_3 = ["mattersim-v1-5m", "grace-oam", "orb-v3", "orb-v3-direct", "esen-30m-oam", "nequip", "eq-v2-m-omat", "pet-oam-xl", "pet-omat-xl", "grace-oam-compiled", "mattersim-v1-5m-compile", "pet-oam-xl-torchscript", "pet-omat-xl-torchscript"]
TIER_4 = ["mace-mh-omat", "mace-mh-omat-compile", "uma-s-omat", "uma-s-omat-compile", "uma-s-omat-turbo", "uma-m-omat", "uma-m-omat-compile", "uma-m-omat-turbo"]

TIER_COLORS = {
    "Tier 1": palette[2],
    "Tier 2": palette[1],
    "Tier 3": palette[3],
    "Tier 4": palette[0],
    "Other": "#757575",
}

DEFAULT_TIMINGS_DIR = Path(
    "../data/timings_fp32_v100"
)
DEFAULT_PRESSURE_SCORES_FILE = Path(
    "../data/results/same-simulation-length/model_pressure_error_metric.csv"
)
DEFAULT_OUTPUT_FILE = Path(
    "plots/plot_SI_pareto_pressure_time_same_length.pdf"
)


def detect_pressure_error_column(pressure_df: pd.DataFrame) -> tuple[str, str]:
    """
    Detect pressure error column and user-facing axis label.

    Returns:
      (column_name_in_df, axis_label)
    """
    cols = set(pressure_df.columns)

    for col in [
        "final_mean_pressure_error_percent",
        "pressure_error_percent",
        "Pressure Error [%]",
    ]:
        if col in cols:
            return col, "Pressure histogram error [%]"

    if "error_GPa" in cols:
        return "error_GPa", "Pressure MAE [GPa]"

    raise ValueError(
        "Could not find pressure error column in pressure scores file. Expected one of: "
        "final_mean_pressure_error_percent, pressure_error_percent, Pressure Error [%], error_GPa."
    )


def detect_model_column(pressure_df: pd.DataFrame) -> str:
    cols = set(pressure_df.columns)
    if "mlip_model" in cols:
        return "mlip_model"
    if "model" in cols:
        return "model"
    if "calculator" in cols:
        return "calculator"

    raise ValueError("Could not find model column in pressure scores file (expected mlip_model/model/calculator).")


def load_model_avg_timings(timings_dir: Path) -> pd.DataFrame:
    """Compute mean ms/step per model from valid per-system timing CSVs."""
    model_timings: dict[str, list[float]] = {}
    for csv_path in sorted(timings_dir.glob("*/md_timing_*.csv")):
        if csv_path.stat().st_size == 0:
            continue  # Empty files mark failed or interrupted runs.
        model_name = csv_path.stem.removeprefix("md_timing_")
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            sps = float(df["seconds_per_step"].iloc[0])
            if not np.isfinite(sps) or sps <= 0:
                continue
            if "n_steps" in df:
                steps = float(df["n_steps"].iloc[0])
                if not np.isfinite(steps) or steps <= 0:
                    continue
            model_timings.setdefault(model_name, []).append(sps)
        except (ValueError, KeyError, IndexError) as exc:
            print(f"Warning: could not read {csv_path}: {exc}")

    rows = [
        {"model": normalize_model_name(name), "mean_time_ms_per_step": np.mean(vals) * 1000}
        for name, vals in model_timings.items()
    ]
    return pd.DataFrame(rows, columns=["model", "mean_time_ms_per_step"])


def load_and_merge(timings_dir: Path, pressure_file: Path) -> tuple[pd.DataFrame, str]:
    timings_df = load_model_avg_timings(timings_dir)
    if timings_df.empty:
        raise ValueError(f"No valid timing records found in {timings_dir}")
    pressure_df = pd.read_csv(pressure_file)

    pressure_model_col = detect_model_column(pressure_df)
    pressure_error_col, y_axis_label = detect_pressure_error_column(pressure_df)

    pressure_small = pressure_df[[pressure_model_col, pressure_error_col]].copy()
    # Normalize model names (strip simulation-length suffixes) so they merge correctly
    pressure_small["model"] = (
        pressure_small[pressure_model_col]
        .astype(str)
        .apply(lambda m: normalize_model_name(re.sub(r"_same-simulation-length$", "", m)))
    )

    merged = pd.merge(
        timings_df[["model", "mean_time_ms_per_step"]],
        pressure_small[["model", pressure_error_col]],
        on="model",
        how="inner",
    )

    merged["mean_time_ms_per_step"] = pd.to_numeric(merged["mean_time_ms_per_step"], errors="coerce")
    merged[pressure_error_col] = pd.to_numeric(merged[pressure_error_col], errors="coerce")

    merged = merged.dropna(subset=["mean_time_ms_per_step", pressure_error_col]).copy()
    merged = merged.rename(columns={pressure_error_col: "pressure_error"})

    if merged.empty:
        raise ValueError("No overlapping models between timing data and pressure scores file.")

    return merged, y_axis_label


def is_pareto_optimal(df: pd.DataFrame) -> np.ndarray:
    """Return mask for non-dominated points.

    Dominance definition for objectives (min time, min pressure error):
    i is dominated by j if:
      time_j <= time_i and err_j <= err_i and at least one is strict.
    """
    times = df["mean_time_ms_per_step"].to_numpy()
    pressure_errors = df["pressure_error"].to_numpy()
    n = len(df)

    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_or_equal_time = times[j] <= times[i]
            better_or_equal_error = pressure_errors[j] <= pressure_errors[i]
            strictly_better_one = (times[j] < times[i]) or (pressure_errors[j] < pressure_errors[i])
            if better_or_equal_time and better_or_equal_error and strictly_better_one:
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


def plot_pareto(df: pd.DataFrame, y_axis_label: str, output_file: Path) -> None:
    pareto_mask = is_pareto_optimal(df)

    all_df = df.copy()
    all_df["tier"] = all_df["model"].map(model_tier)
    pareto_df = df[pareto_mask].copy()
    pareto_df = pareto_df.sort_values("mean_time_ms_per_step")

    fig, ax = plt.subplots(figsize=(3.53 * 1.5, 3.53 * 1.5))

    for tier_name in ["Tier 1", "Tier 2", "Tier 3", "Tier 4", "Other"]:
        tier_df = all_df[all_df["tier"] == tier_name]
        if tier_df.empty:
            continue
        ax.scatter(
            tier_df["mean_time_ms_per_step"],
            tier_df["pressure_error"],
            color=TIER_COLORS[tier_name],
            alpha=0.9,
            s=26,
            label=tier_name,
            zorder=2,
        )

    ax.scatter(
        pareto_df["mean_time_ms_per_step"],
        pareto_df["pressure_error"],
        facecolors="none",
        edgecolors="black",
        alpha=0.95,
        s=58,
        linewidths=1.0,
        label="Pareto-optimal",
        zorder=4,
    )

    ax.plot(
        pareto_df["mean_time_ms_per_step"],
        pareto_df["pressure_error"],
        color="black",
        linewidth=1.2,
        alpha=0.9,
        zorder=3,
    )

    for _, row in all_df.iterrows():
        ax.annotate(
            display_name(row["model"]),
            (row["mean_time_ms_per_step"], row["pressure_error"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=FONT_SIZE,
        )

    ax.set_xlabel("Mean time per step [ms]")
    ax.set_ylabel(y_axis_label)
    # ax.set_title("Pareto Front: Pressure Error vs Force Eval Time")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=True)

    ax.set_xlim(right=590)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved: {output_file}")
    print(f"Models plotted: {len(all_df)}")
    print(f"Pareto-optimal models: {len(pareto_df)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Pareto front of pressure error vs force evaluation time per atom."
    )
    parser.add_argument(
        "--timings-dir",
        default=str(DEFAULT_TIMINGS_DIR),
        help="Directory containing per-system md_timing_<model>.csv files (timings_fp32_v100).",
    )
    parser.add_argument(
        "--pressure-scores-file",
        default=str(DEFAULT_PRESSURE_SCORES_FILE),
        help="CSV with model-level pressure error or pressure similarity columns.",
    )
    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output plot path.",
    )
    args = parser.parse_args()

    merged, y_axis_label = load_and_merge(Path(args.timings_dir), Path(args.pressure_scores_file))
    plot_pareto(merged, y_axis_label, Path(args.output_file))


if __name__ == "__main__":
    main()
