#!/usr/bin/env python3
"""Plot Pareto front for RDF error vs mean MD step time.

Objective:
- minimize RDF error [%]
- minimize mean MD step time [ms]

Timing observations are read from the ``md_timing_<model>.csv`` files written
by the MD runners. Valid ``seconds_per_step`` values are averaged over systems
for each model before they are joined to the source-specific RDF scores.
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

plt.rcParams.update({
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
})

palette = sns.color_palette("deep")
CALCULATOR_DISPLAY_NAMES = MODEL_DISPLAY_NAMES


def normalize_model_name(name: str) -> str:
    return str(name).strip().lower()


def metric_model_key(name: str) -> str:
    """Normalize execution/property tags when joining timing and RDF tables."""
    key = normalize_model_name(name)
    key = key.replace("-force-only", "").replace("-stress", "")
    if key.endswith("-eager"):
        key = key.removesuffix("-eager")
    return {"nequip-oam-l": "nequip"}.get(key, key)


def base_model_key(name: str) -> str:
    """Return the architecture name without accelerated execution suffixes."""
    key = metric_model_key(name)
    for suffix in ("-torchscript", "-compiled", "-compile", "-turbo"):
        if key.endswith(suffix):
            key = key.removesuffix(suffix)
            break
    return key


def display_name(model: str) -> str:
    return display_model_name(model)


TIER_1 = ["chgnet", "mace-mp-0", "grace-mp"]
TIER_2 = ["mace-mpa-0", "orb-v2"]
TIER_3 = ["mattersim-v1-5m", "grace-oam", "orb-v3-omat", "orb-v3-direct-omat", "esen-30m-oam", "nequip", "eq-v2-m-omat", "pet-oam-xl", "pet-omat-xl"]
TIER_4 = ["mace-mh-omat", "uma-s-omat", "uma-m-omat"]

ACCELERATED_TIER_3 = {
    "grace-oam",
    "mattersim-v1-5m",
    "pet-oam-xl",
    "pet-omat-xl",
}
ACCELERATED_SUFFIXES = ("-torchscript", "-compiled", "-compile", "-turbo")

TIER_COLORS = {
    "Tier 1": palette[2],
    "Tier 2": palette[1],
    "Tier 3": palette[3],
    "Tier 4": palette[0],
    "Other": "#757575",
}

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TIMINGS_DIR = SCRIPT_DIR.parent / "data" / "mlip-trajs-torchsim-eager"
DEFAULT_RDF_SCORES_FILE = SCRIPT_DIR / "results" / "rdf_similarity_scores_same_simulation_length.csv"
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "plots" / "figure_SI_14.pdf"


def load_model_avg_timings(timings_dir: Path, include_molecular_crystals: bool = False) -> pd.DataFrame:
    """Load valid timing CSVs and return mean MD step time for each model."""
    if not timings_dir.is_dir():
        raise FileNotFoundError(f"Timing directory does not exist: {timings_dir}")

    model_timings: dict[str, list[float]] = {}
    for csv_path in cohort_timing_files(timings_dir):
        if not include_system(csv_path.parent.name, include_molecular_crystals):
            continue
        if registered_md_failure_reason(csv_path):
            continue
        model = metric_model_key(csv_path.stem.removeprefix("md_timing_"))
        try:
            timing = pd.read_csv(csv_path)
            if len(timing) != 1 or "seconds_per_step" not in timing.columns:
                continue
            seconds_per_step = float(timing["seconds_per_step"].iloc[0])
            if not np.isfinite(seconds_per_step) or seconds_per_step <= 0.0:
                continue
        except Exception:
            # Empty failure markers and malformed timing files are intentionally
            # ignored; they contain no meaningful runtime observation.
            continue
        model_timings.setdefault(model, []).append(seconds_per_step)

    rows = [
        {
            "model": model,
            "mean_time_per_step_ms": float(np.mean(values)) * 1000.0,
            "n_timing_systems": len(values),
        }
        for model, values in sorted(model_timings.items())
    ]
    if not rows:
        raise ValueError(f"No valid md_timing_*.csv files found under {timings_dir}")
    return pd.DataFrame(rows)


def load_and_merge(timings_dir: Path, rdf_file: Path, include_molecular_crystals: bool = False) -> pd.DataFrame:
    timings_df = load_model_avg_timings(timings_dir, include_molecular_crystals)
    rdf_df = pd.read_csv(rdf_file)

    rdf_needed = {"Calculator"}
    missing_rdf = rdf_needed - set(rdf_df.columns)
    if missing_rdf:
        raise ValueError(f"Missing columns in RDF scores file: {sorted(missing_rdf)}")

    rdf_small = rdf_df[["Calculator"]].copy()
    if "Mean RDF Error [%]" in rdf_df.columns:
        rdf_small["RDF Error [%]"] = pd.to_numeric(rdf_df["Mean RDF Error [%]"], errors="coerce")
    elif "Mean Similarity Score [%]" in rdf_df.columns:
        rdf_small["RDF Error [%]"] = 100.0 - pd.to_numeric(
            rdf_df["Mean Similarity Score [%]"], errors="coerce"
        )
    else:
        raise ValueError("RDF scores file must contain Mean RDF Error [%]")

    rdf_small["model"] = rdf_small["Calculator"].map(metric_model_key)

    merged = pd.merge(
        timings_df,
        rdf_small[["model", "RDF Error [%]"]],
        on="model",
        how="inner",
    )
    merged["mean_time_per_step_ms"] = pd.to_numeric(
        merged["mean_time_per_step_ms"], errors="coerce"
    )
    merged = merged.dropna(subset=["mean_time_per_step_ms", "RDF Error [%]"]).copy()

    if merged.empty:
        raise ValueError("No overlapping models between MD timings and RDF scores files.")

    return merged


def is_pareto_optimal(df: pd.DataFrame) -> np.ndarray:
    """Return mask for non-dominated points.

    Dominance definition for objectives (min time, min RDF error):
    i is dominated by j if:
      time_j <= time_i and err_j <= err_i and at least one is strict.
    """
    times = df["mean_time_per_step_ms"].to_numpy()
    rdf_errors = df["RDF Error [%]"].to_numpy()
    n = len(df)

    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_or_equal_time = times[j] <= times[i]
            better_or_equal_rdf_error = rdf_errors[j] <= rdf_errors[i]
            strictly_better_one = (times[j] < times[i]) or (rdf_errors[j] < rdf_errors[i])
            if better_or_equal_time and better_or_equal_rdf_error and strictly_better_one:
                pareto[i] = False
                break

    return pareto


def model_tier(model_name: str) -> str:
    execution_key = metric_model_key(model_name)
    architecture_key = base_model_key(model_name)
    is_accelerated = execution_key.endswith(ACCELERATED_SUFFIXES)

    # Accelerated execution suffixes describe implementation, not model quality.
    # Only architectures explicitly listed above receive an accelerated override;
    # MACE-MH-OMAT and UMA retain their base Tier 4 classification.
    if is_accelerated and architecture_key in ACCELERATED_TIER_3:
        return "Tier 3"
    if architecture_key in TIER_1:
        return "Tier 1"
    if architecture_key in TIER_2:
        return "Tier 2"
    if architecture_key in TIER_3:
        return "Tier 3"
    if architecture_key in TIER_4:
        return "Tier 4"
    return "Other"


def plot_pareto(df: pd.DataFrame, output_file: Path) -> None:
    pareto_mask = is_pareto_optimal(df)

    all_df = df.copy()
    all_df["tier"] = all_df["model"].map(model_tier)
    pareto_df = df[pareto_mask].copy()
    pareto_df = pareto_df.sort_values("mean_time_per_step_ms")
    pareto_df["tier"] = pareto_df["model"].map(model_tier)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for tier_name in ["Tier 1", "Tier 2", "Tier 3", "Tier 4", "Other"]:
        tier_df = all_df[all_df["tier"] == tier_name]
        if tier_df.empty:
            continue
        ax.scatter(
            tier_df["mean_time_per_step_ms"],
            tier_df["RDF Error [%]"],
            color=TIER_COLORS[tier_name],
            alpha=0.9,
            s=26,
            label=tier_name,
            zorder=2,
        )

    ax.scatter(
        pareto_df["mean_time_per_step_ms"],
        pareto_df["RDF Error [%]"],
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
        pareto_df["RDF Error [%]"],
        color="black",
        linewidth=1.2,
        alpha=0.9,
        zorder=3,
    )

    x_vals = all_df["mean_time_per_step_ms"].to_numpy()
    y_vals = all_df["RDF Error [%]"].to_numpy()
    apply_pareto_style(ax, x_vals, y_vals, ylabel="RDF error [%]")
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
        description="Plot Pareto front of RDF error vs mean MD step time."
    )
    parser.add_argument(
        "--timings-dir",
        default=str(DEFAULT_TIMINGS_DIR),
        help="Directory containing per-system md_timing_<model>.csv files.",
    )
    parser.add_argument(
        "--rdf-scores-file",
        default=str(DEFAULT_RDF_SCORES_FILE),
        help="CSV with model-level Mean RDF Error [%%].",
    )
    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output plot path.",
    )
    add_molecular_crystal_option(parser)
    args = parser.parse_args()

    merged = load_and_merge(Path(args.timings_dir), Path(args.rdf_scores_file), args.include_molecular_crystals)
    plot_pareto(merged, Path(args.output_file))


if __name__ == "__main__":
    main()
