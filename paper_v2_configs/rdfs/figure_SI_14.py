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


try:
    from adjustText import adjust_text
except Exception:
    adjust_text = None

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


def add_spread_labels(ax, x_vals, y_vals, labels) -> None:
    """Place labels with deterministic vertical separation without adjustText."""
    ax.figure.canvas.draw()
    points = ax.transData.transform(np.column_stack([x_vals, y_vals]))
    axes_box = ax.get_window_extent()
    gap = FONT_SIZE * ax.figure.dpi / 72.0 * 1.55
    lower = axes_box.y0 + gap / 2
    upper = axes_box.y1 - gap / 2

    order = np.argsort(points[:, 1])
    placed_y = points[:, 1].copy()
    for previous, current in zip(order[:-1], order[1:]):
        placed_y[current] = max(placed_y[current], placed_y[previous] + gap)
    if placed_y[order[-1]] > upper:
        placed_y[order] -= placed_y[order[-1]] - upper
    for current, following in zip(order[-2::-1], order[:0:-1]):
        placed_y[current] = min(placed_y[current], placed_y[following] - gap)
    if placed_y[order[0]] < lower:
        placed_y[order] += lower - placed_y[order[0]]

    inverse = ax.transData.inverted()
    midpoint = (axes_box.x0 + axes_box.x1) / 2
    for (point_x, point_y), label_y, label in zip(points, placed_y, labels):
        align_left = point_x < midpoint
        label_x = point_x + 4 if align_left else point_x - 4
        text_x, text_y = inverse.transform((label_x, label_y))
        ax.annotate(
            label,
            xy=inverse.transform((point_x, point_y)),
            xytext=(text_x, text_y),
            textcoords="data",
            fontsize=FONT_SIZE,
            alpha=0.9,
            ha="left" if align_left else "right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.75),
            arrowprops=dict(arrowstyle="-", color="0.55", lw=0.4, alpha=0.6),
            zorder=5,
        )


TIER_1 = ["chgnet", "mace-mp-0", "grace-mp"]
TIER_2 = ["mace-mpa-0", "orb-v2"]
TIER_3 = ["mattersim-v1-5m", "grace-oam", "orb-v3", "esen-30m-oam", "nequip", "eq-v2-m-omat", "pet-oam-xl"]
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
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "plots" / "plot_SI_pareto_rdf_time_same_length.pdf"


def load_model_avg_timings(timings_dir: Path) -> pd.DataFrame:
    """Load valid timing CSVs and return mean MD step time for each model."""
    if not timings_dir.is_dir():
        raise FileNotFoundError(f"Timing directory does not exist: {timings_dir}")

    model_timings: dict[str, list[float]] = {}
    for csv_path in sorted(timings_dir.glob("*/md_timing_*.csv")):
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


def load_and_merge(timings_dir: Path, rdf_file: Path) -> pd.DataFrame:
    timings_df = load_model_avg_timings(timings_dir)
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

    fig, ax = plt.subplots(figsize=(3.53 * 1.5, 3.53 * 1.5))

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

    # Create labels in data coordinates.  adjustText can then move them and draw
    # leader lines in the same coordinate system; offset-point annotations cause
    # misplaced text and converging arrows after adjustment.
    label_texts = []
    x_vals = all_df["mean_time_per_step_ms"].to_numpy()
    y_vals = all_df["RDF Error [%]"].to_numpy()
    for xi, yi, m in zip(x_vals, y_vals, all_df["model"].values):
        txt = ax.text(
            xi,
            yi,
            display_name(m),
            fontsize=FONT_SIZE,
            alpha=0.9,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.08",
                facecolor="white",
                edgecolor="none",
                alpha=0.7,
            ),
            zorder=5,
        )
        label_texts.append(txt)

    ax.margins(x=0.08, y=0.08)
    if adjust_text is None:
        for txt in label_texts:
            txt.remove()
        add_spread_labels(
            ax, x_vals, y_vals, [display_name(m) for m in all_df["model"]]
        )
    elif label_texts:
        try:
            adjust_text(
                label_texts,
                ax=ax,
                x=x_vals,
                y=y_vals,
                target_x=x_vals,
                target_y=y_vals,
                avoid_self=True,
                prevent_crossings=True,
                ensure_inside_axes=True,
                expand_axes=True,
                force_text=(0.8, 1.2),
                force_static=(0.4, 0.7),
                force_pull=(0.015, 0.025),
                force_explode=(0.7, 1.0),
                expand=(1.35, 1.55),
                max_move=(60, 60),
                min_arrow_len=10,
                iter_lim=3000,
                arrowprops=dict(
                    arrowstyle="-",
                    color="0.55",
                    lw=0.45,
                    alpha=0.65,
                    shrinkA=4,
                    shrinkB=3,
                ),
            )
        except Exception:
            pass

    ax.set_xlabel("Mean MD time per step [ms]")
    ax.set_ylabel("RDF error [%]")
    # ax.set_title("Pareto Front: RDF Error vs MD Step Time")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0.02)
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
    args = parser.parse_args()

    merged = load_and_merge(Path(args.timings_dir), Path(args.rdf_scores_file))
    plot_pareto(merged, Path(args.output_file))


if __name__ == "__main__":
    main()
