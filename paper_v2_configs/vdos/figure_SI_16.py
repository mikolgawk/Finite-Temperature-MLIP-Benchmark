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

try:
    from adjustText import adjust_text
except Exception:
    adjust_text = None

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


TIER_1 = ["chgnet", "mace-mp-0", "mace-mp-0-compile", "grace-mp"]
TIER_2 = ["mace-mpa-0", "mace-mpa-0-compile", "orb-v2"]
TIER_3 = [
    "mattersim-v1-5m",
    "grace-oam",
    "orb-v3",
    "orb-v3-direct",
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
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "plots" / "plot_SI_pareto_vdos_time_same_length.pdf"


def source_input_paths(source: str) -> tuple[Path, Path]:
    """Return timing and VDOS summary paths produced for *source*."""
    return (
        CONFIG_DIR / "data" / source,
        SCRIPT_DIR
        / "results"
        / source
        / "vdos_model_mean_ev_normalized_same_simulation_length.csv",
    )


def load_model_avg_timings(timings_dir: Path) -> pd.DataFrame:
    """Return mean MD step time for every model with valid timing CSVs."""
    if not timings_dir.is_dir():
        raise FileNotFoundError(f"Timing directory does not exist: {timings_dir}")
    model_timings: dict[str, list[float]] = {}
    all_systems = [d for d in timings_dir.iterdir() if d.is_dir()]
    for system_dir in sorted(all_systems):
        for csv_path in sorted(system_dir.glob("md_timing_*.csv")):
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


def load_and_merge(timings_dir: Path, vdos_file: Path) -> pd.DataFrame:
    timings_df = load_model_avg_timings(timings_dir)
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

    fig, ax = plt.subplots(figsize=(3.53 * 1.5, 3.53 * 1.5))

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

    ax.set_xlim(right=590)

    # Keep labels and their leader-line targets in data coordinates so adjustText
    # cannot send labels outside the axes or make arrows converge spuriously.
    label_texts = []
    x_vals = all_df["mean_time_per_step_ms"].to_numpy()
    y_vals = all_df["VDOS Error [%]"].to_numpy()
    for xi, yi, m in zip(x_vals, y_vals, all_df["model"].values):
        txt = ax.text(
            xi,
            yi,
            display_name(str(m)),
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
            ax, x_vals, y_vals, [display_name(str(m)) for m in all_df["model"]]
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

    ax.set_xlabel("Mean time per step [ms]")
    ax.set_ylabel("VDOS error [%]")
    # ax.set_title("Pareto Front: VDOS Error vs Force Eval Time")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
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
    args = parser.parse_args()

    default_timings, default_vdos = source_input_paths(args.source)
    merged = load_and_merge(
        args.timings_dir or default_timings,
        args.vdos_scores_file or default_vdos,
    )
    plot_pareto(merged, args.output_file)


if __name__ == "__main__":
    main()
