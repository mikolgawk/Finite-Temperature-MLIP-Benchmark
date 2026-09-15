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
"""Create RDF/pressure/VDOS correlation panels with only outlier model labels.

The main output keeps the correlation panels clean by labeling only models that
are far from the fitted trend or from the central point cloud. A larger SI
output with every model labeled is written by default for readers who want to
identify all points.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import string

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_SI_7_8_9 import (
    DEFAULT_PRESSURE_FILE,
    DEFAULT_SCORE_DIR,
    DEFAULT_SOURCE,
    FONT_SIZE,
    LEGEND_FONT_SIZE,
    SCRIPT_DIR,
    SOURCES,
    display_name,
    load_joined_data,
    model_color,
    palette,
)

try:
    from adjustText import adjust_text
except Exception:
    adjust_text = None


MAIN_FIGSIZE = (3.53 * 3, 3.53 * 3)
OUTLIER_LABEL_FONT_SIZE = 10
SI_LABEL_FONT_SIZE = FONT_SIZE

X_METRICS = [
    ("force_rmse", "Force RMSE [eV/A]"),
    ("f1_score", "F1 score"),
    ("ksrme_score", r"$\kappa_{\mathrm{SRME}}$ score"),
]
Y_METRICS = [
    ("rdf_error_percent", "RDF error [%]"),
    ("pressure_error_percent", "Pressure histogram error [%]"),
    ("vdos_error_percent", "VDOS error [%]"),
]


def robust_z_scores(values: np.ndarray) -> np.ndarray:
    """Return absolute robust z scores using MAD with a standard-deviation fallback."""
    values = np.asarray(values, dtype=float)
    scores = np.zeros_like(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return scores

    finite_values = values[finite]
    center = np.nanmedian(finite_values)
    mad = np.nanmedian(np.abs(finite_values - center))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0.0:
        scale = np.nanstd(finite_values)
    if not np.isfinite(scale) or scale <= 0.0:
        return scores

    scores[finite] = np.abs(finite_values - center) / scale
    return scores


def panel_outlier_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Score points by residual outlierness and high-leverage x/y outlierness."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    residual_z = np.zeros_like(y, dtype=float)

    if len(x) >= 3 and np.nanstd(x) > 0.0:
        try:
            slope, intercept = np.polyfit(x, y, 1)
            residual_z = robust_z_scores(y - (slope * x + intercept))
        except Exception:
            residual_z = np.zeros_like(y, dtype=float)

    return np.maximum.reduce([residual_z, robust_z_scores(x), robust_z_scores(y)])


def select_outlier_labels(
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    threshold: float,
    max_labels: int,
) -> set[str]:
    if max_labels <= 0:
        return set()

    x = sub[x_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    ranked = sub[["calculator"]].copy()
    ranked["outlier_score"] = panel_outlier_scores(x, y)
    ranked = ranked[ranked["outlier_score"] >= threshold].copy()
    ranked = ranked.sort_values(["outlier_score", "calculator"], ascending=[False, True])
    ranked = ranked.head(max_labels)
    return set(ranked["calculator"].astype(str))


def inward_label_offset(
    ax: plt.Axes,
    x_value: float,
    y_value: float,
) -> tuple[tuple[int, int], str, str]:
    x_min, x_max = ax.get_xlim()

    x_fraction = 0.5
    if x_max > x_min:
        x_fraction = (x_value - x_min) / (x_max - x_min)

    x_offset = 3
    horizontal_alignment = "left"
    if x_fraction >= 0.65:
        x_offset = -3
        horizontal_alignment = "right"
    elif x_fraction <= 0.35:
        x_offset = 12

    y_offset = 3
    vertical_alignment = "bottom"

    return (x_offset, y_offset), horizontal_alignment, vertical_alignment


def add_point_labels(
    ax: plt.Axes,
    sub: pd.DataFrame,
    labels_to_show: set[str] | None,
    label_font_size: int,
) -> None:
    label_texts = []
    for _, row in sub.iterrows():
        model = str(row["calculator"])
        if labels_to_show is not None and model not in labels_to_show:
            continue

        xytext, horizontal_alignment, vertical_alignment = inward_label_offset(
            ax,
            float(row["x_value"]),
            float(row["y_value"]),
        )
        txt = ax.annotate(
            display_name(model),
            xy=(row["x_value"], row["y_value"]),
            xytext=xytext,
            textcoords="offset points",
            fontsize=label_font_size,
            alpha=0.85,
            ha=horizontal_alignment,
            va=vertical_alignment,
            annotation_clip=False,
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.55,
            },
        )
        txt._keep_inside_axes = True
        label_texts.append(txt)

    if adjust_text is not None and label_texts:
        adjust_kwargs = {
            "texts": label_texts,
            "ax": ax,
            "x": sub["x_value"].to_numpy(dtype=float),
            "y": sub["y_value"].to_numpy(dtype=float),
            "avoid_self": True,
            "only_move": {"points": "xy", "text": "xy"},
            "force_text": (1.2, 1.4),
            "force_points": (0.8, 1.0),
            "expand_points": (1.3, 1.4),
            "expand_text": (1.2, 1.3),
            "lim": 500,
            "arrowprops": {"arrowstyle": "-", "color": "0.5", "lw": 0.4, "alpha": 0.45},
            "ensure_inside_axes": True,
            "expand_axes": False,
        }
        try:
            adjust_text(**adjust_kwargs)
        except TypeError:
            adjust_kwargs.pop("ensure_inside_axes")
            adjust_kwargs.pop("expand_axes")
            adjust_text(**adjust_kwargs)


def keep_model_labels_inside_axes(fig: plt.Figure, axes: np.ndarray) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.ravel():
        axis_bbox = ax.get_window_extent(renderer)
        label_texts = [
            text
            for text in ax.texts
            if getattr(text, "_keep_inside_axes", False)
        ]
        for _ in range(3):
            moved_any = False
            for text in label_texts:
                text_bbox = text.get_window_extent(renderer)
                dx_pixels = 0.0
                dy_pixels = 0.0
                padding = 1.0

                if text_bbox.x0 < axis_bbox.x0 + padding:
                    dx_pixels = axis_bbox.x0 + padding - text_bbox.x0
                elif text_bbox.x1 > axis_bbox.x1 - padding:
                    dx_pixels = axis_bbox.x1 - padding - text_bbox.x1

                if text_bbox.y0 < axis_bbox.y0 + padding:
                    dy_pixels = axis_bbox.y0 + padding - text_bbox.y0
                elif text_bbox.y1 > axis_bbox.y1 - padding:
                    dy_pixels = axis_bbox.y1 - padding - text_bbox.y1

                if dx_pixels == 0.0 and dy_pixels == 0.0:
                    continue

                x_offset, y_offset = text.get_position()
                points_per_pixel = 72.0 / fig.dpi
                text.set_position(
                    (
                        x_offset + dx_pixels * points_per_pixel,
                        y_offset + dy_pixels * points_per_pixel,
                    )
                )
                moved_any = True

            if not moved_any:
                break
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            axis_bbox = ax.get_window_extent(renderer)


def plot_correlation_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    tier_colors: dict[str, tuple[float, float, float]],
    label_mode: str,
    threshold: float,
    max_labels: int,
    label_font_size: int,
) -> list[str]:
    sub = sub.copy()
    sub["x_value"] = sub[x_col].astype(float)
    sub["y_value"] = sub[y_col].astype(float)

    x = sub["x_value"].to_numpy(dtype=float)
    y = sub["y_value"].to_numpy(dtype=float)
    colors = [model_color(model, tier_colors) for model in sub["calculator"].to_numpy()]

    ax.scatter(x, y, c=colors, edgecolor="k", linewidth=0.4, s=40)
    ax.margins(x=0.08, y=0.12)

    labels_to_show = None
    if label_mode == "outliers":
        labels_to_show = select_outlier_labels(sub, x_col, y_col, threshold, max_labels)
    add_point_labels(ax, sub, labels_to_show, label_font_size)

    try:
        if np.nanstd(x) > 0.0:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ax.plot(x_line, slope * x_line + intercept, color="gray", linestyle="--", linewidth=1)
    except Exception:
        pass

    try:
        if np.nanstd(x) > 0.0 and np.nanstd(y) > 0.0:
            r = np.corrcoef(x, y)[0, 1]
            ax.text(0.02, 0.80, f"r = {r:.2f}", transform=ax.transAxes, va="bottom", fontsize=FONT_SIZE)
    except Exception:
        pass

    ax.grid()
    return sorted(labels_to_show) if labels_to_show is not None else sorted(sub["calculator"].astype(str))


def add_tier_legend(fig: plt.Figure, tier_colors: dict[str, tuple[float, float, float]]) -> None:
    handles = [
        mlines.Line2D(
            [],
            [],
            color=tier_colors["tier_1"],
            marker="o",
            linestyle="None",
            markersize=8,
            label="Tier 1",
        ),
        mlines.Line2D(
            [],
            [],
            color=tier_colors["tier_2"],
            marker="o",
            linestyle="None",
            markersize=8,
            label="Tier 2",
        ),
        mlines.Line2D(
            [],
            [],
            color=tier_colors["tier_3"],
            marker="o",
            linestyle="None",
            markersize=8,
            label="Tier 3",
        ),
        mlines.Line2D(
            [],
            [],
            color=tier_colors["tier_4"],
            marker="o",
            linestyle="None",
            markersize=8,
            label="Tier 4",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )


def render_figure(
    df: pd.DataFrame,
    output_file: str,
    label_mode: str,
    threshold: float,
    max_labels: int,
    figsize: tuple[float, float],
    label_font_size: int,
) -> dict[str, list[str]]:
    tier_colors = {
        "tier_1": palette[2],
        "tier_2": palette[1],
        "tier_3": palette[3],
        "tier_4": palette[0],
    }

    fig, axes = plt.subplots(
        len(Y_METRICS),
        3,
        figsize=figsize,
        sharex="col",
        sharey="row",
    )
    label_summary: dict[str, list[str]] = {}

    for row_idx, (y_col, y_label) in enumerate(Y_METRICS):
        for col_idx, (x_col, x_label) in enumerate(X_METRICS):
            ax = axes[row_idx, col_idx]
            is_bottom_row = row_idx == len(Y_METRICS) - 1
            is_left_col = col_idx == 0
            ax.tick_params(axis="x", labelbottom=is_bottom_row)
            ax.tick_params(axis="y", labelleft=is_left_col)
            ax.set_ylabel(y_label if is_left_col else "")
            ax.set_xlabel(x_label if is_bottom_row else "")
            ax.set_xlim(auto=True)
            ax.set_ylim(auto=True)

            panel_index = row_idx * len(X_METRICS) + col_idx
            ax.text(
                0.02,
                0.985,
                f"({string.ascii_lowercase[panel_index]})",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=FONT_SIZE,
            )

            sub = df[["calculator", x_col, y_col]].dropna().copy()
            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONT_SIZE)
                ax.grid()
                continue

            panel_key = f"{y_col} vs {x_col}"
            label_summary[panel_key] = plot_correlation_panel(
                ax=ax,
                sub=sub,
                x_col=x_col,
                y_col=y_col,
                tier_colors=tier_colors,
                label_mode=label_mode,
                threshold=threshold,
                max_labels=max_labels,
                label_font_size=label_font_size,
            )

    add_tier_legend(fig, tier_colors)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keep_model_labels_inside_axes(fig, axes)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    keep_model_labels_inside_axes(fig, axes)
    plt.savefig(output_path)
    plt.show()
    plt.close(fig)
    print(f"Saved {output_path}")
    return label_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create RDF/pressure/VDOS correlation panels with clean outlier-only labels, "
            "plus an optional larger SI figure with all labels."
        )
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default=DEFAULT_SOURCE,
        help="Trajectory source used to locate generated force, RDF, VDOS, and pressure metrics.",
    )
    parser.add_argument(
        "--force-rmse-input",
        "--means-file",
        dest="force_rmse_input",
        type=Path,
        default=None,
        help=(
            "Override the source-specific RMSE result directory with a directory "
            "of rmse-results-all_*.csv files or a model-summary CSV."
        ),
    )
    parser.add_argument(
        "--rdf-file",
        type=Path,
        default=None,
        help="Override the source-specific generated RDF model-summary CSV.",
    )
    parser.add_argument(
        "--pressure-file",
        type=Path,
        default=DEFAULT_PRESSURE_FILE,
        help="Generated CSV with model-level pressure histogram metrics.",
    )
    parser.add_argument(
        "--pressure-backend",
        default=None,
        help="Pressure backend to select (default: inferred from --source).",
    )
    parser.add_argument(
        "--pressure-mode",
        default=None,
        help="Pressure mode to select (default: inferred from --source).",
    )
    parser.add_argument(
        "--vdos-model-means-file",
        type=Path,
        default=None,
        help="Override the source-specific generated VDOS model-summary CSV.",
    )
    parser.add_argument(
        "--f1-file",
        type=Path,
        default=DEFAULT_SCORE_DIR / "f1-scores.csv",
        help="CSV with model-level F1 scores.",
    )
    parser.add_argument(
        "--ksrme-file",
        type=Path,
        default=DEFAULT_SCORE_DIR / "ksrme-scores.csv",
        help="CSV with model-level k_SRME scores.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=SCRIPT_DIR
        / "plots"
        / "plot_rdf_pressure_vdos_correlations_outlier_labels_3x3.pdf",
        help="Clean main output path with only outlier labels.",
    )
    parser.add_argument(
        "--si-output-file",
        default=str(
            SCRIPT_DIR
            / "plots"
            / "plot_SI_rdf_pressure_vdos_correlations_all_labels_3x3.pdf"
        ),
        help="Larger SI output path with every model labeled. Use an empty string to skip.",
    )
    parser.add_argument(
        "--si-scale",
        type=float,
        default=1.45,
        help="Scale factor applied to the main figure size for the SI all-label figure.",
    )
    parser.add_argument(
        "--outlier-z-threshold",
        type=float,
        default=2.5,
        help=(
            "Robust z-score threshold for labeling main-figure outliers. "
            "The score combines fit residual, x-value, and y-value outlierness."
        ),
    )
    parser.add_argument(
        "--max-labels-per-panel",
        type=int,
        default=3,
        help="Maximum number of outlier labels to draw in each main-figure panel.",
    )
    parser.add_argument(
        "--show-outlier-summary",
        action="store_true",
        help="Print the models labeled in each main-figure panel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_joined_data(args)

    outlier_summary = render_figure(
        df=df,
        output_file=args.output_file,
        label_mode="outliers",
        threshold=args.outlier_z_threshold,
        max_labels=args.max_labels_per_panel,
        figsize=MAIN_FIGSIZE,
        label_font_size=OUTLIER_LABEL_FONT_SIZE,
    )

    if args.si_output_file:
        si_figsize = (MAIN_FIGSIZE[0] * args.si_scale, MAIN_FIGSIZE[1] * args.si_scale)
        render_figure(
            df=df,
            output_file=args.si_output_file,
            label_mode="all",
            threshold=args.outlier_z_threshold,
            max_labels=args.max_labels_per_panel,
            figsize=si_figsize,
            label_font_size=SI_LABEL_FONT_SIZE,
        )

    if args.show_outlier_summary:
        print("Main-figure outlier labels:")
        for panel, models in outlier_summary.items():
            labels = ", ".join(display_name(model) for model in models) if models else "none"
            print(f"  {panel}: {labels}")


if __name__ == "__main__":
    main()
