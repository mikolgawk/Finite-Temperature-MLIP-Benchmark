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
"""Create RDF/pressure/VDOS correlation panels with only outlier model labels.

The main output labels every model with an absolute externally studentized
regression residual at or above the threshold, without a default limit per
panel. Extreme x/y coordinates alone do not trigger a label. This is a
descriptive regression diagnostic, not a statistical significance test. A larger
SI output with every model labeled is written by default.

Metric means use only each model's completed MD runs that are not registered as
physically unstable, matching the SI 1x3 correlation plots. Benchmark failure
penalties remain in the metric CSVs. F1 and k_SRME values are unchanged.

Use --exclude-hydrogen to exclude hydrogen from force RMSE, RDF, pressure, and
VDOS means. The legacy --exclude-hydrogen-force-rmse option and
_no_hydrogen_force_rmse output suffix remain supported.
Add --exclude-pt-water to also exclude Pt+H2O from the metric averages.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import string

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from .figure_SI_7_8_9 import (
        DEFAULT_SCORE_DIR,
        DEFAULT_SOURCE,
        FONT_SIZE,
        LEGEND_FONT_SIZE,
        SOURCES,
        add_force_rmse_exclusion_option,
        correlation_output_path,
        display_name,
        error_axis_label,
        force_rmse_axis_label,
        load_joined_data as load_correlation_data,
        model_color,
        palette,
    )
else:
    from figure_SI_7_8_9 import (
        DEFAULT_SCORE_DIR,
        DEFAULT_SOURCE,
        FONT_SIZE,
        LEGEND_FONT_SIZE,
        SOURCES,
        add_force_rmse_exclusion_option,
        correlation_output_path,
        display_name,
        error_axis_label,
        force_rmse_axis_label,
        load_joined_data as load_correlation_data,
        model_color,
        palette,
    )
from correlation_labels import inward_label_offset, position_model_labels

# Match v1's text-to-canvas proportions when the figure is fitted into the paper.
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


def load_joined_data(args: argparse.Namespace) -> pd.DataFrame:
    """Reaverage Figure 6's metric inputs over each model's stable MD runs."""
    return load_correlation_data(args, stable_md_only=True, md_data_dir=getattr(args, "md_data_dir", None))


def panel_outlier_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Absolute externally studentized OLS residuals: |e_i|/(s_(-i)*sqrt(1-h_i)).

    The residual and leverage come from the full intercept/slope fit. Each
    deleted fit estimates its residual variance using n-3 degrees of freedom.
    Direct deleted fits avoid cancellation in SSE-e_i**2/(1-h_i). Undefined
    diagnostics and numerically exact full fits do not produce labels.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("Outlier scores require paired one-dimensional metric arrays")
    scores = np.zeros_like(y)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    n = len(x)
    if n < 4:
        return scores
    x, y = x - x.mean(), y - y.mean()
    x_scale, y_scale = np.max(np.abs(x)), np.max(np.abs(y))
    if x_scale == 0 or y_scale == 0:
        return scores
    x, y = x / x_scale, y / y_scale
    design = np.column_stack((np.ones(n), x))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 2:
        return scores
    residuals = y - design @ coefficients
    numerical_residual = 64 * n * np.finfo(float).eps
    sse_tolerance = numerical_residual**2 * np.dot(y, y)
    if np.dot(residuals, residuals) <= sse_tolerance:
        return scores
    q, _ = np.linalg.qr(design, mode="reduced")
    leverage = np.sum(q**2, axis=1)
    indices = np.flatnonzero(finite)
    for i in range(n):
        remaining_leverage = 1 - leverage[i]
        if remaining_leverage <= 64 * np.finfo(float).eps:
            continue
        keep = np.arange(n) != i
        deleted_coefficients, _, deleted_rank, _ = np.linalg.lstsq(design[keep], y[keep], rcond=None)
        if deleted_rank < 2:
            continue
        deleted_residuals = y[keep] - design[keep] @ deleted_coefficients
        deleted_sse = np.dot(deleted_residuals, deleted_residuals)
        if deleted_sse <= sse_tolerance:
            if abs(residuals[i]) > numerical_residual:
                scores[indices[i]] = np.inf
            continue
        deleted_variance = deleted_sse / (n - 3)
        scores[indices[i]] = abs(residuals[i]) / np.sqrt(deleted_variance * remaining_leverage)
    return scores


def finite_panel_data(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Use the same finite model/metric pairs for labels, dots, fits, and r."""
    sub = df.loc[df["calculator"].notna(), ["calculator", x_col, y_col]].copy()
    for column in (x_col, y_col):
        sub[column] = pd.to_numeric(sub[column], errors="coerce")
    return sub.loc[np.isfinite(sub[x_col]) & np.isfinite(sub[y_col])].copy()


def select_outlier_labels(
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    threshold: float,
    max_labels: int | None = None,
) -> set[str]:
    if max_labels is not None and max_labels <= 0:
        return set()

    sub = finite_panel_data(sub, x_col, y_col)
    x = sub[x_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    ranked = sub[["calculator"]].copy()
    ranked["outlier_score"] = panel_outlier_scores(x, y)
    ranked = ranked[ranked["outlier_score"] >= threshold].copy()
    ranked = ranked.sort_values(["outlier_score", "calculator"], ascending=[False, True])
    if max_labels is not None:
        ranked = ranked.head(max_labels)
    return set(ranked["calculator"].astype(str))


def outlier_label_summary(
    df: pd.DataFrame, threshold: float, max_labels: int | None = None,
) -> dict[str, list[str]]:
    """Report exactly which outlier names each panel will show, without plotting."""
    return {
        f"{y_col} vs {x_col}": sorted(select_outlier_labels(df, x_col, y_col, threshold, max_labels))
        for y_col, _ in Y_METRICS for x_col, _ in X_METRICS
    }


def add_point_labels(
    ax: plt.Axes,
    sub: pd.DataFrame,
    labels_to_show: set[str] | None,
    label_font_size: int,
) -> None:
    for _, row in sub.iterrows():
        model = str(row["calculator"])
        if labels_to_show is not None and model not in labels_to_show:
            continue

        xytext, horizontal_alignment, vertical_alignment = inward_label_offset(
            ax,
            float(row["x_value"]),
        )
        txt = ax.annotate(
            display_name(model),
            xy=(row["x_value"], row["y_value"]),
            xytext=xytext,
            textcoords="offset points",
            fontsize=label_font_size,
            alpha=1.0,
            zorder=10,
            ha=horizontal_alignment,
            va=vertical_alignment,
            annotation_clip=False,
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )
        txt._model_point_label = True


def plot_correlation_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    tier_colors: dict[str, tuple[float, float, float]],
    label_mode: str,
    threshold: float,
    max_labels: int | None,
    label_font_size: int,
) -> list[str]:
    sub = finite_panel_data(sub, x_col, y_col)
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
            ax.text(
                0.98, 0.985,
                f"r = {r:.2f}", transform=ax.transAxes,
                va="top", ha="right", fontsize=FONT_SIZE,
            )
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
    max_labels: int | None,
    figsize: tuple[float, float],
    label_font_size: int,
    exclude_hydrogen_force_rmse: bool = False,
    exclude_pt_water: bool = False,
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
        y_label = error_axis_label(y_label, exclude_hydrogen_force_rmse, exclude_pt_water)
        for col_idx, (x_col, x_label) in enumerate(X_METRICS):
            if x_col == "force_rmse":
                x_label = force_rmse_axis_label(exclude_hydrogen_force_rmse, exclude_pt_water)
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

            sub = finite_panel_data(df, x_col, y_col)
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
    label_options = {"max_offset_points": 18}
    if label_mode == "all":
        label_options["max_distance_points"] = 18
    position_model_labels(fig, axes.ravel(), **label_options)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    position_model_labels(fig, axes.ravel(), **label_options)
    plt.savefig(output_path)
    plt.show()
    plt.close(fig)
    print(f"Saved {output_path}")
    return label_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
            "Override the source-specific RMSE input with a per-system CSV or a "
            "directory of rmse-results-all_*.csv files. Stable-MD filtering "
            "requires per-system rows (default: rmse_per_system.csv)."
        ),
    )
    add_force_rmse_exclusion_option(parser)
    parser.add_argument(
        "--md-data-dir",
        type=Path,
        default=None,
        help="MD data root used to identify completed, stable runs (default: paper_v2_configs/data).",
    )
    parser.add_argument(
        "--rdf-file",
        type=Path,
        default=None,
        help="Override the RDF summary or per-system CSV (or per-model CSV directory).",
    )
    parser.add_argument(
        "--pressure-file",
        type=Path,
        default=None,
        help="Generated CSV with model-level or per-system pressure histogram metrics.",
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
        help="Override the source-specific VDOS summary or per-system CSV.",
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
        default=None,
        help="Clean main output path with only outlier labels.",
    )
    parser.add_argument(
        "--si-output-file",
        default=None,
        help="Larger SI output path with every model labeled. Use an empty string to skip.",
    )
    parser.add_argument(
        "--si-scale",
        type=float,
        default=1.8,
        help="Scale factor applied to the main figure size for the SI all-label figure.",
    )
    parser.add_argument(
        "--outlier-threshold",
        "--outlier-z-threshold",
        dest="outlier_z_threshold",
        type=float,
        default=2.5,
        help=(
            "Absolute externally studentized residual threshold for labels (default: 2.5). "
            "Extreme metric coordinates alone are not labeled. This is a descriptive cutoff."
        ),
    )
    parser.add_argument(
        "--max-labels-per-panel",
        type=int,
        default=None,
        help="Optional cap on outlier labels per main-figure panel; default: all qualifying models. Use 0 to hide labels.",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Report outlier names for each panel without creating figures.",
    )
    parser.add_argument(
        "--show-outlier-summary",
        action="store_true",
        help="Print the models labeled in each main-figure panel.",
    )
    args = parser.parse_args(argv)
    if args.output_file is None:
        args.output_file = correlation_output_path(
            "figure_6", args.exclude_hydrogen_force_rmse, args.exclude_pt_water,
        )
    if args.si_output_file is None:
        args.si_output_file = str(correlation_output_path(
            "figure_6_all_labels", args.exclude_hydrogen_force_rmse,
            args.exclude_pt_water,
        ))
    return args


def main() -> None:
    args = parse_args()
    df = load_joined_data(args)

    if args.data_only or args.show_outlier_summary:
        print("Main-figure outlier labels:")
        for panel, models in outlier_label_summary(df, args.outlier_z_threshold, args.max_labels_per_panel).items():
            labels = ", ".join(display_name(model) for model in models) if models else "none"
            print(f"  {panel}: {labels}")
    if args.data_only:
        return

    render_figure(
        df=df,
        output_file=args.output_file,
        label_mode="outliers",
        threshold=args.outlier_z_threshold,
        max_labels=args.max_labels_per_panel,
        figsize=MAIN_FIGSIZE,
        label_font_size=OUTLIER_LABEL_FONT_SIZE,
        exclude_hydrogen_force_rmse=args.exclude_hydrogen_force_rmse,
        exclude_pt_water=args.exclude_pt_water,
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
            exclude_hydrogen_force_rmse=args.exclude_hydrogen_force_rmse,
            exclude_pt_water=args.exclude_pt_water,
        )


if __name__ == "__main__":
    main()
