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
"""Create separate 1x3 correlation panels for RDF, pressure, and VDOS errors.

Rows:
1) RDF error
2) Pressure histogram error
3) VDOS error

Columns:
1) Force RMSE
2) F1 score
3) k_SRME score
"""

from __future__ import annotations

import argparse
from pathlib import Path
import string

import matplotlib.lines as mlines
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


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE = "mlip-trajs-torchsim-eager"
SOURCES = (
    "mlip-trajs-ase",
    "mlip-trajs-torchsim-eager",
    "mlip-trajs-ase-accelerated",
    "mlip-trajs-torchsim-accelerated",
)
DEFAULT_PRESSURE_FILE = (
    CONFIG_DIR / "pressures" / "results" / "model_pressure_error_metric.csv"
)
DEFAULT_SCORE_DIR = CONFIG_DIR / "data" / "matbench-scores"

FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
SPLIT_FIGSIZE = (3.53 * 3, 3.53 * 2)

plt.rcParams.update(
    {
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


EXCLUDED_MODELS = {"pet-mad"}

TIER_1_MODELS = ["chgnet", "mace-mp-0", "mace-mp-0-compile", "grace-mp"]
TIER_2_MODELS = ["mace-mpa-0", "mace-mpa-0-compile", "orb-v2"]
TIER_3_MODELS = [
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
TIER_4_MODELS = [
    "mace-mh-omat",
    "mace-mh-omat-compile",
    "uma-s-omat",
    "uma-s-omat-compile",
    "uma-s-omat-turbo",
    "uma-m-omat",
    "uma-m-omat-compile",
    "uma-m-omat-turbo",
]


def source_paths(source: str) -> tuple[Path, Path, Path]:
    """Return source-specific force-RMSE, RDF, and VDOS inputs."""
    backend_dir = "e-f-predictions" if "torchsim" in source else "e-f-predictions-ase"
    mode_dir = "md-accelerated" if source.endswith("-accelerated") else "md_eager"
    return (
        CONFIG_DIR / "e_f_rmses" / "data" / backend_dir / mode_dir,
        CONFIG_DIR
        / "rdfs"
        / "results"
        / source
        / "rdf_similarity_scores_same_simulation_length.csv",
        SCRIPT_DIR
        / "results"
        / source
        / "vdos_model_mean_ev_normalized_same_simulation_length.csv",
    )


def source_pressure_provenance(source: str) -> tuple[str, str]:
    backend = "torchsim" if "torchsim" in source else "ase"
    mode = "md_accelerated" if source.endswith("-accelerated") else "md_eager"
    return backend, mode


def pick_first_column(df: pd.DataFrame, candidates: list[str], description: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Missing {description}. Expected one of: {', '.join(candidates)}. "
        f"Found columns: {list(df.columns)}"
    )


def values_to_percent(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return values
    if float(values.max()) <= 1.5:
        return values * 100.0
    return values


def metric_values_to_percent(series: pd.Series, column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if "percent" in column.lower() or "[%]" in column:
        return values
    return values_to_percent(values)


def aggregate_models(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("calculator", as_index=False).mean(numeric_only=True)


def standardize_rdf(df_rdf_raw: pd.DataFrame) -> pd.DataFrame:
    model_col = pick_first_column(
        df_rdf_raw,
        ["calculator", "Calculator", "model", "mlip_model"],
        "RDF model column",
    )
    error_col = next(
        (
            col
            for col in (
                "rdf_error_percent",
                "Mean RDF Error [%]",
                "RDF Error [%]",
                "rdf_error_mean",
            )
            if col in df_rdf_raw.columns
        ),
        None,
    )
    similarity_col = next(
        (
            col
            for col in (
                "rdf_similarity_percent",
                "Mean Similarity Score [%]",
                "RDF Similarity [%]",
                "rdf_similarity_mean",
                "rdf_similarity",
            )
            if col in df_rdf_raw.columns
        ),
        None,
    )
    if error_col is None and similarity_col is None:
        raise ValueError("RDF table needs an RDF error or similarity column.")
    metric_col = error_col or similarity_col
    out = df_rdf_raw[[model_col, metric_col]].copy()
    out = out.rename(columns={model_col: "calculator"})
    out["calculator"] = (
        out["calculator"].astype(str).str.strip().map(normalize_model_name)
    )
    if error_col is not None:
        out["rdf_error_percent"] = metric_values_to_percent(out[metric_col], metric_col)
    else:
        similarity = metric_values_to_percent(out[metric_col], metric_col)
        out["rdf_error_percent"] = 100.0 - similarity
    return aggregate_models(out[["calculator", "rdf_error_percent"]])


def standardize_vdos(df_vdos_raw: pd.DataFrame) -> pd.DataFrame:
    model_col = pick_first_column(
        df_vdos_raw,
        ["model", "mlip_model", "calculator", "Calculator"],
        "VDOS model column",
    )
    error_col = next(
        (
            col
            for col in (
                "vdos_error_percent",
                "VDOS Error [%]",
                "Mean VDOS Error [%]",
                "final_mean_vdos_error",
                "mean_vdos_error",
                "vdos_error",
            )
            if col in df_vdos_raw.columns
        ),
        None,
    )
    similarity_col = next(
        (
            col
            for col in (
                "vdos_similarity_percent",
                "Mean Similarity Score [%]",
                "VDOS Similarity [%]",
                "final_mean_vdos_similarity",
                "mean_vdos_similarity",
                "vdos_similarity",
            )
            if col in df_vdos_raw.columns
        ),
        None,
    )
    if error_col is None and similarity_col is None:
        raise ValueError("VDOS table needs a VDOS error or similarity column.")
    metric_col = error_col or similarity_col
    out = df_vdos_raw[[model_col, metric_col]].copy()
    out = out.rename(columns={model_col: "calculator"})
    out["calculator"] = (
        out["calculator"].astype(str).str.strip().map(normalize_model_name)
    )
    metric = metric_values_to_percent(out[metric_col], metric_col)
    out["vdos_error_percent"] = metric if error_col is not None else 100.0 - metric
    return aggregate_models(out[["calculator", "vdos_error_percent"]])


def standardize_pressure(
    df_pressure_raw: pd.DataFrame,
    *,
    backend: str | None = None,
    mode: str | None = None,
) -> pd.DataFrame:
    for column, value in (("backend", backend), ("mode", mode)):
        if value is not None and column in df_pressure_raw.columns:
            df_pressure_raw = df_pressure_raw.loc[
                df_pressure_raw[column].astype(str).str.lower() == value.lower()
            ].copy()
    if df_pressure_raw.empty:
        raise ValueError(
            f"No pressure rows remain for backend={backend!r}, mode={mode!r}."
        )
    model_col = pick_first_column(
        df_pressure_raw,
        ["model", "calculator", "Calculator", "mlip_model"],
        "pressure model column",
    )
    error_col = next(
        (
            col
            for col in (
                "pressure_error_percent",
                "final_mean_pressure_error_percent",
                "mean_pressure_error_percent",
                "Pressure Error [%]",
            )
            if col in df_pressure_raw.columns
        ),
        None,
    )
    similarity_col = next(
        (
            col
            for col in (
                "pressure_similarity_percent",
                "final_mean_pressure_similarity_percent",
                "mean_pressure_similarity_percent",
                "Pressure Similarity [%]",
                "final_mean_pressure_similarity",
            )
            if col in df_pressure_raw.columns
        ),
        None,
    )
    if error_col is None and similarity_col is None:
        raise ValueError("Pressure table needs a pressure error or similarity column.")
    metric_col = error_col or similarity_col
    out = df_pressure_raw[[model_col, metric_col]].copy()
    out = out.rename(columns={model_col: "calculator"})
    out["calculator"] = (
        out["calculator"].astype(str).str.strip().map(normalize_model_name)
    )
    metric = metric_values_to_percent(out[metric_col], metric_col)
    out["pressure_error_percent"] = metric if error_col is not None else 100.0 - metric
    return aggregate_models(out[["calculator", "pressure_error_percent"]])


def model_color(model: str, tier_colors: dict[str, tuple[float, float, float]]) -> str:
    model = normalize_model_name(model)
    if model in TIER_1_MODELS:
        return tier_colors["tier_1"]
    if model in TIER_2_MODELS:
        return tier_colors["tier_2"]
    if model in TIER_3_MODELS:
        return tier_colors["tier_3"]
    if model in TIER_4_MODELS:
        return tier_colors["tier_4"]
    return "#757575"


def inward_label_offset(
    ax: plt.Axes,
    x_value: float,
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

    return (x_offset, 3), horizontal_alignment, "bottom"


def keep_model_labels_inside_axes(fig: plt.Figure, axes: np.ndarray) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.ravel():
        axis_bbox = ax.get_window_extent(renderer)
        label_texts = [
            text for text in ax.texts if getattr(text, "_keep_inside_axes", False)
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


def standardize_force_rmse(df_means_raw: pd.DataFrame) -> pd.DataFrame:
    means_model_col = pick_first_column(
        df_means_raw,
        ["calculator", "Calculator", "model", "mlip_model"],
        "means model column",
    )
    force_rmse_col = pick_first_column(
        df_means_raw,
        ["force_rmse", "force_RMSE", "mean_force_rmse"],
        "force RMSE column",
    )
    out = df_means_raw[[means_model_col, force_rmse_col]].copy()
    out = out.rename(
        columns={means_model_col: "calculator", force_rmse_col: "force_rmse"}
    )
    out["calculator"] = out["calculator"].map(normalize_model_name)
    out["force_rmse"] = pd.to_numeric(out["force_rmse"], errors="coerce")
    return aggregate_models(out)


def load_force_rmse(path: Path) -> pd.DataFrame:
    """Load a summary CSV or aggregate source-specific per-model RMSE CSVs."""
    if path.is_file():
        return standardize_force_rmse(pd.read_csv(path))
    if not path.is_dir():
        raise FileNotFoundError(f"Force-RMSE input not found: {path}")

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(path.rglob("rmse-results-all_*.csv")):
        frame = pd.read_csv(csv_path)
        if "force_rmse" not in frame.columns:
            continue
        frame = frame[["force_rmse"]].copy()
        frame["calculator"] = csv_path.stem.removeprefix("rmse-results-all_")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No rmse-results-all_*.csv files with force_rmse found under {path}"
        )
    return standardize_force_rmse(pd.concat(frames, ignore_index=True))


def standardize_score(
    frame: pd.DataFrame,
    *,
    value_candidates: list[str],
    output_column: str,
    description: str,
) -> pd.DataFrame:
    model_col = pick_first_column(
        frame,
        ["calculator", "Calculator", "model", "mlip_model"],
        f"{description} model column",
    )
    value_col = pick_first_column(frame, value_candidates, description)
    out = frame[[model_col, value_col]].rename(
        columns={model_col: "calculator", value_col: output_column}
    )
    out["calculator"] = out["calculator"].map(normalize_model_name)
    out[output_column] = pd.to_numeric(out[output_column], errors="coerce")
    return aggregate_models(out)


def read_csv(path: Path, description: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path}. Supply its command-line override."
        )
    return pd.read_csv(path)


def load_joined_data(args: argparse.Namespace) -> pd.DataFrame:
    default_force, default_rdf, default_vdos = source_paths(args.source)
    pressure_backend, pressure_mode = source_pressure_provenance(args.source)

    df_means = load_force_rmse(args.force_rmse_input or default_force)
    df_f1 = standardize_score(
        read_csv(args.f1_file, "F1 score CSV"),
        value_candidates=["f1_score", "f1", "F1"],
        output_column="f1_score",
        description="F1 score column",
    )
    df_ksrme = standardize_score(
        read_csv(args.ksrme_file, "k_SRME score CSV"),
        value_candidates=[
            "ksrme_score",
            "k_srme_score",
            "k_SRME_score",
            "k_SRME",
            "κ_SRME",
        ],
        output_column="ksrme_score",
        description="k_SRME score column",
    )
    df_rdf = standardize_rdf(
        read_csv(args.rdf_file or default_rdf, "RDF model-summary CSV")
    )
    df_pressure = standardize_pressure(
        read_csv(args.pressure_file, "Pressure metric CSV"),
        backend=args.pressure_backend or pressure_backend,
        mode=args.pressure_mode or pressure_mode,
    )
    df_vdos = standardize_vdos(
        read_csv(
            args.vdos_model_means_file or default_vdos,
            "VDOS model-summary CSV",
        )
    )

    joined = df_means.copy()
    joined = joined.merge(df_f1, on="calculator", how="left")
    joined = joined.merge(df_ksrme, on="calculator", how="left")
    joined = joined.merge(df_rdf, on="calculator", how="left")
    joined = joined.merge(df_pressure, on="calculator", how="left")
    joined = joined.merge(df_vdos, on="calculator", how="left")

    for col in [
        "force_rmse",
        "f1_score",
        "ksrme_score",
        "rdf_error_percent",
        "pressure_error_percent",
        "vdos_error_percent",
    ]:
        if col in joined.columns:
            joined[col] = pd.to_numeric(joined[col], errors="coerce")

    excluded_models_lower = {model.lower() for model in EXCLUDED_MODELS}
    joined = joined[
        ~joined["calculator"].str.lower().isin(excluded_models_lower)
    ].copy()
    if joined.empty:
        raise ValueError("No models left after exclusions.")

    empty_metrics = [
        col
        for col in (
            "force_rmse",
            "f1_score",
            "ksrme_score",
            "rdf_error_percent",
            "pressure_error_percent",
            "vdos_error_percent",
        )
        if col not in joined.columns or joined[col].notna().sum() == 0
    ]
    if empty_metrics:
        raise ValueError(
            "No model overlap produced finite values for: " + ", ".join(empty_metrics)
        )

    joined = joined.sort_values("calculator").reset_index(drop=True)
    return joined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create separate 1x3 correlation panels for RDF error, pressure histogram error, and VDOS error "
            "against [Force RMSE, F1 score, k_SRME score]."
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
        help=(
            "CSV with model-level F1 scores. The repository-local default can be "
            "overridden when these external benchmark scores are stored elsewhere."
        ),
    )
    parser.add_argument(
        "--ksrme-file",
        type=Path,
        default=DEFAULT_SCORE_DIR / "ksrme-scores.csv",
        help=(
            "CSV with model-level k_SRME scores. The repository-local default can be "
            "overridden when these external benchmark scores are stored elsewhere."
        ),
    )
    parser.add_argument(
        "--rdf-output-file",
        type=Path,
        default=SCRIPT_DIR / "plots" / "plot_rdf_correlations_1x3.pdf",
        help="Output path for the RDF correlation plot.",
    )
    parser.add_argument(
        "--pressure-output-file",
        type=Path,
        default=SCRIPT_DIR
        / "plots"
        / "plot_pressure_histogram_correlations_1x3_same_length.pdf",
        help="Output path for the pressure histogram correlation plot.",
    )
    parser.add_argument(
        "--vdos-output-file",
        type=Path,
        default=SCRIPT_DIR / "plots" / "plot_vdos_correlations_1x3_same_length.pdf",
        help="Output path for the VDOS correlation plot.",
    )
    return parser.parse_args()


def add_tier_legend(
    fig: plt.Figure, tier_colors: dict[str, tuple[float, float, float]]
) -> None:
    h1 = mlines.Line2D(
        [],
        [],
        color=tier_colors["tier_1"],
        marker="o",
        linestyle="None",
        markersize=8,
        label="Tier 1",
    )
    h2 = mlines.Line2D(
        [],
        [],
        color=tier_colors["tier_2"],
        marker="o",
        linestyle="None",
        markersize=8,
        label="Tier 2",
    )
    h3 = mlines.Line2D(
        [],
        [],
        color=tier_colors["tier_3"],
        marker="o",
        linestyle="None",
        markersize=8,
        label="Tier 3",
    )
    h4 = mlines.Line2D(
        [],
        [],
        color=tier_colors["tier_4"],
        marker="o",
        linestyle="None",
        markersize=8,
        label="Tier 4",
    )
    fig.legend(
        handles=[h1, h2, h3, h4],
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )


def plot_single_error_figure(
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    output_file: str | Path,
    tier_colors: dict[str, tuple[float, float, float]],
) -> None:
    x_metrics = [
        ("force_rmse", r"Force RMSE [eV/$\AA{}$]"),
        ("f1_score", "F1 score"),
        ("ksrme_score", r"$\kappa_{\mathrm{SRME}}$ score"),
    ]

    fig, axes_arr = plt.subplots(
        1,
        3,
        figsize=SPLIT_FIGSIZE,
        sharey=True,
    )
    axes = np.asarray(axes_arr)

    for col_idx, (x_col, x_label) in enumerate(x_metrics):
        ax = axes[col_idx]
        ax.tick_params(axis="y", labelleft=col_idx == 0)
        ax.set_ylabel(y_label if col_idx == 0 else "")
        ax.set_xlabel(x_label)
        ax.text(
            0.02,
            0.95,
            f"({string.ascii_lowercase[col_idx]})",
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

        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        colors = [
            model_color(model, tier_colors) for model in sub["calculator"].to_numpy()
        ]

        ax.scatter(x, y, c=colors, edgecolor="k", linewidth=0.4, s=40)
        ax.margins(x=0.08, y=0.12)

        label_texts = []
        for xi, yi, model in zip(x, y, sub["calculator"].to_numpy()):
            xytext, horizontal_alignment, vertical_alignment = inward_label_offset(
                ax, xi
            )
            txt = ax.annotate(
                display_name(str(model)),
                xy=(xi, yi),
                xytext=xytext,
                textcoords="offset points",
                fontsize=FONT_SIZE,
                alpha=0.8,
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
                "x": x,
                "y": y,
                "avoid_self": True,
                "only_move": {"points": "xy", "text": "xy"},
                "force_text": (1.2, 1.4),
                "force_points": (0.8, 1.0),
                "expand_points": (1.3, 1.4),
                "expand_text": (1.2, 1.3),
                "lim": 400,
                "arrowprops": {
                    "arrowstyle": "-",
                    "color": "0.5",
                    "lw": 0.4,
                    "alpha": 0.45,
                },
                "ensure_inside_axes": True,
                "expand_axes": False,
            }
            try:
                adjust_text(**adjust_kwargs)
            except TypeError:
                adjust_kwargs.pop("ensure_inside_axes")
                adjust_kwargs.pop("expand_axes")
                adjust_text(**adjust_kwargs)

        try:
            if np.nanstd(x) > 0.0:
                coef = np.polyfit(x, y, 1)
                x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                y_line = coef[0] * x_line + coef[1]
                ax.plot(x_line, y_line, color="gray", linestyle="--", linewidth=1)
        except Exception:
            pass

        try:
            if np.nanstd(x) > 0.0 and np.nanstd(y) > 0.0:
                r = np.corrcoef(x, y)[0, 1]
                ax.text(
                    0.02,
                    0.80,
                    f"r = {r:.2f}",
                    transform=ax.transAxes,
                    va="bottom",
                    fontsize=FONT_SIZE,
                )
        except Exception:
            pass

        ax.grid()

    add_tier_legend(fig, tier_colors)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keep_model_labels_inside_axes(fig, axes)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    keep_model_labels_inside_axes(fig, axes)
    fig.savefig(output_path)
    print(f"Saved {output_path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_joined_data(args)

    tier_colors = {
        "tier_1": palette[2],
        "tier_2": palette[1],
        "tier_3": palette[3],
        "tier_4": palette[0],
    }

    plot_single_error_figure(
        df=df,
        y_col="rdf_error_percent",
        y_label="RDF error [%]",
        output_file=args.rdf_output_file,
        tier_colors=tier_colors,
    )
    plot_single_error_figure(
        df=df,
        y_col="pressure_error_percent",
        y_label="Pressure histogram error [%]",
        output_file=args.pressure_output_file,
        tier_colors=tier_colors,
    )
    plot_single_error_figure(
        df=df,
        y_col="vdos_error_percent",
        y_label="VDOS error [%]",
        output_file=args.vdos_output_file,
        tier_colors=tier_colors,
    )


if __name__ == "__main__":
    main()
