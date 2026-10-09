#!/usr/bin/env python3
"""Combined pressure figure for same-simulation-length pressure-MAE views.

The top row adds the global pressure MAE bar plot with tier medians.
The lower panels reproduce the per-system-type stacked pressure histograms,
selecting best/worst models by pressure MAE.

Requires: numpy, pandas, matplotlib, seaborn
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns

from get_model_pressure_errors import (
    filter_completed_pressure_md_rows,
    pressure_histogram_similarity,
    pressure_source,
    resolve_model_reference_pressure_file,
)
from pressure_axis_breaks import (
    distribution_axis_windows,
    histogram_segments,
    mae_axis_windows,
    pressure_axes,
    pressure_tick,
)

import sys

PAPER_V2_CONFIG_DIR = Path(__file__).resolve().parents[1]
if str(PAPER_V2_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_V2_CONFIG_DIR))
from model_display_names import MODEL_DISPLAY_NAMES, display_model_name
from system_filters import add_molecular_crystal_option, filter_pressure_systems
from subplot_filters import display_error_percent


FONT_SIZE = 10
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
    name = re.sub(r"_same-simulation-length$", "", str(name))
    return name.strip().lower()


def display_name(model: str) -> str:
    return display_model_name(model)

SYSTEMS = {
    "Pure metals": ["bulkAu_1500K_Kapil", "bulkAg_600K_Kapil", "bulkCu_1000K_Kapil"],
    "Perovskites": ["CsSnI3_500K_Ivor_VASP", "MAPbBr3_300K_Ivor_VASP"],
    "Metal dichalcogenides": [
        "bulkMoS2_300K_NO-VdW_J.Kioseoglou_VASP",
        "TiSe2_400K_Ivor_VASP",
    ],
    "Metal alloys": [
        "bulkLiMgAlZnSn_900K_J_Schmidt_VASP",
        "bulkPt3Co_300K_J.Kioseoglou_VASP",
        "bulkCuAu_500K-Artrith_VASP",
        "bulkCuZrAl_1500K_A.Wadowski-J.Schmidt_VASP",
        "bulkLiMgAlZnSn_600K_J_Schmidt_VASP",
    ],
    "Molecular crystals": ["anthracene_293K_Sharma_S", "naphthalene_295K_Sharma_S", "pentacene_295K_Sharma_S", "picene_295K_Sharma_S", "tetracene_295K_Sharma_S"],

    "Hydrogen": ["H_1050K_Rupp_QE"],

}

STRUCTURE_TO_TYPE = {
    structure: system_type
    for system_type, structure_list in SYSTEMS.items()
    for structure in structure_list
}

TIER_1 = ["chgnet", "mace-mp-0", "mace-mp-0-compile", "grace-mp"]
TIER_2 = ["mace-mpa-0", "mace-mpa-0-compile", "orb-v2"]
TIER_3 = ["mattersim-v1-5m", "grace-oam", "orb-v3-omat", "orb-v3-direct-omat", "esen-30m-oam", "nequip", "eq-v2-m-omat", "pet-oam-xl", "pet-omat-xl", "grace-oam-compiled", "mattersim-v1-5m-compile", "pet-oam-xl-torchscript", "pet-omat-xl-torchscript"]
TIER_4 = ["mace-mh-omat", "mace-mh-omat-compile", "uma-s-omat", "uma-s-omat-compile", "uma-s-omat-turbo", "uma-m-omat", "uma-m-omat-compile", "uma-m-omat-turbo"]

TIER_DEFS = [
    ("Tier 1", TIER_1, palette[2]),
    ("Tier 2", TIER_2, palette[1]),
    ("Tier 3", TIER_3, palette[3]),
    ("Tier 4", TIER_4, palette[0]),
]
TIER_ORDER = TIER_1 + TIER_2 + TIER_3 + TIER_4
PER_FRAME_SUFFIX = "_pressure_per_frame.csv"


def parse_model_name(file_path: Path) -> str:
    name = file_path.name.removesuffix(PER_FRAME_SUFFIX)
    return normalize_model_name(name)


def structure_from_trajectory_file(path_like: str) -> str:
    return Path(str(path_like)).parent.name


def legacy_reference_files() -> tuple[Path, ...]:
    return (
        Path("../data/results/same-simulation-length/reference_pressure_per_frame_same_simulation_length.csv"),
        Path(__file__).resolve().parent / "results-new" / "reference_pressure_per_frame.csv",
    )


def pressure_column_name(columns: Iterable[str]) -> str:
    cols = set(columns)
    if "pressure_GPa" in cols:
        return "pressure_GPa"
    if "pressure_ref_GPa" in cols:
        return "pressure_ref_GPa"
    raise ValueError("No pressure column found. Expected one of: pressure_GPa, pressure_ref_GPa")


def load_pressure_per_frame_csv(csv_path: Path, deduplicate_reference: bool = False, include_molecular_crystals: bool = False) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    pcol = pressure_column_name(header.columns)

    usecols = ["trajectory_file", pcol]
    if "frame_index" in set(header.columns):
        usecols.append("frame_index")

    df = pd.read_csv(csv_path, usecols=usecols)
    df = df.rename(columns={pcol: "pressure_GPa"})
    df["pressure_GPa"] = pd.to_numeric(df["pressure_GPa"], errors="coerce")
    df = df.dropna(subset=["trajectory_file", "pressure_GPa"]).copy()

    df["structure"] = df["trajectory_file"].apply(structure_from_trajectory_file)
    df["system_type"] = df["structure"].map(STRUCTURE_TO_TYPE)
    df = df.dropna(subset=["system_type"]).copy()

    if deduplicate_reference:
        if "frame_index" in df.columns:
            df = df.drop_duplicates(subset=["trajectory_file", "frame_index"], keep="first")
        else:
            df = df.drop_duplicates(subset=["trajectory_file", "pressure_GPa"], keep="first")

    return filter_pressure_systems(df, include_molecular_crystals)


def choose_best_worst_from_scores(
    models: list[str],
    model_scores: dict[str, float],
    visible_models: set[str] | None = None,
) -> tuple[str | None, str | None, dict[str, float]]:
    scores = {
        model: score
        for model, score in model_scores.items()
        if model in models and np.isfinite(score)
    }
    visible_scores = {
        model: score for model, score in scores.items()
        if visible_models is None or model in visible_models
    }
    if not visible_scores:
        return None, None, scores

    best = min(visible_scores, key=visible_scores.get)
    worst = max(visible_scores, key=visible_scores.get)
    return best, worst, scores


def visible_pressure_models(
    references_by_model: dict[str, np.ndarray],
    values_by_model: dict[str, np.ndarray],
    bins: int,
) -> set[str]:
    """Use histogram percentages to filter curves in the pressure-MAE panel."""
    visible_models = set()
    for model, values in values_by_model.items():
        reference_values = references_by_model.get(model)
        if reference_values is None:
            continue
        score = pressure_histogram_similarity(reference_values, values, bins)
        if score and display_error_percent(score["pressure_error_percent"]):
            visible_models.add(model)
    return visible_models


def make_bin_edges(reference_values: np.ndarray, candidate_arrays: list[np.ndarray], bins: int) -> np.ndarray:
    arrays = [reference_values] + [arr for arr in candidate_arrays if arr.size > 0]
    lo = min(float(np.min(arr)) for arr in arrays)
    hi = max(float(np.max(arr)) for arr in arrays)

    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = -1.0, 1.0
    if hi <= lo:
        hi = lo + 1e-6

    return np.linspace(lo, hi, bins + 1)


def format_model_label(model_name: str, mae_gpa: float | None, prefix: str | None = None) -> str:
    model_key = normalize_model_name(model_name)
    label = display_name(model_key)
    if prefix:
        label = f"{prefix}: {label}"
    if mae_gpa is None or not np.isfinite(mae_gpa):
        return label
    return f"{label} ({format_mae_value(mae_gpa)})"


def format_mae_value(mae_gpa: float | None) -> str:
    if mae_gpa is None or not np.isfinite(mae_gpa):
        return "n/a"
    value = f"{mae_gpa:.2e}" if abs(mae_gpa) >= 1e4 else f"{mae_gpa:.2f}"
    return f"{value} GPa"


def format_chemical_formula(formula: str) -> str:
    return re.sub(r"(\d+)", r"$_{\1}$", formula)


def format_system_name(system_name: str) -> str:
    tokens = system_name.split("_")
    temp_value = None
    temp_idx = None

    for idx, token in enumerate(tokens):
        match = re.search(r"(\d+)K", token)
        if match:
            temp_value = match.group(1)
            temp_idx = idx
            break

    if temp_idx is None:
        base_name = system_name
    else:
        base_name = "_".join(tokens[:temp_idx])

    if base_name.startswith("bulk") and len(base_name) > 4:
        base_name = f"bulk {base_name[4:]}"
    base_name = format_chemical_formula(base_name)

    no_vdw = any("NO-VdW" in token for token in tokens)

    if temp_value is not None:
        suffix = f" ({temp_value} K"
        if no_vdw:
            # suffix += ", no vdW"
            suffix += ""
        suffix += ")"
        return f"{base_name}{suffix}"

    if no_vdw:
        # return f"{base_name} (no vdW)"
        return base_name
    return base_name


def get_tier_color(model: str):
    if model in TIER_1:
        return palette[2]
    if model in TIER_2:
        return palette[1]
    if model in TIER_3:
        return palette[3]
    if model in TIER_4:
        return palette[0]
    return "#757575"


def get_overall_pressure_mae_per_model(df: pd.DataFrame) -> pd.Series:
    for col in ["error_GPa", "pressure_mae_GPa", "Pressure MAE [GPa]"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")

    mae_suffix = "_error_GPa"
    class_mae_cols = [
        col for col in df.columns if col.endswith(mae_suffix) and col != "error_GPa"
    ]
    if class_mae_cols:
        class_errors = df[class_mae_cols].apply(pd.to_numeric, errors="coerce")
        return class_errors.mean(axis=1, skipna=True)

    if any(col in df.columns for col in ["final_mean_pressure_error_percent", "pressure_error_percent"]):
        raise ValueError(
            "Ranking CSV contains pressure histogram error percent, not pressure MAE in GPa. "
            "Use model_mean_pressure_comparison.csv from "
            "compute-model-pressure-means-same-simulation-length.py."
        )

    raise ValueError(
        "Input CSV must contain pressure MAE columns, e.g. "
        "'error_GPa', 'pressure_mae_GPa', or '*_error_GPa'."
    )


def pressure_mean_absolute_error_gpa(
    reference_values: np.ndarray,
    model_values: np.ndarray | None,
) -> float:
    if model_values is None or reference_values.size == 0 or model_values.size == 0:
        return float("nan")
    ref_mean = float(np.mean(reference_values))
    model_mean = float(np.mean(model_values))
    if not np.isfinite(ref_mean) or not np.isfinite(model_mean):
        return float("nan")
    return abs(model_mean - ref_mean)


def mean_finite(values: Iterable[float | None]) -> float:
    finite_values = [
        float(value)
        for value in values
        if value is not None and np.isfinite(value)
    ]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def collect_histogram_panels(
    pressures_dir: Path,
    reference_file: Path | None,
    bins: int,
    include_molecular_crystals: bool = False,
) -> list[tuple[str, str, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float], np.ndarray]]:
    model_files = sorted(pressures_dir.glob(f"*{PER_FRAME_SUFFIX}"))
    model_files = [path for path in model_files if not path.name.startswith("reference_")]
    if not model_files:
        raise FileNotFoundError(f"No model per-frame files found in {pressures_dir}")

    model_data: dict[str, dict[str, np.ndarray]] = {}
    reference_data: dict[str, dict[str, np.ndarray]] = {}
    for model_file in model_files:
        model_name = parse_model_name(model_file)
        try:
            matched_reference = resolve_model_reference_pressure_file(
                pressures_dir, model_file, reference_file, legacy_reference_files()
            )
            df_model = load_pressure_per_frame_csv(model_file, include_molecular_crystals=include_molecular_crystals)
            df_model = filter_completed_pressure_md_rows(
                df_model, model_name, pressure_source(model_file)
            )
            df_ref = load_pressure_per_frame_csv(matched_reference, deduplicate_reference=True, include_molecular_crystals=include_molecular_crystals)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"[WARN] Skipping {model_file.name}: {exc}")
            continue
        if df_model.empty or df_ref.empty:
            continue

        def by_structure(df: pd.DataFrame) -> dict[str, np.ndarray]:
            return {
                str(structure): group["pressure_GPa"].to_numpy(dtype=float)
                for structure, group in df.groupby("structure", sort=False)
                if not group.empty
            }

        model_data[model_name] = by_structure(df_model)
        reference_data[model_name] = by_structure(df_ref)

    if not model_data:
        raise RuntimeError("No usable model/reference per-frame pairs could be loaded.")

    available_panels = []
    skipped_types: list[str] = []
    for system_type, ordered_structures in SYSTEMS.items():
        structure_scores_by_model: dict[str, dict[str, float]] = {}
        structure_avg_mae: dict[str, float] = {}
        for structure in ordered_structures:
            scores = {}
            for model_name, values_by_structure in model_data.items():
                ref_values = reference_data[model_name].get(structure)
                if ref_values is None:
                    continue
                score = pressure_mean_absolute_error_gpa(
                    ref_values, values_by_structure.get(structure)
                )
                if np.isfinite(score):
                    scores[model_name] = score
            if scores:
                structure_scores_by_model[structure] = scores
                structure_avg_mae[structure] = mean_finite(scores.values())

        if not structure_avg_mae:
            skipped_types.append(system_type)
            continue
        representative_system = max(structure_avg_mae, key=structure_avg_mae.get)
        model_scores = structure_scores_by_model[representative_system]
        values_by_model = {
            model: model_data[model][representative_system]
            for model in model_scores if model in TIER_ORDER
        }
        references_by_model = {
            model: reference_data[model][representative_system]
            for model in values_by_model
        }
        if not values_by_model:
            skipped_types.append(system_type)
            continue
        edges = make_bin_edges(
            next(iter(references_by_model.values())),
            list(references_by_model.values()) + list(values_by_model.values()),
            bins=bins,
        )
        print(
            f"[INFO] Selected {representative_system} for {system_type}: "
            f"mean model pressure MAE = {structure_avg_mae[representative_system]:.2f} GPa "
            f"across {len(model_scores)} models."
        )
        available_panels.append((
            system_type, representative_system, references_by_model,
            values_by_model, model_scores, edges,
        ))

    if skipped_types:
        print(f"[INFO] Skipping system types without enough data: {', '.join(skipped_types)}")
    if not available_panels:
        raise RuntimeError("No system types have both reference and model pressure data for plotting.")
    return available_panels


def draw_overall_pressure_mae_plot(ax, ranking_df: pd.DataFrame, panel_label: str) -> None:
    if "model" not in ranking_df.columns:
        raise ValueError("Missing required column in input CSV: 'model'")
    df = ranking_df.copy()
    df["pressure_mae_GPa"] = get_overall_pressure_mae_per_model(df)
    df["model_key"] = df["model"].apply(lambda model: normalize_model_name(str(model)))
    df = df.dropna(subset=["pressure_mae_GPa"])
    df = df[df["model_key"].isin(TIER_ORDER)].copy()
    if df.empty:
        raise ValueError("No tier-listed models with valid pressure MAE values found to plot.")
    df["tier_order"] = df["model_key"].map(TIER_ORDER.index)
    df = df.sort_values(["tier_order", "model_key"]).reset_index(drop=True)
    models = df["model_key"].to_list()
    values = df["pressure_mae_GPa"].to_numpy(dtype=float)
    tiers = []
    start = 0
    for label, tier_models, color in TIER_DEFS:
        tier_values = df.loc[df["model_key"].isin(tier_models), "pressure_mae_GPa"]
        count = len(tier_values)
        if count:
            tiers.append((label, start, start + count, float(tier_values.median()), color))
            start += count
    windows = mae_axis_windows(np.r_[values, [tier[3] for tier in tiers]])
    if len(windows) > 1:
        fig, spec = ax.figure, ax.get_subplotspec()
        ax.remove()
        axes = pressure_axes(fig, spec, windows, orientation="y")
    else:
        axes = [ax]
        ax.set_ylim(windows[0])
    x = np.arange(len(models))
    for segment_index, segment_ax in enumerate(axes):
        segment_ax.bar(x, values, width=0.65, color=[get_tier_color(m) for m in models],
                       alpha=0.8, edgecolor="black", linewidth=0.5)
        segment_ax.set_xlim(-0.6, len(models) - 0.4)
        segment_ax.grid(axis="y")
        segment_ax.grid(axis="x", visible=False)
        segment_ax.yaxis.set_major_locator(MaxNLocator(
            nbins=3 if segment_index == 0 else 2,
            min_n_ticks=2, prune="upper" if segment_index == 0 else "both",
        ))
        for _, start, end, median, color in tiers:
            if end < len(models):
                segment_ax.axvline(end - 0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
            segment_ax.hlines(median, start - 0.5, end - 0.5,
                             colors=color, linestyles="--", linewidth=1.5, alpha=0.9)
    bottom, top = axes[0], axes[-1]
    bottom.set_xlabel("Model")
    bottom.set_ylabel("Pressure MAE [GPa]")
    bottom.set_xticks(x)
    bottom.set_xticklabels([display_name(model) for model in models], rotation=45,
                           ha="right", fontsize=FONT_SIZE)
    top.text(-0.035, 1.05, panel_label, transform=top.transAxes,
             ha="left", va="bottom", fontsize=FONT_SIZE)
    for label, start, end, median, color in tiers:
        top.text((start + end - 1) / 2, 1.06,
                 f"{label}\nMedian: {pressure_tick(median)}",
                 transform=top.get_xaxis_transform(), ha="center", va="bottom",
                 fontsize=FONT_SIZE - 2, color=color)
    print("Pressure MAE medians -> " + ", ".join(
        f"{label}: {format_mae_value(median)}" for label, _, _, median, _ in tiers
    ))


def plot_combined(
    pressures_dir: Path,
    reference_file: Path | None,
    ranking_df: pd.DataFrame,
    bins: int,
    output: str | Path,
    include_molecular_crystals: bool = False,
) -> None:
    available_panels = collect_histogram_panels(
        pressures_dir=pressures_dir,
        reference_file=reference_file,
        bins=bins,
        include_molecular_crystals=include_molecular_crystals,
    )

    n_panels = len(available_panels)
    n_hist_cols = 3
    n_hist_rows = int(np.ceil(n_panels / n_hist_cols))

    fig = plt.figure(figsize=(3.53 * 3.0, 3.53 * (1.40 + 1.15 * n_hist_rows)))
    outer_gs = gridspec.GridSpec(
        nrows=n_hist_rows + 2,
        ncols=n_hist_cols * 2,
        figure=fig,
        height_ratios=[1.05, 0.35] + [1.15] * n_hist_rows,
        hspace=0.48,
        wspace=0.30,
    )

    panel_labels = [f"({chr(97 + i)})" for i in range(n_panels + 1)]

    overall_ax = fig.add_subplot(outer_gs[0, :])
    draw_overall_pressure_mae_plot(overall_ax, ranking_df, panel_labels[0])

    for idx, (
        system_type,
        representative_system,
        references_by_model,
        values_by_model,
        model_scores,
        _bin_edges,
    ) in enumerate(available_panels):
        visible_models = visible_pressure_models(references_by_model, values_by_model, bins)
        references = list(references_by_model.values())
        plot_arrays = references + [values_by_model[model] for model in visible_models]
        windows = distribution_axis_windows(plot_arrays, references)
        reference_center = float(np.median(references[0]))
        core_index = next(i for i, (lo, hi) in enumerate(windows)
                          if lo <= reference_center <= hi)
        row = idx // n_hist_cols + 2
        col = idx % n_hist_cols
        hist_grid_row = idx // n_hist_cols
        is_bottom_hist_row = hist_grid_row == n_hist_rows - 1
        panels_in_row = min(n_hist_cols, n_panels - hist_grid_row * n_hist_cols)
        start_col = n_hist_cols - panels_in_row + col * 2
        panel_spec = outer_gs[row, start_col:start_col + 2]

        sub_gs = gridspec.GridSpecFromSubplotSpec(
            nrows=len(TIER_DEFS),
            ncols=1,
            subplot_spec=panel_spec,
            hspace=0.05,
        )

        is_left_col = (col == 0)
        is_right_col = (col == panels_in_row - 1)

        panel_ax = fig.add_subplot(panel_spec, frame_on=False)
        panel_ax.tick_params(
            labelcolor="none",
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        panel_ax.grid(False)
        if is_left_col:
            panel_ax.set_ylabel("Density", labelpad=8)
        else:
            panel_ax.set_ylabel("")
        if is_bottom_hist_row:
            panel_ax.set_xlabel("Pressure [GPa]", labelpad=18)
        else:
            panel_ax.set_xlabel("")

        for tier_idx, (tier_label, tier_models, tier_color) in enumerate(TIER_DEFS):
            axes = pressure_axes(fig, sub_gs[tier_idx], windows,
                                 orientation="x", core_index=core_index)
            ax = axes[core_index]

            def draw_histogram(values, **style):
                for segment_ax, (density, edges) in zip(
                    axes, histogram_segments(values, windows, bins)
                ):
                    segment_ax.stairs(density, edges, linewidth=1.5, **style)

            if tier_idx == 0:
                ax.set_title(f"{system_type}\n{format_system_name(representative_system)}")
                axes[0].text(
                    0.02,
                    0.95,
                    panel_labels[idx + 1],
                    transform=axes[0].transAxes,
                    ha="left",
                    va="top",
                    fontsize=FONT_SIZE,
                )

            best_model, worst_model, scores = choose_best_worst_from_scores(
                tier_models, model_scores, visible_models=visible_models
            )
            tier_mean_error = mean_finite(scores.values())

            reference_choices = (
                (best_model, "Reference (best)", "-"),
                (worst_model, "Reference (worst)", "--"),
            )
            if best_model is None:
                reference_model = next(
                    (model for model in tier_models if model in references_by_model),
                    next(iter(references_by_model), None),
                )
                reference_choices = ((reference_model, "Reference", "-"),)
            for model, label, style in reference_choices:
                if model is None or model not in references_by_model:
                    continue
                if model == best_model and style == "--":
                    continue
                draw_histogram(
                    references_by_model[model], color="black",
                    linestyle=style, label=label if best_model != worst_model else "Reference",
                )

            if best_model is not None and best_model in values_by_model:
                best_prefix = "Best/worst" if worst_model == best_model else "Best"
                draw_histogram(
                    values_by_model[best_model],
                    color=tier_color,
                    linestyle="-",
                    label=format_model_label(best_model, scores.get(best_model), prefix=best_prefix),
                )

            if worst_model is not None and worst_model != best_model and worst_model in values_by_model:
                draw_histogram(
                    values_by_model[worst_model],
                    color=tier_color,
                    linestyle="--",
                    label=format_model_label(worst_model, scores.get(worst_model), prefix="Worst"),
                )

            if is_right_col:
                axes[-1].set_ylabel(tier_label, labelpad=2)
                axes[-1].yaxis.set_label_position("right")
            else:
                ax.set_ylabel("")

            ax.set_xlabel("")
            for segment_ax in axes:
                segment_ax.tick_params(axis="x", labelsize=FONT_SIZE - 2)
                if tier_idx != len(TIER_DEFS) - 1:
                    segment_ax.tick_params(labelbottom=False)
                    segment_ax.xaxis.get_offset_text().set_visible(False)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                handles = handles + [Line2D([], [], color="none", linestyle="none", linewidth=0)]
                labels = labels + [
                    f"{tier_label} mean MAE: {format_mae_value(tier_mean_error)}"
                ]
                # Span the complete tier row, even when its x axis is broken.
                axes[-1].legend(
                    handles,
                    labels,
                    loc="upper right",
                    fontsize=LEGEND_FONT_SIZE,
                    handlelength=1.0,
                    handletextpad=0.3,
                    borderpad=0.2,
                    labelspacing=0.2,
                    borderaxespad=0.2,
                )

            for segment_ax in axes:
                segment_ax.grid(True)
            # Limit number of major y ticks to avoid overlapping numeric labels
            try:
                axes[0].yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=1, prune='both'))
            except Exception:
                pass
            axes[0].tick_params(axis="y", labelsize=FONT_SIZE, pad=1)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved combined pressure panel plot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a combined pressure-MAE figure from same-simulation-length views."
    )
    parser.add_argument(
        "--pressures-dir",
        default="../data/results/same-simulation-length",
        help="Directory with per-frame pressure CSV files.",
    )
    parser.add_argument(
        "--reference-file",
        default=None,
        help="Optional explicit path to reference per-frame pressure CSV.",
    )
    parser.add_argument(
        "--ranking-file",
        default="../data/results/same-simulation-length/model_mean_pressure_comparison.csv",
        help="CSV used for tier ranking and overall pressure MAE summary.",
    )
    parser.add_argument("--bins", type=int, default=80, help="Number of histogram bins.")
    parser.add_argument(
        "--output-file",
        default="plots/plot_pressure_panel_combined_pressure_mae.pdf",
        help="Output plot file path.",
    )
    add_molecular_crystal_option(parser)
    args = parser.parse_args()

    pressures_dir = Path(args.pressures_dir)
    if not pressures_dir.is_dir():
        raise NotADirectoryError(f"Pressures directory not found: {pressures_dir}")

    reference_file = Path(args.reference_file) if args.reference_file else None

    ranking_file = Path(args.ranking_file)
    if not ranking_file.is_file():
        raise FileNotFoundError(f"Ranking CSV not found: {ranking_file}")

    ranking_df = pd.read_csv(ranking_file)
    if "model" not in ranking_df.columns:
        raise ValueError(f"Ranking CSV missing 'model' column: {ranking_file}")

    plot_combined(
        pressures_dir=pressures_dir,
        reference_file=reference_file,
        ranking_df=ranking_df,
        bins=args.bins,
        output=args.output_file,
        include_molecular_crystals=args.include_molecular_crystals,
    )


if __name__ == "__main__":
    main()
