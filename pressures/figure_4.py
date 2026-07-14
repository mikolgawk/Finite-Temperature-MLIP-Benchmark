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
CALCULATOR_DISPLAY_NAMES = {
    'chgnet': 'CHGNet',
    'mace-mp-0': 'MACE-MP-0',
    'grace-mp': 'GRACE-2L-MPtrj',
    'mace-mpa-0': 'MACE-MPA-0',
    'orb-v2': 'orb-v2',
    'eq-v2-m-omat': 'EquiformerV2',
    'mattersim-v1-5m': 'MatterSim-v1.0.0-5M',
    'grace-oam': 'GRACE-2L-OAM',
    'orb-v3': 'orb-v3-conservative-inf-mpa',
    'orb-v3-direct': 'orb-v3-direct-20-mpa',
    'nequip': 'NequIP-OAM-XL',
    'esen-30m-oam': 'eSEN-30M-OAM',
    'pet-oam-xl': 'PET-OAM-XL',
    'pet-omat-xl': 'PET-OMAT-XL',
    'mace-mh-omat': 'MACE-MH-1-OMAT',
    'uma-s-omat': 'UMA-S-P1',
    'uma-m-omat': 'UMA-M-P1',
}


def normalize_model_name(name: str) -> str:
    name = re.sub(r"_same-simulation-length$", "", str(name))
    return name.strip().lower()


def display_name(model: str) -> str:
    normalized = normalize_model_name(model)
    return CALCULATOR_DISPLAY_NAMES.get(normalized, model)

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
    # "Molecular crystals": ["anthracene_293K_Sharma_S", "naphthalene_295K_Sharma_S", "pentacene_295K_Sharma_S", "picene_295K_Sharma_S", "tetracene_295K_Sharma_S"],

    "Metal-water interfaces": ["Pt111w24H2O_380K_Heenen_VASP"],
    "Hydrogen": ["H_1050K_Rupp_QE"],

}

STRUCTURE_TO_TYPE = {
    structure: system_type
    for system_type, structure_list in SYSTEMS.items()
    for structure in structure_list
}

TIER_1 = ["chgnet", "mace-mp-0", "grace-mp"]
TIER_2 = ["mace-mpa-0", "orb-v2"]
TIER_3 = ["mattersim-v1-5M", "grace-oam", "orb-v3", "orb-v3-direct", "eSEN-30M-OAM", "nequip", "eq-v2-M-omat", "pet-oam-xl", "pet-omat-xl"]
TIER_4 = ["mace-mh-omat", "uma-s-omat", "uma-m-omat"]

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


def find_reference_file(pressures_dir: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        candidate = Path(explicit_path)
        if not candidate.is_file():
            raise FileNotFoundError(f"Reference per-frame CSV not found: {candidate}")
        return candidate

    local_candidate = Path(
        "../data/results/same-simulation-length/reference_pressure_per_frame_same_simulation_length.csv"
    )
    if local_candidate.is_file():
        return local_candidate

    fallback_candidate = Path(__file__).resolve().parent / "results-new" / "reference_pressure_per_frame.csv"
    if fallback_candidate.is_file():
        print(f"[INFO] Reference per-frame CSV not found in {pressures_dir}; using fallback {fallback_candidate}")
        return fallback_candidate

    raise FileNotFoundError(
        "Could not find reference per-frame pressure CSV. Looked for: "
        f"{local_candidate} and {fallback_candidate}"
    )


def pressure_column_name(columns: Iterable[str]) -> str:
    cols = set(columns)
    if "pressure_GPa" in cols:
        return "pressure_GPa"
    if "pressure_ref_GPa" in cols:
        return "pressure_ref_GPa"
    raise ValueError("No pressure column found. Expected one of: pressure_GPa, pressure_ref_GPa")


def load_pressure_per_frame_csv(csv_path: Path, deduplicate_reference: bool = False) -> pd.DataFrame:
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

    return df


def choose_best_worst_from_scores(
    models: list[str],
    model_scores: dict[str, float],
) -> tuple[str | None, str | None, dict[str, float]]:
    scores = {
        model: score
        for model, score in model_scores.items()
        if model in models and np.isfinite(score)
    }
    if not scores:
        return None, None, scores

    best = min(scores, key=scores.get)
    worst = max(scores, key=scores.get)
    return best, worst, scores


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
    return f"{label} ({mae_gpa:.2f} GPa)"


def format_mae_value(mae_gpa: float | None) -> str:
    if mae_gpa is None or not np.isfinite(mae_gpa):
        return "n/a"
    return f"{mae_gpa:.2f} GPa"


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


def annotate_median(ax, x_center: float, y_value: float, y_text: float, color, fmt: str = "{:.2f}") -> None:
    ax.annotate(
        fmt.format(y_value),
        xy=(x_center, y_value),
        xytext=(x_center, y_text),
        textcoords="data",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE,
        fontweight="bold",
        color=color,
        zorder=10,
        annotation_clip=False,
        arrowprops=dict(
            arrowstyle="-",
            color=color,
            linewidth=0.7,
            alpha=0.8,
            shrinkA=0,
            shrinkB=0,
        ),
    )


def tier_center(start_idx: float, end_idx: float) -> float:
    return (start_idx + end_idx) / 2


def tier_label_y(y_max: float) -> float:
    return y_max * 1.31


def median_value_label_y(y_max: float) -> float:
    return y_max * 1.08


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
    reference_file: Path,
    bins: int,
) -> list[tuple[str, str, np.ndarray, dict[str, np.ndarray], dict[str, float], np.ndarray]]:
    ref_df = load_pressure_per_frame_csv(reference_file, deduplicate_reference=True)
    if ref_df.empty:
        raise RuntimeError(f"No usable reference rows in {reference_file}")

    model_files = sorted(pressures_dir.glob(f"*{PER_FRAME_SUFFIX}"))
    model_files = [path for path in model_files if not path.name.startswith("reference_")]
    if not model_files:
        raise FileNotFoundError(f"No model per-frame files found in {pressures_dir}")

    model_data: dict[str, dict[str, np.ndarray]] = {}
    for model_file in model_files:
        model_name = parse_model_name(model_file)
        try:
            df_model = load_pressure_per_frame_csv(model_file, deduplicate_reference=False)
        except Exception as exc:
            print(f"[WARN] Skipping {model_file.name}: {exc}")
            continue

        if df_model.empty:
            continue

        normalized_name = normalize_model_name(model_name)
        values_by_structure: dict[str, np.ndarray] = {}
        for structure, structure_df in df_model.groupby("structure", sort=False):
            values = structure_df["pressure_GPa"].to_numpy(dtype=float)
            if values.size > 0:
                values_by_structure[str(structure)] = values

        if values_by_structure:
            model_data[normalized_name] = values_by_structure

    if not model_data:
        raise RuntimeError("No usable model per-frame data could be loaded.")

    available_panels: list[
        tuple[str, str, np.ndarray, dict[str, np.ndarray], dict[str, float], np.ndarray]
    ] = []
    skipped_types: list[str] = []

    for system_type in SYSTEMS:
        ref_sub = ref_df.loc[ref_df["system_type"] == system_type].copy()
        if ref_sub.empty:
            skipped_types.append(system_type)
            continue

        structures_present = set(ref_sub["structure"].astype(str).tolist())
        ordered_structures = [s for s in SYSTEMS[system_type] if s in structures_present]
        if not ordered_structures:
            skipped_types.append(system_type)
            continue

        ref_values_by_structure: dict[str, np.ndarray] = {
            structure: ref_sub.loc[
                ref_sub["structure"] == structure, "pressure_GPa"
            ].to_numpy(dtype=float)
            for structure in ordered_structures
        }

        structure_scores_by_model: dict[str, dict[str, float]] = {}
        structure_avg_mae: dict[str, float] = {}

        for structure in ordered_structures:
            ref_values = ref_values_by_structure[structure]
            if ref_values.size == 0:
                continue

            scores: dict[str, float] = {}
            for model_name, values_by_structure in model_data.items():
                mae_gpa = pressure_mean_absolute_error_gpa(
                    ref_values,
                    values_by_structure.get(structure),
                )
                if np.isfinite(mae_gpa):
                    scores[model_name] = mae_gpa

            if scores:
                structure_scores_by_model[structure] = scores
                structure_avg_mae[structure] = mean_finite(scores.values())

        if not structure_avg_mae:
            skipped_types.append(system_type)
            continue

        representative_system = max(structure_avg_mae, key=structure_avg_mae.get)
        ref_vals_plot = ref_values_by_structure[representative_system]
        if ref_vals_plot.size == 0:
            skipped_types.append(system_type)
            continue

        values_by_model_plot: dict[str, np.ndarray] = {}
        for model_name in TIER_ORDER:
            values_by_structure = model_data.get(model_name)
            if values_by_structure is None:
                continue

            rep_vals = values_by_structure.get(representative_system)

            if rep_vals is not None and rep_vals.size > 0:
                values_by_model_plot[model_name] = rep_vals

        model_scores = {
            model: score for model, score in structure_scores_by_model[representative_system].items()
            if model in values_by_model_plot
        }

        if not model_scores or not values_by_model_plot:
            skipped_types.append(system_type)
            continue

        print(
            "[INFO] Selected "
            f"{representative_system} for {system_type}: "
            f"mean model pressure MAE = {structure_avg_mae[representative_system]:.2f} GPa "
            f"across {len(structure_scores_by_model[representative_system])} models."
        )

        bin_edges_plot = make_bin_edges(ref_vals_plot, list(values_by_model_plot.values()), bins=bins)
        available_panels.append(
            (
                system_type,
                representative_system,
                ref_vals_plot,
                values_by_model_plot,
                model_scores,
                bin_edges_plot,
            )
        )

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
    df = df.dropna(subset=["pressure_mae_GPa"]).copy()

    if df.empty:
        raise ValueError("No valid model pressure MAE values found to plot.")

    df["model_key"] = df["model"].apply(lambda m: normalize_model_name(str(m)))
    # Keep only models explicitly defined in the tier lists.
    df = df[df["model_key"].isin(TIER_ORDER)].copy()
    if df.empty:
        raise ValueError("No tier-listed models with valid pressure MAE values found to plot.")

    df["tier_order"] = df["model_key"].apply(lambda model: TIER_ORDER.index(model))
    df = df.sort_values(["tier_order", "model_key"]).reset_index(drop=True)

    models = df["model_key"].to_list()
    x = np.arange(len(models))
    bar_colors = [get_tier_color(model) for model in models]

    ax.bar(
        x,
        df["pressure_mae_GPa"],
        width=0.65,
        color=bar_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.text(0.01, 0.95, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=FONT_SIZE)
    ax.set_xlabel("Model")
    ax.set_ylabel("Pressure MAE [GPa]")
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(model) for model in models], rotation=45, ha="right", fontsize=FONT_SIZE)
    ax.grid(axis="y")

    tier1_df = df[df["model_key"].isin(TIER_1)].copy()
    tier2_df = df[df["model_key"].isin(TIER_2)].copy()
    tier3_df = df[df["model_key"].isin(TIER_3)].copy()
    tier4_df = df[df["model_key"].isin(TIER_4)].copy()

    t1_count = len(tier1_df)
    t2_count = len(tier2_df)
    t3_count = len(tier3_df)
    t4_count = len(tier4_df)

    t1_med = tier1_df["pressure_mae_GPa"].median() if t1_count > 0 else np.nan
    t2_med = tier2_df["pressure_mae_GPa"].median() if t2_count > 0 else np.nan
    t3_med = tier3_df["pressure_mae_GPa"].median() if t3_count > 0 else np.nan
    t4_med = tier4_df["pressure_mae_GPa"].median() if t4_count > 0 else np.nan

    print(
        "Pressure MAE medians -> "
        f"Tier 1: {format_mae_value(t1_med)}, "
        f"Tier 2: {format_mae_value(t2_med)}, "
        f"Tier 3: {format_mae_value(t3_med)}, "
        f"Tier 4: {format_mae_value(t4_med)}"
    )

    finite_vals = df["pressure_mae_GPa"].to_numpy(dtype=float)
    med_vals = np.array([t1_med, t2_med, t3_med, t4_med], dtype=float)
    all_for_ylim = np.concatenate([
        finite_vals[np.isfinite(finite_vals)],
        med_vals[np.isfinite(med_vals)],
    ]) if (np.isfinite(finite_vals).any() or np.isfinite(med_vals).any()) else np.array([1.0])

    metric_ymax = float(np.nanmax(all_for_ylim))
    if not np.isfinite(metric_ymax) or metric_ymax <= 0:
        metric_ymax = 1.0
    ax.set_ylim(0, metric_ymax * 1.35)

    tier1_end = t1_count - 0.5
    tier2_end = t1_count + t2_count - 0.5
    tier3_end = t1_count + t2_count + t3_count - 0.5
    tier4_end = t1_count + t2_count + t3_count + t4_count - 0.5

    if t1_count > 0 and (t2_count > 0 or t3_count > 0 or t4_count > 0):
        ax.axvline(x=tier1_end, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    if t2_count > 0 and (t3_count > 0 or t4_count > 0):
        ax.axvline(x=tier2_end, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    if t3_count > 0 and t4_count > 0:
        ax.axvline(x=tier3_end, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    y_top = tier_label_y(metric_ymax)
    if t1_count > 0:
        ax.text((t1_count - 1) / 2, y_top, "Tier 1", ha="center", va="top", fontsize=FONT_SIZE, color=palette[2])
    if t2_count > 0:
        ax.text(t1_count + (t2_count - 1) / 2, y_top, "Tier 2", ha="center", va="top", fontsize=FONT_SIZE, color=palette[1])
    if t3_count > 0:
        ax.text(t1_count + t2_count + (t3_count - 1) / 2, y_top, "Tier 3", ha="center", va="top", fontsize=FONT_SIZE, color=palette[3])
    if t4_count > 0:
        ax.text(
            t1_count + t2_count + t3_count + (t4_count - 1) / 2,
            y_top,
            "Tier 4",
            ha="center",
            va="top",
            fontsize=FONT_SIZE,
            color=palette[0],
        )

    if np.isfinite(t1_med):
        ax.hlines(
            t1_med, xmin=-0.5, xmax=tier1_end,
            colors=palette[2], linestyles="--", linewidth=2, alpha=0.9
        )
    if np.isfinite(t2_med):
        ax.hlines(
            t2_med, xmin=tier1_end, xmax=tier2_end,
            colors=palette[1], linestyles="--", linewidth=2, alpha=0.9
        )
    if np.isfinite(t3_med):
        ax.hlines(
            t3_med, xmin=tier2_end, xmax=tier3_end,
            colors=palette[3], linestyles="--", linewidth=2, alpha=0.9
        )
    if np.isfinite(t4_med):
        ax.hlines(
            t4_med, xmin=tier3_end, xmax=tier4_end,
            colors=palette[0], linestyles="--", linewidth=2, alpha=0.9
        )

    median_label_y = median_value_label_y(metric_ymax)

    # Reduce number of y-axis ticks on overall pressure-MAE plot to avoid overlap
    try:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
    except Exception:
        pass

    if np.isfinite(t1_med):
        annotate_median(ax, tier_center(0, t1_count - 1), t1_med, median_label_y, palette[2], fmt="{:.2f}")
    if np.isfinite(t2_med):
        annotate_median(
            ax,
            tier_center(t1_count, t1_count + t2_count - 1),
            t2_med,
            median_label_y,
            palette[1],
            fmt="{:.2f}",
        )
    if np.isfinite(t3_med):
        annotate_median(
            ax,
            tier_center(t1_count + t2_count, t1_count + t2_count + t3_count - 1),
            t3_med,
            median_label_y,
            palette[3],
            fmt="{:.2f}",
        )
    if np.isfinite(t4_med):
        annotate_median(
            ax,
            tier_center(t1_count + t2_count + t3_count, len(models) - 1),
            t4_med,
            median_label_y,
            palette[0],
            fmt="{:.2f}",
        )


def plot_combined(
    pressures_dir: Path,
    reference_file: Path,
    ranking_df: pd.DataFrame,
    bins: int,
    output: str | Path,
) -> None:
    available_panels = collect_histogram_panels(
        pressures_dir=pressures_dir,
        reference_file=reference_file,
        bins=bins,
    )

    if len(available_panels) > 4:
        print(f"[INFO] Using first 4 histogram panels for 2x2 layout (out of {len(available_panels)} available).")
        available_panels = available_panels[:4]

    n_panels = len(available_panels)
    n_hist_cols = 2
    n_hist_rows = int(np.ceil(n_panels / n_hist_cols))

    fig = plt.figure(figsize=(3.53 * 3.0, 3.53 * 3.55))
    outer_gs = gridspec.GridSpec(
        nrows=n_hist_rows + 2,
        ncols=n_hist_cols,
        figure=fig,
        height_ratios=[0.78, 0.22] + [1.15] * n_hist_rows,
        hspace=0.48,
        wspace=0.30,
    )

    panel_labels = [f"({chr(97 + i)})" for i in range(n_panels + 1)]

    overall_ax = fig.add_subplot(outer_gs[0, :])
    draw_overall_pressure_mae_plot(overall_ax, ranking_df, panel_labels[0])

    for idx, (
        system_type,
        representative_system,
        ref_values,
        values_by_model,
        model_scores,
        bin_edges,
    ) in enumerate(available_panels):
        row = idx // n_hist_cols + 2
        col = idx % n_hist_cols
        hist_grid_row = idx // n_hist_cols
        is_bottom_hist_row = hist_grid_row == n_hist_rows - 1

        sub_gs = gridspec.GridSpecFromSubplotSpec(
            nrows=len(TIER_DEFS),
            ncols=1,
            subplot_spec=outer_gs[row, col],
            hspace=0.05,
        )

        is_left_col = (col == 0)
        is_right_col = (col == n_hist_cols - 1)

        panel_ax = fig.add_subplot(outer_gs[row, col], frame_on=False)
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
            ax = fig.add_subplot(sub_gs[tier_idx])

            if tier_idx == 0:
                ax.set_title(f"{system_type}\n{format_system_name(representative_system)}")
                ax.text(
                    0.02,
                    0.95,
                    panel_labels[idx + 1],
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=FONT_SIZE,
                )

            best_model, worst_model, scores = choose_best_worst_from_scores(tier_models, model_scores)
            tier_mean_error = mean_finite(scores.values())

            ax.hist(
                ref_values,
                bins=bin_edges,
                density=True,
                histtype="step",
                color="black",
                linewidth=1.5,
                linestyle="-",
                label="Reference",
            )

            if best_model is not None and best_model in values_by_model:
                best_prefix = "Best/worst" if worst_model == best_model else "Best"
                ax.hist(
                    values_by_model[best_model],
                    bins=bin_edges,
                    density=True,
                    histtype="step",
                    color=tier_color,
                    linewidth=1.5,
                    linestyle="-",
                    label=format_model_label(best_model, scores.get(best_model), prefix=best_prefix),
                )

            if worst_model is not None and worst_model != best_model and worst_model in values_by_model:
                ax.hist(
                    values_by_model[worst_model],
                    bins=bin_edges,
                    density=True,
                    histtype="step",
                    color=tier_color,
                    linewidth=1.5,
                    linestyle="--",
                    label=format_model_label(worst_model, scores.get(worst_model), prefix="Worst"),
                )

            if is_right_col:
                ax.set_ylabel(tier_label, labelpad=2)
                ax.yaxis.set_label_position("right")
            else:
                ax.set_ylabel("")

            ax.set_xlabel("")
            if tier_idx != len(TIER_DEFS) - 1:
                ax.tick_params(labelbottom=False)
            
            # ax.set_ylim()
            

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                handles = handles + [Line2D([], [], color="none", linestyle="none", linewidth=0)]
                labels = labels + [
                    f"{tier_label} mean MAE: {format_mae_value(tier_mean_error)}"
                ]
                ax.legend(
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

            ax.grid(True)
            # Limit number of major y ticks to avoid overlapping numeric labels
            try:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=1, prune='both'))
            except Exception:
                pass
            ax.tick_params(axis="y", labelsize=FONT_SIZE, pad=1)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.show()
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
    args = parser.parse_args()

    pressures_dir = Path(args.pressures_dir)
    if not pressures_dir.is_dir():
        raise NotADirectoryError(f"Pressures directory not found: {pressures_dir}")

    reference_file = find_reference_file(pressures_dir, args.reference_file)

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
    )


if __name__ == "__main__":
    main()
