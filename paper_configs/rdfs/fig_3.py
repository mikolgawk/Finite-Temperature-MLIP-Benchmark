#!/usr/bin/env python3
"""Combined figure for plot-2 and plot-3 same-simulation-length views.

The top two rows reproduce the per-system-type stacked RDF comparisons from
plot-2-same-simulation-length.py. The bottom row adds the global RDF-error bar
plot with tier medians from plot-3-same-simulation-length.py.

Requires: numpy, pandas, matplotlib, seaborn
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
LEGACY_DATA_ROOT = Path.home() / "Finite-Temperature-MLIP-Benchmarks" / "data"


def first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def first_existing_matching_dir(pattern: str, *candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob(pattern)):
            return candidate
    return candidates[0]

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
    'mattersim-v1-1m': 'MatterSim-v1.0.0-1M',
    'grace-oam': 'GRACE-2L-OAM',
    'orb-v3': 'orb-v3-conservative-inf-mpa',

    'nequip': 'NequIP-OAM-XL',
    'esen-30m-oam': 'eSEN-30M-OAM',
    'pet-oam-xl': 'PET-OAM-XL',
    'mace-mh-omat': 'MACE-MH-1-OMAT',
    'uma-s-omat': 'UMA-S-P1',
    'uma-m-omat': 'UMA-M-P1',
}


def normalize_model_name(name: str) -> str:
    return str(name).strip().lower()


def display_name(model: str) -> str:
    normalized = normalize_model_name(model)
    return CALCULATOR_DISPLAY_NAMES.get(normalized, model)
EXCLUDED_MODELS = {"pet-mad"}

SYSTEMS = {
    "Pure metals": ["bulkAu_1500K_Kapil", "bulkAg_600K_Kapil", "bulkCu_1000K_Kapil"],
    "Perovskites": ["CsSnI3_500K_Ivor_VASP", "MAPbBr3_300K_Ivor_VASP"],
    "Metal dichalcogenides": ["bulkMoS2_300K_NO-VdW_J.Kioseoglou_VASP", "TiSe2_400K_Ivor_VASP"],
    "Metal alloys": ["bulkCuAu_500K-Artrith_VASP", "bulkCuZrAl_1500K_A.Wadowski-J.Schmidt_VASP", "bulkLiMgAlZnSn_600K_J_Schmidt_VASP", 
                     "bulkLiMgAlZnSn_900K_J_Schmidt_VASP", "bulkPt3Co_300K_J.Kioseoglou_VASP"],
    "Molecular crystals": [
        "picene_295K_Sharma_S",
        "tetracene_295K_Sharma_S",
        "anthracene_293K_Sharma_S",
        "naphthalene_295K_Sharma_S",
        "pentacene_295K_Sharma_S",
        
        
    ],
}

TIER_1 = ["chgnet", "mace-mp-0", "grace-mp"]
TIER_2 = ["mace-mpa-0", "orb-v2"]
TIER_3 = ["mattersim-v1-1M", "grace-oam", "orb-v3", "eSEN-30M-OAM", "nequip", "eq-v2-M-omat", "pet-oam-xl"]
TIER_4 = ["mace-mh-omat", "uma-s-omat", "uma-m-omat"]

TIER_1_NORM = [normalize_model_name(model) for model in TIER_1]
TIER_2_NORM = [normalize_model_name(model) for model in TIER_2]
TIER_3_NORM = [normalize_model_name(model) for model in TIER_3]
TIER_4_NORM = [normalize_model_name(model) for model in TIER_4]

TIER_DEFS = [
    ("Tier 1", TIER_1, palette[2]),
    ("Tier 2", TIER_2, palette[1]),
    ("Tier 3", TIER_3, palette[3]),
    ("Tier 4", TIER_4, palette[0]),
]

def annotate_median(ax, x_center, y_value, y_text, fmt="{:.2f}%", color="black"):
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


def tier_center(start_idx, end_idx):
    return (start_idx + end_idx) / 2


def tier_label_y(y_max):
    return y_max * 1.15


def median_value_label_y(y_max):
    return y_max * 1.02



DEFAULT_RDF_CSV_DIR = "/home/mjgawkowski/phd_mlip_matbench_benchmarks/scripts/rdf-scripts-final-copy/results"
DEFAULT_RDF_MEANS_FILE = "/home/mjgawkowski/phd_mlip_matbench_benchmarks/scripts/rdf-scripts-final-copy/results/rdf_similarity_scores_same_simulation_length.csv"
DEFAULT_REF_BASE = "/home/mjgawkowski/phd_mlip_matbench_benchmarks/ref-trajs"
DEFAULT_MLIP_BASE = "/home/mjgawkowski/Finite-Temperature-MLIP-Benchmarks/plots/mlip_trajectories"
DEFAULT_RDF_SAVE_DIR = "/home/mjgawkowski/phd_mlip_matbench_benchmarks/scripts/rdf-scripts-final-copy/results/rdf_same_simulation_length_saved"
DEFAULT_OUTPUT_FILE = "/home/mjgawkowski/phd_mlip_matbench_benchmarks/scripts/rdf-scripts-final-copy/plots/plot_rdf_panel_combined.pdf"
def load_rdf_csv(path: str):
    if not os.path.exists(path):
        return None
    try:
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        if arr.ndim == 1 and arr.size == 2:
            arr = arr.reshape(1, 2)
        return arr[:, 0], arr[:, 1]
    except Exception:
        return None


def _normalize_system_key(name: str) -> str:
    key = name.lower().replace("_no-vdw", "").replace("_vasp", "")
    return re.sub(r"[^a-z0-9]", "", key)


def _build_system_aliases(system: str) -> list[str]:
    aliases = {system}
    queue = [system]

    while queue:
        current = queue.pop()
        variants = set()

        if "_NO-VdW_" in current:
            variants.add(current.replace("_NO-VdW_", "_"))
        if "_NO-VdW" in current:
            variants.add(current.replace("_NO-VdW", ""))

        if current.endswith("_VASP"):
            variants.add(current[: -len("_VASP")])
        else:
            variants.add(f"{current}_VASP")

        for variant in variants:
            if variant and variant not in aliases:
                aliases.add(variant)
                queue.append(variant)

    return sorted(aliases, key=len)


@lru_cache(maxsize=None)
def resolve_saved_rdf_path(system: str, model_name: str | None, rdf_dir: str) -> Path | None:
    base_dir = Path(rdf_dir) / "reference" if model_name is None else Path(rdf_dir) / "mlip" / model_name
    if not base_dir.is_dir():
        return None

    aliases = _build_system_aliases(system)
    for alias in aliases:
        candidate = base_dir / f"{alias}.csv"
        if candidate.is_file():
            return candidate

    alias_norms = {_normalize_system_key(alias) for alias in aliases}
    csv_files = sorted(base_dir.glob("*.csv"))

    for candidate in csv_files:
        if _normalize_system_key(candidate.stem) in alias_norms:
            return candidate

    for candidate in csv_files:
        normalized = _normalize_system_key(candidate.stem)
        if any(normalized.startswith(alias_norm) or alias_norm.startswith(normalized) for alias_norm in alias_norms):
            return candidate

    return None


def load_overall_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Calculator" not in df.columns:
        raise ValueError("Missing column in overall RDF file: Calculator")

    out = df[["Calculator"]].copy()
    out["Calculator"] = out["Calculator"].astype(str)
    excluded_models_lower = {model.lower() for model in EXCLUDED_MODELS}
    out = out[~out["Calculator"].str.lower().isin(excluded_models_lower)].copy()

    if "Mean RDF Error [%]" in df.columns:
        rdf_errors = pd.to_numeric(df["Mean RDF Error [%]"], errors="coerce")
    elif "Mean Similarity Score [%]" in df.columns:
        rdf_errors = 100.0 - pd.to_numeric(df["Mean Similarity Score [%]"], errors="coerce")
    else:
        raise ValueError("Overall RDF file must contain Mean RDF Error [%]")

    out["RDF Error [%]"] = rdf_errors
    return out.dropna(subset=["RDF Error [%]"]).reset_index(drop=True)


def load_model_errors(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        warnings.warn(f"Could not read {path}; skipping")
        return None

    if "System" not in df.columns:
        warnings.warn(f"Missing System column in {path}; skipping")
        return None

    if "RDF_Error" in df.columns:
        rdf_errors = pd.to_numeric(df["RDF_Error"], errors="coerce")
    elif "Similarity_Score" in df.columns:
        rdf_errors = 100.0 - pd.to_numeric(df["Similarity_Score"], errors="coerce")
    else:
        warnings.warn(f"Missing RDF_Error column in {path}; skipping")
        return None

    out = df[["System"]].copy()
    out["RDF_Error"] = rdf_errors
    return out.dropna(subset=["System", "RDF_Error"])


def find_model_error_means(system_list, csv_dir: Path):
    files = [
        f for f in os.listdir(csv_dir)
        if f.startswith("rdf_similarity_scores_same_simulation_length_") and f.endswith(".csv")
    ]
    model_means = {}
    for filename in files:
        model = filename.replace("rdf_similarity_scores_same_simulation_length_", "").replace(".csv", "")
        path = csv_dir / filename
        df = load_model_errors(path)
        if df is None:
            continue

        subset = df[df["System"].isin(system_list)]
        model_means[model] = np.nan if subset.empty else subset["RDF_Error"].mean()
    return model_means


def find_model_system_errors(system_name: str, csv_dir: Path):
    files = [
        f for f in os.listdir(csv_dir)
        if f.startswith("rdf_similarity_scores_same_simulation_length_") and f.endswith(".csv")
    ]
    model_errors = {}
    for filename in files:
        model = filename.replace("rdf_similarity_scores_same_simulation_length_", "").replace(".csv", "")
        path = csv_dir / filename
        df = load_model_errors(path)
        if df is None:
            continue

        subset = df[df["System"] == system_name]
        model_errors[model] = np.nan if subset.empty else subset["RDF_Error"].mean()
    return model_errors


def format_chemical_subscripts(name: str) -> str:
    return re.sub(r"(?<=[A-Za-z])(\d+)", r"$_{\1}$", name)


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

    base_name = format_chemical_subscripts(base_name)

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


def load_saved_rdf(system: str, model_name: str | None, rdf_dir: str):
    csv_path = resolve_saved_rdf_path(system, model_name, rdf_dir)
    if csv_path is None:
        target_name = f"{system}/{model_name}" if model_name is not None else system
        warnings.warn(f"Saved RDF not found for {target_name} in {rdf_dir}")
        return None

    loaded = load_rdf_csv(str(csv_path))
    if loaded is not None:
        return loaded

    target_name = f"{system}/{model_name}" if model_name is not None else system
    warnings.warn(f"Saved RDF unreadable for {target_name}: {csv_path}")
    return None


def format_model_label(model_name: str, system_errors: dict[str, float]) -> str:
    name = display_name(model_name)
    rdf_error = system_errors.get(model_name, np.nan)
    suffix = ""
    if np.isfinite(rdf_error):
        suffix = f" ({rdf_error:.1f}%)"
    return f"{name}{suffix}"


def tier_mean_rdf_error(tier_models, model_means: dict[str, float]) -> float:
    errors = []
    for model in tier_models:
        rdf_error = model_means.get(model, np.nan)
        if np.isfinite(rdf_error):
            errors.append(float(rdf_error))
    if not errors:
        return np.nan
    return float(np.nanmean(errors))


def mean_system_rdf_error(system: str, rdf_csv_dir: Path, rdf_dir: str) -> float:
    system_errors = find_model_system_errors(system, rdf_csv_dir)
    errors = []
    for model, rdf_error in system_errors.items():
        if not np.isfinite(rdf_error):
            continue
        if resolve_saved_rdf_path(system, model, rdf_dir) is None:
            continue
        errors.append(float(rdf_error))

    if not errors:
        return np.nan

    return float(np.nanmean(errors))


def select_system_with_worst_mean_rdf_error(system_list, rdf_csv_dir: Path, rdf_dir: str):
    worst_choice = None
    for system in system_list:
        mean_error = mean_system_rdf_error(system, rdf_csv_dir, rdf_dir)
        if not np.isfinite(mean_error):
            continue

        if worst_choice is None or mean_error > worst_choice[0]:
            worst_choice = (mean_error, system)

    if worst_choice is None:
        return system_list[0]

    _, system = worst_choice
    return system


def build_system_title(system_type: str, system_name: str) -> str:
    return f"{system_type}\n{system_name}"


def draw_overall_error_plot(ax, overall_df: pd.DataFrame) -> None:
    tier_order = TIER_1 + TIER_2 + TIER_3 + TIER_4
    tier_order_norm = TIER_1_NORM + TIER_2_NORM + TIER_3_NORM + TIER_4_NORM
    df = overall_df.copy()
    df["calculator_norm"] = df["Calculator"].map(normalize_model_name)
    df["tier_order"] = df["calculator_norm"].apply(
        lambda x: tier_order_norm.index(x) if x in tier_order_norm else len(tier_order_norm)
    )
    df = df.sort_values("tier_order").reset_index(drop=True)

    models = df["Calculator"].to_list()
    model_norms = df["calculator_norm"].to_list()
    x_pos = np.arange(len(models))
    tier_colors = {
        "tier_1": palette[2],
        "tier_2": palette[1],
        "tier_3": palette[3],
        "tier_4": palette[0],
    }

    colors = []
    for model_norm in model_norms:
        if model_norm in TIER_1_NORM:
            colors.append(tier_colors["tier_1"])
        elif model_norm in TIER_2_NORM:
            colors.append(tier_colors["tier_2"])
        elif model_norm in TIER_3_NORM:
            colors.append(tier_colors["tier_3"])
        elif model_norm in TIER_4_NORM:
            colors.append(tier_colors["tier_4"])
        else:
            colors.append("#757575")

    ax.bar(x_pos, df["RDF Error [%]"], color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.text(0.01, 0.95, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=FONT_SIZE)
    ax.set_xlabel("Model")
    ax.set_ylabel("RDF error [%]")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([display_name(model) for model in models], rotation=45, ha="right", fontsize=FONT_SIZE)
    ax.grid(axis="y")

    t1_med = np.nanmedian(df[df["calculator_norm"].isin(TIER_1_NORM)]["RDF Error [%]"]) if df[df["calculator_norm"].isin(TIER_1_NORM)].shape[0] else np.nan
    t2_med = np.nanmedian(df[df["calculator_norm"].isin(TIER_2_NORM)]["RDF Error [%]"]) if df[df["calculator_norm"].isin(TIER_2_NORM)].shape[0] else np.nan
    t3_med = np.nanmedian(df[df["calculator_norm"].isin(TIER_3_NORM)]["RDF Error [%]"]) if df[df["calculator_norm"].isin(TIER_3_NORM)].shape[0] else np.nan
    t4_med = np.nanmedian(df[df["calculator_norm"].isin(TIER_4_NORM)]["RDF Error [%]"]) if df[df["calculator_norm"].isin(TIER_4_NORM)].shape[0] else np.nan

    print(f"Tier 1 median RDF error: {t1_med:.6f}%" if np.isfinite(t1_med) else "Tier 1 median RDF error: N/A")
    print(f"Tier 2 median RDF error: {t2_med:.2f}%" if np.isfinite(t2_med) else "Tier 2 median RDF error: N/A")
    print(f"Tier 3 median RDF error: {t3_med:.2f}%" if np.isfinite(t3_med) else "Tier 3 median RDF error: N/A")
    print(f"Tier 4 median RDF error: {t4_med:.2f}%" if np.isfinite(t4_med) else "Tier 4 median RDF error: N/A")

    finite_vals = df["RDF Error [%]"].to_numpy()
    med_vals = np.array([t1_med, t2_med, t3_med, t4_med])
    y_max = 1.0
    if np.isfinite(finite_vals).any():
        max_val = np.nanmax(np.concatenate([finite_vals[np.isfinite(finite_vals)], med_vals[np.isfinite(med_vals)]]))
        y_max = max(1.0, float(max_val))
        ax.set_ylim(0, y_max * 1.25)

    tier1_end = len(TIER_1) - 0.5
    tier2_end = len(TIER_1) + len(TIER_2) - 0.5
    tier3_end = len(TIER_1) + len(TIER_2) + len(TIER_3) - 0.5
    ax.axvline(x=tier1_end, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(x=tier2_end, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(x=tier3_end, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    if np.isfinite(finite_vals).any():
        ax.text(tier_center(0, len(TIER_1) - 1), tier_label_y(y_max), "Tier 1", ha="center", fontsize=FONT_SIZE, color=tier_colors["tier_1"])
        ax.text(
            tier_center(len(TIER_1), len(TIER_1) + len(TIER_2) - 1),
            tier_label_y(y_max),
            "Tier 2",
            ha="center",
            fontsize=FONT_SIZE,
            color=tier_colors["tier_2"],
        )
        ax.text(
            tier_center(len(TIER_1) + len(TIER_2), len(TIER_1) + len(TIER_2) + len(TIER_3) - 1),
            tier_label_y(y_max),
            "Tier 3",
            ha="center",
            fontsize=FONT_SIZE,
            color=tier_colors["tier_3"],
        )
        ax.text(
            tier_center(
                len(TIER_1) + len(TIER_2) + len(TIER_3),
                len(TIER_1) + len(TIER_2) + len(TIER_3) + len(TIER_4) - 1,
            ),
            tier_label_y(y_max),
            "Tier 4",
            ha="center",
            fontsize=FONT_SIZE,
            color=tier_colors["tier_4"],
        )

    if np.isfinite(t1_med):
        ax.hlines(t1_med, xmin=-0.5, xmax=tier1_end, colors=tier_colors["tier_1"], linestyles="--", linewidth=2, alpha=0.9)
    if np.isfinite(t2_med):
        ax.hlines(t2_med, xmin=tier1_end, xmax=tier2_end, colors=tier_colors["tier_2"], linestyles="--", linewidth=2, alpha=0.9)
    if np.isfinite(t3_med):
        ax.hlines(t3_med, xmin=tier2_end, xmax=tier3_end, colors=tier_colors["tier_3"], linestyles="--", linewidth=2, alpha=0.9)
    if np.isfinite(t4_med):
        ax.hlines(t4_med, xmin=tier3_end, xmax=len(models) - 0.5, colors=tier_colors["tier_4"], linestyles="--", linewidth=2, alpha=0.9)

    if np.isfinite(t1_med):
        annotate_median(
            ax,
            tier_center(0, len(TIER_1) - 1),
            t1_med,
            median_value_label_y(y_max),
            color=tier_colors["tier_1"],
        )
    if np.isfinite(t2_med):
        annotate_median(
            ax,
            tier_center(len(TIER_1), len(TIER_1) + len(TIER_2) - 1),
            t2_med,
            median_value_label_y(y_max),
            color=tier_colors["tier_2"],
        )
    if np.isfinite(t3_med):
        annotate_median(
            ax,
            tier_center(len(TIER_1) + len(TIER_2), len(TIER_1) + len(TIER_2) + len(TIER_3) - 1),
            t3_med,
            median_value_label_y(y_max),
            color=tier_colors["tier_3"],
        )
    if np.isfinite(t4_med):
        annotate_median(
            ax,
            tier_center(
                len(TIER_1) + len(TIER_2) + len(TIER_3),
                len(TIER_1) + len(TIER_2) + len(TIER_3) + len(TIER_4) - 1,
            ),
            t4_med,
            median_value_label_y(y_max),
            color=tier_colors["tier_4"],
        )


def plot_combined(
    overall_rdf_file: Path,
    rdf_csv_dir: Path,
    ref_base: str,
    mlip_base: str,
    rdf_dir: str,
    output: str,
) -> None:
    overall_df = load_overall_scores(overall_rdf_file)
    tier_count = len(TIER_DEFS)

    fig = plt.figure(figsize=(3.53 * 3.0, 3.53 * 3.55))
    outer_gs = gridspec.GridSpec(
        4,
        6,
        figure=fig,
        wspace=0.30,
        hspace=0.48,
        height_ratios=[0.78, 0.22, 1.15, 1.15],
    )
    panel_labels = ["(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]

    top_row_spans = [(0, 2), (2, 4), (4, 6)]
    # When only two RDF panels exist in row 2, center them with equal margins.
    if len(SYSTEMS) == 5:
        second_row_spans = [(1, 3), (3, 5)]
    else:
        second_row_spans = [(0, 2), (2, 4), (4, 6)]

    for idx, (system_type, system_list) in enumerate(SYSTEMS.items()):
        if idx < 3:
            panel_row_idx = 2
            panel_col_idx = idx
            panels_in_row = len(top_row_spans)
            start_col, end_col = top_row_spans[idx]
        else:
            panel_row_idx = 3
            panel_col_idx = idx - 3
            panels_in_row = len(second_row_spans)
            if panel_col_idx >= panels_in_row:
                warnings.warn(f"No subplot slot configured for {system_type}; skipping")
                continue
            start_col, end_col = second_row_spans[panel_col_idx]

        system = select_system_with_worst_mean_rdf_error(system_list, rdf_csv_dir, rdf_dir)
        system_display_name = format_system_name(system)

        sub_gs = outer_gs[panel_row_idx, start_col:end_col].subgridspec(tier_count, 1, hspace=0.05)
        is_left_col = (panel_col_idx == 0)
        is_right_col = (panel_col_idx == panels_in_row - 1)

        model_means = find_model_error_means(system_list, rdf_csv_dir)
        system_errors = find_model_system_errors(system, rdf_csv_dir)

        def pick_best_worst(models_in_tier):
            scored_models = []
            for model in models_in_tier:
                rdf_error = system_errors.get(model, np.nan)
                if np.isnan(rdf_error):
                    continue
                if resolve_saved_rdf_path(system, model, rdf_dir) is None:
                    continue
                scored_models.append((model, rdf_error))

            if not scored_models:
                return None, None

            scored_models.sort(key=lambda item: item[1])
            best_model = scored_models[0][0]
            worst_model = scored_models[-1][0]
            return best_model, worst_model

        ref_data = load_saved_rdf(system, None, rdf_dir)
        if ref_data is None:
            for tier_idx in range(tier_count):
                rdf_ax = fig.add_subplot(sub_gs[tier_idx])
                rdf_ax.text(0.5, 0.5, "Reference RDF unavailable", ha="center", va="center", fontsize=FONT_SIZE)
                rdf_ax.axis("off")
            continue

        r_ref, g_ref = ref_data

        for tier_idx, (tier_label, tier_models, tier_color) in enumerate(TIER_DEFS):
            rdf_ax = fig.add_subplot(sub_gs[tier_idx])
            if tier_idx == 0:
                rdf_ax.set_title(build_system_title(system_type, system_display_name))
                rdf_ax.text(0.02, 0.95, panel_labels[idx], transform=rdf_ax.transAxes, ha="left", va="top", fontsize=FONT_SIZE)

            mean_rdf_error = tier_mean_rdf_error(tier_models, model_means)
            best_model, worst_model = pick_best_worst(tier_models)
            rdf_ax.plot(r_ref, g_ref, color="black", linewidth=2.0, label="Reference")

            if best_model:
                best_data = load_saved_rdf(system, best_model, rdf_dir)
                if best_data is not None:
                    rdf_ax.plot(
                        best_data[0],
                        best_data[1],
                        color=tier_color,
                        linewidth=1.6,
                        alpha=0.9,
                        linestyle="-",
                        label=format_model_label(best_model, system_errors),
                    )

            if worst_model and worst_model != best_model:
                worst_data = load_saved_rdf(system, worst_model, rdf_dir)
                if worst_data is not None:
                    rdf_ax.plot(
                        worst_data[0],
                        worst_data[1],
                        color=tier_color,
                        linewidth=1.4,
                        alpha=0.9,
                        linestyle="--",
                        label=format_model_label(worst_model, system_errors),
                    )

            rdf_ax.set_xlim(0, None)
            if is_right_col:
                rdf_ax.set_ylabel(tier_label)
                rdf_ax.yaxis.set_label_position("right")
            elif is_left_col:
                rdf_ax.set_ylabel("g(r)")
            else:
                rdf_ax.set_ylabel("")

            if tier_idx == len(TIER_DEFS) - 1:
                rdf_ax.set_xlabel("r [Å]")
            else:
                rdf_ax.set_xlabel("")
                rdf_ax.set_xticklabels([])
            rdf_ax.grid()

            handles, labels = rdf_ax.get_legend_handles_labels()
            if np.isfinite(mean_rdf_error):
                mean_handle = Line2D([], [], color="none", linestyle="none")
                handles.append(mean_handle)
                labels.append(f"{tier_label} mean RDF error: {mean_rdf_error:.1f}%")
            if handles:
                rdf_ax.legend(
                    handles,
                    labels,
                    loc="upper right",
                    # loc="best",
                    fontsize=LEGEND_FONT_SIZE,
                    handlelength=1.0,
                    handletextpad=0.3,
                    borderpad=0.2,
                    labelspacing=0.2,
                    borderaxespad=0.2,
                )

    overall_ax = fig.add_subplot(outer_gs[0, :])
    draw_overall_error_plot(overall_ax, overall_df)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.show()
    print(f"Saved combined RDF panel plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a combined figure from plot-2 and plot-3 same-simulation-length views."
    )
    parser.add_argument(
        "--overall-rdf-file",
        default=str(DEFAULT_RDF_MEANS_FILE),
        help="CSV containing Calculator and Mean RDF Error [%].",
    )
    parser.add_argument(
        "--rdf-csv-dir",
        default=str(DEFAULT_RDF_CSV_DIR),
        help="Directory containing per-model same-simulation-length RDF error CSVs.",
    )
    parser.add_argument(
        "--ref-base",
        default=str(DEFAULT_REF_BASE),
        help="Unused legacy option. RDFs are loaded from precomputed CSV files.",
    )
    parser.add_argument(
        "--mlip-base",
        default=str(DEFAULT_MLIP_BASE),
        help="Unused legacy option. RDFs are loaded from precomputed CSV files.",
    )
    parser.add_argument(
        "--rdf-save-dir",
        default=str(DEFAULT_RDF_SAVE_DIR),
        help="Directory containing precomputed RDF CSV files.",
    )
    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output plot file path.",
    )
    args = parser.parse_args()

    plot_combined(
        overall_rdf_file=Path(args.overall_rdf_file),
        rdf_csv_dir=Path(args.rdf_csv_dir),
        ref_base=args.ref_base,
        mlip_base=args.mlip_base,
        rdf_dir=args.rdf_save_dir,
        output=args.output_file,
    )


if __name__ == "__main__":
    main()
