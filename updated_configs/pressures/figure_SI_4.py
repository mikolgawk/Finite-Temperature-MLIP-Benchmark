#!/usr/bin/env python3
"""Violin plot comparing reference vs predicted pressure distributions.

For each model, the violin shows the distribution of raw pressure values
(pooled across all systems that have reference data).  A reference violin
is drawn first (grey) so the visual overlap/shift is immediately visible.

Requires: numpy, pandas, matplotlib, seaborn
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde


FONT_SIZE = 8
LEGEND_FONT_SIZE = 8

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

SYSTEMS = {
    "Pure metals": [
        "bulkAu_1500K_Kapil",
        "bulkAg_600K_Kapil",
        "bulkCu_1000K_Kapil",
    ],
    "Perovskites": [
        "CsSnI3_500K_Ivor_VASP",
        "MAPbBr3_300K_Ivor_VASP",
    ],
    "Metal dichalcogenides": [
        "bulkMoS2_300K_NO-VdW_J.Kioseoglou_VASP",
        "TiSe2_400K_Ivor_VASP",
    ],
    "Metal alloys": [
        "bulkCuAu_500K-Artrith_VASP",
        "bulkCuZrAl_1500K_A.Wadowski-J.Schmidt_VASP",
        "bulkLiMgAlZnSn_600K_J_Schmidt_VASP",
        "bulkLiMgAlZnSn_900K_J_Schmidt_VASP",
        "bulkPt3Co_300K_J.Kioseoglou_VASP",
    ],
    "Molecular crystals": [
        "picene_295K_Sharma_S",
        "tetracene_295K_Sharma_S",
        "anthracene_293K_Sharma_S",
        "naphthalene_295K_Sharma_S",
        "pentacene_295K_Sharma_S",
    ],
    "Molecular crystals": ["anthracene_293K_Sharma_S", "naphthalene_295K_Sharma_S", "pentacene_295K_Sharma_S", "picene_295K_Sharma_S", "tetracene_295K_Sharma_S"],
    "Metal-water interfaces": ["Pt111w24H2O_380K_Heenen_VASP"],
    "Hydrogen": ["H_1050K_Rupp_QE"],
}

STRUCTURE_TO_TYPE = {
    structure: sys_type
    for sys_type, sys_list in SYSTEMS.items()
    for structure in sys_list
}

TIER_1 = ["chgnet", "mace-mp-0", "grace-mp"]
TIER_2 = ["mace-mpa-0", "orb-v2"]
TIER_3 = ["mattersim-v1-5M", "grace-oam", "orb-v3", "orb-v3-direct", "eSEN-30M-OAM", "nequip", "eq-v2-M-omat", "pet-oam-xl", "pet-omat-xl"]
TIER_4 = ["mace-mh-omat", "uma-s-omat", "uma-m-omat"]
TIER_ORDER = TIER_1 + TIER_2 + TIER_3 + TIER_4

TIER_DEFS = [
    ("Tier 1", TIER_1, palette[2]),
    ("Tier 2", TIER_2, palette[1]),
    ("Tier 3", TIER_3, palette[3]),
    ("Tier 4", TIER_4, palette[0]),
]

PER_FRAME_SUFFIX = "_pressure_per_frame.csv"

REF_LABEL = "Reference"
REF_COLOR = "#888888"


# ── Utilities ──────────────────────────────────────────────────────────────────

def normalize_model_name(name: str) -> str:
    name = re.sub(r"_same-simulation-length$", "", str(name))
    return name.strip().lower()


def display_name(model: str) -> str:
    return CALCULATOR_DISPLAY_NAMES.get(normalize_model_name(model), model)


def structure_from_trajectory_file(path_like: str) -> str:
    return Path(str(path_like)).parent.name


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


def pressure_column_name(columns: Iterable[str]) -> str:
    cols = set(columns)
    if "pressure_GPa" in cols:
        return "pressure_GPa"
    if "pressure_ref_GPa" in cols:
        return "pressure_ref_GPa"
    raise ValueError("No pressure column found.")


def load_pressure_per_frame_csv(csv_path: Path, deduplicate: bool = False) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    pcol = pressure_column_name(header.columns)
    usecols = ["trajectory_file", pcol]
    if "frame_index" in set(header.columns):
        usecols.append("frame_index")
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    df = df.rename(columns={pcol: "pressure_GPa"})
    df["pressure_GPa"] = pd.to_numeric(df["pressure_GPa"], errors="coerce")
    df = df.dropna(subset=["trajectory_file", "pressure_GPa"]).copy()
    df["structure"] = df["trajectory_file"].apply(structure_from_trajectory_file)
    df["sys_type"] = df["structure"].map(STRUCTURE_TO_TYPE)
    df = df.dropna(subset=["sys_type"]).copy()
    if deduplicate:
        if "frame_index" in df.columns:
            df = df.drop_duplicates(subset=["trajectory_file", "frame_index"], keep="first")
        else:
            df = df.drop_duplicates(subset=["trajectory_file", "pressure_GPa"], keep="first")
    return df


def find_reference_file(pressures_dir: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Reference CSV not found: {p}")
    local = Path(
        "../data/results/same-simulation-length/reference_pressure_per_frame_same_simulation_length.csv"
    )
    if local.is_file():
        return local
    raise FileNotFoundError(f"Reference CSV not found: {local}")


# ── Data collection ────────────────────────────────────────────────────────────

def build_pressure_dataframe(pressures_dir: Path, reference_file: Path) -> pd.DataFrame:
    """Return a long-format DataFrame with one row per frame, label = 'Reference' or display name."""
    ref_df = load_pressure_per_frame_csv(reference_file, deduplicate=True)
    # Only keep systems that have reference data and are in our SYSTEMS dict
    ref_df = ref_df[ref_df["structure"].isin(STRUCTURE_TO_TYPE)].copy()
    ref_df["label"] = REF_LABEL
    ref_df["model"] = "__reference__"

    valid_structures = set(ref_df["structure"].unique())
    print(f"[INFO] Reference: {len(ref_df):,} frames across {len(valid_structures)} systems.")

    model_files = sorted(pressures_dir.glob(f"*{PER_FRAME_SUFFIX}"))
    model_files = [f for f in model_files if not f.name.startswith("reference_")]

    parts = [ref_df[["model", "label", "pressure_GPa"]]]

    for model_file in model_files:
        model_name = normalize_model_name(model_file.name.removesuffix(PER_FRAME_SUFFIX))
        if model_name not in TIER_ORDER:
            continue
        try:
            df_m = load_pressure_per_frame_csv(model_file)
        except Exception as exc:
            print(f"[WARN] Skipping {model_file.name}: {exc}")
            continue
        # Keep only structures that have reference data
        df_m = df_m[df_m["structure"].isin(valid_structures)].copy()
        if df_m.empty:
            continue
        df_m["label"] = display_name(model_name)
        df_m["model"] = model_name
        parts.append(df_m[["model", "label", "pressure_GPa"]])
        print(f"[INFO] {display_name(model_name)}: {len(df_m):,} frames")

    df = pd.concat(parts, ignore_index=True)
    return df


# ── Plotting ───────────────────────────────────────────────────────────────────

def _half_violin(ax, x_center: float, data: np.ndarray, side: str,
                 color, alpha: float, half_width: float, lw: float = 0.6) -> None:
    """Draw one half of a split violin using a KDE.

    side = 'left'  → fill from (x_center - density) to x_center
    side = 'right' → fill from x_center to (x_center + density)
    """
    kde = gaussian_kde(data, bw_method="scott")
    y_vals = np.linspace(data.min(), data.max(), 400)
    density = kde(y_vals)
    density = density / density.max() * half_width  # normalise to max half-width

    if side == "left":
        ax.fill_betweenx(y_vals, x_center - density, x_center,
                         color=color, alpha=alpha, linewidth=0)
        ax.plot(x_center - density, y_vals, color=color, lw=lw)
    else:
        ax.fill_betweenx(y_vals, x_center, x_center + density,
                         color=color, alpha=alpha, linewidth=0)
        ax.plot(x_center + density, y_vals, color=color, lw=lw)


def plot_violin(df: pd.DataFrame, output: str | Path) -> None:
    models_present = [m for m in TIER_ORDER if m in df["model"].unique()]
    display_order = [display_name(m) for m in models_present]

    ref_data = df[df["model"] == "__reference__"]["pressure_GPa"].values
    HALF_WIDTH = 0.42

    _, ax = plt.subplots(figsize=(3.53 * 2, 3.53))

    for i, model_name in enumerate(models_present):
        tier_color = get_tier_color(model_name)
        model_data = df[df["model"] == model_name]["pressure_GPa"].values

        # Left half: reference (grey), right half: prediction (tier colour)
        _half_violin(ax, i, ref_data,   side="left",  color=REF_COLOR,  alpha=0.65, half_width=HALF_WIDTH)
        _half_violin(ax, i, model_data, side="right", color=tier_color, alpha=0.65, half_width=HALF_WIDTH)

        # Median markers
        ax.hlines(np.median(ref_data),   i - HALF_WIDTH, i, colors="black", lw=1.0, linestyles="dashed", zorder=3)
        ax.hlines(np.median(model_data), i, i + HALF_WIDTH, colors="black", lw=1.0, linestyles="dashed", zorder=3)

        # Centre spine
        ax.vlines(i, ref_data.min(), ref_data.max(), color="black", lw=0.4, alpha=0.25, zorder=2)

    # ── Tier boundaries and labels ────────────────────────────────────────────
    tier_counts = [
        sum(1 for m in tier_models if m in models_present)
        for _, tier_models, _ in TIER_DEFS
    ]
    cumulative = np.cumsum([0] + tier_counts)

    ymin = df["pressure_GPa"].min()
    ymax = df["pressure_GPa"].max()
    ax.set_ylim(ymin - (ymax - ymin) * 0.02, ymax + (ymax - ymin) * 0.18)

    for boundary in cumulative[1:-1]:
        if 0 < boundary < len(models_present):
            ax.axvline(boundary - 0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.55)

    tier_label_y = ymax + (ymax - ymin) * 0.05

    for i, (tier_label, _, tier_color) in enumerate(TIER_DEFS):
        start = cumulative[i]
        end = cumulative[i + 1]
        if start >= end:
            continue
        center = (start + end - 1) / 2
        ax.text(
            center, tier_label_y, tier_label,
            ha="center", va="bottom",
            fontsize=FONT_SIZE - 1, color=tier_color, fontweight="bold",
        )

    ax.set_xlabel("")
    ax.set_ylabel("Pressure [GPa]")
    ax.set_xticks(range(len(display_order)))
    ax.set_xticklabels(display_order, rotation=45, ha="right", fontsize=FONT_SIZE)
    ax.set_xlim(-0.6, len(display_order) - 0.4)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    # ── Legend ────────────────────────────────────────────────────────────────
    # Place legend top just below the tier label baseline (in axes fraction)
    ymin_ax, ymax_ax = ax.get_ylim()
    legend_top_frac = (tier_label_y - ymin_ax) / (ymax_ax - ymin_ax) - 0.01

    legend_handles = [
        mpatches.Patch(facecolor=REF_COLOR, alpha=0.65, label="Reference"),
    ]
    for tier_label, _, tier_color in TIER_DEFS:
        legend_handles.append(mpatches.Patch(facecolor=tier_color, alpha=0.65, label=tier_label))
    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(0.01, legend_top_frac),
        loc="upper left",
        fontsize=LEGEND_FONT_SIZE,
        framealpha=0.9,
    )

    plt.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.show()
    print(f"Saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Violin plot of reference vs predicted pressure distributions."
    )
    parser.add_argument(
        "--pressures-dir",
        default="../data/results/same-simulation-length",
    )
    parser.add_argument("--reference-file", default=None)
    parser.add_argument(
        "--output-file",
        default="plots/plot_pressure_violin_distributions.pdf",
    )
    args = parser.parse_args()

    pressures_dir = Path(args.pressures_dir)
    if not pressures_dir.is_dir():
        raise NotADirectoryError(f"Pressures directory not found: {pressures_dir}")

    reference_file = find_reference_file(pressures_dir, args.reference_file)
    print(f"[INFO] Using reference file: {reference_file}")

    df = build_pressure_dataframe(pressures_dir, reference_file)
    print(f"[INFO] Total rows: {len(df):,}")

    plot_violin(df, args.output_file)


if __name__ == "__main__":
    main()
