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

from get_model_pressure_errors import (
    filter_completed_pressure_md_rows,
    pressure_source,
    resolve_model_reference_pressure_file,
)
from pressure_axis_breaks import distribution_axis_windows, finite_values, pressure_axes

import sys

PAPER_V2_CONFIG_DIR = Path(__file__).resolve().parents[1]
if str(PAPER_V2_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_V2_CONFIG_DIR))
from model_display_names import MODEL_DISPLAY_NAMES, display_model_name
from system_filters import add_molecular_crystal_option, filter_pressure_systems


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

CALCULATOR_DISPLAY_NAMES = MODEL_DISPLAY_NAMES

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
    "Molecular crystals": ["anthracene_293K_Sharma_S", "naphthalene_295K_Sharma_S", "pentacene_295K_Sharma_S", "picene_295K_Sharma_S", "tetracene_295K_Sharma_S"],
    "Hydrogen": ["H_1050K_Rupp_QE"],
}

STRUCTURE_TO_TYPE = {
    structure: sys_type
    for sys_type, sys_list in SYSTEMS.items()
    for structure in sys_list
}

TIER_1 = ["chgnet", "mace-mp-0", "mace-mp-0-compile", "grace-mp"]
TIER_2 = ["mace-mpa-0", "mace-mpa-0-compile", "orb-v2"]
TIER_3 = ["mattersim-v1-5m", "grace-oam", "orb-v3-omat", "orb-v3-direct-omat", "esen-30m-oam", "nequip", "eq-v2-m-omat", "pet-oam-xl", "pet-omat-xl", "grace-oam-compiled", "mattersim-v1-5m-compile", "pet-oam-xl-torchscript", "pet-omat-xl-torchscript"]
TIER_4 = ["mace-mh-omat", "mace-mh-omat-compile", "uma-s-omat", "uma-s-omat-compile", "uma-s-omat-turbo", "uma-m-omat", "uma-m-omat-compile", "uma-m-omat-turbo"]
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
    return display_model_name(model)


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


def load_pressure_per_frame_csv(csv_path: Path, deduplicate: bool = False, include_molecular_crystals: bool = False) -> pd.DataFrame:
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
    return filter_pressure_systems(df, include_molecular_crystals)


def legacy_reference_files() -> tuple[Path, ...]:
    return (Path(
        "../data/results/same-simulation-length/reference_pressure_per_frame_same_simulation_length.csv"
    ),)


# ── Data collection ────────────────────────────────────────────────────────────

def build_pressure_dataframe(pressures_dir: Path, reference_file: Path | None, include_molecular_crystals: bool = False) -> pd.DataFrame:
    """Return per-frame model and model-matched reference pressure values."""
    model_files = sorted(pressures_dir.glob(f"*{PER_FRAME_SUFFIX}"))
    model_files = [f for f in model_files if not f.name.startswith("reference_")]
    parts: list[pd.DataFrame] = []
    for model_file in model_files:
        model_name = normalize_model_name(model_file.name.removesuffix(PER_FRAME_SUFFIX))
        if model_name not in TIER_ORDER:
            continue
        try:
            matched_reference = resolve_model_reference_pressure_file(
                pressures_dir, model_file, reference_file, legacy_reference_files()
            )
            df_m = load_pressure_per_frame_csv(model_file, include_molecular_crystals=include_molecular_crystals)
            df_m = filter_completed_pressure_md_rows(df_m, model_name, pressure_source(model_file))
            df_ref = load_pressure_per_frame_csv(matched_reference, deduplicate=True, include_molecular_crystals=include_molecular_crystals)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"[WARN] Skipping {model_file.name}: {exc}")
            continue
        shared_structures = set(df_m["structure"]) & set(df_ref["structure"])
        df_m = df_m[df_m["structure"].isin(shared_structures)].copy()
        df_ref = df_ref[df_ref["structure"].isin(shared_structures)].copy()
        if df_m.empty or df_ref.empty:
            continue
        for df, label, kind in ((df_ref, REF_LABEL, "reference"),
                                (df_m, display_name(model_name), "model")):
            df["label"] = label
            df["model"] = model_name
            df["kind"] = kind
            parts.append(df[["model", "kind", "label", "pressure_GPa"]])
        print(f"[INFO] {display_name(model_name)}: {len(df_m):,} frames; reference: {len(df_ref):,} frames")

    if not parts:
        raise RuntimeError("No usable model/reference pressure pairs could be loaded.")
    return pd.concat(parts, ignore_index=True)


# ── Plotting ───────────────────────────────────────────────────────────────────

def _violin_profile(data: np.ndarray, windows):
    """Sample every visible range and use a bandwidth robust to extreme tails.

    Quantile sampling bounds rendering cost for long trajectories.  The full
    frame arrays still determine the median, range, quantiles and Scott factor.
    """
    data = finite_values(data)
    if data.size == 0:
        return [(np.array([]), np.array([])) for _ in windows]
    if np.ptp(data) == 0:
        return [(np.array([data[0]]), np.array([1.0])) if lo <= data[0] <= hi
                else (np.array([]), np.array([])) for lo, hi in windows]
    sample = (np.quantile(data, np.linspace(0, 1, 8192))
              if data.size > 8192 else data)
    std = np.std(sample, ddof=1)
    q25, q75 = np.quantile(data, [0.25, 0.75])
    robust_std = (q75 - q25) / 1.349
    scale = min(std, robust_std) if robust_std > 0 else std
    bandwidth = max(scale * data.size ** (-0.2), np.finfo(float).eps * std)
    kde = gaussian_kde(sample, bw_method=bandwidth / std)
    profiles = []
    for lo, hi in windows:
        local = sample[(sample >= lo) & (sample <= hi)]
        if local.size == 0:
            profiles.append((np.array([]), np.array([])))
            continue
        y = np.unique(np.r_[np.linspace(max(lo, data.min()), min(hi, data.max()), 400),
                            np.quantile(local, np.linspace(0, 1, min(local.size, 200)))])
        profiles.append((y, kde(y)))
    peak = max((density.max() for _, density in profiles if density.size), default=1.0)
    return [(y, density / peak) for y, density in profiles]


def _half_violin(ax, x_center: float, profile, side: str,
                 color, alpha: float, half_width: float, lw: float = 0.6) -> None:
    y_vals, density = profile
    if not y_vals.size:
        return
    density = density * half_width
    edge = x_center - density if side == "left" else x_center + density
    if y_vals.size == 1:
        ax.hlines(y_vals[0], min(edge[0], x_center), max(edge[0], x_center), color=color, lw=lw)
        return
    ax.fill_betweenx(y_vals, edge, x_center, color=color, alpha=alpha, linewidth=0)
    ax.plot(edge, y_vals, color=color, lw=lw)


def plot_violin(df: pd.DataFrame, output: str | Path) -> None:
    models_present = [m for m in TIER_ORDER if m in df["model"].unique()]
    data_by_model = {
        model: tuple(finite_values(df.loc[(df["model"] == model) & (df["kind"] == kind),
                                         "pressure_GPa"].values)
                     for kind in ("reference", "model"))
        for model in models_present
    }
    references = [pair[0] for pair in data_by_model.values()]
    windows = distribution_axis_windows([values for pair in data_by_model.values()
                                         for values in pair], references)
    reference_center = float(np.median(np.concatenate(references)))
    core_index = next(i for i, (lo, hi) in enumerate(windows) if lo <= reference_center <= hi)
    height = 3.53 + 0.55 * (len(windows) - 1)
    fig = plt.figure(figsize=(3.53 * 2, height),
                     layout="constrained")
    spec = fig.add_gridspec(1, 1)[0]
    axes = pressure_axes(fig, spec, windows, orientation="y", core_index=core_index)
    half_width = 0.42
    for i, (model, (ref_data, model_data)) in enumerate(data_by_model.items()):
        color = get_tier_color(model)
        ref_profiles = _violin_profile(ref_data, windows)
        model_profiles = _violin_profile(model_data, windows)
        for index, ax in enumerate(axes):
            _half_violin(ax, i, ref_profiles[index], side="left", color=REF_COLOR,
                         alpha=0.65, half_width=half_width)
            _half_violin(ax, i, model_profiles[index], side="right", color=color,
                         alpha=0.65, half_width=half_width)
            # Mark the FULL distributions' medians and extents on each segment.
            for values, xmin, xmax, spine_color in (
                (ref_data, i - half_width, i, REF_COLOR),
                (model_data, i, i + half_width, color),
            ):
                if values.size:
                    ax.hlines(np.median(values), xmin, xmax, colors="black", lw=1.0,
                              linestyles="dashed", zorder=3)
                    ax.vlines(i, values.min(), values.max(), color=spine_color,
                              lw=0.6, alpha=0.6, zorder=2)
                    ax.hlines([values.min(), values.max()], i - 0.06, i + 0.06,
                              color=spine_color, lw=0.6)
    tier_counts = [sum(model in models_present for model in tier_models)
                   for _, tier_models, _ in TIER_DEFS]
    cumulative = np.cumsum([0] + tier_counts)
    for ax in axes:
        for boundary in np.unique(cumulative[1:-1]):
            if 0 < boundary < len(models_present):
                ax.axvline(boundary - 0.5, color="black", linestyle="--",
                           linewidth=1.2, alpha=0.55)
        ax.set_xlim(-0.6, len(models_present) - 0.4)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    top, bottom = axes[-1], axes[0]
    for i, (tier_label, _, tier_color) in enumerate(TIER_DEFS):
        start, end = cumulative[i:i + 2]
        if start < end:
            top.text((start + end - 1) / 2, 1.04, tier_label,
                     transform=top.get_xaxis_transform(), ha="center", va="bottom",
                     fontsize=FONT_SIZE, color=tier_color, fontweight="bold")
    bottom.set_xticks(range(len(models_present)))
    bottom.set_xticklabels([display_name(model) for model in models_present],
                          rotation=45, ha="right", fontsize=FONT_SIZE)
    fig.supylabel("Pressure [GPa]", fontsize=FONT_SIZE)
    handles = [mpatches.Patch(facecolor=REF_COLOR, alpha=0.65, label="Reference")]
    handles.extend(mpatches.Patch(facecolor=color, alpha=0.65, label=label)
                   for (label, _, color), count in zip(TIER_DEFS, tier_counts) if count)
    top.legend(handles=handles, loc="upper left", fontsize=LEGEND_FONT_SIZE - 1,
               framealpha=0.9, ncol=3)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
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
        default="plots/figure_SI_4.pdf",
    )
    add_molecular_crystal_option(parser)
    args = parser.parse_args()

    pressures_dir = Path(args.pressures_dir)
    if not pressures_dir.is_dir():
        raise NotADirectoryError(f"Pressures directory not found: {pressures_dir}")

    reference_file = Path(args.reference_file) if args.reference_file else None

    df = build_pressure_dataframe(pressures_dir, reference_file, include_molecular_crystals=args.include_molecular_crystals)
    print(f"[INFO] Total rows: {len(df):,}")

    plot_violin(df, args.output_file)


if __name__ == "__main__":
    main()
