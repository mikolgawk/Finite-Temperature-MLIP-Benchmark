#!/usr/bin/env python3
"""Correlate generated VDOS and RDF model-level mean errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _figure_data import (
    DEFAULT_SOURCE,
    SCRIPT_DIR,
    SOURCES,
    load_rdf as load_rdf_table,
    load_vdos as load_vdos_table,
    source_paths,
)

import sys

PAPER_V2_CONFIG_DIR = Path(__file__).resolve().parents[1]
if str(PAPER_V2_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_V2_CONFIG_DIR))
from model_display_names import MODEL_DISPLAY_NAMES, display_model_name


FONT_SIZE = 6


CALCULATOR_DISPLAY_NAMES = MODEL_DISPLAY_NAMES


def normalize_model_name(name: str) -> str:
    return str(name).strip().lower()


def display_name(model: str) -> str:
    return display_model_name(model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate VDOS model mean errors with RDF model mean errors.",
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default=DEFAULT_SOURCE,
        help="Trajectory source used to locate generated RDF and VDOS summaries.",
    )
    parser.add_argument(
        "--mode",
        choices=["same", "different"],
        default="same",
        help="Use matched-length generated files; 'different' requires explicit input files.",
    )
    parser.add_argument(
        "--vdos-file",
        type=Path,
        default=None,
        help="Override VDOS model means CSV path.",
    )
    parser.add_argument(
        "--rdf-file",
        type=Path,
        default=None,
        help="Override RDF model means CSV path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=SCRIPT_DIR / "results/vdos_rdf_model_means_merged_same_simulation_length.csv",
        help="Path to save merged table.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=SCRIPT_DIR / "plots/correlation_SI_rdf_vdos_same_length.pdf",
        help="Path to save correlation scatter plot.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=SCRIPT_DIR / "results/vdos_rdf_model_means_correlation.json",
        help="Path to save correlation stats as JSON.",
    )
    return parser.parse_args()


def _values_to_percent(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.dropna().empty:
        return vals
    if float(vals.max()) <= 1.5:
        return vals * 100.0
    return vals


def load_vdos(path: Path) -> pd.DataFrame:
    return load_vdos_table(path)


def load_rdf(path: Path) -> pd.DataFrame:
    return load_rdf_table(path)


def plot_scatter(merged: pd.DataFrame, output_plot: Path, mode: str) -> None:
    x = merged["vdos_error_percent"].to_numpy(dtype=float)
    y = merged["rdf_error_percent"].to_numpy(dtype=float)

    plt.rcParams.update({
        'lines.markersize': 4,
        'lines.linewidth': 1.5,
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE,
        'figure.titlesize': FONT_SIZE,
        'axes.grid': True,
        'grid.linewidth': 0.5,
        'grid.alpha': 1.0,
    })

    palette = sns.color_palette("deep")

    tier_1 = ["chgnet", "mace-mp-0", "mace-mp-0-compile", "grace-mp"]
    tier_2 = ["mace-mpa-0", "mace-mpa-0-compile", "orb-v2"]
    tier_3 = ["mattersim-v1-5m", "grace-oam", "orb-v3", "esen-30m-oam", "nequip", "eq-v2-m-omat", "pet-oam-xl", "pet-omat-xl", "grace-oam-compiled", "mattersim-v1-5m-compile", "pet-oam-xl-torchscript", "pet-omat-xl-torchscript"]
    tier_4 = ["mace-mh-omat", "mace-mh-omat-compile", "uma-s-omat", "uma-s-omat-compile", "uma-s-omat-turbo", "uma-m-omat", "uma-m-omat-compile", "uma-m-omat-turbo"]
    tier_colors = {
        "tier_1": palette[2],
        "tier_2": palette[1],
        "tier_3": palette[3],
        "tier_4": palette[0],
    }

    def model_color(m: str) -> str:
        m = normalize_model_name(m)
        if m in tier_1:
            return tier_colors["tier_1"]
        if m in tier_2:
            return tier_colors["tier_2"]
        if m in tier_3:
            return tier_colors["tier_3"]
        if m in tier_4:
            return tier_colors["tier_4"]
        return "#757575"

    try:
        from adjustText import adjust_text
    except Exception:
        adjust_text = None

    fig, ax = plt.subplots(figsize=(3.53 * 1.5, 3.53 * 1.5))

    colors = [model_color(m) for m in merged["model"].to_numpy()]
    ax.scatter(x, y, c=colors, edgecolor="k", linewidth=0.4, s=40)

    label_texts = []
    for xi, yi, model in zip(x, y, merged["model"]):
        txt = ax.annotate(
            display_name(model),
            xy=(xi, yi),
            xytext=(2, 2),
            textcoords='offset points',
            fontsize=FONT_SIZE,
            alpha=0.85,
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.08', facecolor='white', edgecolor='none', alpha=0.55),
        )
        label_texts.append(txt)

    if adjust_text is not None and label_texts:
        adjust_text(label_texts, ax=ax, x=x, y=y, only_move={'points': 'xy', 'text': 'xy'})

    if len(merged) >= 2:
        try:
            coef = np.polyfit(x, y, 1)
            x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
            y_line = coef[0] * x_line + coef[1]
            ax.plot(x_line, y_line, color="gray", linestyle="--", linewidth=1)
        except Exception:
            pass

    try:
        pearson = merged["vdos_error_percent"].corr(merged["rdf_error_percent"], method="pearson")
        ax.text(0.02, 0.03, f"r = {pearson:.2f}", transform=ax.transAxes, va="bottom")
    except Exception:
        pass

    # legend for tiers
    import matplotlib.lines as mlines
    legend_handles = []
    legend_handles.append(mlines.Line2D([], [], color=tier_colors['tier_1'], marker='o', linestyle='None', markersize=6, label='Tier 1'))
    legend_handles.append(mlines.Line2D([], [], color=tier_colors['tier_2'], marker='o', linestyle='None', markersize=6, label='Tier 2'))
    legend_handles.append(mlines.Line2D([], [], color=tier_colors['tier_3'], marker='o', linestyle='None', markersize=6, label='Tier 3'))
    legend_handles.append(mlines.Line2D([], [], color=tier_colors['tier_4'], marker='o', linestyle='None', markersize=6, label='Tier 4'))
    ax.legend(handles=legend_handles, frameon=False, loc='upper left')

    # ax.set_title(f"VDOS vs RDF model errors ({mode} sim. length)")
    ax.set_xlabel("VDOS error [%]")
    ax.set_ylabel("RDF error [%]")
    ax.grid(True, alpha=0.25)

    ax.set_xlim(right=32.2)
    ax.set_ylim(bottom=2.4)

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_plot)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.mode == "different" and (args.vdos_file is None or args.rdf_file is None):
        raise ValueError("--mode different requires both --vdos-file and --rdf-file.")
    _, default_rdf, default_vdos = source_paths(args.source)
    args.vdos_file = args.vdos_file or default_vdos
    args.rdf_file = args.rdf_file or default_rdf

    df_vdos = load_vdos(args.vdos_file)
    df_rdf = load_rdf(args.rdf_file)

    merged = df_vdos.merge(df_rdf, on="model", how="inner")
    merged = merged.sort_values("model").reset_index(drop=True)

    if merged.empty:
        raise ValueError(
            "No overlapping models found between VDOS and RDF tables. "
            "Check model names and input files."
        )

    corr = {
        "mode": args.mode,
        "n_models": int(len(merged)),
        "pearson_similarity": float(
            merged["vdos_similarity_percent"].corr(merged["rdf_similarity_percent"], method="pearson")
        ),
        "spearman_similarity": float(
            merged["vdos_similarity_percent"].corr(merged["rdf_similarity_percent"], method="spearman")
        ),
        "pearson_error": float(
            merged["vdos_error_percent"].corr(merged["rdf_error_percent"], method="pearson")
        ),
        "spearman_error": float(
            merged["vdos_error_percent"].corr(merged["rdf_error_percent"], method="spearman")
        ),
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(args.output_csv, index=False)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(corr, f, indent=2)

    plot_scatter(merged, args.output_plot, args.mode)

    print(f"Merged rows: {len(merged)}")
    print(f"Saved merged CSV: {args.output_csv}")
    print(f"Saved plot: {args.output_plot}")
    print(f"Saved correlation JSON: {args.output_json}")
    print(f"Pearson(error): {corr['pearson_error']:.4f}")
    print(f"Spearman(error): {corr['spearman_error']:.4f}")
    print(f"Pearson(similarity): {corr['pearson_similarity']:.4f}")
    print(f"Spearman(similarity): {corr['spearman_similarity']:.4f}")


if __name__ == "__main__":
    main()
