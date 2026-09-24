#!/usr/bin/env python3
"""Correlate RDF model mean error with pressure error.

Reads an RDF summary CSV and a pressure error CSV, merges on model name,
and computes Pearson/Spearman correlations between RDF error (%) and
pressure error (%).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _figure_data import (
    pressure_path,
    DEFAULT_SOURCE,
    SCRIPT_DIR,
    SOURCES,
    load_pressure as load_pressure_table,
    load_rdf as load_rdf_table,
    source_paths,
    source_pressure_provenance,
)

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
    parser = argparse.ArgumentParser(description="Correlate RDF error with pressure error")
    parser.add_argument("--source", choices=SOURCES, default=DEFAULT_SOURCE)
    parser.add_argument("--rdf-file", default=None, type=Path, help="Override the generated RDF model summary CSV.")
    parser.add_argument("--pressure-file", default=None, type=Path)
    parser.add_argument("--pressure-backend", default=None, help="Default: inferred from --source.")
    parser.add_argument("--pressure-mode", default=None, help="Default: inferred from --source.")
    parser.add_argument("--output-csv", type=Path, default=SCRIPT_DIR / "results/rdf_pressure_merged_same_simulation_length.csv")
    parser.add_argument("--output-json", type=Path, default=SCRIPT_DIR / "results/rdf_pressure_correlation_same_simulation_length.json")
    parser.add_argument("--output-plot", type=Path, default=SCRIPT_DIR / "plots/correlation_SI_rdf_pressure_same_length.pdf")
    return parser.parse_args()


def _values_to_percent(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.dropna().empty:
        return vals
    if float(vals.max()) <= 1.5:
        return vals * 100.0
    return vals


def load_rdf(path: Path) -> pd.DataFrame:
    return load_rdf_table(path)


def load_pressure(path: Path, backend: str | None = None, mode: str | None = None) -> pd.DataFrame:
    return load_pressure_table(path, backend=backend, mode=mode)


def plot_scatter(merged: pd.DataFrame, output_plot: Path) -> None:
    x = merged["pressure_error_percent"].to_numpy(dtype=float)
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
            ax.plot(x_line, y_line, color="gray", linestyle='--', linewidth=1)
        except Exception:
            pass

    try:
        pearson = merged["rdf_error_percent"].corr(merged["pressure_error_percent"], method="pearson")
        ax.text(0.02, 0.03, f"r = {pearson:.2f}", transform=ax.transAxes, va="bottom")
    except Exception:
        pass

    import matplotlib.lines as mlines
    legend_handles = []
    legend_handles.append(mlines.Line2D([], [], color=tier_colors['tier_1'], marker='o', linestyle='None', markersize=6, label='Tier 1'))
    legend_handles.append(mlines.Line2D([], [], color=tier_colors['tier_2'], marker='o', linestyle='None', markersize=6, label='Tier 2'))
    legend_handles.append(mlines.Line2D([], [], color=tier_colors['tier_3'], marker='o', linestyle='None', markersize=6, label='Tier 3'))
    legend_handles.append(mlines.Line2D([], [], color=tier_colors['tier_4'], marker='o', linestyle='None', markersize=6, label='Tier 4'))
    ax.legend(handles=legend_handles, frameon=False, loc='upper left')

    ax.set_xlim(right=79)
    # ax.set_ylim(bottom=1.9)

    ax.set_xlabel("Pressure histogram error [%]")
    ax.set_ylabel("RDF error [%]")
    ax.grid(True, alpha=0.25)



    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_plot)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _, default_rdf, _ = source_paths(args.source)
    backend, mode = source_pressure_provenance(args.source)
    rdf = load_rdf(args.rdf_file or default_rdf)
    pressure = load_pressure(
        args.pressure_file or pressure_path(args.source),
        backend=args.pressure_backend or backend,
        mode=args.pressure_mode or mode,
    )

    merged = rdf.merge(pressure, on="model", how="inner")
    merged = merged.sort_values("model").reset_index(drop=True)

    if merged.empty:
        print("No overlapping models after merge", file=sys.stderr)
        sys.exit(2)

    corr = {
        "n_models": int(len(merged)),
        "pearson_error_vs_pressure": float(merged["rdf_error_percent"].corr(merged["pressure_error_percent"], method="pearson")),
        "spearman_error_vs_pressure": float(merged["rdf_error_percent"].corr(merged["pressure_error_percent"], method="spearman")),
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(args.output_csv, index=False)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(corr, f, indent=2)

    plot_scatter(merged, args.output_plot)

    print(f"Merged rows: {len(merged)}")
    print(f"Saved merged CSV: {args.output_csv}")
    print(f"Saved plot: {args.output_plot}")
    print(f"Saved correlation JSON: {args.output_json}")
    print(f"Pearson(error vs pressure error): {corr['pearson_error_vs_pressure']:.4f}")
    print(f"Spearman(error vs pressure error): {corr['spearman_error_vs_pressure']:.4f}")


if __name__ == "__main__":
    main()
