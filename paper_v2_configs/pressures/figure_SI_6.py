#!/usr/bin/env python3
"""Combined pressure figure for same-simulation-length percentage-error views.

The top row plots model-level pressure histogram error by tier. The lower
panels show the reference distribution and the best/worst model distribution
in each tier for one representative system of each system type.
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
    load_pressure_per_frame_csv,
    normalize_model_name,
    parse_model_name,
    pressure_histogram_similarity,
)


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
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "grid.alpha": 1.0,
})

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRESSURES_DIR = SCRIPT_DIR.parent / "data" / "results" / "same-simulation-length"
DEFAULT_REFERENCE_FILE = DEFAULT_PRESSURES_DIR / "reference_pressure_per_frame_same_simulation_length.csv"
DEFAULT_RANKING_FILE = DEFAULT_PRESSURES_DIR / "model_pressure_error_metric.csv"
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "plots" / "plot_pressure_panel_combined_pressure_errors.pdf"
PER_FRAME_SUFFIX = "_same-simulation-length_pressure_per_frame.csv"

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
    "Metal-water interfaces": ["Pt111w24H2O_380K_Heenen_VASP"],
    "Hydrogen": ["H_1050K_Rupp_QE"],
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


def display_name(model: str) -> str:
    return CALCULATOR_DISPLAY_NAMES.get(normalize_model_name(model), model)


def format_error_value(error_percent: float | None) -> str:
    if error_percent is None or not np.isfinite(error_percent):
        return "n/a"
    return f"{error_percent:.1f}%"


def format_model_label(model: str, error_percent: float | None, prefix: str | None = None) -> str:
    label = display_name(model)
    if prefix:
        label = f"{prefix}: {label}"
    if error_percent is None or not np.isfinite(error_percent):
        return label
    return f"{label} ({error_percent:.1f}%)"


def format_chemical_formula(formula: str) -> str:
    return re.sub(r"(\d+)", r"$_{\1}$", formula)


def format_system_name(system_name: str) -> str:
    tokens = system_name.split("_")
    temperature = None
    temperature_index = None
    for idx, token in enumerate(tokens):
        match = re.search(r"(\d+)K", token)
        if match:
            temperature = match.group(1)
            temperature_index = idx
            break

    base_name = system_name if temperature_index is None else "_".join(tokens[:temperature_index])
    if base_name.startswith("bulk") and len(base_name) > 4:
        base_name = f"bulk {base_name[4:]}"
    base_name = format_chemical_formula(base_name)
    if temperature is None:
        return base_name
    return f"{base_name} ({temperature} K)"


def get_tier_color(model: str):
    for _, tier_models, tier_color in TIER_DEFS:
        if model in tier_models:
            return tier_color
    return "#757575"


def find_reference_file(pressures_dir: Path, explicit_path: str | None) -> Path:
    path = Path(explicit_path) if explicit_path else DEFAULT_REFERENCE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Reference per-frame CSV not found: {path}")
    return path


def make_bin_edges(reference_values: np.ndarray, candidates: Iterable[np.ndarray], bins: int) -> np.ndarray:
    arrays = [reference_values] + [values for values in candidates if values.size > 0]
    lo = min(float(np.min(values)) for values in arrays)
    hi = max(float(np.max(values)) for values in arrays)
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = -1.0, 1.0
    if hi <= lo:
        hi = lo + 1e-6
    return np.linspace(lo, hi, bins + 1)


def pressure_error_percent(reference_values: np.ndarray, model_values: np.ndarray | None, bins: int) -> float:
    if model_values is None or reference_values.size == 0 or model_values.size == 0:
        return float("nan")
    score = pressure_histogram_similarity(reference_values, model_values, bins=bins)
    if not score:
        return float("nan")
    return float(score["pressure_error_percent"])


def mean_finite(values: Iterable[float | None]) -> float:
    finite_values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def choose_best_worst(
    tier_models: list[str],
    model_scores: dict[str, float],
) -> tuple[str | None, str | None, dict[str, float]]:
    scores = {
        model: score
        for model, score in model_scores.items()
        if model in tier_models and np.isfinite(score)
    }
    if not scores:
        return None, None, scores
    return min(scores, key=scores.get), max(scores, key=scores.get), scores


def load_model_values(pressures_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    model_values: dict[str, dict[str, np.ndarray]] = {}
    model_files = sorted(pressures_dir.glob(f"*{PER_FRAME_SUFFIX}"))
    model_files = [path for path in model_files if not path.name.startswith("reference_")]
    if not model_files:
        raise FileNotFoundError(f"No model per-frame CSV files found in: {pressures_dir}")

    for model_file in model_files:
        model = parse_model_name(model_file, PER_FRAME_SUFFIX)
        if model not in TIER_ORDER:
            continue
        try:
            model_df = load_pressure_per_frame_csv(model_file)
        except Exception as exc:
            print(f"[WARN] Skipping {model_file.name}: {exc}")
            continue

        values_by_system = {
            str(system): group["pressure_GPa"].to_numpy(dtype=float)
            for system, group in model_df.groupby("system", sort=False)
            if not group.empty
        }
        if values_by_system:
            model_values[model] = values_by_system

    if not model_values:
        raise RuntimeError("No usable model per-frame data could be loaded.")
    return model_values


def collect_histogram_panels(
    pressures_dir: Path,
    reference_file: Path,
    bins: int,
) -> list[tuple[str, str, np.ndarray, dict[str, np.ndarray], dict[str, float], np.ndarray]]:
    reference_df = load_pressure_per_frame_csv(reference_file, deduplicate_reference=True)
    if reference_df.empty:
        raise RuntimeError(f"No usable reference rows in {reference_file}")

    model_values = load_model_values(pressures_dir)
    panels: list[tuple[str, str, np.ndarray, dict[str, np.ndarray], dict[str, float], np.ndarray]] = []

    for system_type, ordered_systems in SYSTEMS.items():
        reference_by_system = {
            system: reference_df.loc[
                reference_df["system"] == system, "pressure_GPa"
            ].to_numpy(dtype=float)
            for system in ordered_systems
            if not reference_df.loc[reference_df["system"] == system].empty
        }

        system_scores: dict[str, dict[str, float]] = {}
        for system, reference_values in reference_by_system.items():
            scores: dict[str, float] = {}
            for model, values_by_system in model_values.items():
                error = pressure_error_percent(reference_values, values_by_system.get(system), bins=bins)
                if np.isfinite(error):
                    scores[model] = error
            if scores:
                system_scores[system] = scores

        if not system_scores:
            print(f"[INFO] Skipping system type without enough data: {system_type}")
            continue

        representative_system = max(
            system_scores,
            key=lambda system: mean_finite(system_scores[system].values()),
        )
        reference_values = reference_by_system[representative_system]
        values_by_model = {
            model: values_by_system[representative_system]
            for model, values_by_system in model_values.items()
            if representative_system in values_by_system
        }
        scores = {
            model: score
            for model, score in system_scores[representative_system].items()
            if model in values_by_model
        }
        edges = make_bin_edges(reference_values, values_by_model.values(), bins=bins)

        print(
            "[INFO] Selected "
            f"{representative_system} for {system_type}: mean model pressure error = "
            f"{format_error_value(mean_finite(scores.values()))} across {len(scores)} models."
        )
        panels.append(
            (system_type, representative_system, reference_values, values_by_model, scores, edges)
        )

    if not panels:
        raise RuntimeError("No system types have both reference and model pressure data for plotting.")
    return panels


def extract_overall_errors(ranking_df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in ranking_df.columns:
        raise ValueError("Ranking CSV is missing required column: 'model'")
    candidates = [
        "final_mean_pressure_error_percent",
        "pressure_error_percent",
        "Pressure Error [%]",
    ]
    error_col = next((col for col in candidates if col in ranking_df.columns), None)
    if error_col is None:
        raise ValueError(
            "Ranking CSV must contain 'final_mean_pressure_error_percent' or "
            "'pressure_error_percent'."
        )

    df = ranking_df[["model", error_col]].copy()
    df["model"] = df["model"].map(normalize_model_name)
    df["pressure_error_percent"] = pd.to_numeric(df[error_col], errors="coerce")
    df = df[df["model"].isin(TIER_ORDER)].dropna(subset=["pressure_error_percent"]).copy()
    df["tier_order"] = df["model"].map(TIER_ORDER.index)
    return df.sort_values("tier_order").reset_index(drop=True)


def annotate_median(ax, x: float, y: float, text_y: float, color) -> None:
    ax.annotate(
        f"{y:.1f}%",
        xy=(x, y),
        xytext=(x, text_y),
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE,
        fontweight="bold",
        color=color,
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


def draw_overall_pressure_error_plot(ax, ranking_df: pd.DataFrame, panel_label: str) -> None:
    df = extract_overall_errors(ranking_df)
    if df.empty:
        raise ValueError("No tier-listed models with valid pressure errors found to plot.")

    models = df["model"].to_list()
    x = np.arange(len(models))
    values = df["pressure_error_percent"].to_numpy(dtype=float)
    ax.bar(
        x,
        values,
        width=0.65,
        color=[get_tier_color(model) for model in models],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.text(0.01, 0.95, panel_label, transform=ax.transAxes, ha="left", va="top")
    ax.set_xlabel("Model")
    ax.set_ylabel("Pressure hist. error [%]")
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(model) for model in models], rotation=45, ha="right")
    ax.grid(axis="y")

    boundaries: list[tuple[str, list[str], object, int, int, float]] = []
    start = 0
    for tier_label, tier_models, color in TIER_DEFS:
        tier_df = df[df["model"].isin(tier_models)]
        count = len(tier_df)
        if count:
            median = float(tier_df["pressure_error_percent"].median())
            boundaries.append((tier_label, tier_models, color, start, start + count - 1, median))
            start += count

    print(
        "Pressure error medians -> "
        + ", ".join(f"{label}: {format_error_value(median)}" for label, _, _, _, _, median in boundaries)
    )

    metric_ymax = max(float(np.max(values)), max(median for _, _, _, _, _, median in boundaries))
    ax.set_ylim(0, metric_ymax * 1.35)
    tier_text_y = metric_ymax * 1.28
    median_text_y = metric_ymax * 1.08

    for idx, (label, _, color, first, last, median) in enumerate(boundaries):
        center = (first + last) / 2
        if idx < len(boundaries) - 1:
            ax.axvline(last + 0.5, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(center, tier_text_y, label, ha="center", va="top", color=color)
        ax.hlines(median, first - 0.5, last + 0.5, colors=color, linestyles="--", linewidth=2)
        annotate_median(ax, center, median, median_text_y, color)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))


def plot_combined(
    pressures_dir: Path,
    reference_file: Path,
    ranking_df: pd.DataFrame,
    bins: int,
    output: Path,
) -> None:
    panels = collect_histogram_panels(pressures_dir, reference_file, bins)
    if len(panels) > 4:
        panels = panels[:4]

    n_hist_cols = 2
    n_hist_rows = int(np.ceil(len(panels) / n_hist_cols))
    fig = plt.figure(figsize=(3.53 * 3.0, 3.53 * 3.55))
    outer_gs = gridspec.GridSpec(
        nrows=n_hist_rows + 2,
        ncols=n_hist_cols,
        figure=fig,
        height_ratios=[0.78, 0.22] + [1.15] * n_hist_rows,
        hspace=0.48,
        wspace=0.30,
    )
    panel_labels = [f"({chr(97 + idx)})" for idx in range(len(panels) + 1)]

    overall_ax = fig.add_subplot(outer_gs[0, :])
    draw_overall_pressure_error_plot(overall_ax, ranking_df, panel_labels[0])

    for idx, (system_type, system, reference_values, values_by_model, model_scores, edges) in enumerate(panels):
        row = idx // n_hist_cols + 2
        col = idx % n_hist_cols
        bottom_row = idx // n_hist_cols == n_hist_rows - 1
        sub_gs = gridspec.GridSpecFromSubplotSpec(
            nrows=len(TIER_DEFS),
            ncols=1,
            subplot_spec=outer_gs[row, col],
            hspace=0.05,
        )

        panel_ax = fig.add_subplot(outer_gs[row, col], frame_on=False)
        panel_ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
        panel_ax.grid(False)
        panel_ax.set_ylabel("Density" if col == 0 else "", labelpad=8)
        panel_ax.set_xlabel("Pressure [GPa]" if bottom_row else "", labelpad=18)

        for tier_idx, (tier_label, tier_models, tier_color) in enumerate(TIER_DEFS):
            ax = fig.add_subplot(sub_gs[tier_idx])
            if tier_idx == 0:
                ax.set_title(f"{system_type}\n{format_system_name(system)}")
                ax.text(
                    0.02,
                    0.95,
                    panel_labels[idx + 1],
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                )

            best, worst, scores = choose_best_worst(tier_models, model_scores)
            ax.hist(
                reference_values,
                bins=edges,
                density=True,
                histtype="step",
                color="black",
                linewidth=1.5,
                label="Reference",
            )
            if best is not None:
                prefix = "Best/worst" if best == worst else "Best"
                ax.hist(
                    values_by_model[best],
                    bins=edges,
                    density=True,
                    histtype="step",
                    color=tier_color,
                    linewidth=1.5,
                    label=format_model_label(best, scores.get(best), prefix),
                )
            if worst is not None and worst != best:
                ax.hist(
                    values_by_model[worst],
                    bins=edges,
                    density=True,
                    histtype="step",
                    color=tier_color,
                    linewidth=1.5,
                    linestyle="--",
                    label=format_model_label(worst, scores.get(worst), "Worst"),
                )

            if col == n_hist_cols - 1:
                ax.set_ylabel(tier_label, labelpad=2)
                ax.yaxis.set_label_position("right")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            if tier_idx != len(TIER_DEFS) - 1:
                ax.tick_params(labelbottom=False)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                handles.append(Line2D([], [], color="none", linestyle="none", linewidth=0))
                labels.append(f"{tier_label} mean error: {format_error_value(mean_finite(scores.values()))}")
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
            ax.yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=1, prune="both"))
            ax.tick_params(axis="y", labelsize=FONT_SIZE, pad=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.show()
    plt.close(fig)
    print(f"Saved combined pressure panel plot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a combined pressure percentage-error figure for same-simulation-length data."
    )
    parser.add_argument(
        "--pressures-dir",
        type=Path,
        default=DEFAULT_PRESSURES_DIR,
        help="Directory with model per-frame pressure CSV files.",
    )
    parser.add_argument(
        "--reference-file",
        default=None,
        help="Optional explicit path to the reference per-frame pressure CSV.",
    )
    parser.add_argument(
        "--ranking-file",
        type=Path,
        default=DEFAULT_RANKING_FILE,
        help="CSV with model-level regular pressure error percentages.",
    )
    parser.add_argument("--bins", type=int, default=80, help="Number of shared histogram bins.")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output plot file path.",
    )
    args = parser.parse_args()

    if not args.pressures_dir.is_dir():
        raise NotADirectoryError(f"Pressures directory not found: {args.pressures_dir}")
    if not args.ranking_file.is_file():
        raise FileNotFoundError(f"Ranking CSV not found: {args.ranking_file}")
    if args.bins < 2:
        raise ValueError("--bins must be >= 2")

    reference_file = find_reference_file(args.pressures_dir, args.reference_file)
    ranking_df = pd.read_csv(args.ranking_file)
    plot_combined(args.pressures_dir, reference_file, ranking_df, args.bins, args.output_file)


if __name__ == "__main__":
    main()
