#!/usr/bin/env python3
"""Plot Pareto front for RDF error vs force evaluation time per atom.

Objective:
- minimize RDF error [%]
- minimize mean force evaluation time per atom [s]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


try:
    from adjustText import adjust_text
except Exception:
    adjust_text = None

FONT_SIZE = 6

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

palette = sns.color_palette("deep")
CALCULATOR_DISPLAY_NAMES = {
    "chgnet": "CHGNet",
    "mace-mp-0": "MACE-MP-0",
    "grace-mp": "GRACE-2L-MPtrj",
    "mace-mpa-0": "MACE-MPA-0",
    "orb-v2": "orb-v2",
    "eq-v2-m-omat": "EquiformerV2",
    "mattersim-v1-5m": "MatterSim-v1.0.0-5M",
    "orb-v3": "orb-v3-conservative-inf-mpa",
    "grace-oam": "GRACE-2L-OAM",
    "nequip": "NequIP-OAM-XL",
    "pet-oam-xl": "PET-OAM-XL",
    "esen-30m-oam": "eSEN-30M-OAM",
    "mace-mh-omat": "MACE-MH-1-OMAT",
    "uma-s-omat": "UMA-S-P1",
    "uma-m-omat": "UMA-M-P1",
}


def normalize_model_name(name: str) -> str:
    return str(name).strip().lower()


def display_name(model: str) -> str:
    normalized = normalize_model_name(model)
    return CALCULATOR_DISPLAY_NAMES.get(normalized, model)


TIER_1 = ["chgnet", "mace-mp-0", "grace-mp"]
TIER_2 = ["mace-mpa-0", "orb-v2"]
TIER_3 = ["mattersim-v1-5m", "grace-oam", "orb-v3", "esen-30m-oam", "nequip", "eq-v2-m-omat", "pet-oam-xl"]
TIER_4 = ["mace-mh-omat", "uma-s-omat", "uma-m-omat"]

TIER_COLORS = {
    "Tier 1": palette[2],
    "Tier 2": palette[1],
    "Tier 3": palette[3],
    "Tier 4": palette[0],
    "Other": "#757575",
}

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RMSE_METRICS_FILE = SCRIPT_DIR.parent / "e_f_rmses" / "results" / "mean_metrics_by_model.csv"
DEFAULT_RDF_SCORES_FILE = SCRIPT_DIR / "results" / "rdf_similarity_scores_same_simulation_length.csv"
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "plots" / "plot_SI_pareto_rdf_time_same_length.pdf"


def load_and_merge(rmse_file: Path, rdf_file: Path) -> pd.DataFrame:
    rmse_df = pd.read_csv(rmse_file)
    rdf_df = pd.read_csv(rdf_file)

    rmse_needed = {"calculator", "mean_force_eval_time_per_atom_s"}
    rdf_needed = {"Calculator"}

    missing_rmse = rmse_needed - set(rmse_df.columns)
    missing_rdf = rdf_needed - set(rdf_df.columns)

    if missing_rmse:
        raise ValueError(f"Missing columns in RMSE metrics file: {sorted(missing_rmse)}")
    if missing_rdf:
        raise ValueError(f"Missing columns in RDF scores file: {sorted(missing_rdf)}")

    rmse_small = rmse_df[["calculator", "mean_force_eval_time_per_atom_s"]].copy()
    rdf_small = rdf_df[["Calculator"]].copy()
    if "Mean RDF Error [%]" in rdf_df.columns:
        rdf_small["RDF Error [%]"] = pd.to_numeric(rdf_df["Mean RDF Error [%]"], errors="coerce")
    elif "Mean Similarity Score [%]" in rdf_df.columns:
        rdf_small["RDF Error [%]"] = 100.0 - pd.to_numeric(
            rdf_df["Mean Similarity Score [%]"], errors="coerce"
        )
    else:
        raise ValueError("RDF scores file must contain Mean RDF Error [%]")

    rmse_small["model"] = rmse_small["calculator"].map(normalize_model_name)
    rdf_small["model"] = rdf_small["Calculator"].map(normalize_model_name)

    merged = pd.merge(
        rmse_small[["model", "mean_force_eval_time_per_atom_s"]],
        rdf_small[["model", "RDF Error [%]"]],
        on="model",
        how="inner",
    )

    merged["mean_force_eval_time_per_atom_s"] = 1000 * pd.to_numeric(
        merged["mean_force_eval_time_per_atom_s"], errors="coerce"
    )
    merged = merged.dropna(subset=["mean_force_eval_time_per_atom_s", "RDF Error [%]"]).copy()

    if merged.empty:
        raise ValueError("No overlapping models between RMSE metrics and RDF scores files.")

    return merged


def is_pareto_optimal(df: pd.DataFrame) -> np.ndarray:
    """Return mask for non-dominated points.

    Dominance definition for objectives (min time, min RDF error):
    i is dominated by j if:
      time_j <= time_i and err_j <= err_i and at least one is strict.
    """
    times = df["mean_force_eval_time_per_atom_s"].to_numpy()
    rdf_errors = df["RDF Error [%]"].to_numpy()
    n = len(df)

    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_or_equal_time = times[j] <= times[i]
            better_or_equal_rdf_error = rdf_errors[j] <= rdf_errors[i]
            strictly_better_one = (times[j] < times[i]) or (rdf_errors[j] < rdf_errors[i])
            if better_or_equal_time and better_or_equal_rdf_error and strictly_better_one:
                pareto[i] = False
                break

    return pareto


def model_tier(model_name: str) -> str:
    model_name = normalize_model_name(model_name)
    if model_name in TIER_1:
        return "Tier 1"
    if model_name in TIER_2:
        return "Tier 2"
    if model_name in TIER_3:
        return "Tier 3"
    if model_name in TIER_4:
        return "Tier 4"
    return "Other"


def plot_pareto(df: pd.DataFrame, output_file: Path) -> None:
    pareto_mask = is_pareto_optimal(df)

    all_df = df.copy()
    all_df["tier"] = all_df["model"].map(model_tier)
    pareto_df = df[pareto_mask].copy()
    pareto_df = pareto_df.sort_values("mean_force_eval_time_per_atom_s")
    pareto_df["tier"] = pareto_df["model"].map(model_tier)

    fig, ax = plt.subplots(figsize=(3.53 * 1.5, 3.53 * 1.5))

    for tier_name in ["Tier 1", "Tier 2", "Tier 3", "Tier 4", "Other"]:
        tier_df = all_df[all_df["tier"] == tier_name]
        if tier_df.empty:
            continue
        ax.scatter(
            tier_df["mean_force_eval_time_per_atom_s"],
            tier_df["RDF Error [%]"],
            color=TIER_COLORS[tier_name],
            alpha=0.9,
            s=26,
            label=tier_name,
            zorder=2,
        )

    ax.scatter(
        pareto_df["mean_force_eval_time_per_atom_s"],
        pareto_df["RDF Error [%]"],
        facecolors="none",
        edgecolors="black",
        alpha=0.95,
        s=58,
        linewidths=1.0,
        label="Pareto-optimal",
        zorder=4,
    )

    ax.plot(
        pareto_df["mean_force_eval_time_per_atom_s"],
        pareto_df["RDF Error [%]"],
        color="black",
        linewidth=1.2,
        alpha=0.9,
        zorder=3,
    )

    # Annotate model names with a light/faded style and optional overlap adjustment
    label_texts = []
    x_vals = all_df["mean_force_eval_time_per_atom_s"].to_numpy()
    y_vals = all_df["RDF Error [%]"].to_numpy()
    for xi, yi, m in zip(x_vals, y_vals, all_df["model"].values):
        txt = ax.annotate(
            display_name(m),
            xy=(xi, yi),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=FONT_SIZE,
            alpha=0.8,
            ha="left",
            va="bottom",
            rotation=0,
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.55),
        )
        label_texts.append(txt)

    if adjust_text is not None and label_texts:
        try:
            adjust_text(
                label_texts,
                ax=ax,
                x=x_vals,
                y=y_vals,
                avoid_self=True,
                only_move={"points": "xy", "text": "xy"},
                force_text=(1.2, 1.4),
                force_points=(0.8, 1.0),
                expand_points=(1.3, 1.4),
                expand_text=(1.2, 1.3),
                lim=400,
                arrowprops=dict(arrowstyle='-', color='0.5', lw=0.4, alpha=0.45),
            )
        except Exception:
            pass

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("RDF error [%]")
    # ax.set_title("Pareto Front: RDF Error vs Force Eval Time")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=True)

    ax.set_xlim(right=4.4)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0.02)
    plt.show()
    plt.close(fig)

    print(f"Saved: {output_file}")
    print(f"Models plotted: {len(all_df)}")
    print(f"Pareto-optimal models: {len(pareto_df)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Pareto front of RDF error vs force evaluation time per atom."
    )
    parser.add_argument(
        "--rmse-metrics-file",
        default=str(DEFAULT_RMSE_METRICS_FILE),
        help="CSV with model-level mean_force_eval_time_per_atom_s.",
    )
    parser.add_argument(
        "--rdf-scores-file",
        default=str(DEFAULT_RDF_SCORES_FILE),
        help="CSV with model-level Mean RDF Error [%].",
    )
    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output plot path.",
    )
    args = parser.parse_args()

    merged = load_and_merge(Path(args.rmse_metrics_file), Path(args.rdf_scores_file))
    plot_pareto(merged, Path(args.output_file))


if __name__ == "__main__":
    main()
