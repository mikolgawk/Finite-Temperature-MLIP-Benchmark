#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib>=3.9",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "seaborn>=0.13",
# ]
# ///
"""Combined figure for plot-9 and plot-10 same-simulation-length views.

The top section reproduces per-system-type stacked VDOS comparisons from
plot-10-same-simulation-length.py. The bottom row adds the global VDOS-error
bar plot with tier medians from plot-9-same-simulation-length.py.

Requires: numpy, pandas, matplotlib, seaborn
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

import sys

PAPER_V2_CONFIG_DIR = Path(__file__).resolve().parents[1]
if str(PAPER_V2_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_V2_CONFIG_DIR))
from model_display_names import MODEL_DISPLAY_NAMES, display_model_name


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = "mlip-trajs-torchsim-eager"
SOURCES = (
    "mlip-trajs-ase",
    "mlip-trajs-torchsim-eager",
    "mlip-trajs-ase-accelerated",
    "mlip-trajs-torchsim-accelerated",
)

CM_INV_TO_EV = 1.2398419843320026e-4
FONT_SIZE = 10
LEGEND_FONT_SIZE = 6

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


REF_COLOR = "black"
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

TIER_PANELS = [
    ("Tier 1", TIER_1_MODELS, palette[2]),
    ("Tier 2", TIER_2_MODELS, palette[1]),
    ("Tier 3", TIER_3_MODELS, palette[3]),
    ("Tier 4", TIER_4_MODELS, palette[0]),
]

MODEL_ORDER = [model for _, models, _ in TIER_PANELS for model in models]
ALLOWED_MODELS = set(MODEL_ORDER)
CANONICAL_MODEL_BY_LOWER = {model.strip().lower(): model for model in MODEL_ORDER}

SYSTEM_TYPES = [
    "Pure metals",
    "Perovskites",
    "Metal dichalcogenides",
    "Metal alloys",
    "Molecular crystals",
    "Metal-water interfaces",
    "Hydrogen",
]

SYSTEMS_BY_TYPE = {
    "Pure metals": [
        "bulkAu_1500K_Kapil",
        "bulkAg_600K_Kapil",
        "bulkCu_1000K_Kapil",
    ],
    "Perovskites": ["MAPbBr3_300K_Ivor_VASP", "CsSnI3_500K_Ivor_VASP"],
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
        "naphthalene_295K_Sharma_S",
        "pentacene_295K_Sharma_S",
        "picene_295K_Sharma_S",
        "tetracene_295K_Sharma_S",
        "anthracene_293K_Sharma_S",
    ],
    "Metal-water interfaces": ["Pt111w24H2O_380K_Heenen_VASP"],
    "Hydrogen": ["H_1050K_Rupp_QE"],
}

DEFAULT_XLIM_BY_SYSTEM_TYPE = {
    "Pure metals": (0.0, 0.03),
    "Perovskites": (0.0, 0.02),
    "Metal dichalcogenides": (0.0, 0.06),
    "Metal alloys": (0.0, 0.04),
    "Molecular crystals": (0.0, 0.5),
    "Metal-water interfaces": (0.0, 0.2),
    "Hydrogen": (0.0, 0.7),
}

DEFAULT_YLIM_BY_SYSTEM_TYPE = {
    "Perovskites": (0.0, 240.0),
    "Metal dichalcogenides": (0.0, 290.0),
    "Metal alloys": (0.0, 140.0),
    "Molecular crystals": (0.0, 60.0),
}


def trapz_integral(y: np.ndarray, x: np.ndarray) -> float:
    trapz_fn = getattr(np, "trapezoid", None)
    if trapz_fn is None:
        trapz_fn = np.trapz
    return float(trapz_fn(y, x=x))


def infer_system_type(system: str) -> str:
    s = system.lower()
    if s.startswith("bulkau") or s.startswith("bulkag") or s.startswith("bulkcu_"):
        return "Pure metals"
    if s.startswith("cssni3") or s.startswith("mapbbr3"):
        return "Perovskites"
    if s.startswith("bulkmos2") or s.startswith("tise2"):
        return "Metal dichalcogenides"
    if (
        s.startswith("bulkcuau")
        or s.startswith("bulkcuzral")
        or s.startswith("bulklimgalznsn")
        or s.startswith("bulkpt3co")
    ):
        return "Metal alloys"
    if (
        s.startswith("anthracene")
        or s.startswith("naphthalene")
        or s.startswith("pentacene")
        or s.startswith("picene")
        or s.startswith("tetracene")
    ):
        return "Molecular crystals"
    if s.startswith("pt111w24h2o"):
        return "Metal-water interfaces"
    if s.startswith("h_1050k"):
        return "Hydrogen"
    return "other"


def load_vdos(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"VDOS spectrum not found: {path}")
    with path.open(encoding="utf-8") as handle:
        first_line = handle.readline()
    if "," in first_line:
        has_header = any(character.isalpha() for character in first_line)
        arr = np.loadtxt(path, delimiter=",", skiprows=int(has_header))
    else:
        arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        raise ValueError(f"File has fewer than 2 columns: {path}")

    x_cm = np.asarray(arr[:, 0], dtype=float)
    # Pipeline spectra contain wavenumber, energy, raw intensity, and
    # normalized intensity. Legacy spectra contain wavenumber and intensity.
    intensity_col = 3 if arr.shape[1] >= 4 else (2 if arr.shape[1] >= 3 else 1)
    y = np.asarray(arr[:, intensity_col], dtype=float)

    keep = np.isfinite(x_cm) & np.isfinite(y) & (x_cm >= 0.0)
    x_cm = x_cm[keep]
    y = y[keep]
    if x_cm.size < 2:
        raise ValueError(f"Not enough valid points in {path}")

    x_ev = x_cm * CM_INV_TO_EV
    order = np.argsort(x_ev)
    x_ev = x_ev[order]
    y = y[order]

    x_ev_unique, idx = np.unique(x_ev, return_index=True)
    return x_ev_unique, y[idx]


def normalize_area(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    area = trapz_integral(y, x)
    if area <= 0.0 or not np.isfinite(area):
        return y
    return y / area


def format_chemical_subscripts(name: str) -> str:
    name = re.sub(
        r"Pt(\d{3})w(\d+)H2O",
        r"Pt(\1) + \2 H$_{2}$O",
        name,
    )
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
            suffix += ""
        suffix += ")"
        return f"{base_name}{suffix}"

    if no_vdw:
        return f"{base_name}"
    return base_name


def choose_representative_system(subset: pd.DataFrame, system_type: str) -> str:
    available_systems = set(subset["system"].astype(str).tolist())
    preferred_systems = SYSTEMS_BY_TYPE.get(system_type, [])
    for system in preferred_systems:
        if system in available_systems:
            return system
    return sorted(available_systems)[0]


def pick_best_worst_models_for_tier(
    subset: pd.DataFrame, tier_models: list[str]
) -> tuple[str | None, str | None]:
    tier_subset = subset[subset["mlip_model"].astype(str).isin(tier_models)]
    if tier_subset.empty:
        return None, None

    tier_subset = tier_subset.copy()
    model_means = (
        tier_subset.groupby("mlip_model", as_index=False)["vdos_error_percent"]
        .mean()
        .sort_values(["vdos_error_percent", "mlip_model"])
        .reset_index(drop=True)
    )
    best_model = str(model_means.iloc[0]["mlip_model"])
    worst_model = str(model_means.iloc[-1]["mlip_model"])
    return best_model, worst_model


def pick_model_row(
    subset: pd.DataFrame,
    system: str,
    model: str,
    prefer_low_error: bool,
) -> pd.Series | None:
    rows = subset[
        (subset["system"].astype(str) == system)
        & (subset["mlip_model"].astype(str) == model)
    ]
    if rows.empty:
        return None
    return rows.sort_values("vdos_error_percent", ascending=prefer_low_error).iloc[0]


def pick_reference_row(subset: pd.DataFrame, system: str) -> pd.Series | None:
    rows = subset[subset["system"].astype(str) == system]
    if rows.empty:
        return None
    rows = rows[
        rows["ref_file"].map(
            lambda value: bool(str(value).strip()) and Path(str(value)).is_file()
        )
    ]
    if rows.empty:
        return None
    return rows.sort_values("vdos_error_percent").iloc[0]


def full_penalty_models(
    subset: pd.DataFrame, tier_models: list[str]
) -> list[str]:
    """Return tier models assigned the explicit 100% no-data penalty."""
    tier_subset = subset[subset["mlip_model"].isin(tier_models)].copy()
    if tier_subset.empty:
        return []
    errors = tier_subset.groupby("mlip_model")["vdos_error_percent"].mean()
    return sorted(
        str(model)
        for model, error in errors.items()
        if np.isfinite(error) and np.isclose(float(error), 100.0)
    )


def error_to_percent(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return values
    if float(values.max()) <= 1.5:
        return values * 100.0
    return values


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number):
        return None
    return number


def format_model_label_with_vdos_error(model_name: str, error_value: object) -> str:
    error_percent = finite_float(error_value)
    display = display_name(model_name)
    if error_percent is None:
        return display

    return f"{display} ({error_percent:.1f}%)"


def mean_system_vdos_error(system_subset: pd.DataFrame) -> float:
    errors = pd.to_numeric(system_subset["vdos_error_percent"], errors="coerce")
    errors = errors[np.isfinite(errors)]
    if errors.empty:
        return float("nan")
    return float(errors.mean())


def select_system_with_worst_mean_vdos_error(
    subset: pd.DataFrame, system_type: str
) -> str:
    worst_choice: tuple[float, str] | None = None
    for system, system_subset in subset.groupby("system", sort=False):
        mean_error = mean_system_vdos_error(system_subset)
        if not np.isfinite(mean_error):
            continue
        if worst_choice is None or mean_error > worst_choice[0]:
            worst_choice = (mean_error, str(system))

    if worst_choice is None:
        return choose_representative_system(subset, system_type)

    mean_error, system = worst_choice
    print(
        f"[INFO] Selected {system} for {system_type}: "
        f"mean model VDOS error = {mean_error:.1f}%."
    )
    return system


def model_mean_vdos_error_by_category(subset: pd.DataFrame) -> dict[str, float]:
    df = subset[["mlip_model", "vdos_error_percent"]].copy()
    df = df.dropna(subset=["mlip_model", "vdos_error_percent"])
    if df.empty:
        return {}
    return {
        str(model): float(value)
        for model, value in df.groupby("mlip_model")["vdos_error_percent"]
        .mean()
        .items()
        if np.isfinite(value)
    }


def tier_mean_vdos_error(
    tier_models: list[str], model_means: dict[str, float]
) -> float:
    errors = []
    for model in tier_models:
        score = model_means.get(model, np.nan)
        if np.isfinite(score):
            errors.append(float(score))
    if not errors:
        return float("nan")
    return float(np.mean(errors))


def annotate_median(
    ax: plt.Axes,
    x_center: float,
    y_value: float,
    y_text: float,
    color: str,
    fmt: str = "{:.2f}",
) -> None:
    ax.annotate(
        f"{fmt.format(y_value)}%",
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


def tier_center(start_idx: int, end_idx: int) -> float:
    return (start_idx + end_idx) / 2


def format_tier_median_summary(
    tier_medians: list[tuple[str, int, int, float, str]],
) -> str:
    return ", ".join(
        f"{tier_name}: {median_val:.2f}%"
        for tier_name, _, _, median_val, _ in tier_medians
    )


def canonicalize_model_name(value: object) -> str | None:
    if value is None:
        return None
    normalized = normalize_model_name(str(value))
    if not normalized:
        return None
    return CANONICAL_MODEL_BY_LOWER.get(normalized)


def load_model_means(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"VDOS model means file not found: {path}")
    df = pd.read_csv(path)
    model_col = next(
        (
            column
            for column in ("model", "mlip_model", "Calculator", "calculator")
            if column in df.columns
        ),
        None,
    )
    error_col = next(
        (
            column
            for column in (
                "vdos_error_percent",
                "Mean VDOS Error [%]",
                "VDOS Error [%]",
                "final_mean_vdos_error",
                "mean_vdos_error",
                "vdos_error",
            )
            if column in df.columns
        ),
        None,
    )
    if model_col is None or error_col is None:
        raise ValueError(
            f"VDOS model means file {path} needs a model column and a VDOS error column."
        )

    errors = pd.to_numeric(df[error_col], errors="coerce")
    if "percent" not in error_col.lower() and "[%]" not in error_col:
        errors = error_to_percent(errors)
    out = pd.DataFrame(
        {
            "mlip_model": df[model_col],
            "vdos_error_percent": errors,
        }
    )
    out["mlip_model"] = out["mlip_model"].astype(str)
    excluded_models_lower = {model.lower() for model in EXCLUDED_MODELS}
    out = out[
        ~out["mlip_model"].map(normalize_model_name).isin(excluded_models_lower)
    ].copy()
    out["mlip_model"] = out["mlip_model"].map(canonicalize_model_name)
    out = out.dropna(subset=["mlip_model"])
    out = out[out["mlip_model"].isin(ALLOWED_MODELS)].copy()
    out = out.dropna(subset=["vdos_error_percent"])
    out = out.groupby("mlip_model", as_index=False)["vdos_error_percent"].mean()
    if out.empty:
        raise ValueError("No valid model mean values found after excluding models.")

    order_map = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    out["_model_order"] = out["mlip_model"].map(order_map)
    out["_orig_order"] = np.arange(len(out))
    out["_model_order"] = out["_model_order"].fillna(
        len(MODEL_ORDER) + out["_orig_order"]
    )
    out = out.sort_values(["_model_order", "_orig_order"]).drop(
        columns=["_model_order", "_orig_order"]
    )
    return out.reset_index(drop=True)


def draw_overall_vdos_error_plot(
    ax: plt.Axes, model_means_df: pd.DataFrame, panel_label: str
) -> None:
    df = model_means_df.copy()

    selected_rows = []
    selected_labels: list[str] = []
    selected_colors: list[str] = []
    tier_medians: list[tuple[str, int, int, float, str]] = []
    tier_label_ranges: list[tuple[str, int, int, str]] = []
    current_pos = 0

    for tier_name, tier_models, tier_color in TIER_PANELS:
        tier_sub = df[df["mlip_model"].isin(tier_models)].copy()
        if tier_sub.empty:
            continue

        tier_sub["_tier_order"] = pd.Categorical(
            tier_sub["mlip_model"], categories=tier_models, ordered=True
        )
        tier_sub = (
            tier_sub.sort_values("_tier_order")
            .drop(columns=["_tier_order"])
            .reset_index(drop=True)
        )

        selected_rows.extend([row for _, row in tier_sub.iterrows()])
        selected_labels.extend(
            [display_name(str(model)) for model in tier_sub["mlip_model"]]
        )
        selected_colors.extend([tier_color] * len(tier_sub))

        tier_median = float(tier_sub["vdos_error_percent"].median())
        tier_start = current_pos
        tier_end = current_pos + len(tier_sub) - 1
        tier_medians.append((tier_name, tier_start, tier_end, tier_median, tier_color))
        tier_label_ranges.append((tier_name, tier_start, tier_end, tier_color))
        current_pos += len(tier_sub)

    plot_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    if plot_df.empty:
        raise ValueError("No model data available for overall VDOS error panel.")

    x = np.arange(len(plot_df))

    ax.bar(
        x,
        plot_df["vdos_error_percent"],
        color=selected_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.text(
        0.01,
        0.95,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZE,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("VDOS error [%]")
    ax.set_xticks(x)
    ax.set_xticklabels(selected_labels, rotation=45, ha="right", fontsize=FONT_SIZE)
    ax.grid(axis="y")
    data_ymax = float(plot_df["vdos_error_percent"].max())
    ax.set_ylim(0.0, max(1.0, data_ymax * 1.20))

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min if y_max > y_min else 1.0
    median_label_y = y_min + 0.81 * y_range
    tier_label_y = y_min + 0.91 * y_range
    max_label_y = y_max

    for _, start_idx, end_idx, median_val, line_color in tier_medians:
        ax.hlines(
            y=median_val,
            xmin=start_idx - 0.35,
            xmax=end_idx + 0.35,
            colors=line_color,
            linestyles="--",
            linewidth=2,
            alpha=0.9,
            zorder=5,
        )
        annotate_median(
            ax,
            tier_center(start_idx, end_idx),
            median_val,
            median_label_y,
            line_color,
        )
        max_label_y = max(max_label_y, median_label_y)

    if max_label_y > y_max:
        ax.set_ylim(y_min, max_label_y)

    for _, _, end_idx, _, _ in tier_medians[:-1]:
        ax.axvline(
            x=end_idx + 0.5, color="black", linestyle="--", linewidth=1.5, alpha=0.7
        )

    for tier_name, start_idx, end_idx, tier_color in tier_label_ranges:
        ax.text(
            tier_center(start_idx, end_idx),
            tier_label_y,
            tier_name,
            ha="center",
            fontsize=FONT_SIZE,
            color=tier_color,
        )

    if tier_medians:
        print(f"VDOS Error medians [%] -> {format_tier_median_summary(tier_medians)}")


def load_normalized_pairs(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"VDOS pair table not found: {path}")
    df = pd.read_csv(path)
    columns = set(df.columns)
    model_col = next((col for col in ("model", "mlip_model") if col in columns), None)
    error_col = next(
        (col for col in ("vdos_error_percent", "vdos_error") if col in columns),
        None,
    )
    mlip_col = next(
        (col for col in ("mlip_spectrum", "mlip_file") if col in columns), None
    )
    ref_col = next(
        (col for col in ("reference_spectrum", "ref_file") if col in columns), None
    )
    if "system" not in columns or None in (model_col, error_col, mlip_col, ref_col):
        raise ValueError(
            f"VDOS pair table {path} needs system/model/error and reference/MLIP spectrum columns."
        )

    errors = pd.to_numeric(df[error_col], errors="coerce")
    if error_col != "vdos_error_percent":
        errors = error_to_percent(errors)
    out = pd.DataFrame(
        {
            "system": df["system"],
            "mlip_model": df[model_col],
            "vdos_error_percent": errors,
            "mlip_file": df[mlip_col],
            "ref_file": df[ref_col],
        }
    )
    if "system_type" in df.columns:
        out["system_type"] = df["system_type"]
    out["system"] = out["system"].astype(str)
    out["mlip_model"] = out["mlip_model"].astype(str)
    excluded_models_lower = {model.lower() for model in EXCLUDED_MODELS}
    out = out[
        ~out["mlip_model"].str.strip().str.lower().isin(excluded_models_lower)
    ].copy()
    out["mlip_model"] = out["mlip_model"].map(canonicalize_model_name)
    out = out.dropna(subset=["mlip_model"])
    out = out[out["mlip_model"].isin(ALLOWED_MODELS)].copy()
    # Keep explicit 100% penalty rows even though they intentionally have no
    # spectrum paths. The panel legend reports them without drawing a curve.
    out = out.dropna(subset=["vdos_error_percent"])
    out["mlip_file"] = out["mlip_file"].fillna("").astype(str)
    out["ref_file"] = out["ref_file"].fillna("").astype(str)
    if out.empty:
        raise ValueError(
            "No valid normalized pairs found for models defined in the tier lists."
        )
    if "system_type" not in out.columns:
        out["system_type"] = out["system"].map(infer_system_type)
    else:
        inferred_types = out["system"].map(infer_system_type)
        out["system_type"] = out["system_type"].where(
            out["system_type"].isin(SYSTEM_TYPES), inferred_types
        )
    return out


def plot_combined(
    norm_df: pd.DataFrame, model_means_df: pd.DataFrame, output_file: Path
) -> None:
    available_system_types = [
        system_type
        for system_type in SYSTEM_TYPES
        if not norm_df[norm_df["system_type"] == system_type].empty
    ]
    if not available_system_types:
        raise ValueError(
            "No recognized system-type data found in normalized pairs file."
        )

    n_system_types = len(available_system_types)
    n_cols = 3
    n_top_rows = int(np.ceil(n_system_types / n_cols))
    center_last_two = n_top_rows > 1 and (n_system_types % n_cols == 2)
    outer_wspace = 0.45
    bar_system_gap_ratio = 0.22

    fig = plt.figure(figsize=(3.53 * 3.0, 3.53 * (1.55 * n_top_rows + 0.65)))
    panel_slots: list[tuple[int, slice | int, int, int]] = []
    if center_last_two:
        outer_gs = gridspec.GridSpec(
            n_top_rows + 2,
            n_cols * 2,
            figure=fig,
            wspace=outer_wspace,
            hspace=0.48,
            height_ratios=[0.78, bar_system_gap_ratio] + [1.15] * n_top_rows,
        )
        for idx in range(n_system_types):
            system_row_idx = idx // n_cols
            row_idx = system_row_idx + 2
            logical_col = idx % n_cols
            items_in_row = min(n_cols, n_system_types - system_row_idx * n_cols)
            if system_row_idx == n_top_rows - 1 and items_in_row == 2:
                col_start = 1 + logical_col * 2
            else:
                col_start = logical_col * 2
            panel_slots.append(
                (row_idx, slice(col_start, col_start + 2), logical_col, items_in_row)
            )
    else:
        outer_gs = gridspec.GridSpec(
            n_top_rows + 2,
            n_cols,
            figure=fig,
            wspace=outer_wspace,
            hspace=0.48,
            height_ratios=[0.78, bar_system_gap_ratio] + [1.15] * n_top_rows,
        )
        for idx in range(n_system_types):
            system_row_idx = idx // n_cols
            row_idx = system_row_idx + 2
            logical_col = idx % n_cols
            items_in_row = min(n_cols, n_system_types - system_row_idx * n_cols)
            panel_slots.append((row_idx, logical_col, logical_col, items_in_row))

    panel_labels = [f"({chr(97 + idx)})" for idx in range(n_system_types + 1)]

    overall_ax = fig.add_subplot(outer_gs[0, :])
    draw_overall_vdos_error_plot(overall_ax, model_means_df, panel_labels[0])

    for idx, system_type in enumerate(available_system_types):
        row_idx, col_sel, logical_col, items_in_row = panel_slots[idx]

        is_left_col = logical_col == 0
        is_right_col = logical_col == items_in_row - 1

        subset = norm_df[norm_df["system_type"] == system_type].copy()
        representative_system = select_system_with_worst_mean_vdos_error(
            subset, system_type
        )
        representative_subset = subset[
            subset["system"].astype(str) == representative_system
        ].copy()
        system_display_name = format_system_name(representative_system)
        model_means = model_mean_vdos_error_by_category(subset)

        sub_gs = outer_gs[row_idx, col_sel].subgridspec(
            len(TIER_PANELS), 1, hspace=0.05
        )
        tier_axes = [
            fig.add_subplot(sub_gs[tier_idx]) for tier_idx in range(len(TIER_PANELS))
        ]

        ref_row = pick_reference_row(representative_subset, representative_system)
        if ref_row is None:
            for ax in tier_axes:
                ax.text(
                    0.5,
                    0.5,
                    "No reference data",
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZE,
                )
                ax.axis("off")
            continue

        ref_path = Path(str(ref_row["ref_file"]))
        try:
            x_ref, y_ref = load_vdos(ref_path)
            ref_n = normalize_area(x_ref, y_ref)
        except Exception as exc:
            for ax in tier_axes:
                ax.text(
                    0.5,
                    0.5,
                    f"Load failed:\n{exc}",
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZE,
                )
                ax.axis("off")
            continue

        for tier_idx, (tier_label, tier_models, tier_color) in enumerate(TIER_PANELS):
            ax = tier_axes[tier_idx]

            if tier_idx == 0:
                ax.set_title(f"{system_type}\n{system_display_name}")
                ax.text(
                    0.02,
                    0.95,
                    panel_labels[idx + 1],
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=FONT_SIZE,
                )

            tier_mean_error = tier_mean_vdos_error(tier_models, model_means)
            best_model, worst_model = pick_best_worst_models_for_tier(
                representative_subset, tier_models
            )
            plotted_models: set[str] = set()

            ax.plot(
                x_ref,
                ref_n,
                label="Reference",
                color=REF_COLOR,
                linewidth=2.0,
                linestyle="-",
            )

            if best_model is not None:
                best_row = pick_model_row(
                    subset=representative_subset,
                    system=representative_system,
                    model=best_model,
                    prefer_low_error=True,
                )
                if best_row is not None:
                    try:
                        x_best, y_best = load_vdos(Path(str(best_row["mlip_file"])))
                        ax.plot(
                            x_best,
                            normalize_area(x_best, y_best),
                            label=format_model_label_with_vdos_error(
                                best_model, best_row["vdos_error_percent"]
                            ),
                            color=tier_color,
                            linewidth=1.6,
                            alpha=0.9,
                            linestyle="-",
                        )
                        plotted_models.add(best_model)
                    except Exception:
                        pass

            if worst_model is not None and worst_model != best_model:
                worst_row = pick_model_row(
                    subset=representative_subset,
                    system=representative_system,
                    model=worst_model,
                    prefer_low_error=False,
                )
                if worst_row is not None:
                    try:
                        x_worst, y_worst = load_vdos(Path(str(worst_row["mlip_file"])))
                        ax.plot(
                            x_worst,
                            normalize_area(x_worst, y_worst),
                            label=format_model_label_with_vdos_error(
                                worst_model, worst_row["vdos_error_percent"]
                            ),
                            color=tier_color,
                            linewidth=1.4,
                            linestyle="--",
                            alpha=0.9,
                        )
                        plotted_models.add(worst_model)
                    except Exception:
                        pass

            if system_type in DEFAULT_XLIM_BY_SYSTEM_TYPE:
                ax.set_xlim(*DEFAULT_XLIM_BY_SYSTEM_TYPE[system_type])
            if system_type in DEFAULT_YLIM_BY_SYSTEM_TYPE:
                ax.set_ylim(*DEFAULT_YLIM_BY_SYSTEM_TYPE[system_type])

            if is_right_col:
                ax.set_ylabel(tier_label)
                ax.yaxis.set_label_position("right")
            elif is_left_col:
                ax.set_ylabel("VDOS")
            else:
                ax.set_ylabel("")

            if tier_idx == len(TIER_PANELS) - 1:
                ax.set_xlabel("Energy [eV]")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            handles, labels = ax.get_legend_handles_labels()
            for model in full_penalty_models(representative_subset, tier_models):
                if model in plotted_models:
                    continue
                handles.append(
                    Line2D([], [], color=tier_color, linestyle=":", linewidth=1.4)
                )
                labels.append(f"{display_name(model)} (100.0%; no VDOS data)")
            if np.isfinite(tier_mean_error):
                handles = handles + [
                    Line2D([], [], color="none", linestyle="none", linewidth=0)
                ]
                labels = labels + [
                    f"{tier_label} mean VDOS error: {tier_mean_error:.1f}%"
                ]
            if handles:
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
            ax.grid()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a combined figure from plot-9 and plot-10 same-simulation-length views."
    )
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default=DEFAULT_SOURCE,
        help="Trajectory source used to locate generated VDOS tables.",
    )
    parser.add_argument(
        "--normalized-pairs-file",
        type=Path,
        default=None,
        help="Override the generated pairwise VDOS error CSV.",
    )
    parser.add_argument(
        "--model-mean-file",
        type=Path,
        default=None,
        help="Override the generated model-level VDOS error CSV.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=SCRIPT_DIR / "plots" / "plot_vdos_panel_combined.pdf",
        help="Output plot file path.",
    )
    args = parser.parse_args()

    source_results = SCRIPT_DIR / "results" / args.source
    normalized_pairs_path = args.normalized_pairs_file or (
        source_results / "vdos_pair_errors_ev_normalized_same_simulation_length.csv"
    )
    model_means_path = args.model_mean_file or (
        source_results / "vdos_model_mean_ev_normalized_same_simulation_length.csv"
    )
    normalized_pairs = load_normalized_pairs(normalized_pairs_path)
    model_means = load_model_means(model_means_path)

    plot_combined(normalized_pairs, model_means, args.output_file)
    print(f"Saved combined VDOS panel plot to {args.output_file}")


if __name__ == "__main__":
    main()
