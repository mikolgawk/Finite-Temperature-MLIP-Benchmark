#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXCLUDED_MODELS = {"pet-mad"}

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRESSURES_DIR = SCRIPT_DIR / "results" / "same-simulation-length"
DEFAULT_REFERENCE_FILE = DEFAULT_PRESSURES_DIR / "reference_pressure_per_frame_same_simulation_length.csv"
DEFAULT_OUTPUT_FILE = DEFAULT_PRESSURES_DIR / "model_pressure_error_metric.csv"
DEFAULT_PAIR_OUTPUT_FILE = DEFAULT_PRESSURES_DIR / "pressure_pair_similarity_same_simulation_length.csv"
DEFAULT_SYSTEM_MODEL_MEAN_OUTPUT_FILE = (
    DEFAULT_PRESSURES_DIR / "pressure_system_model_mean_similarity_same_simulation_length.csv"
)
DEFAULT_MODEL_SYSTEM_TYPE_MEAN_OUTPUT_FILE = (
    DEFAULT_PRESSURES_DIR / "pressure_model_system_type_mean_similarity_same_simulation_length.csv"
)
DEFAULT_PRESSURE_COMPARISON_FILE = DEFAULT_PRESSURES_DIR / "model_mean_pressure_comparison.csv"


def infer_system_type(system: str) -> str:
    s = system.lower()

    if (
        s.startswith("bulkcuau")
        or s.startswith("bulkcuzral")
        or s.startswith("bulklimgalznsn")
        or s.startswith("bulkpt3co")
    ):
        return "metal alloys"

    if s.startswith("bulkau") or s.startswith("bulkag") or s.startswith("bulkcu_"):
        return "pure metals"
    if s.startswith("cssni3") or s.startswith("mapbbr3"):
        return "perovskites"
    if s.startswith("bulkmos2") or s.startswith("tise2"):
        return "metal dichalcogenides"
    if (
        s.startswith("anthracene")
        or s.startswith("naphthalene")
        or s.startswith("pentacene")
        or s.startswith("picene")
        or s.startswith("tetracene")
    ):
        return "molecular crystals"
    if s.startswith("pt111w24h2o"):
        return "metal-water interfaces"
    return "other"


def normalize_model_name(name: str) -> str:
    return str(name).strip().lower()


def pressure_column_name(columns: list[str]) -> str:
    cols = set(columns)
    if "pressure_GPa" in cols:
        return "pressure_GPa"
    if "pressure_ref_GPa" in cols:
        return "pressure_ref_GPa"
    raise ValueError("No pressure column found. Expected one of: pressure_GPa, pressure_ref_GPa")


def structure_from_trajectory_file(path_like: str) -> str:
    return Path(str(path_like)).parent.name


def parse_model_name(file_path: Path, suffix: str) -> str:
    return normalize_model_name(file_path.name.removesuffix(suffix))


def load_pressure_per_frame_csv(csv_path: Path, deduplicate_reference: bool = False) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    pcol = pressure_column_name(list(header.columns))

    usecols = ["trajectory_file", pcol]
    if "frame_index" in set(header.columns):
        usecols.append("frame_index")

    df = pd.read_csv(csv_path, usecols=usecols)
    df = df.rename(columns={pcol: "pressure_GPa"})
    df["pressure_GPa"] = pd.to_numeric(df["pressure_GPa"], errors="coerce")
    df = df.dropna(subset=["trajectory_file", "pressure_GPa"]).copy()

    df["system"] = df["trajectory_file"].apply(structure_from_trajectory_file)
    df["system_type"] = df["system"].map(infer_system_type)

    if deduplicate_reference:
        if "frame_index" in df.columns:
            df = df.drop_duplicates(subset=["trajectory_file", "frame_index"], keep="first")
        else:
            df = df.drop_duplicates(subset=["trajectory_file", "pressure_GPa"], keep="first")

    return df


def build_bin_edges(ref_values: np.ndarray, mlip_values: np.ndarray, bins: int) -> np.ndarray:
    lo = float(min(np.min(ref_values), np.min(mlip_values)))
    hi = float(max(np.max(ref_values), np.max(mlip_values)))

    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = -1.0, 1.0
    if hi <= lo:
        hi = lo + 1e-6

    return np.linspace(lo, hi, bins + 1)


def pressure_histogram_similarity(
    ref_values: np.ndarray,
    mlip_values: np.ndarray,
    bins: int,
) -> dict[str, float]:
    """
    Score pressure distributions using the same normalized L1 form as VDOS.

    The VDOS scorer computes:
        similarity = 1 - integral(|ref - mlip|) / (integral(ref) + integral(mlip))

    Here ref and mlip are area-normalized pressure histograms on shared bins.
    """
    if ref_values.size == 0 or mlip_values.size == 0:
        return {}

    edges = build_bin_edges(ref_values, mlip_values, bins=bins)
    widths = np.diff(edges)
    if widths.size == 0 or np.any(widths <= 0.0):
        return {}

    ref_hist, _ = np.histogram(ref_values, bins=edges, density=True)
    mlip_hist, _ = np.histogram(mlip_values, bins=edges, density=True)

    ref_area = float(np.sum(ref_hist * widths))
    mlip_area = float(np.sum(mlip_hist * widths))
    if ref_area <= 0.0 or mlip_area <= 0.0:
        return {}

    ref_hist = ref_hist / ref_area
    mlip_hist = mlip_hist / mlip_area
    ref_area = float(np.sum(ref_hist * widths))
    mlip_area = float(np.sum(mlip_hist * widths))

    numerator = float(np.sum(np.abs(ref_hist - mlip_hist) * widths))
    denominator = ref_area + mlip_area
    if denominator <= 0.0 or not np.isfinite(denominator):
        return {}

    distance = numerator / denominator
    similarity = float(np.clip(1.0 - distance, 0.0, 1.0))
    error_fraction = 1.0 - similarity

    return {
        "pressure_similarity": similarity,
        "pressure_similarity_percent": 100.0 * similarity,
        "pressure_error_fraction": error_fraction,
        "pressure_error_percent": 100.0 * error_fraction,
        "pressure_histogram_l1_area": numerator,
        "pressure_histogram_distance": distance,
        "pressure_ref_histogram_area": ref_area,
        "pressure_mlip_histogram_area": mlip_area,
        "pressure_histogram_min_GPa": float(edges[0]),
        "pressure_histogram_max_GPa": float(edges[-1]),
    }


def build_pair_rows(
    pressures_dir: Path,
    reference_file: Path,
    model_file_suffix: str,
    bins: int,
) -> pd.DataFrame:
    ref_df = load_pressure_per_frame_csv(reference_file, deduplicate_reference=True)
    if ref_df.empty:
        raise RuntimeError(f"No usable reference rows in {reference_file}")

    model_files = sorted(pressures_dir.glob(f"*{model_file_suffix}"))
    model_files = [p for p in model_files if not p.name.startswith("reference_")]
    if not model_files:
        raise FileNotFoundError(
            f"No model per-frame files found in {pressures_dir} matching *{model_file_suffix}"
        )

    excluded_models_lower = {m.lower() for m in EXCLUDED_MODELS}
    rows: list[dict[str, object]] = []

    for model_file in model_files:
        model_name = parse_model_name(model_file, model_file_suffix)
        if model_name.lower() in excluded_models_lower:
            continue

        try:
            model_df = load_pressure_per_frame_csv(model_file, deduplicate_reference=False)
        except Exception as exc:
            print(f"[WARN] Skipping {model_file.name}: {exc}")
            continue

        if model_df.empty:
            continue

        common_systems = sorted(set(ref_df["system"]) & set(model_df["system"]))
        if not common_systems:
            print(f"[WARN] {model_name}: no overlapping systems with reference")
            continue

        for system in common_systems:
            ref_vals = ref_df.loc[ref_df["system"] == system, "pressure_GPa"].to_numpy(dtype=float)
            mlip_vals = model_df.loc[model_df["system"] == system, "pressure_GPa"].to_numpy(dtype=float)
            score = pressure_histogram_similarity(ref_vals, mlip_vals, bins=bins)
            if not score or not np.isfinite(score["pressure_similarity"]):
                continue

            rows.append(
                {
                    "system": system,
                    "system_type": infer_system_type(system),
                    "mlip_model": model_name,
                    **score,
                    "n_ref_frames": int(ref_vals.size),
                    "n_mlip_frames": int(mlip_vals.size),
                    "bins": int(bins),
                    "reference_file": str(reference_file),
                    "model_file": str(model_file),
                }
            )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise RuntimeError("No pressure histogram similarity rows were computed.")

    return out_df.sort_values(["system", "mlip_model"]).reset_index(drop=True)


def load_pressure_mae_columns(pressure_comparison_file: Path | None) -> pd.DataFrame | None:
    if pressure_comparison_file is None or not pressure_comparison_file.is_file():
        return None

    df = pd.read_csv(pressure_comparison_file)
    if "model" not in df.columns:
        return None

    keep_cols = ["model"]
    rename_cols: dict[str, str] = {}
    if "error_GPa" in df.columns:
        keep_cols.append("error_GPa")
        rename_cols["error_GPa"] = "pressure_mae_GPa"
    elif "pressure_mae_GPa" in df.columns:
        keep_cols.append("pressure_mae_GPa")

    for col in df.columns:
        if col.endswith("_error_GPa") and col != "error_GPa":
            keep_cols.append(col)
            rename_cols[col] = f"{col.removesuffix('_error_GPa')}_pressure_mae_GPa"

    out = df[keep_cols].copy()
    out["model"] = out["model"].map(normalize_model_name)
    out = out.rename(columns=rename_cols)
    return out


def write_metric_outputs(
    pair_df: pd.DataFrame,
    pair_output_file: Path,
    system_model_mean_output_file: Path,
    model_mean_output_file: Path,
    model_system_type_mean_output_file: Path,
    pressure_comparison_file: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_output_file.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(pair_output_file, index=False)

    system_model_mean_df = (
        pair_df.groupby(["system", "mlip_model"], as_index=False)["pressure_similarity"]
        .mean()
        .rename(columns={"pressure_similarity": "mean_pressure_similarity"})
        .sort_values(["system", "mlip_model"])
        .reset_index(drop=True)
    )
    system_model_mean_df["mean_pressure_similarity_percent"] = (
        100.0 * system_model_mean_df["mean_pressure_similarity"]
    )
    system_model_mean_df["mean_pressure_error_percent"] = (
        100.0 - system_model_mean_df["mean_pressure_similarity_percent"]
    )
    system_model_mean_df["bins"] = pair_df["bins"].iloc[0]
    system_model_mean_output_file.parent.mkdir(parents=True, exist_ok=True)
    system_model_mean_df.to_csv(system_model_mean_output_file, index=False)

    model_mean_df = (
        system_model_mean_df.groupby("mlip_model", as_index=False)["mean_pressure_similarity"]
        .mean()
        .rename(columns={"mean_pressure_similarity": "final_mean_pressure_similarity"})
        .sort_values("final_mean_pressure_similarity", ascending=False)
        .reset_index(drop=True)
    )
    model_mean_df.insert(0, "model", model_mean_df["mlip_model"])
    model_mean_df["final_mean_pressure_similarity_percent"] = (
        100.0 * model_mean_df["final_mean_pressure_similarity"]
    )
    model_mean_df["final_mean_pressure_error_percent"] = (
        100.0 - model_mean_df["final_mean_pressure_similarity_percent"]
    )
    model_mean_df["pressure_similarity_percent"] = model_mean_df[
        "final_mean_pressure_similarity_percent"
    ]
    model_mean_df["pressure_error_percent"] = model_mean_df[
        "final_mean_pressure_error_percent"
    ]
    model_mean_df["pressure_score_percent"] = model_mean_df[
        "final_mean_pressure_similarity_percent"
    ]
    model_mean_df["bins"] = pair_df["bins"].iloc[0]

    pressure_mae_df = load_pressure_mae_columns(pressure_comparison_file)
    if pressure_mae_df is not None:
        model_mean_df = model_mean_df.merge(pressure_mae_df, on="model", how="left")

    model_mean_output_file.parent.mkdir(parents=True, exist_ok=True)
    model_mean_df.to_csv(model_mean_output_file, index=False)

    model_system_type_mean_df = (
        pair_df.groupby(["mlip_model", "system_type"], as_index=False)["pressure_similarity"]
        .mean()
        .rename(columns={"pressure_similarity": "mean_pressure_similarity"})
        .sort_values(["mlip_model", "system_type"])
        .reset_index(drop=True)
    )
    model_system_type_mean_df["mean_pressure_similarity_percent"] = (
        100.0 * model_system_type_mean_df["mean_pressure_similarity"]
    )
    model_system_type_mean_df["mean_pressure_error_percent"] = (
        100.0 - model_system_type_mean_df["mean_pressure_similarity_percent"]
    )
    model_system_type_mean_df["bins"] = pair_df["bins"].iloc[0]
    model_system_type_mean_output_file.parent.mkdir(parents=True, exist_ok=True)
    model_system_type_mean_df.to_csv(model_system_type_mean_output_file, index=False)

    return system_model_mean_df, model_mean_df, model_system_type_mean_df


def compute_pressure_metric(
    pressures_dir: Path,
    reference_file: Path,
    model_file_suffix: str,
    bins: int,
    pair_output_file: Path,
    system_model_mean_output_file: Path,
    model_mean_output_file: Path,
    model_system_type_mean_output_file: Path,
    pressure_comparison_file: Path | None,
) -> pd.DataFrame:
    if bins < 2:
        raise ValueError("--bins must be >= 2")
    if not pressures_dir.is_dir():
        raise NotADirectoryError(f"Pressures directory not found: {pressures_dir}")
    if not reference_file.is_file():
        raise FileNotFoundError(f"Reference per-frame CSV not found: {reference_file}")

    pair_df = build_pair_rows(
        pressures_dir=pressures_dir,
        reference_file=reference_file,
        model_file_suffix=model_file_suffix,
        bins=bins,
    )

    _, model_mean_df, _ = write_metric_outputs(
        pair_df=pair_df,
        pair_output_file=pair_output_file,
        system_model_mean_output_file=system_model_mean_output_file,
        model_mean_output_file=model_mean_output_file,
        model_system_type_mean_output_file=model_system_type_mean_output_file,
        pressure_comparison_file=pressure_comparison_file,
    )

    print("Saved pressure histogram-similarity outputs:")
    print(f"- Pair-level:              {pair_output_file.resolve()}")
    print(f"- System-model mean:       {system_model_mean_output_file.resolve()}")
    print(f"- Model mean/error metric: {model_mean_output_file.resolve()}")
    print(f"- Model x system-type:     {model_system_type_mean_output_file.resolve()}")
    print()
    print(
        model_mean_df[
            [
                "model",
                "final_mean_pressure_similarity",
                "final_mean_pressure_similarity_percent",
                "final_mean_pressure_error_percent",
            ]
        ].to_string(index=False)
    )

    return model_mean_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute pressure error and similarity from per-frame pressure histograms "
            "using the same normalized L1 score used for VDOS."
        )
    )
    parser.add_argument(
        "--pressures-dir",
        type=Path,
        default=DEFAULT_PRESSURES_DIR,
        help="Directory containing model per-frame pressure CSV files.",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=DEFAULT_REFERENCE_FILE,
        help="Reference per-frame pressure CSV file.",
    )
    parser.add_argument(
        "--model-file-suffix",
        default="_same-simulation-length_pressure_per_frame.csv",
        help="Suffix used to identify model per-frame pressure CSV files.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of shared pressure bins used for each reference/model histogram pair.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Model-level output CSV. Kept for compatibility with earlier workflows.",
    )
    parser.add_argument(
        "--model-mean-output-file",
        type=Path,
        default=None,
        help="Alias for --output-file. If provided, this path is used for model-level output.",
    )
    parser.add_argument(
        "--pair-output-file",
        type=Path,
        default=DEFAULT_PAIR_OUTPUT_FILE,
        help="Pair-level pressure histogram similarity output CSV.",
    )
    parser.add_argument(
        "--system-model-mean-output-file",
        type=Path,
        default=DEFAULT_SYSTEM_MODEL_MEAN_OUTPUT_FILE,
        help="Mean similarity per (system, model).",
    )
    parser.add_argument(
        "--model-system-type-mean-output-file",
        type=Path,
        default=DEFAULT_MODEL_SYSTEM_TYPE_MEAN_OUTPUT_FILE,
        help="Mean similarity per (model, system type).",
    )
    parser.add_argument(
        "--pressure-comparison-file",
        type=Path,
        default=DEFAULT_PRESSURE_COMPARISON_FILE,
        help=(
            "Optional pressure MAE CSV to merge into the model-level output for compatibility. "
            "The histogram score does not use these MAE values."
        ),
    )

    args = parser.parse_args()

    model_mean_output_file = args.model_mean_output_file or args.output_file
    compute_pressure_metric(
        pressures_dir=Path(args.pressures_dir).resolve(),
        reference_file=Path(args.reference_file).resolve(),
        model_file_suffix=args.model_file_suffix,
        bins=args.bins,
        pair_output_file=Path(args.pair_output_file).resolve(),
        system_model_mean_output_file=Path(args.system_model_mean_output_file).resolve(),
        model_mean_output_file=Path(model_mean_output_file).resolve(),
        model_system_type_mean_output_file=Path(args.model_system_type_mean_output_file).resolve(),
        pressure_comparison_file=Path(args.pressure_comparison_file).resolve()
        if args.pressure_comparison_file
        else None,
    )


if __name__ == "__main__":
    main()
