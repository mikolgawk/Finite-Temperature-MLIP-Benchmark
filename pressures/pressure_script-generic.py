#!/usr/bin/env python3
"""
Generic pressure script: computes per-frame stress and trajectory-average
pressure for MLIP trajectories, matching MLIP and reference trajectories to
the same simulation length, using a calculator selected at runtime from
model_calculators.json.

Matching policy:
- For each MLIP trajectory, read the corresponding reference trajectory.
- Use ONLY dt from the CSV timing tables (ignore stride for matching).
- Compare total simulated times:
      ref_time_fs  = n_ref_total  * ref_dt_fs
      mlip_time_fs = n_mlip_total * mlip_dt_fs
- Trim the longer trajectory so that both cover the same matched time,
  exactly as in the RDF analysis.

Notes:
- Only the MLIP-trimmed frames are evaluated for MLIP pressure.
- The reference trajectory is only used to determine the matched length.
- The original trajectory files are not modified on disk.

Usage:
    MODEL_NAME=mace-mp-0 python pressure_script-generic.py [--traj-dir ...] [--ref-dir ...]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import math
from fractions import Fraction

import numpy as np
import pandas as pd
from ase.io import read

MODEL_CATALOG_PATH = os.path.join(os.path.dirname(__file__), 'model_calculators.json')

EV_PER_A3_TO_GPA = 160.21766208

SKIP_PARENT_PREFIXES = ("Pt111w24H2O_",)
ONLY_2D_SYSTEMS = {"TiSe2"}
PLANE_2D = "xy"

PRESSURE_SETTINGS_DEFAULT = "pressure_settings_ref.csv"

REFERENCE_FILE_PREFERENCES = (
    "traj.extxyz",
)

# Models whose atoms.info must carry integer charge/spin tags before
# evaluation (fairchem UMA calculators require these to be present).
CHARGE_SPIN_NORMALIZED_MODELS = {"uma-m-omat", "uma-s-omat"}


def load_model_catalog(path: str = MODEL_CATALOG_PATH) -> dict:
    """Loads the model calculator catalog from JSON."""
    with open(path) as f:
        catalog = json.load(f)
    return {model['name']: model for model in catalog['models']}


def build_calculator(model_entry: dict):
    """
    Executes a model's import statements and evaluates its calculator
    expression, returning the constructed ASE calculator instance.
    """
    namespace = {}
    for import_line in model_entry['imports']:
        exec(import_line, namespace)
    return eval(model_entry['calculator_expr'], namespace)


def normalize_charge_spin(atoms):
    charge = atoms.info.get("charge", 0)
    spin = atoms.info.get("spin", 0)
    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = int(spin)
    return atoms


def should_skip(traj_path: str) -> bool:
    parent = Path(traj_path).parent.name
    return parent.startswith(SKIP_PARENT_PREFIXES)


def is_2d_system(system_name: str) -> bool:
    return system_name in ONLY_2D_SYSTEMS


def as_traj_list(frames):
    if isinstance(frames, list):
        return frames
    return [frames]


def find_reference_trajectory_file(reference_system_dir: Path) -> Path:
    for name in REFERENCE_FILE_PREFERENCES:
        candidate = reference_system_dir / name
        if candidate.is_file():
            return candidate

    files = sorted(
        [p for p in reference_system_dir.glob("*.extxyz")]
        + [p for p in reference_system_dir.glob("*.xyz")]
    )
    if not files:
        raise FileNotFoundError(f"No reference trajectory found in {reference_system_dir}")

    traj_files = [p for p in files if "traj" in p.stem]
    if traj_files:
        return traj_files[0]

    return files[0]


def _normalize_temperature(value: object) -> int:
    if pd.isna(value):
        raise ValueError("Missing temperature in settings CSV")
    return int(round(float(value)))


def load_timing_settings(settings_csv: Path, table_name: str) -> dict[tuple[str, int], dict[str, float]]:
    if not settings_csv.is_file():
        raise FileNotFoundError(f"{table_name} settings file not found: {settings_csv}")

    df = pd.read_csv(settings_csv)
    required = {"System", "temperature", "stride", "dt"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{table_name} settings file {settings_csv} is missing columns: {sorted(missing)}"
        )

    settings: dict[tuple[str, int], dict[str, float]] = {}

    for _, row in df.iterrows():
        system = str(row["System"]).strip()
        temperature_k = _normalize_temperature(row["temperature"])
        stride = float(row["stride"])
        dt_fs = float(row["dt"])
        padding = float(row["padding"]) if "padding" in df.columns and not pd.isna(row["padding"]) else np.nan

        if stride <= 0 or dt_fs <= 0:
            raise ValueError(
                f"Invalid stride/dt in {table_name} settings for {(system, temperature_k)}: "
                f"stride={stride}, dt={dt_fs}"
            )

        key_st = (system, temperature_k)
        if key_st in settings:
            raise ValueError(f"Duplicate {table_name} settings for {key_st} in {settings_csv}")

        settings[key_st] = {
            "stride": stride,
            "dt_fs": dt_fs,
            "padding": padding,
        }

    return settings


def get_file_names(directory: str, prefix: str, extension: str = ".extxyz") -> list[str]:
    base_path = Path(directory)
    file_paths = [str(path) for path in base_path.rglob(f"{prefix}*{extension}")]
    file_paths.sort()
    print(f"Found {len(file_paths)} trajectory files in {directory}")
    return file_paths


def collect_trajectory_settings_keys(
    traj_paths: list[str],
) -> tuple[set[tuple[str, int]], list[str]]:
    required_keys: set[tuple[str, int]] = set()
    missing_temperature_paths: list[str] = []

    for traj_path in traj_paths:
        if should_skip(traj_path):
            continue

        system_name, temperature_k, _ = parse_system_info(traj_path)
        if temperature_k is None:
            missing_temperature_paths.append(traj_path)
            continue

        required_keys.add((system_name, int(temperature_k)))

    return required_keys, missing_temperature_paths


def validate_settings_coverage(
    required_keys: set[tuple[str, int]],
    settings: dict[tuple[str, int], dict[str, float]],
) -> None:
    missing = sorted(k for k in required_keys if k not in settings)

    if not missing:
        print(f"Settings coverage OK for {len(required_keys)} system/temperature pairs.")
        return

    lines = [f"Settings coverage check failed. Missing ({len(missing)}):"]
    lines.extend([f"  - {system}_{temp}K" for system, temp in missing])

    raise SystemExit("\n".join(lines))


def parse_system_info(file_path: str) -> tuple[str, int | None, str]:
    parent_dir = Path(file_path).parent.name

    m = re.search(r"^(?P<sys>.+?)_(?P<T>\d+)K(?P<rest>[_-].*)?$", parent_dir)
    if m:
        system_name = m.group("sys")
        temperature_k = int(m.group("T"))
        reference_key = f"{system_name}_{temperature_k}K"
        return system_name, temperature_k, reference_key

    return parent_dir, None, parent_dir


def _lcm_int(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def _lcm_fraction(a: Fraction, b: Fraction) -> Fraction:
    return Fraction(_lcm_int(a.numerator, b.numerator), math.gcd(a.denominator, b.denominator))


def matched_frame_counts(
    n_ref_total: int,
    n_mlip_total: int,
    ref_dt_fs: float,
    mlip_dt_fs: float,
) -> tuple[int, int, float]:
    """
    Compute (n_ref_use, n_mlip_use, matched_time_fs) so that:
      n_ref_use * ref_dt_fs == n_mlip_use * mlip_dt_fs
    and matched time does not exceed either trajectory's available simulation time.
    """
    if n_ref_total <= 0 or n_mlip_total <= 0:
        return 0, 0, 0.0

    ref_dt = Fraction(str(ref_dt_fs))
    mlip_dt = Fraction(str(mlip_dt_fs))

    common_time = _lcm_fraction(ref_dt, mlip_dt)
    ref_time_max = n_ref_total * ref_dt
    mlip_time_max = n_mlip_total * mlip_dt

    n_common = min(ref_time_max // common_time, mlip_time_max // common_time)
    if n_common <= 0:
        return 0, 0, 0.0

    matched_time = n_common * common_time
    n_ref_use = int(matched_time / ref_dt)
    n_mlip_use = int(matched_time / mlip_dt)

    return n_ref_use, n_mlip_use, float(matched_time)


def _stress_as_3x3(stress: np.ndarray) -> np.ndarray:
    arr = np.asarray(stress, dtype=float)
    if arr.shape == (3, 3):
        return arr

    flat = arr.reshape(-1)
    if flat.size == 6:
        xx, yy, zz, yz, xz, xy = flat.tolist()
        return np.array(
            [[xx, xy, xz],
             [xy, yy, yz],
             [xz, yz, zz]],
            dtype=float,
        )

    if flat.size == 9:
        return flat.reshape(3, 3)

    raise ValueError(f"Unsupported stress shape: {arr.shape}")


def pressure_from_stress_gpa_3d(stress: np.ndarray) -> float:
    s = _stress_as_3x3(stress)
    sigma_hyd = float(np.trace(s) / 3.0)
    return (-sigma_hyd) * EV_PER_A3_TO_GPA


def pressure_from_stress_gpa_2d_inplane(stress: np.ndarray, plane: str = "xy") -> float:
    s = _stress_as_3x3(stress)
    plane = plane.lower()

    if plane == "xy":
        sigma_in = 0.5 * (s[0, 0] + s[1, 1])
    elif plane == "xz":
        sigma_in = 0.5 * (s[0, 0] + s[2, 2])
    elif plane == "yz":
        sigma_in = 0.5 * (s[1, 1] + s[2, 2])
    else:
        raise ValueError("plane must be one of: 'xy', 'xz', 'yz'")

    return (-float(sigma_in)) * EV_PER_A3_TO_GPA


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    model_name = os.environ.get('MODEL_NAME')
    catalog = load_model_catalog()
    if model_name not in catalog:
        raise SystemExit(f"Set MODEL_NAME to one of: {', '.join(sorted(catalog))}")
    key = model_name

    parser = argparse.ArgumentParser(
        description=f"Compute per-frame stress and average trajectory pressure for {key} with RDF-style equal-time matching."
    )
    parser.add_argument("--traj-dir", default="../data/mlip-trajs-20fs-tau", help="Root directory with MLIP trajectory files (searched recursively, so it should cover all systems).")
    parser.add_argument("--ref-dir", default="../data/ref-trajs", help="Root directory with reference trajectories.")
    parser.add_argument(
        "--settings",
        default=str(script_dir / PRESSURE_SETTINGS_DEFAULT),
        help="CSV file with timing settings (System, temperature, stride, dt), used for both reference and MLIP matching.",
    )
    parser.add_argument("--prefix", default=None, help="Trajectory filename prefix (default: nvt_<MODEL_NAME>).")
    parser.add_argument("--output-dir", default="results-20fs-tau", help="Output directory for CSV files.")
    args = parser.parse_args()

    prefix = args.prefix or f"nvt_{key}"

    if PLANE_2D not in {"xy", "xz", "yz"}:
        raise SystemExit("Error: PLANE_2D must be one of: 'xy', 'xz', 'yz'.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_root = Path(args.ref_dir)
    if not reference_root.is_dir():
        raise SystemExit(f"Reference directory not found: {reference_root}")

    pressure_settings = load_timing_settings(Path(args.settings), "pressure")

    traj_paths = get_file_names(directory=args.traj_dir, prefix=prefix)
    if not traj_paths:
        raise SystemExit("No trajectory files found.")

    required_keys, missing_temperature_paths = collect_trajectory_settings_keys(traj_paths)
    if missing_temperature_paths:
        lines = [
            "Could not parse temperature from trajectory folder name for these paths:",
            *[f"  - {path}" for path in missing_temperature_paths],
            "Expected folder names like '<System>_<Temperature>K_...'.",
        ]
        raise SystemExit("\n".join(lines))

    validate_settings_coverage(required_keys, pressure_settings)

    print(f"\nInitializing calculator '{key}'...")
    try:
        calculator = build_calculator(catalog[key])
    except Exception as exc:
        raise SystemExit(f"Failed to initialize calculator '{key}': {exc}")
    print(f"  ✓ Loaded {key}")

    apply_charge_spin_fix = key in CHARGE_SPIN_NORMALIZED_MODELS

    per_frame_rows: list[dict] = []
    summary_rows: list[dict] = []

    n_skipped_policy = 0
    n_skipped_read = 0
    n_skipped_reference = 0

    for idx, traj_path in enumerate(traj_paths, start=1):
        if should_skip(traj_path):
            print(f"[{idx}/{len(traj_paths)}] Skipping {traj_path} (Pt111w24H2O policy)")
            n_skipped_policy += 1
            continue

        parent_dir_name = Path(traj_path).parent.name
        system_name, temperature_k, reference_key = parse_system_info(traj_path)
        mode = "2d" if is_2d_system(system_name) else "3d"

        reference_system_dir = reference_root / parent_dir_name
        if not reference_system_dir.is_dir():
            print(
                f"[{idx}/{len(traj_paths)}] Skipping {traj_path} "
                f"(missing reference directory: {reference_system_dir})"
            )
            n_skipped_reference += 1
            continue

        try:
            reference_traj_file = find_reference_trajectory_file(reference_system_dir)

            if temperature_k is None:
                raise ValueError(
                    f"Temperature missing in trajectory folder name; cannot match CSV settings for {traj_path}"
                )

            settings_key = (system_name, int(temperature_k))
            if settings_key not in pressure_settings:
                raise KeyError(f"Missing timing settings for {settings_key}")

            # Reference and MLIP trajectories share the same dt (single settings file).
            dt_fs = float(pressure_settings[settings_key]["dt_fs"])
            ref_dt_fs = dt_fs
            mlip_dt_fs = dt_fs

            ref_frames_all = as_traj_list(read(reference_traj_file, ":"))
            mlip_frames_all = as_traj_list(read(traj_path, ":"))

            n_ref_total = len(ref_frames_all)
            n_mlip_total = len(mlip_frames_all)

            if n_ref_total <= 0:
                raise ValueError(f"Reference trajectory is empty: {reference_traj_file}")
            if n_mlip_total <= 0:
                raise ValueError(f"MLIP trajectory is empty: {traj_path}")

            n_ref_use, n_mlip_use, matched_time_fs = matched_frame_counts(
                n_ref_total=n_ref_total,
                n_mlip_total=n_mlip_total,
                ref_dt_fs=ref_dt_fs,
                mlip_dt_fs=mlip_dt_fs,
            )

            if n_ref_use <= 0 or n_mlip_use <= 0:
                raise ValueError("Could not find positive matched simulation time.")

            ref_time_used = n_ref_use * ref_dt_fs
            mlip_time_used = n_mlip_use * mlip_dt_fs

            print(
                f"[{idx}/{len(traj_paths)}] Processing {traj_path} "
                f"(mode={mode})"
            )
            print(
                f"  matched time={matched_time_fs:.3f} fs | "
                f"ref frames used={n_ref_use}/{n_ref_total} | "
                f"mlip frames used={n_mlip_use}/{n_mlip_total}"
            )

            if abs(ref_time_used - mlip_time_used) > 1e-12:
                print(
                    f"  [WARN] Time mismatch after truncation: "
                    f"ref={ref_time_used:.6f} fs, mlip={mlip_time_used:.6f} fs"
                )

            mlip_frames = mlip_frames_all[:n_mlip_use]

        except Exception as exc:
            print(f"[{idx}/{len(traj_paths)}] Skipping {traj_path} (reference timing/read error: {exc})")
            n_skipped_reference += 1
            continue

        pressures_gpa: list[float] = []
        n_failed = 0

        for frame_index, atoms in enumerate(mlip_frames):
            try:
                atoms_eval = atoms.copy()
                if apply_charge_spin_fix:
                    atoms_eval = normalize_charge_spin(atoms_eval)
                atoms_eval.calc = calculator
                stress = atoms_eval.get_stress(voigt=False)

                if mode == "2d":
                    pressure_gpa = pressure_from_stress_gpa_2d_inplane(stress, plane=PLANE_2D)
                    plane_used = PLANE_2D
                else:
                    pressure_gpa = pressure_from_stress_gpa_3d(stress)
                    plane_used = None

                pressures_gpa.append(pressure_gpa)

                per_frame_rows.append(
                    {
                        "system": system_name,
                        "temperature_K": temperature_k,
                        "reference_key": reference_key,
                        "trajectory_file": traj_path,
                        "reference_trajectory_file": str(reference_traj_file),
                        "settings_key": settings_key,
                        "reference_frames_total": int(n_ref_total),
                        "reference_frames_used": int(n_ref_use),
                        "reference_frames_trimmed": int(n_ref_total - n_ref_use),
                        "reference_dt_fs": float(ref_dt_fs),
                        "reference_time_used_fs": float(ref_time_used),
                        "mlip_frames_total": int(n_mlip_total),
                        "mlip_frames_used": int(n_mlip_use),
                        "mlip_frames_trimmed": int(n_mlip_total - n_mlip_use),
                        "mlip_dt_fs": float(mlip_dt_fs),
                        "mlip_time_used_fs": float(mlip_time_used),
                        "matched_time_fs": float(matched_time_fs),
                        "frame_index": frame_index,
                        "pressure_GPa": pressure_gpa,
                        "pressure_mode": mode,
                        "plane_used": plane_used,
                    }
                )
            except Exception:
                n_failed += 1

        if not pressures_gpa:
            print("  Skipping (no valid stress/pressure frames).")
            continue

        arr = np.asarray(pressures_gpa, dtype=float)

        summary_rows.append(
            {
                "system": system_name,
                "temperature_K": temperature_k,
                "reference_key": reference_key,
                "trajectory_file": traj_path,
                "reference_trajectory_file": str(reference_traj_file),
                "settings_key": settings_key,
                "reference_frames_total": int(n_ref_total),
                "reference_frames_used": int(n_ref_use),
                "reference_frames_trimmed": int(n_ref_total - n_ref_use),
                "reference_dt_fs": float(ref_dt_fs),
                "reference_time_used_fs": float(ref_time_used),
                "mlip_frames_total": int(n_mlip_total),
                "mlip_frames_used": int(arr.size),
                "mlip_frames_failed": int(n_failed),
                "mlip_frames_trimmed": int(n_mlip_total - n_mlip_use),
                "mlip_dt_fs": float(mlip_dt_fs),
                "mlip_time_used_fs": float(mlip_time_used),
                "matched_time_fs": float(matched_time_fs),
                "pressure_mode": mode,
                "plane_used": (PLANE_2D if mode == "2d" else None),
                "pressure_mean_GPa": float(np.mean(arr)),
                "pressure_std_GPa": float(np.std(arr, ddof=0)),
                "pressure_min_GPa": float(np.min(arr)),
                "pressure_max_GPa": float(np.max(arr)),
            }
        )

        print(
            f"  mean pressure = {np.mean(arr):.4f} GPa "
            f"(mlip frames used={arr.size}/{n_mlip_total}, "
            f"ref frames used={n_ref_use}/{n_ref_total})"
        )

    if not summary_rows:
        raise SystemExit("No trajectory pressure summaries were produced.")

    per_frame_path = output_dir / f"{key}_same-simulation-length_pressure_per_frame.csv"
    summary_path = output_dir / f"{key}_same-simulation-length_pressure_trajectory_summary.csv"

    pd.DataFrame(per_frame_rows).to_csv(per_frame_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Skipped by policy: {n_skipped_policy} trajectories")
    print(f"Skipped due to read errors: {n_skipped_read} trajectories")
    print(f"Skipped due to reference/equal-time issues: {n_skipped_reference} trajectories")
    print(f"Saved per-frame pressures: {per_frame_path}")
    print(f"Saved trajectory summaries: {summary_path}")


if __name__ == "__main__":
    main()
