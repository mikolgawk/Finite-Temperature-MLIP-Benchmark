#!/usr/bin/env python3

import os
import csv
import numpy as np
import pandas as pd
import mdtraj as mdt
from ase.io import read
from fractions import Fraction
import math
import re


# ============================================================
# ASE → MDTraj (NO disk I/O, exact physics)
# ============================================================

def ase_to_mdtraj(traj):
    """
    Convert ASE trajectory (list[ase.Atoms]) to MDTraj Trajectory
    """
    # Positions: Å → nm
    xyz = np.array([a.get_positions() for a in traj], dtype=np.float64) / 10.0

    # Build topology from first frame
    symbols = traj[0].get_chemical_symbols()
    top = mdt.Topology()
    chain = top.add_chain()
    res = top.add_residue("SYS", chain)

    for s in symbols:
        top.add_atom(
            s,
            element=mdt.element.get_by_symbol(s),
            residue=res
        )

    md_traj = mdt.Trajectory(xyz=xyz, topology=top)

    # Unit cell: Å → nm
    cells = np.array([a.get_cell().array for a in traj], dtype=np.float64) / 10.0
    md_traj.unitcell_vectors = cells
    md_traj.unitcell_lengths = np.linalg.norm(cells, axis=2)

    return md_traj


# ============================================================
# RDF computation (ALL PAIRS, exact physics)
# ============================================================

def compute_rdf(md_traj, nbins=500):
    """
    Time-averaged RDF using all atom pairs (O(N^2))
    Physics identical to original code.
    """
    n_atoms = md_traj.n_atoms
    atoms = np.arange(n_atoms)

    # ALL pairs (same as before)
    pairs = md_traj.top.select_pairs(atoms, atoms)

    # r_max = half minimum box length
    cell_lengths_A = md_traj.unitcell_lengths[0] * 10.0
    r_max_nm = (np.min(cell_lengths_A) / 2.0) / 10.0

    r, g = mdt.compute_rdf(
        md_traj,
        pairs=pairs,
        r_range=(0.0, r_max_nm),
        n_bins=nbins,
        periodic=True
    )

    return r * 10.0, g  # Å


# ============================================================
# RDF error metric
# ============================================================

def rdf_error(ref_rdf, test_rdf):
    r_ref, g_ref = ref_rdf
    r_test, g_test = test_rdf

    g_test_interp = np.interp(r_ref, r_test, g_test)

    numerator = np.sum(np.abs(g_ref - g_test_interp))
    denominator = np.sum(np.abs(g_ref - 1.0))

    if denominator == 0.0:
        return 100.0

    return min(1.0, numerator / denominator) * 100.0


def save_rdf_csv(r, g, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, np.column_stack([r, g]), delimiter=',', header='r_A,g_r', comments='')


# ============================================================
# Utilities
# ============================================================

REF_SETTINGS_CSV = "/home/mjgawkowski/phd_mlip_matbench_benchmarks/scripts/rdf-scripts-final-copy/vdos_settings_ref.csv"
MLIP_SETTINGS_CSV = "/home/mjgawkowski/phd_mlip_matbench_benchmarks/scripts/rdf-scripts-final-copy/vdos_settings_mlip.csv"

RESULTS_DIR = "/home/mjgawkowski/phd_mlip_matbench_benchmarks/scripts/rdf-scripts-final-copy/results"


def _normalize_system_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def parse_system_temperature_from_dirname(dirname: str) -> tuple[str, int] | None:
    # Example names: bulkAg_600K_Kapil, bulkCuAu_500K-Artrith_VASP
    match = re.match(r"^(?P<system>[^_]+)_(?P<temp>\d+)K(?:\b|[_-].*)$", dirname)
    if match is None:
        return None

    system = _normalize_system_name(match.group("system"))
    temperature = int(match.group("temp"))
    return system, temperature


def load_vdos_settings(path: str) -> dict[tuple[str, int], dict[str, float | int]]:
    settings: dict[tuple[str, int], dict[str, float | int]] = {}

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            system = _normalize_system_name(row["System"])
            temperature = int(float(row["temperature"]))
            stride = int(float(row["stride"]))
            dt_fs = float(row["dt"])
            padding = int(float(row["padding"]))
            settings[(system, temperature)] = {
                "stride": stride,
                "dt_fs": dt_fs,
                "padding": padding,
            }

    return settings


def _lcm_int(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def _lcm_fraction(a: Fraction, b: Fraction) -> Fraction:
    """Least common multiple for positive rational numbers."""
    return Fraction(_lcm_int(a.numerator, b.numerator), math.gcd(a.denominator, b.denominator))


def as_traj_list(frames):
    if isinstance(frames, list):
        return frames
    return [frames]


def contains_hydrogen(traj) -> bool:
    if not traj:
        return False
    return "H" in set(traj[0].get_chemical_symbols())


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


def obtain_system_names(data_dir):
    return [
        name for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    ]


# ============================================================
# System-type aggregation
# ============================================================

SYSTEMS = {
    "Pure metals": ["bulkAu_1500K_Kapil", "bulkAg_600K_Kapil", "bulkCu_1000K_Kapil"],
    "Perovskites": ["CsSnI3_500K_Ivor_VASP", "MAPbBr3_300K_Ivor_VASP"],
    "Metal dichalcogenides": ["bulkMoS2_300K_NO-VdW_J.Kioseoglou_VASP", "TiSe2_400K_Ivor_VASP"],
    "Metal alloys": ["bulkCuAu_500K-Artrith_VASP", "bulkCuZrAl_1500K_A.Wadowski-J.Schmidt_VASP", "bulkLiMgAlZnSn_600K_J_Schmidt_VASP", "bulkLiMgAlZnSn_900K_J_Schmidt_VASP", "bulkPt3Co_300K_J.Kioseoglou_VASP"],
    "Molecular crystals": ["anthracene_293K_Sharma_S", "naphthalene_295K_Sharma_S", "pentacene_295K_Sharma_S", "picene_295K_Sharma_S", "tetracene_295K_Sharma_S"],
    # "Metal-water interfaces": ["Pt111w24H2O_380K_Heenen_VASP"],
}

BY_SYSTEM_TYPE_OUTPUT_FILE = os.path.join(RESULTS_DIR, "rdf_similarity_scores_by_system_type_same_simulation_length.csv")


def aggregate_by_system_type(detailed_results: dict[str, list[dict]]) -> pd.DataFrame:
    """Aggregate per-system RDF errors into per-system-type means, per model."""
    results: dict[str, dict[str, float]] = {}

    for model, rows in detailed_results.items():
        if not rows:
            continue

        df = pd.DataFrame(rows)
        results[model] = {}

        print(f"\nModel: {model}")
        print(f"  Available systems: {df['System'].nunique()}")

        for system_type, system_list in SYSTEMS.items():
            print(f"  Processing system type: {system_type}")

            type_df = df[df["System"].isin(system_list)]

            systems_found = sorted(type_df["System"].unique().tolist())
            print(f"    Systems found: {systems_found}")
            print(f"    Number of rows: {len(type_df)}")

            mean_error = type_df["RDF_Error"].mean()
            results[model][system_type] = np.round(mean_error, 3)

            if pd.isna(mean_error):
                print(f"    Mean RDF error for {system_type}: NaN (no matching systems)")
            else:
                print(f"    Mean RDF error for {system_type}: {mean_error:.3f}")

    if not results:
        raise RuntimeError("No valid model results were produced.")

    results_df = pd.DataFrame(results).T  # Transpose so models are rows
    results_df.index.name = "model"
    results_df = results_df.sort_index()
    return results_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    ref_base = "/home/mjgawkowski/phd_mlip_matbench_benchmarks/ref-trajs"
    mlip_base = "/home/mjgawkowski/Finite-Temperature-MLIP-Benchmarks/plots/mlip_trajectories"

    model_names = [
        'chgnet', 'mace-mp-0', 'grace-mp',
        'mace-mpa-0', 'orb-v2',
        'mattersim-v1-5M', 'grace-oam', 'orb-v3', 'eSEN-30M-OAM', 'nequip', 'eq-v2-M-omat', 'pet-oam-xl',
        "mace-mh-omat", "uma-s-omat", "uma-m-omat"
    ]

    results = {m: [] for m in model_names}
    detailed_results = {m: [] for m in model_names}

    ref_settings = load_vdos_settings(REF_SETTINGS_CSV)
    mlip_settings = load_vdos_settings(MLIP_SETTINGS_CSV)

    systems = obtain_system_names(ref_base)
    print(f"Found {len(systems)} systems")

    for system in systems:
        print(f"\n=== System: {system} ===")

        system_key = parse_system_temperature_from_dirname(system)
        if system_key is None:
            print(f"  [SKIP] Could not parse system/temperature from directory name: {system}")
            continue

        if system_key not in ref_settings:
            print(f"  [SKIP] Missing reference CSV settings for {system}")
            continue

        if system_key not in mlip_settings:
            print(f"  [SKIP] Missing MLIP CSV settings for {system}")
            continue

        ref_cfg = ref_settings[system_key]
        mlip_cfg = mlip_settings[system_key]

        ref_frame_dt_fs = float(ref_cfg["dt_fs"])
        mlip_frame_dt_fs = float(mlip_cfg["dt_fs"])

        # ---------- Reference ----------
        ref_traj_path = os.path.join(ref_base, system, "traj.extxyz")
        if not os.path.exists(ref_traj_path):
            print(f"  [SKIP] Missing reference trajectory: {ref_traj_path}")
            continue

        ref_traj_all = as_traj_list(read(ref_traj_path, ":"))
        if len(ref_traj_all) == 0:
            print(f"  [SKIP] Empty reference trajectory: {ref_traj_path}")
            continue

        has_hydrogen = contains_hydrogen(ref_traj_all)

        print(
            f"  Reference frames={len(ref_traj_all)}, dt={ref_frame_dt_fs:.3f} fs "
            f"(contains H: {'yes' if has_hydrogen else 'no'})"
        )

        # Cache RDFs for truncated reference lengths used by multiple models.
        ref_rdf_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # ---------- Models ----------
        for model in model_names:
            print(f"  Model: {model}")

            mlip_path = os.path.join(
                mlip_base, system, f"nvt_{model}.extxyz"
            )

            if not os.path.exists(mlip_path):
                print(f"    [SKIP] Missing MLIP trajectory: {mlip_path}")
                continue

            mlip_traj_all = as_traj_list(read(mlip_path, ":"))
            if len(mlip_traj_all) == 0:
                print(f"    [SKIP] Empty MLIP trajectory: {mlip_path}")
                continue

            n_ref_use, n_mlip_use, matched_time_fs = matched_frame_counts(
                n_ref_total=len(ref_traj_all),
                n_mlip_total=len(mlip_traj_all),
                ref_dt_fs=ref_frame_dt_fs,
                mlip_dt_fs=mlip_frame_dt_fs,
            )

            if n_ref_use <= 0 or n_mlip_use <= 0:
                print("    [SKIP] Could not find positive matched simulation time.")
                continue

            print(
                f"    matched time={matched_time_fs:.3f} fs | "
                f"ref frames={n_ref_use}/{len(ref_traj_all)} | "
                f"mlip frames={n_mlip_use}/{len(mlip_traj_all)}"
            )

            ref_time_used = n_ref_use * ref_frame_dt_fs
            mlip_time_used = n_mlip_use * mlip_frame_dt_fs
            if abs(ref_time_used - mlip_time_used) > 1e-12:
                print(
                    f"    [WARN] Time mismatch after truncation: "
                    f"ref={ref_time_used:.6f} fs, mlip={mlip_time_used:.6f} fs"
                )

            ref_traj = ref_traj_all[:n_ref_use]
            mlip_traj = mlip_traj_all[:n_mlip_use]

            if n_ref_use not in ref_rdf_cache:
                md_ref = ase_to_mdtraj(ref_traj)
                ref_rdf_cache[n_ref_use] = compute_rdf(md_ref)

            ref_rdf = ref_rdf_cache[n_ref_use]
            save_rdf_csv(
                ref_rdf[0],
                ref_rdf[1],
                os.path.join(RESULTS_DIR, "rdf_same_simulation_length_saved", "reference", f"{system}__{model}.csv"),
            )

            md_mlip = ase_to_mdtraj(mlip_traj)

            mlip_rdf = compute_rdf(md_mlip)
            save_rdf_csv(mlip_rdf[0], mlip_rdf[1], os.path.join(RESULTS_DIR, "rdf_same_simulation_length_saved", "mlip", model, f"{system}.csv"))

            error = rdf_error(ref_rdf, mlip_rdf)

            print(f"    RDF error: {error:.6f} %")

            results[model].append(error)
            detailed_results[model].append({
                "System": system,
                "RDF_Error": error
            })

    # ---------- Aggregate (per model, mean over all systems) ----------
    print("\n================ FINAL SCORES ================")
    for model in model_names:
        if results[model]:
            mean_error = float(np.mean(results[model]))
        else:
            mean_error = float("nan")
        results[model] = mean_error
        print(f"{model:20s} : {mean_error:10.6f} %")

    # -------- Save CSVs --------
    with open(os.path.join(RESULTS_DIR, "rdf_similarity_scores_same_simulation_length.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Calculator", "Mean RDF Error [%]"])
        for model, error in results.items():
            writer.writerow([model, f"{error:.3f}"])

    for model in model_names:
        with open(os.path.join(RESULTS_DIR, f"rdf_similarity_scores_same_simulation_length_{model}.csv"), "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["System", "RDF_Error"]
            )
            writer.writeheader()
            for row in detailed_results[model]:
                writer.writerow(row)

    print("\nCSV files written successfully.")

    # ---------- Aggregate by system type (directly from in-memory results) ----------
    print("\n================ AGGREGATING BY SYSTEM TYPE ================")
    results_by_type_df = aggregate_by_system_type(detailed_results)
    results_by_type_df.to_csv(BY_SYSTEM_TYPE_OUTPUT_FILE)

    print(f"\nResults saved to {BY_SYSTEM_TYPE_OUTPUT_FILE}")
    print("\nSummary:")
    print(results_by_type_df.to_string())
