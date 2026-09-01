#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "ase>=3.26",
#   "h5py>=3.11",
#   "mdtraj>=1.10",
#   "numpy>=1.26",
#   "pandas>=2.2",
# ]
# ///

from __future__ import annotations

import argparse
import csv
import numpy as np
import pandas as pd
import mdtraj as mdt
import h5py
from ase.data import chemical_symbols
from ase.io import iread
from pathlib import Path


# ============================================================
# Paths (all relative to this script: settings/outputs live here,
# trajectories come from ../data)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

RESULTS_DIR = BASE_DIR / "results"

REF_TRAJ_BASE_DIR = DATA_DIR / "ref-trajs"
MLIP_TRAJ_BASE_DIR = DATA_DIR / "mlip-trajs-torchsim-matched"

# ============================================================
# Trajectory loading
# ============================================================

def build_topology(symbols: list[str]) -> mdt.Topology:
    top = mdt.Topology()
    chain = top.add_chain()
    res = top.add_residue("SYS", chain)

    for s in symbols:
        top.add_atom(
            s,
            element=mdt.element.get_by_symbol(s),
            residue=res
        )

    return top


def make_mdtraj(xyz_angstrom: np.ndarray, cells_angstrom: np.ndarray, symbols: list[str]) -> mdt.Trajectory:
    """Build an MDTraj trajectory from arrays stored in Angstrom."""
    if len(xyz_angstrom) == 0:
        raise ValueError("trajectory contains no frames")

    xyz_nm = np.asarray(xyz_angstrom, dtype=np.float32) / np.float32(10.0)
    cells_nm = np.asarray(cells_angstrom, dtype=np.float32) / np.float32(10.0)
    md_traj = mdt.Trajectory(xyz=xyz_nm, topology=build_topology(symbols))
    md_traj.unitcell_vectors = cells_nm
    return md_traj


def load_reference_trajectory(path: Path) -> mdt.Trajectory:
    """Stream an ASE-readable reference trajectory into MDTraj."""
    positions: list[np.ndarray] = []
    cells: list[np.ndarray] = []
    symbols: list[str] | None = None

    for atoms in iread(path, index=":"):
        if symbols is None:
            symbols = atoms.get_chemical_symbols()
        positions.append(np.asarray(atoms.positions, dtype=np.float32))
        cells.append(np.asarray(atoms.cell.array, dtype=np.float32))

    if symbols is None:
        raise ValueError("trajectory contains no frames")

    return make_mdtraj(np.stack(positions), np.stack(cells), symbols)


def h5_frame_count(path: Path) -> int:
    """Return the number of complete position frames in a TorchSim HDF5 file."""
    with h5py.File(path, "r") as h5:
        if "data/positions" not in h5:
            raise ValueError("missing data/positions dataset")
        return int(h5["data/positions"].shape[0])


def load_mlip_trajectory(path: Path, n_frames: int) -> mdt.Trajectory:
    """Load a TorchSim HDF5 trajectory without requiring torch-sim."""
    with h5py.File(path, "r") as h5:
        required = ("data/positions", "data/cell", "data/atomic_numbers")
        missing = [name for name in required if name not in h5]
        if missing:
            raise ValueError(f"missing HDF5 datasets: {', '.join(missing)}")

        positions = h5["data/positions"][:n_frames]
        cells_dataset = h5["data/cell"]
        cells = cells_dataset[:n_frames]
        if len(cells) == 1 and n_frames > 1:
            cells = np.repeat(cells, n_frames, axis=0)

        atomic_numbers = np.asarray(h5["data/atomic_numbers"][0], dtype=int)

    if len(positions) != n_frames or len(cells) != n_frames:
        raise ValueError(
            f"requested {n_frames} frames but found "
            f"{len(positions)} position and {len(cells)} cell frames"
        )

    symbols = [chemical_symbols[number] for number in atomic_numbers]
    return make_mdtraj(positions, cells, symbols)


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


def save_rdf_csv(r, g, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([r, g]), delimiter=',', header='r_A,g_r', comments='')


# ============================================================
# Utilities
# ============================================================

def matched_frame_count(n_ref_total: int, n_mlip_total: int) -> int:
    """Return the common prefix length for same-timestep trajectories."""
    if n_ref_total <= 0 or n_mlip_total <= 0:
        return 0
    return min(n_ref_total, n_mlip_total)


# ============================================================
# System-type aggregation
# ============================================================

SYSTEMS = {
    "Pure metals": ["bulkAu_1500K_Kapil", "bulkAg_600K_Kapil", "bulkCu_1000K_Kapil"],
    "Perovskites": ["CsSnI3_500K_Ivor_VASP", "MAPbBr3_300K_Ivor_VASP"],
    "Metal dichalcogenides": ["bulkMoS2_300K_NO-VdW_J.Kioseoglou_VASP", "TiSe2_400K_Ivor_VASP"],
    "Metal alloys": ["bulkCuAu_500K-Artrith_VASP", "bulkCuZrAl_1500K_A.Wadowski-J.Schmidt_VASP", "bulkLiMgAlZnSn_600K_J_Schmidt_VASP", "bulkLiMgAlZnSn_900K_J_Schmidt_VASP", "bulkPt3Co_300K_J.Kioseoglou_VASP"],
    "Molecular crystals": ["anthracene_293K_Sharma_S", "naphthalene_295K_Sharma_S", "pentacene_295K_Sharma_S", "picene_295K_Sharma_S", "tetracene_295K_Sharma_S"],
    "Metal-water interfaces": ["Pt111w24H2O_380K_Heenen_VASP"],
    "Hydrogen": ["H_1050K_Rupp_QE"],
}

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
# Discovery and main program
# ============================================================

def discover_mlip_trajectories(base_dir: Path) -> dict[str, dict[str, Path]]:
    """Discover every completed ``nvt_<model>.h5`` trajectory."""
    trajectories: dict[str, dict[str, Path]] = {}
    for path in sorted(base_dir.glob("*/nvt_*.h5")):
        model = path.stem.removeprefix("nvt_")
        trajectories.setdefault(path.parent.name, {})[model] = path
    return trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RDFs and RDF errors for matched TorchSim MLIP and reference trajectories."
    )
    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help="process only this system (repeatable; default: all)",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="process only this model (repeatable; default: all discovered models)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"output directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show discovered inputs without loading trajectories or computing RDFs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()

    mlip_trajectories = discover_mlip_trajectories(MLIP_TRAJ_BASE_DIR)
    reference_trajectories = {
        path.parent.name: path
        for path in sorted(REF_TRAJ_BASE_DIR.glob("*/traj.extxyz"))
    }

    selected_systems = set(args.systems) if args.systems else None
    selected_models = set(args.models) if args.models else None
    if selected_systems is not None:
        reference_trajectories = {
            system: path
            for system, path in reference_trajectories.items()
            if system in selected_systems
        }
        mlip_trajectories = {
            system: models
            for system, models in mlip_trajectories.items()
            if system in selected_systems
        }
    if selected_models is not None:
        mlip_trajectories = {
            system: {
                model: path
                for model, path in models.items()
                if model in selected_models
            }
            for system, models in mlip_trajectories.items()
        }

    model_names = sorted({
        model
        for system_models in mlip_trajectories.values()
        for model in system_models
    })

    print(
        f"Found {len(reference_trajectories)} reference trajectories and "
        f"{sum(len(models) for models in mlip_trajectories.values())} completed MLIP trajectories"
    )
    print(f"Models ({len(model_names)}): {', '.join(model_names)}")

    if args.dry_run:
        for system in sorted(reference_trajectories):
            models = sorted(mlip_trajectories.get(system, {}))
            print(f"{system}: {len(models)} MLIP trajectories")
        return

    if not reference_trajectories:
        raise RuntimeError(f"No reference trajectories found in {REF_TRAJ_BASE_DIR}")
    if not model_names:
        raise RuntimeError(f"No completed MLIP trajectories found in {MLIP_TRAJ_BASE_DIR}")

    results_dir.mkdir(parents=True, exist_ok=True)
    rdf_save_dir = results_dir / "rdf_same_simulation_length_saved"

    results: dict[str, list[float] | float] = {model: [] for model in model_names}
    detailed_results: dict[str, list[dict]] = {model: [] for model in model_names}

    mlip_only_systems = sorted(set(mlip_trajectories) - set(reference_trajectories))
    for system in mlip_only_systems:
        print(f"[WARN] MLIP trajectories have no reference trajectory: {system}")

    for system, ref_path in sorted(reference_trajectories.items()):
        print(f"\n=== System: {system} ===")

        try:
            md_ref_all = load_reference_trajectory(ref_path)
            ref_rdf_full = compute_rdf(md_ref_all)
        except Exception as exc:
            print(f"  [SKIP] Could not load/compute reference RDF: {exc}")
            continue

        n_ref_total = md_ref_all.n_frames
        print(f"  Reference frames={n_ref_total}")
        save_rdf_csv(
            ref_rdf_full[0],
            ref_rdf_full[1],
            rdf_save_dir / "reference" / f"{system}.csv",
        )

        # Almost all matched HDF5 files contain one additional initial frame.
        # Cache the full reference RDF and any shorter prefixes needed by
        # genuinely short but otherwise valid MLIP trajectories.
        ref_rdf_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {
            n_ref_total: ref_rdf_full,
        }

        system_models = mlip_trajectories.get(system, {})
        for model, mlip_path in sorted(system_models.items()):
            print(f"  Model: {model}")

            try:
                n_mlip_total = h5_frame_count(mlip_path)
                n_frames_use = matched_frame_count(n_ref_total, n_mlip_total)
                if n_frames_use <= 0:
                    raise ValueError("no common frames")

                print(
                    f"    matched frames={n_frames_use} | "
                    f"ref={n_frames_use}/{n_ref_total} | "
                    f"mlip={n_frames_use}/{n_mlip_total}"
                )

                if n_frames_use not in ref_rdf_cache:
                    ref_rdf_cache[n_frames_use] = compute_rdf(md_ref_all[:n_frames_use])
                ref_rdf = ref_rdf_cache[n_frames_use]

                # Preserve the exact shorter reference RDF used for a score.
                if n_frames_use != n_ref_total:
                    save_rdf_csv(
                        ref_rdf[0],
                        ref_rdf[1],
                        rdf_save_dir / "reference_matched" / model / f"{system}.csv",
                    )

                md_mlip = load_mlip_trajectory(mlip_path, n_frames_use)
                mlip_rdf = compute_rdf(md_mlip)
                save_rdf_csv(
                    mlip_rdf[0],
                    mlip_rdf[1],
                    rdf_save_dir / "mlip" / model / f"{system}.csv",
                )
                error = rdf_error(ref_rdf, mlip_rdf)
            except Exception as exc:
                print(f"    [SKIP] Could not load/compute MLIP RDF: {exc}")
                continue

            print(f"    RDF error: {error:.6f} %")
            assert isinstance(results[model], list)
            results[model].append(error)
            detailed_results[model].append({"System": system, "RDF_Error": error})

    print("\n================ FINAL SCORES ================")
    for model in model_names:
        model_errors = results[model]
        assert isinstance(model_errors, list)
        mean_error = float(np.mean(model_errors)) if model_errors else float("nan")
        results[model] = mean_error
        print(f"{model:20s} : {mean_error:10.6f} %")

    with open(results_dir / "rdf_similarity_scores_same_simulation_length.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Calculator", "Mean RDF Error [%]"])
        for model, error in results.items():
            writer.writerow([model, f"{error:.3f}"])

    for model in model_names:
        with open(
            results_dir / f"rdf_similarity_scores_same_simulation_length_{model}.csv",
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["System", "RDF_Error"])
            writer.writeheader()
            writer.writerows(detailed_results[model])

    print("\nCSV files written successfully.")
    print("\n================ AGGREGATING BY SYSTEM TYPE ================")
    results_by_type_df = aggregate_by_system_type(detailed_results)
    by_type_output_file = results_dir / "rdf_similarity_scores_by_system_type_same_simulation_length.csv"
    results_by_type_df.to_csv(by_type_output_file)

    print(f"\nResults saved to {by_type_output_file}")
    print("\nSummary:")
    print(results_by_type_df.to_string())


if __name__ == "__main__":
    main()
