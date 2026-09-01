"""
Generic RMSE script: reads trajectory files, extracts temperatures,
and computes MLIP energies/forces RMSE against reference data using a
calculator selected at runtime from model_calculators.json.

Reads reference trajectories from ../data/ref-trajs and saves detailed
per-trajectory results (histograms, comparison plots) into this script's
own outputs/ directory, one subdirectory per trajectory. The per-model
summary CSV is written to ../data/e-f-predictions/ as
rmse-results-all_<model>.csv, with one row per evaluated system/temperature.

Usage:
    MODEL_NAME=mace-mpa-0 uv run rmse_script-generic.py [--debug]

The model-specific ``rmse_*.py`` entry points in this directory provide the
dependencies and set ``MODEL_NAME`` automatically.
"""

import os
import re
import json
import argparse
from collections import Counter
import numpy as np
from ase.io import read
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / 'data'
REF_TRAJ_DIR = DATA_DIR / 'ref-trajs'

MODEL_CATALOG_PATH = BASE_DIR / 'model_calculators.json'

# Per-system isolated-atom reference files, keyed by the system name parsed by
# parse_system_info() (the directory prefix before the first '_'). Each entry
# maps element symbol -> path to a single-atom .extxyz file with a REF_energy
# info tag, computed at the same level of theory as that system's trajectory.
# Isolated-atom energies are DFT-code/settings-specific, so files are NOT
# shared across systems computed with different codes (e.g. the aromatic
# systems use VASP references, H_*_Rupp_QE uses Quantum Espresso references,
# and the two H atom energies differ accordingly).
# Systems with no entry here are evaluated without any energy correction.
_AROMATIC_ISOLATED_ATOMS = {
    'C': str(REF_TRAJ_DIR / 'naphthalene_295K_Sharma_S' / 'isolated_atom_C.extxyz'),
    'H': str(REF_TRAJ_DIR / 'naphthalene_295K_Sharma_S' / 'isolated_atom_H.extxyz'),
}
_RUPP_QE_HYDROGEN_ISOLATED_ATOM = str(
    DATA_DIR / 'Hydrogen_E0' / 'isolated_atom_H.extxyz'
)
ISOLATED_ATOM_FILES = {
    **{name: _AROMATIC_ISOLATED_ATOMS for name in
       ('anthracene', 'picene', 'naphthalene', 'pentacene', 'tetracene')},
    'H': {'H': _RUPP_QE_HYDROGEN_ISOLATED_ATOM},
}

# Global dictionary to track issues
issues = {
    'failed_file_reads': [],
    'failed_evaluations': [],
}


def load_model_catalog(path=MODEL_CATALOG_PATH):
    """Loads the model calculator catalog from JSON."""
    with open(path) as f:
        catalog = json.load(f)
    return {model['name']: model for model in catalog['models']}


def build_calculator(model_entry):
    """
    Executes a model's import statements and evaluates its calculator
    expression, returning the constructed ASE calculator instance.
    """
    namespace = {}
    for import_line in model_entry['imports']:
        exec(import_line, namespace)
    # Checkpoint paths in the catalog are written relative to this directory
    # ('../data/...'). Resolve them against DATA_DIR so the calculators load
    # regardless of the current working directory the script was launched from.
    # as_posix() keeps forward slashes, avoiding backslash escapes in the
    # eval'd string literal on non-POSIX shells.
    expr = model_entry['calculator_expr'].replace('../data/', f'{DATA_DIR.as_posix()}/')
    return eval(expr, namespace)


def log(message, level='info', debug=False):
    """Controlled logging."""
    if level == 'info' or debug:
        print(message)


def read_trajectory(file_path, debug=False):
    """Reads a trajectory file."""
    try:
        # removing index=':' reads only last frame by default in some ASE versions,
        # ensuring we read all frames:
        structures = read(file_path, index=':')
        log(f"Successfully read {len(structures)} frames from {file_path}", debug=debug)
        return structures
    except Exception as e:
        error_msg = f"Error reading {file_path}: {e}"
        print(error_msg)
        issues['failed_file_reads'].append((file_path, str(e)))
        return None


def get_file_names(directory=REF_TRAJ_DIR, extension='.extxyz', prefix='traj'):
    """Recursive search for trajectory files."""
    base_path = Path(directory)

    file_paths = []
    for file_path in base_path.rglob(f"{prefix}*{extension}"):
        file_paths.append(str(file_path))

    print(f"\nFound {len(file_paths)} trajectory files.")
    return sorted(file_paths)


def parse_system_info(file_path):
    """
    Extract system name and temperature from directory name.

    Args:
        file_path: Path to trajectory file

    Returns:
        tuple: (system_name, temperature_K, reference_key)
    """
    parent_dir = os.path.basename(os.path.dirname(file_path))

    # Extract system name (everything before first underscore)
    parts = parent_dir.split('_')
    system_name = parts[0] if parts else "unknown"

    # Extract temperature
    match = re.search(r'(\d+)K', parent_dir)
    if match:
        tempK = int(match.group(1))
    else:
        tempK = 0

    reference_key = f"{system_name}_{tempK}K"

    return system_name, tempK, reference_key


def histogram_energies(energies, bins=50):
    """Computes histogram."""
    hist, bin_edges = np.histogram(energies, bins=bins, density=True)
    return hist, bin_edges


def plot_histogram(hist, bin_edges, title, output_dir, debug=False):
    """Plots histogram to specific directory."""
    try:
        plt.figure(figsize=(8, 6))
        plt.bar(
            bin_edges[:-1],
            hist,
            width=np.diff(bin_edges),
            edgecolor='black',
            alpha=0.7,
            color='steelblue'
        )
        plt.title(title, fontsize=14)
        plt.xlabel('Energy (eV)', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.grid(alpha=0.3, linestyle='--')

        output_path = output_dir / f"{title}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        log(f"      Saved plot to {output_path}", 'debug', debug)
    except Exception as e:
        print(f"      Warning: Could not create plot: {e}")


def plot_comparison(energies_mlip, energies_ref, title, output_dir):
    """Plots comparison histogram to specific directory."""
    try:
        plt.figure(figsize=(10, 6))
        all_e = np.concatenate([energies_mlip, energies_ref])
        bins = np.linspace(all_e.min(), all_e.max(), 50)

        plt.hist(energies_ref, bins=bins, alpha=0.5, label='Reference',
                 color='green', density=True)
        plt.hist(energies_mlip, bins=bins, alpha=0.5, label='MLIP',
                 color='red', density=True)

        plt.title(title)
        plt.legend()

        output_path = output_dir / f"{title}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"      Warning: Could not create comparison plot: {e}")


def print_summary():
    """Print summary."""
    print(f"\n{'='*60}\nEVALUATION SUMMARY\n{'='*60}")
    if issues['failed_file_reads']:
        print(f"❌ Failed to read {len(issues['failed_file_reads'])} files")
    if issues['failed_evaluations']:
        print(f"❌ Failed to evaluate {len(issues['failed_evaluations'])} trajectories")
    if not any(issues.values()):
        print("✅ All evaluations completed without issues!")


def load_isolated_atom_energies(isolated_atom_files, calculator, calc_name=None):
    """
    Reads each element's isolated-atom reference structure and evaluates the
    calculator on it once. Returns (ref_e0, mlip_e0), each a dict of
    element symbol -> single-atom energy.
    """
    ref_e0 = {}
    mlip_e0 = {}
    for element, path in isolated_atom_files.items():
        atom = read(path, 0)
        ref_e0[element] = atom.info["REF_energy"]
        if calc_name == 'chgnet':
            # CHGNet requires a defined periodic cell; give the isolated atom
            # a large non-periodic box instead of the file's empty/zero cell.
            atom.set_cell([100, 100, 100])
            atom.center()
            atom.set_pbc([False, False, False])
        atom.calc = calculator
        mlip_e0[element] = atom.get_potential_energy()
    return ref_e0, mlip_e0


def normalize_energies(
    frames_list,
    isolated_atom_files,
    calculator,
    calc_name=None,
    show_progress=True,
):
    """
    Normalizes energies by subtracting per-element isolated-atom references
    (energy_rmse becomes an atomization-energy RMSE rather than a raw total
    energy RMSE).

    isolated_atom_files: dict of element symbol -> path to a single-atom
    .extxyz file with a REF_energy info tag. Elements present in a frame but
    absent from isolated_atom_files are left uncorrected (zero offset), so
    passing {} reproduces a raw, unnormalized comparison.
    """
    ref_e0, mlip_e0 = load_isolated_atom_energies(isolated_atom_files, calculator, calc_name)

    uncorrected_elements = {
        symbol for frame in frames_list for symbol in frame.get_chemical_symbols()
    } - ref_e0.keys()
    # if uncorrected_elements:
    #     print(
    #         f"    [WARN] No isolated-atom reference for element(s) "
    #         f"{sorted(uncorrected_elements)}; leaving them uncorrected."
    #     )

    def isolated_atom_offset(frame, e0):
        counts = Counter(frame.get_chemical_symbols())
        return sum(counts[element] * e0[element] for element in e0)

    # Extract reference data, keyed by frame index so a frame that fails here
    # drops from the comparison on BOTH sides rather than shifting the pairing.
    ref_data = {}  # frame index -> (energy, forces array)
    for i, frame in enumerate(frames_list):
        try:
            ref_data[i] = (frame.get_potential_energy(), frame.get_forces())
        except Exception as e:
            print(f"Warning: Could not extract reference data from frame {i}: {e}")
            continue

    if not ref_data:
        raise ValueError("No valid reference energies found")

    # Compute MLIP predictions, also keyed by frame index.
    mlip_data = {}  # frame index -> (energy, forces array)

    frame_progress = tqdm(
        frames_list,
        desc=f'{calc_name or "MLIP"} frames',
        unit='frame',
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for i, frame in enumerate(frame_progress):
        try:
            frame.calc = calculator
            forces = frame.get_forces()
            energy = frame.get_potential_energy()

            mlip_data[i] = (energy, forces)

        except Exception as e:
            print(f"Warning: Could not compute MLIP predictions for frame {i}: {e}")
            continue

    # Pair reference and MLIP results only on frames where BOTH succeeded, so a
    # mid-trajectory failure removes that frame from both arrays instead of
    # silently misaligning every later frame (positional truncation would not).
    common_indices = sorted(ref_data.keys() & mlip_data.keys())
    if not common_indices:
        raise ValueError("No frames where both reference and MLIP evaluations succeeded")

    if len(common_indices) != len(frames_list):
        print(
            f"Warning: aligned {len(common_indices)}/{len(frames_list)} frames "
            f"(dropped {len(ref_data) - len(common_indices)} where MLIP failed, "
            f"{len(mlip_data) - len(common_indices)} where reference failed)"
        )

    normalized_ref_energies = []
    normalized_mlip_energies = []
    forces_ref_list = []
    forces_mlip_list = []
    for i in common_indices:
        frame = frames_list[i]
        ref_energy, ref_force = ref_data[i]
        mlip_energy, mlip_force = mlip_data[i]
        normalized_ref_energy = ref_energy - isolated_atom_offset(frame, ref_e0)
        normalized_mlip_energy = mlip_energy - isolated_atom_offset(frame, mlip_e0)
        normalized_ref_energies.append(normalized_ref_energy)
        normalized_mlip_energies.append(normalized_mlip_energy)
        forces_ref_list.append(ref_force.flatten())
        forces_mlip_list.append(mlip_force.flatten())

    energies_normalized_ref = np.array(normalized_ref_energies)
    energies_mlip = np.array(normalized_mlip_energies)
    forces_ref = np.concatenate(forces_ref_list)
    forces_mlip = np.concatenate(forces_mlip_list)

    natoms = frames_list[0].get_global_number_of_atoms()
    energy_rmse = np.sqrt(np.mean((energies_mlip - energies_normalized_ref)**2)) * (1 / natoms)
    force_rmse = np.sqrt(np.mean((forces_mlip - forces_ref)**2))

    return (
        energies_mlip,
        forces_mlip,
        energies_normalized_ref,
        forces_ref,
        energy_rmse,
        force_rmse,
        len(common_indices),
    )


def main():
    parser = argparse.ArgumentParser(description='Evaluate MLIP in-place')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Global summary location. Default: '
                             '../data/e-f-predictions, which the figure scripts '
                             'read from. Relative paths resolve against this script.')
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable trajectory and frame progress bars.')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if this model\'s summary CSV already '
                             'exists (default: skip models already done).')
    args = parser.parse_args()

    model_name = os.environ.get('MODEL_NAME')
    catalog = load_model_catalog()
    if model_name not in catalog:
        raise SystemExit(f"Set MODEL_NAME to one of: {', '.join(sorted(catalog))}")

    # By default each model's summary CSV lands in ../data/e-f-predictions/
    # under the canonical rmse-results-all_<model>.csv name. Each row contains
    # the RMSE summary for one evaluated system/temperature.
    if args.output_dir is None:
        global_output_dir = DATA_DIR / 'e-f-predictions'
    else:
        global_output_dir = Path(args.output_dir)
        if not global_output_dir.is_absolute():
            global_output_dir = BASE_DIR / global_output_dir
    global_output_dir.mkdir(exist_ok=True, parents=True)

    file_paths = get_file_names()

    if not file_paths:
        print("No files found.")
        return

    # Skip models whose summary CSV already exists (lets a batch sweep resume
    # after an interruption without redoing finished models).
    results_path = global_output_dir / f'rmse-results-all_{model_name}.csv'
    if results_path.exists() and not args.force:
        print(f"✓ {results_path.name} already exists; skipping {model_name}. "
              f"Use --force to recompute.")
        return

    try:
        calc = build_calculator(catalog[model_name])
        calc_name = model_name
        print(f"✓ Loaded {calc_name}")
    except Exception as e:
        raise SystemExit(f"✗ Failed to load calculator {model_name}: {e}") from e

    all_metrics = []

    print("\nProcessing files...")
    trajectory_progress = tqdm(
        file_paths,
        desc=f'{model_name} trajectories',
        unit='traj',
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    for i, file_path in enumerate(trajectory_progress, 1):
        print(f"\n[{i}/{len(file_paths)}] Processing: {file_path}")

        # Output directory
        parent_dir = os.path.basename(os.path.dirname(file_path))
        local_output_dir = BASE_DIR / 'outputs' / model_name / parent_dir
        local_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"   ↳ Saving results to: {local_output_dir}")

        frames = read_trajectory(file_path, debug=args.debug)
        if frames is None:
            continue

        try:
            sys_name, temp, ref_key = parse_system_info(file_path)
        except Exception:
            sys_name, temp, ref_key = "Unknown", 0, f"unknown_{i}"

        # Evaluate (every system goes through isolated-atom-energy normalization;
        # systems with no registered isolated-atom files fall back to a raw,
        # uncorrected comparison)
        isolated_atom_files = ISOLATED_ATOM_FILES.get(sys_name, {})
        try:
            (
                e_mlip,
                f_mlip,
                e_ref,
                f_ref,
                rmse_e,
                rmse_f,
                n_evaluated_frames,
            ) = normalize_energies(
                frames,
                isolated_atom_files,
                calc,
                calc_name,
                show_progress=not args.no_progress,
            )

            # Metrics
            if e_mlip is not None and e_ref is not None:
                bias = np.mean(e_mlip) - np.mean(e_ref)
            else:
                bias = 0.0

            natoms = frames[0].get_global_number_of_atoms()

            metrics = {
                'system': sys_name,
                'temperature_K': temp,
                'reference_key': ref_key,
                'calculator': calc_name,
                'natoms': natoms,
                'n_reference_frames': len(frames),
                'n_evaluated_frames': n_evaluated_frames,
                'energy_rmse': rmse_e,
                'force_rmse': rmse_f,
                # 'bias': bias,
                # 'n_frames': len(e_mlip) if e_mlip is not None else len(frames)
            }

            print(
                f"   E_RMSE: {rmse_e:.6f} | "
                f"F_RMSE: {rmse_f:.6f} | "
                f"Bias: {bias:.6f}"
            )

            all_metrics.append(metrics)

            # Save local plots
            if e_mlip is not None and e_ref is not None:
                hist, edges = histogram_energies(e_mlip, bins=args.bins)
                plot_histogram(hist, edges, f"Hist_{sys_name}_{temp}K", local_output_dir, debug=args.debug)
                plot_comparison(e_mlip, e_ref, f"Compare_{sys_name}_{temp}K", local_output_dir)

        except Exception as e:
            print(f"   Error evaluating: {e}")
            issues['failed_evaluations'].append((file_path, str(e)))
            if args.debug:
                import traceback
                traceback.print_exc()

    # Save all results to single CSV (path resolved above, before evaluation)
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        df.to_csv(results_path, index=False)
        print(f"\n✓ All results saved to: {results_path}")

    print_summary()

    if issues['failed_file_reads'] or issues['failed_evaluations']:
        raise SystemExit(
            f"Evaluation incomplete: produced {len(all_metrics)}/{len(file_paths)} "
            "trajectory results"
        )


if __name__ == "__main__":
    main()
