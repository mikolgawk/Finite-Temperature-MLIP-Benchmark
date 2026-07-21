"""
Generic RMSE script: reads trajectory files, extracts temperatures,
and computes MLIP energies/forces RMSE against reference data using a
calculator selected at runtime from model_calculators.json.

Saves detailed results (plots, metrics) inside the SAME directory
as the input trajectory file.

Usage:
    MODEL_NAME=mace-mpa-0 python rmse_script-generic.py [--debug] [--output-dir results]
"""

import os
import re
import json
import time
import argparse
from collections import Counter
import numpy as np
from ase.io import read
import pandas as pd
from pathlib import Path
from scipy.stats import entropy
import matplotlib.pyplot as plt


MODEL_CATALOG_PATH = os.path.join(os.path.dirname(__file__), 'model_calculators.json')

# Per-system isolated-atom reference files, keyed by the system name parsed by
# parse_system_info() (the directory prefix before the first '_'). Each entry
# maps element symbol -> path to a single-atom .extxyz file with a REF_energy
# info tag, computed at the same level of theory as that system's trajectory.
# Isolated-atom energies are DFT-code/settings-specific, so files are NOT
# shared across systems computed with different codes (e.g. the aromatic
# systems use VASP references).
# Systems with no entry here are evaluated without any energy correction.
_AROMATIC_ISOLATED_ATOMS = {
    'C': '../ref-trajs/naphthalene_295K_Sharma_S/isolated_atom_C.extxyz',
    'H': '../ref-trajs/naphthalene_295K_Sharma_S/isolated_atom_H.extxyz',
}
ISOLATED_ATOM_FILES = {
    name: _AROMATIC_ISOLATED_ATOMS for name in
    ('anthracene', 'picene', 'naphthalene', 'pentacene', 'tetracene')
}

# Global dictionary to track issues
issues = {
    'failed_file_reads': [],
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
    return eval(model_entry['calculator_expr'], namespace)


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


def get_file_names(directory='../ref-trajs/', extension='.extxyz', prefix='traj'):
    """Recursive search for trajectory files."""
    base_path = Path(directory)

    file_paths = []
    for file_path in base_path.rglob(f"{prefix}*{extension}"):
        file_paths.append(str(file_path))

    print(f"\nFound {len(file_paths)} trajectory files.")
    return file_paths


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


def kl_divergence_scipy(energies_mlip, energies_ref, bins=50):
    """Calculate KL divergence."""
    epsilon = 1e-10

    all_e = np.concatenate([energies_mlip, energies_ref])
    bin_edges = np.linspace(all_e.min(), all_e.max(), bins + 1)

    hist_mlip, _ = np.histogram(energies_mlip, bins=bin_edges, density=True)
    hist_ref, _ = np.histogram(energies_ref, bins=bin_edges, density=True)

    hist_mlip = np.maximum(hist_mlip, epsilon)
    hist_ref = np.maximum(hist_ref, epsilon)

    return entropy(hist_ref, hist_mlip)


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


def normalize_energies(frames_list, isolated_atom_files, calculator, calc_name=None):
    """
    Normalizes energies by subtracting per-element isolated-atom references
    (energy_rmse becomes an atomization-energy RMSE rather than a raw total
    energy RMSE) and also measures MLIP force evaluation cost.

    isolated_atom_files: dict of element symbol -> path to a single-atom
    .extxyz file with a REF_energy info tag. Elements present in a frame but
    absent from isolated_atom_files are left uncorrected (zero offset), so
    passing {} reproduces a raw, unnormalized comparison.
    """
    ref_e0, mlip_e0 = load_isolated_atom_energies(isolated_atom_files, calculator, calc_name)

    uncorrected_elements = {
        symbol for frame in frames_list for symbol in frame.get_chemical_symbols()
    } - ref_e0.keys()
    if uncorrected_elements:
        print(
            f"    [WARN] No isolated-atom reference for element(s) "
            f"{sorted(uncorrected_elements)}; leaving them uncorrected."
        )

    def isolated_atom_offset(frame, e0):
        counts = Counter(frame.get_chemical_symbols())
        return sum(counts[element] * e0[element] for element in e0)

    # Extract reference data
    ref_energies = []
    ref_forces = []
    for frame in frames_list:
        try:
            ref_energies.append(frame.get_potential_energy())
            ref_forces.append(frame.get_forces())
        except Exception as e:
            print(f"Warning: Could not extract reference data from frame: {e}")
            continue

    if not ref_energies:
        raise ValueError("No valid reference energies found")

    normalized_energies = [
        energy - isolated_atom_offset(frame, ref_e0)
        for frame, energy in zip(frames_list, ref_energies)
    ]

    # Compute MLIP predictions + time FORCE call
    mlip_energies = []
    mlip_forces = []
    mlip_frames = []
    total_force_eval_time_s = 0.0
    successful_force_evals = 0

    for i, frame in enumerate(frames_list):
        try:
            frame.calc = calculator

            t0 = time.perf_counter()
            forces = frame.get_forces()
            t1 = time.perf_counter()

            energy = frame.get_potential_energy()

            mlip_energies.append(energy)
            mlip_forces.append(forces)
            mlip_frames.append(frame)

            total_force_eval_time_s += (t1 - t0)
            successful_force_evals += 1

        except Exception as e:
            print(f"Warning: Could not compute MLIP predictions for frame {i}: {e}")
            continue

    if len(mlip_energies) != len(ref_energies):
        print(
            f"Warning: Mismatch in number of frames - ref: {len(ref_energies)}, "
            f"mlip: {len(mlip_energies)}"
        )

    normalized_mlip_energies = [
        energy - isolated_atom_offset(frame, mlip_e0)
        for frame, energy in zip(mlip_frames, mlip_energies)
    ]

    energies_mlip = np.array(normalized_mlip_energies)
    energies_normalized_ref = np.array(normalized_energies[:len(normalized_mlip_energies)])

    forces_mlip = np.concatenate([f.flatten() for f in mlip_forces])
    forces_ref = np.concatenate([f.flatten() for f in ref_forces[:len(mlip_forces)]])

    natoms = frames_list[0].get_global_number_of_atoms()
    energy_rmse = np.sqrt(np.mean((energies_mlip - energies_normalized_ref)**2)) * (1 / natoms)
    force_rmse = np.sqrt(np.mean((forces_mlip - forces_ref)**2))

    force_eval_time_per_call_s = (
        total_force_eval_time_s / successful_force_evals if successful_force_evals > 0 else np.nan
    )
    force_eval_time_per_call_per_atom_s = (
        force_eval_time_per_call_s / natoms if successful_force_evals > 0 and natoms > 0 else np.nan
    )

    return (
        energies_mlip,
        forces_mlip,
        energies_normalized_ref,
        forces_ref,
        energy_rmse,
        force_rmse,
        total_force_eval_time_s,
        successful_force_evals,
        force_eval_time_per_call_s,
        force_eval_time_per_call_per_atom_s,
    )


def main():
    parser = argparse.ArgumentParser(description='Evaluate MLIP in-place')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--output-dir', type=str, default='results-rmses', help='Global summary location')
    parser.add_argument('--bins', type=int, default=50)
    args = parser.parse_args()

    # Setup global output directory
    global_output_dir = Path(args.output_dir)
    global_output_dir.mkdir(exist_ok=True, parents=True)

    file_paths = get_file_names()

    if not file_paths:
        print("No files found.")
        return

    # Initialize Calculator from the model catalog (model_calculators.json)
    model_name = os.environ.get('MODEL_NAME')
    catalog = load_model_catalog()
    if model_name not in catalog:
        raise SystemExit(f"Set MODEL_NAME to one of: {', '.join(sorted(catalog))}")

    try:
        calc = build_calculator(catalog[model_name])
        calc_name = model_name
        print(f"✓ Loaded {calc_name}")
    except Exception as e:
        print(f"✗ Failed to load calculator: {e}")
        return

    all_metrics = []

    print("\nProcessing files...")
    for i, file_path in enumerate(file_paths, 1):
        print(f"\n[{i}/{len(file_paths)}] Processing: {file_path}")

        # Output directory
        parent_dir = os.path.basename(os.path.dirname(file_path))
        local_output_dir = Path('../outputs') / parent_dir
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
                total_force_eval_time_s,
                num_force_evals,
                force_eval_time_per_call_s,
                force_eval_time_per_call_per_atom_s,
            ) = normalize_energies(frames, isolated_atom_files, calc, calc_name)

            # Metrics
            if e_mlip is not None and e_ref is not None:
                bias = np.mean(e_mlip) - np.mean(e_ref)
                kl_raw = kl_divergence_scipy(e_mlip, e_ref, args.bins)
                kl_shift = kl_divergence_scipy(
                    e_mlip - np.mean(e_mlip),
                    e_ref - np.mean(e_ref),
                    args.bins
                )
            else:
                bias = 0.0
                kl_raw = 0.0
                kl_shift = 0.0

            natoms = frames[0].get_global_number_of_atoms()

            metrics = {
                'system': sys_name,
                'temperature_K': temp,
                'reference_key': ref_key,
                'calculator': calc_name,
                'natoms': natoms,
                'energy_rmse': rmse_e,
                'force_rmse': rmse_f,
                'mlip_total_force_eval_time_s': total_force_eval_time_s,
                'mlip_num_force_evals': num_force_evals,
                'mlip_force_eval_time_per_call_s': force_eval_time_per_call_s,
                'mlip_force_eval_time_per_call_per_atom_s': force_eval_time_per_call_per_atom_s,
                # 'bias': bias,
                # 'kl_not_shifted': kl_raw,
                # 'kl_shifted': kl_shift,
                # 'n_frames': len(e_mlip) if e_mlip is not None else len(frames)
            }

            print(
                f"   E_RMSE: {rmse_e:.6f} | "
                f"F_RMSE: {rmse_f:.6f} | "
                f"Bias: {bias:.6f} | "
                f"force_t/call: {force_eval_time_per_call_s:.6f} s | "
                f"force_t/call/atom: {force_eval_time_per_call_per_atom_s:.6e} s"
            )

            all_metrics.append(metrics)

            # Save local plots
            if e_mlip is not None and e_ref is not None:
                hist, edges = histogram_energies(e_mlip, bins=args.bins)
                plot_histogram(hist, edges, f"Hist_{sys_name}_{temp}K", local_output_dir, debug=args.debug)
                plot_comparison(e_mlip, e_ref, f"Compare_{sys_name}_{temp}K", local_output_dir)

        except Exception as e:
            print(f"   Error evaluating: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()

    # Save all results to single CSV
    if all_metrics:
        results_path = global_output_dir / f'rmse-results-ref-trajs-{calc_name}.csv'
        df = pd.DataFrame(all_metrics)
        df.to_csv(results_path, index=False)
        print(f"\n✓ All results saved to: {results_path}")

    print_summary()


if __name__ == "__main__":
    main()
