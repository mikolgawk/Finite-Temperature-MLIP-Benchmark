'''
Generic MD timing script: reads trajectory files, extracts temperatures, and runs a
short NVT MD benchmark using a calculator selected at runtime from
model_calculators.json.

Usage:
    MODEL_NAME=mace-mpa-0 python md_script-generic.py
'''

import os
import re
import json
import time
import subprocess

import numpy as np
import pandas as pd
import torch
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase import units
from ase.md import MDLogger

# Configuration parameters
NVT_SIMULATION_LENGTH_PS = 0.2  # ps
NVT_TIMESTEP = 1.0  # fs default
NVT_HYDROGEN_TIMESTEP = 0.5  # fs
NVT_CUAU_TIMESTEP = 2.0  # fs
NVT_TAU = 25.0  # fs
NVT_RECORD_INTERVAL = 1

TRAJ_DIR = '../ref-trajs/'
OUTPUT_DIR = '../data/output-trajs-timings-paper/'
SKIP_SYSTEMS = [
    'anthracene', 'naphthalene', 'pentacene', 'picene', 'tetracene',
    'H_1050K_Rupp_QE', 'Pt111w24H2O_380K_Heenen_VASP',
]

# Systems outside the paper panel. Trajectories are discovered by scanning
# TRAJ_DIR, so these are excluded by name in case they are present there.
EXCLUDED_SYSTEMS = ['H_1050K_Rupp_QE', 'Pt111w24H2O_380K_Heenen_VASP']

MODEL_CATALOG_PATH = os.path.join(os.path.dirname(__file__), 'model_calculators.json')


def print_cuda_diagnostics():
    '''Prints CUDA visibility/availability and nvidia-smi output for the job log.'''
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    try:
        print("nvidia-smi output:")
        subprocess.run(["nvidia-smi"], check=True)
    except Exception as e:
        print("nvidia-smi failed:", e)


def load_model_catalog(path=MODEL_CATALOG_PATH):
    '''Loads the model calculator catalog from JSON.'''
    with open(path) as f:
        catalog = json.load(f)
    return {model['name']: model for model in catalog['models']}


def build_calculator(model_entry):
    '''
    Executes a model's import statements and evaluates its calculator
    expression, returning the constructed ASE calculator instance.
    '''
    namespace = {}
    for import_line in model_entry['imports']:
        exec(import_line, namespace)
    return eval(model_entry['calculator_expr'], namespace)


def get_nvt_timestep(system_name, atoms):
    if 'CuAu' in system_name:
        return NVT_CUAU_TIMESTEP
    if 'H' in set(atoms.get_chemical_symbols()):
        return NVT_HYDROGEN_TIMESTEP
    return NVT_TIMESTEP


def get_nvt_n_steps(time_step):
    return int(round(NVT_SIMULATION_LENGTH_PS * 1000.0 / time_step))


def synchronize_cuda():
    '''Synchronizes CUDA work before/after wall-clock timing, when available.'''
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def read_trajectory(file_path):
    '''Reads a trajectory file and returns a list of ASE Atoms objects.'''
    try:
        frames = read(file_path, index=':')
        print(f"  ✓ Read {len(frames)} frames from {file_path}")
        return frames
    except Exception as e:
        print(f"  ✗ Error reading {file_path}: {e}")
        return []


def get_file_names(directory=TRAJ_DIR, extension='.extxyz'):
    '''Gets a list of file names with a specific extension from subdirectories.'''
    directories = [d for d in os.listdir(directory)
                   if os.path.isdir(os.path.join(directory, d))]

    files_names = []

    print(f"Scanning {len(directories)} directories for {extension} files...")

    for dir_name in directories:
        dir_path = os.path.join(directory, dir_name)
        found_files = []

        for file in os.listdir(dir_path):
            if file.endswith(extension):
                file_path = os.path.join(dir_path, file)
                files_names.append(file_path)
                found_files.append(file)

        if found_files:
            print(f"  {dir_name}: found {len(found_files)} file(s)")

    return files_names, directories


def extract_temperatures(directories):
    '''Extracts temperatures from directory names and saves to CSV.'''
    data = []
    for dir_name in directories:
        match = re.search(r'(\d+)K', dir_name)
        if match:
            temperature = int(match.group(1))
            data.append({'Directory': dir_name, 'Temperature': temperature})
        else:
            print(f"  ⚠ No temperature found in directory name: {dir_name}")

    df = pd.DataFrame(data)
    df.to_csv('temperatures.csv', index=False)

    if not df.empty:
        print(f"\nTemperature summary:")
        print(f"  Range: {df['Temperature'].min()}-{df['Temperature'].max()} K")
        print(f"  Values: {sorted(df['Temperature'].unique())}")

    return df


def nvt_simulation(init_structure, temperature, n_steps, time_step,
                   calculator, calculator_name, tdamp, record_interval, file_path):
    '''Runs an NVT molecular dynamics simulation and times the MD loop.'''
    atoms = init_structure.copy()
    atoms.calc = calculator

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    dyn = NoseHooverChainNVT(
        atoms,
        temperature_K=temperature,
        timestep=time_step * units.fs,
        tdamp=tdamp * units.fs,
        tchain=1,
        tloop=1
    )

    file_path_output = os.path.basename(os.path.dirname(file_path))
    os.makedirs(f'{OUTPUT_DIR}{file_path_output}/', exist_ok=True)
    log_file = f'{OUTPUT_DIR}{file_path_output}/md_{calculator_name}.log'
    print(log_file)
    dyn.attach(
        MDLogger(dyn, atoms, log_file, header=True, stress=False,
                 peratom=False, mode="a"),
        interval=1
    )

    traj_frames = []

    def _store():
        traj_frames.append(atoms.copy())

    dyn.attach(_store, interval=record_interval)

    print(f"  Running {n_steps} MD steps...")
    synchronize_cuda()
    start_time = time.perf_counter()
    dyn.run(n_steps)
    synchronize_cuda()
    elapsed_seconds = time.perf_counter() - start_time
    seconds_per_step = elapsed_seconds / n_steps if n_steps else np.nan

    print(f"  ✓ Simulation complete: {len(traj_frames)} frames recorded")
    print(
        f"  ✓ MD simulation time: {elapsed_seconds:.2f} s "
        f"({seconds_per_step:.6f} s/step)"
    )

    timing_file = os.path.join(
        os.path.dirname(log_file),
        f'md_timing_{calculator_name}.csv'
    )
    pd.DataFrame([{
        'calculator': calculator_name,
        'source_file': file_path,
        'temperature_K': temperature,
        'n_steps': n_steps,
        'time_step_fs': time_step,
        'record_interval': record_interval,
        'frames_recorded': len(traj_frames),
        'elapsed_seconds': elapsed_seconds,
        'seconds_per_step': seconds_per_step,
    }]).to_csv(timing_file, index=False)
    print(f"  ✓ Saved MD timing to {timing_file}")

    return traj_frames


def main():
    '''Main execution function'''

    print("="*70)
    print("MLIP MD Timing Benchmark")
    print("="*70)

    print_cuda_diagnostics()

    model_name = os.environ.get('MODEL_NAME')
    catalog = load_model_catalog()
    if model_name not in catalog:
        raise SystemExit(f"Set MODEL_NAME to one of: {', '.join(sorted(catalog))}")

    print(f"\nInitializing calculator '{model_name}'...")
    calculator = build_calculator(catalog[model_name])
    print(f"  ✓ Loaded {model_name}")

    file_names, directories = get_file_names()

    if not file_names:
        print("\n⚠ No trajectory files found!")
        return

    print(f"\n✓ Found {len(file_names)} trajectory file(s)\n")

    print("Extracting temperatures from directory names...")
    extract_temperatures(directories)

    print(f"\n{'='*70}")
    print("Processing trajectory files")
    print(f"{'='*70}\n")

    for file_path in file_names:
        print(f"\n📁 Processing: {file_path}")
        parent_dir = os.path.basename(os.path.dirname(file_path))

        if any(s in parent_dir for s in SKIP_SYSTEMS):
            print(f"Skipping {parent_dir} (organic molecule).")
            continue

        if any(s in parent_dir for s in EXCLUDED_SYSTEMS):
            print(f"Skipping {parent_dir} (not part of the paper panel).")
            continue

        frames = read_trajectory(file_path)
        if not frames:
            print("  ⚠ Skipping due to read error")
            continue

        match = re.search(r'(\d+)K', parent_dir)
        if not match:
            print("  ⚠ Cannot extract temperature from directory name")
            continue

        tempK = int(match.group(1))
        init_structure = frames[0]
        nvt_timestep = get_nvt_timestep(parent_dir, init_structure)

        print(f"\nCalculator: {model_name}")
        print(f"Temperature: {tempK} K")

        file_path_output = os.path.basename(os.path.dirname(file_path))
        nvt_output = f'{OUTPUT_DIR}{file_path_output}/nvt_{model_name}.extxyz'
        if os.path.exists(nvt_output):
            print(f"  ⚠ NVT trajectory already exists: {nvt_output}, skipping simulation.")
            continue

        if os.path.basename(file_path) in ('isolated_atom_C.xyz', 'isolated_atom_H.xyz'):
            continue

        nvt_frames = nvt_simulation(
            init_structure,
            temperature=tempK,
            n_steps=get_nvt_n_steps(nvt_timestep),
            time_step=nvt_timestep,
            calculator=calculator,
            calculator_name=model_name,
            tdamp=NVT_TAU,
            record_interval=NVT_RECORD_INTERVAL,
            file_path=file_path
        )

        try:
            write(nvt_output, nvt_frames)
            print(f"  ✓ Saved NVT trajectory to {nvt_output}")
        except Exception as e:
            print(f"  ✗ Error saving trajectory: {e}")

    print(f"\n{'='*70}")
    print("✓ Timing benchmark complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
