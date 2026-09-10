# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch-sim-atomistic[mattersim]==0.6.1",
#   "mattersim",
#   "ase>=3.26",
#   "torch==2.11.0+cu128",
#   "torchvision==0.26.0+cu128",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
# mattersim = { git = "https://github.com/microsoft/mattersim.git", rev = "df68847af1d66c6725bfadc58c4e3d3ec705bc5f" }
# ///
"""Stress-enabled md_mattersim_v1_5M.py: record potential stress for every saved trajectory frame."""

FAILED_SYSTEMS = []

import csv
import json
import os
import time
from pathlib import Path

import torch


def frame_stress(state, model):
    """Record one finite potential stress tensor per reported system."""
    stress = model(state)["stress"].detach()
    if stress.numel() != state.n_systems * 9 or not bool(torch.isfinite(stress).all()):
        raise ValueError("Model returned an invalid per-frame stress tensor")
    return stress.reshape(state.n_systems, 3, 3)


# TorchInductor's generated C++ precompiled header includes CUDA headers without
# always adding the wheel's CUDA runtime include directory. Make that directory
# visible before MatterSim invokes the in-process compiler.
CUDA_HOME = Path(os.environ.get("MATTERSIM_CUDA_HOME", "/usr/local/cuda")).resolve()
_site_packages = Path(torch.__file__).resolve().parent.parent
_wheel_cuda_include = _site_packages / "nvidia" / "cuda_runtime" / "include"
_cuda_include = (
    _wheel_cuda_include
    if (_wheel_cuda_include / "cuda_fp8.h").is_file()
    else CUDA_HOME / "include"
)
if not (_cuda_include / "cuda_fp8.h").is_file():
    raise FileNotFoundError(
        f"CUDA headers not found under {_cuda_include}; set MATTERSIM_CUDA_HOME "
        "to a complete CUDA toolkit"
    )
os.environ["CUDA_HOME"] = str(CUDA_HOME)
os.environ["CUDA_PATH"] = str(CUDA_HOME)
os.environ["CUDAToolkit_ROOT"] = str(CUDA_HOME)
os.environ["CPATH"] = os.pathsep.join(
    value
    for value in (
        str(_cuda_include),
        str(CUDA_HOME / "include"),
        os.environ.get("CPATH", ""),
    )
    if value
)

# no TF32, no autotuned kernels
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

import torch_sim as ts
from ase.io import read

from mattersim.forcefield import Potential
from mattersim.torchsim.graph_construction import build_graph_from_simstate
from mattersim.torchsim.torchsim_wrapper import TorchSimWrapper


class WithStressTorchSimWrapper(TorchSimWrapper):
    """Use native MatterSim stress conversion from GPa to eV/A^3."""
    pass


# settings
MODEL_NAME = "mattersim-v1-5M-compile-stress"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
METADATA_FILE = REPO / "updated_configs" /  "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-accelerated-stress"

CHAIN_LENGTH = 1              # Nose-Hoover chain settings, as in the ASE benchmark
CHAIN_STEPS = 1
SY_STEPS = 3
SEED = 42                     # Maxwell-Boltzmann velocity seed
STATE_DTYPE = torch.float32

device = torch.device("cuda")

# model
# Checkpoint name resolved/downloaded by MatterSim itself, same as the ASE
# benchmark. Compile the M3GNet forward pass before wrapping it for TorchSim.
# system pays the one-time TorchInductor compilation cost. This revision includes
# the self-image fix from merged PR #164.
potential = Potential.from_checkpoint(
    load_path="MatterSim-v1.0.0-5M.pth", model_name="m3gnet",
    device=device, load_training_state=False,
)
potential.model.forward = torch.compile(potential.model.forward)
model = WithStressTorchSimWrapper(
    model=potential,
    device=device,
)

# NVT MD loop, one entry per system in the metadata file
METADATA = json.loads(METADATA_FILE.read_text())

for name, meta in METADATA.items():
    out_dir = OUT_ROOT / name
    out_h5 = out_dir / f"nvt_{MODEL_NAME}.h5"
    out_csv = out_dir / f"md_timing_{MODEL_NAME}.csv"
    if out_h5.is_file() and out_csv.is_file() and out_csv.stat().st_size > 0:
        print(f"[{MODEL_NAME}] {name}: output exists, skipping")
        continue

    try:
        init_file = next((p for p in (REPO / meta["initfile_path"], REPO / "updated_configs" / meta["initfile_path"]) if p.is_file()), REPO / meta["initfile_path"])
        if not init_file.is_file():
            print(f"[{MODEL_NAME}] {name}: init file missing, skipping ({init_file})")
            continue

        temp_k = float(meta["temperature"])
        dt_fs = float(meta["timestep"])                        # timestep_units: fs
        tau_fs = float(meta["thermostat_coupling_constant"])   # coupling units: fs
        thermostat = meta["thermostat_type"]
        stride = int(meta["position_print_stride"] or 1)
        trajectory_length_ps = float(meta["trajectory_length_ps"])
        n_steps = round(trajectory_length_ps * 1000.0 / dt_fs)

        if thermostat == "Langevin":
            integrator = ts.Integrator.nvt_langevin
            init_kwargs = {}
            step_kwargs = {"gamma": 1000.0 / tau_fs}           # 1/tau, in ps^-1
        elif thermostat == "Nose-Hoover":  # Nose-Hoover
            integrator = ts.Integrator.nvt_nose_hoover
            init_kwargs = {"tau": tau_fs / 1000.0,             # ps
                           "chain_length": CHAIN_LENGTH,
                           "chain_steps": CHAIN_STEPS, "sy_steps": SY_STEPS}
            step_kwargs = {}
        else:  # velocity rescaling (CSVR)
            integrator = ts.Integrator.nvt_vrescale
            init_kwargs = {}
            step_kwargs = {"tau": tau_fs / 1000.0}              # ps

        atoms0 = read(init_file, index=0)                      # reference frame 0
        out_dir.mkdir(parents=True, exist_ok=True)

        state = ts.initialize_state(atoms0, device, STATE_DTYPE)
        state.rng = SEED

        print(f"[{MODEL_NAME}] {name}: T={temp_k:.0f} K, dt={dt_fs} fs, {thermostat} "
              f"tau={tau_fs} fs, {n_steps} steps, {len(atoms0)} atoms, stride {stride}")
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        ts.integrate(
            system=state,
            model=model,
            integrator=integrator,
            n_steps=n_steps,
            temperature=temp_k,                  # Kelvin
            timestep=dt_fs / 1000.0,             # picoseconds, torch-sim metal units
            init_kwargs=init_kwargs,
            trajectory_reporter={
                "filenames": [str(out_h5)],
                "state_frequency": stride,
                    "prop_calculators": {stride: {"stress": frame_stress}},
                    "metadata": {"stress_units": "eV/Angstrom^3",
                                 "stress_kind": "potential; tensile-positive; no kinetic term"},
                "state_kwargs": {"save_velocities": True, "save_forces": False},
            },
            pbar=True,
            **step_kwargs,
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["calculator", "system", "temperature_K",
                                                   "n_steps", "time_step_fs", "thermostat",
                                                   "tau_fs", "record_interval",
                                                   "elapsed_seconds", "seconds_per_step",
                                                   "engine", "seed"])
            writer.writeheader()
            writer.writerow({"calculator": MODEL_NAME, "system": name,
                             "temperature_K": temp_k, "n_steps": n_steps,
                             "time_step_fs": dt_fs, "thermostat": thermostat,
                             "tau_fs": tau_fs, "record_interval": stride,
                             "elapsed_seconds": f"{elapsed:.2f}",
                             "seconds_per_step": f"{elapsed / n_steps:.6f}",
                             "engine": "torch-sim-0.6.1+mattersim-torch-compile-stress",
                             "seed": SEED})
        print(f"[{MODEL_NAME}] {name}: saved {out_h5} "
              f"({elapsed:.1f} s, {elapsed / n_steps * 1e3:.2f} ms/step)")
    except Exception as exc:
        print(f"[{MODEL_NAME}] {name}: FAILED ({type(exc).__name__}: {exc})")
        FAILED_SYSTEMS.append(name)
        # A zero-byte timing CSV marks this model/system as failed.
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv.write_bytes(b"")
        except OSError as marker_error:
            print(f"[{MODEL_NAME}] {name}: could not write failure CSV: {marker_error}")

print(f"[{MODEL_NAME}] done.")

if FAILED_SYSTEMS:

    raise RuntimeError(f"MD failed for {FAILED_SYSTEMS}")
