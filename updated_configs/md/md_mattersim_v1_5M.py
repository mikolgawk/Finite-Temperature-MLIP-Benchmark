# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch-sim-atomistic[mattersim]==0.6.1",
#   "ase>=3.26",
#   "torch",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""TorchSim NVT production MD — mattersim-v1-5M.

Per-system MD parameters come from data/ref-trajs/md_metadata.json, which
records how each reference AIMD was run. The trajectory is saved as HDF5
with positions and velocities:

    <OUT_ROOT>/<system>/nvt_mattersim-v1-5M.h5

Run:  uv run md_mattersim_v1_5M.py
"""

import csv
import json
import time
from pathlib import Path

import torch

# no TF32, no autotuned kernels
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

import torch_sim as ts
from ase.io import read

from mattersim.forcefield import Potential
from torch_sim.models.mattersim import MatterSimModel

# settings
MODEL_NAME = "mattersim-v1-5M"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
METADATA_FILE = REPO / "updated_configs" /  "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim"

SIMULATION_LENGTH_PS = 22.0   # production length, same as the ASE benchmark
CHAIN_LENGTH = 1              # Nose-Hoover chain settings, as in the ASE benchmark
CHAIN_STEPS = 1
SY_STEPS = 3
SEED = 42                     # Maxwell-Boltzmann velocity seed
STATE_DTYPE = torch.float32

device = torch.device("cuda")

# model
# checkpoint name resolved/downloaded by mattersim itself, same as the ASE benchmark
potential = Potential.from_checkpoint(
    load_path="MatterSim-v1.0.0-5M.pth", model_name="m3gnet",
    device=device, load_training_state=False,
)
model = MatterSimModel(model=potential, device=device, dtype=torch.float32)

# NVT MD loop, one entry per system in the metadata file
METADATA = json.loads(METADATA_FILE.read_text())

for name, meta in METADATA.items():
    init_file = REPO / meta["initfile_path"]
    if not init_file.is_file():
        print(f"[{MODEL_NAME}] {name}: init file missing, skipping ({init_file})")
        continue

    out_dir = OUT_ROOT / name
    out_h5 = out_dir / f"nvt_{MODEL_NAME}.h5"
    out_csv = out_dir / f"md_timing_{MODEL_NAME}.csv"
    if out_csv.exists():
        print(f"[{MODEL_NAME}] {name}: output exists, skipping")
        continue

    temp_k = float(meta["temperature"])
    dt_fs = float(meta["timestep"])                        # timestep_units: fs
    tau_fs = float(meta["thermostat_coupling_constant"])   # coupling units: fs
    thermostat = meta["thermostat_type"]
    stride = int(meta["position_print_stride"] or 1)
    n_steps = round(SIMULATION_LENGTH_PS * 1000.0 / dt_fs)

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
                         "engine": "torch-sim-0.6.1", "seed": SEED})
    print(f"[{MODEL_NAME}] {name}: saved {out_h5} "
          f"({elapsed:.1f} s, {elapsed / n_steps * 1e3:.2f} ms/step)")

print(f"[{MODEL_NAME}] done.")
