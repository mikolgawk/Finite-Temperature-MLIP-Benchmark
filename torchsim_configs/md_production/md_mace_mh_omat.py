# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch-sim-atomistic[mace,vesin]==0.6.1",
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
"""TorchSim NVT production MD — mace-mh-omat.

Port of updated_configs/md_production/md_script-generic.py: same protocol
(Nose-Hoover chain, tchain=1, tdamp=20 fs, 22 ps, per-system timestep,
temperature from the system directory name, initial structure = reference
frame 0), run on torch-sim's GPU integrator. The trajectory is saved as HDF5 with
positions and velocities every RECORD_INTERVAL steps:

    <OUT_ROOT>/<system>/nvt_mace-mh-omat.h5

Run:  uv run md_mace_mh_omat.py
"""

import csv
import re
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

from mace.calculators import mace_mp
from torch_sim.models.mace import MaceModel
from torch_sim.neighbors import vesin_nl_ts

# settings
MODEL_NAME = "mace-mh-omat"
HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent.parent / "updated_configs" / "data"
MODELS_DIR = DATA_ROOT / "models"
REF_ROOT = HERE.parent / "ref-trajs"
OUT_ROOT = DATA_ROOT / "mlip-trajs-torchsim"

SIMULATION_LENGTH_PS = 22.0   # NVT_SIMULATION_LENGTH_PS in the ASE script
TAU_PS = 0.020                # == ASE tdamp = 20 fs, same Q_j = kT*tau^2 convention
CHAIN_LENGTH = 1              # == ASE tchain = 1
CHAIN_STEPS = 1               # ~  ASE tloop = 1
SY_STEPS = 3                  # 4th-order Suzuki-Yoshida, matches ASE FOURTH_ORDER_COEFFS
RECORD_INTERVAL = 1           # save every Nth step
SEED = 42                     # Maxwell-Boltzmann velocity seed
STATE_DTYPE = torch.float32

device = torch.device("cuda")

# model
# mh-1 with the omat_pbe head
model = MaceModel(
    model=mace_mp(model="mh-1", return_raw_model=True, default_dtype="float32"),
    device=device, dtype=torch.float32, compute_forces=True, compute_stress=False,
    head="omat_pbe",
    neighbor_list_fn=vesin_nl_ts,  # default NVIDIA NL overflows on dense fluids like H at 1050 K
)


def timestep_fs(system_name: str, atoms) -> float:
    """Per-system timestep rule, identical to the ASE script's get_nvt_timestep."""
    if "CuAu" in system_name:
        return 2.0
    if "H" in set(atoms.get_chemical_symbols()):
        return 0.5
    return 1.0


# NVT MD loop
# Only traj*.extxyz counts — isolated_atom_*.extxyz can never be picked up.
for system_dir in sorted(p for p in REF_ROOT.iterdir() if p.is_dir()):
    hits = sorted(system_dir.glob("traj*.extxyz"))
    match = re.search(r"(\d+)K", system_dir.name)
    if not hits or not match:
        continue

    out_dir = OUT_ROOT / system_dir.name
    out_h5 = out_dir / f"nvt_{MODEL_NAME}.h5"
    if out_h5.exists():
        print(f"[{MODEL_NAME}] {system_dir.name}: output exists, skipping")
        continue

    atoms0 = read(hits[0], index=0)              # reference frame 0
    temp_k = int(match.group(1))
    dt_fs = timestep_fs(system_dir.name, atoms0)
    n_steps = round(SIMULATION_LENGTH_PS * 1000.0 / dt_fs)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = ts.initialize_state(atoms0, device, STATE_DTYPE)
    state.rng = SEED             # seeded Maxwell-Boltzmann momenta, COM removed

    print(f"[{MODEL_NAME}] {system_dir.name}: T={temp_k} K, dt={dt_fs} fs, "
          f"{n_steps} steps, {len(atoms0)} atoms")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    ts.integrate(
        system=state,
        model=model,
        integrator=ts.Integrator.nvt_nose_hoover,
        n_steps=n_steps,
        temperature=temp_k,                  # Kelvin
        timestep=dt_fs / 1000.0,             # picoseconds, torch-sim metal units
        init_kwargs={"tau": TAU_PS, "chain_length": CHAIN_LENGTH,
                     "chain_steps": CHAIN_STEPS, "sy_steps": SY_STEPS},
        trajectory_reporter={
            "filenames": [str(out_h5)],
            "state_frequency": RECORD_INTERVAL,
            "state_kwargs": {"save_velocities": True, "save_forces": False},
        },
        pbar=True,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    with open(out_dir / f"md_timing_{MODEL_NAME}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["calculator", "system", "temperature_K",
                                               "n_steps", "time_step_fs", "record_interval",
                                               "elapsed_seconds", "seconds_per_step",
                                               "engine", "seed"])
        writer.writeheader()
        writer.writerow({"calculator": MODEL_NAME, "system": system_dir.name,
                         "temperature_K": temp_k, "n_steps": n_steps,
                         "time_step_fs": dt_fs, "record_interval": RECORD_INTERVAL,
                         "elapsed_seconds": f"{elapsed:.2f}",
                         "seconds_per_step": f"{elapsed / n_steps:.6f}",
                         "engine": "torch-sim-0.6.1", "seed": SEED})
    print(f"[{MODEL_NAME}] {system_dir.name}: saved {out_h5} "
          f"({elapsed:.1f} s, {elapsed / n_steps * 1e3:.2f} ms/step)")

print(f"[{MODEL_NAME}] done.")
