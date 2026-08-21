# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch-sim-atomistic==0.6.1",
#   "fairchem-core>=2.20.0",
#   "ase>=3.26",
# ]
# ///
"""TorchSim NVT production MD — uma-m-omat.

Port of updated_configs/md_production/md_script-generic.py: same protocol
(Nose-Hoover chain, tchain=1, tdamp=20 fs, 22 ps, per-system timestep,
temperature from the system directory name, initial structure = reference
frame 0), run on torch-sim's GPU integrator. The trajectory is saved as HDF5 with
positions and velocities every RECORD_INTERVAL steps:

    <OUT_ROOT>/<system>/nvt_uma-m-omat.h5

Run:  uv run md_uma_m_omat.py
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

from torch_sim.models.fairchem import FairChemModel

# settings
MODEL_NAME = "uma-m-omat"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-accelerated"

SIMULATION_LENGTH_PS = 22.0   # production length, same as the ASE benchmark
CHAIN_LENGTH = 1              # Nose-Hoover chain settings, as in the ASE benchmark
CHAIN_STEPS = 1
SY_STEPS = 3
ACCELERATION = True           # torch.compile + no activation checkpointing; fp32 numerics unchanged
SEED = 42                     # Maxwell-Boltzmann velocity seed
STATE_DTYPE = torch.float32

device = torch.device("cuda")

# model
# fairchem's torch-sim wrapper has no inference_settings passthrough; it calls
# pretrained_mlip.get_predict_unit(...) internally, so inject settings at the factory.
if ACCELERATION:
    from dataclasses import replace
    from fairchem.core import pretrained_mlip
    from fairchem.core.units.mlip_unit import InferenceSettings

    _get_predict_unit = pretrained_mlip.get_predict_unit

    def _accelerated(model_name, **kwargs):
        kwargs.setdefault("inference_settings", replace(
            InferenceSettings(), compile=True, activation_checkpointing=False))
        return _get_predict_unit(model_name, **kwargs)

    pretrained_mlip.get_predict_unit = _accelerated

model = FairChemModel(model="uma-m-1p1", task_name="omat", device=device, compute_stress=False)

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
    elif thermostat == "Nose-Hoover":
        integrator = ts.Integrator.nvt_nose_hoover
        init_kwargs = {"tau": tau_fs / 1000.0,             # ps
                       "chain_length": CHAIN_LENGTH,
                       "chain_steps": CHAIN_STEPS, "sy_steps": SY_STEPS}
        step_kwargs = {}
    elif thermostat.lower().startswith("velocity"):        # velocity rescaling, Bussi CSVR
        integrator = ts.Integrator.nvt_vrescale
        init_kwargs = {}
        step_kwargs = {"tau": tau_fs / 1000.0}             # ps
    else:
        raise SystemExit(f"{name}: unknown thermostat type {thermostat!r} in metadata")

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

    with open(out_dir / f"md_timing_{MODEL_NAME}.csv", "w", newline="") as f:
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
                         "engine": f"torch-sim-0.6.1{'+compile' if ACCELERATION else ''}", "seed": SEED})
    print(f"[{MODEL_NAME}] {name}: saved {out_h5} "
          f"({elapsed:.1f} s, {elapsed / n_steps * 1e3:.2f} ms/step)")

print(f"[{MODEL_NAME}] done.")
