# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch-sim-atomistic==0.6.1",
#   "fairchem-core",
#   "ase>=3.26",
# ]
# ///
"""Stress-enabled md_uma_m_omat_turbo.py: record potential stress for every saved trajectory frame."""

FAILED_SYSTEMS = []

import csv
import json
import time
from pathlib import Path

import torch


def frame_stress(state, model):
    """Record one finite potential stress tensor per reported system."""
    stress = model(state)["stress"].detach()
    if stress.numel() != state.n_systems * 9 or not bool(torch.isfinite(stress).all()):
        raise ValueError("Model returned an invalid per-frame stress tensor")
    return stress.reshape(state.n_systems, 3, 3)


# FairChem temporarily configures TF32 inside the UMA predictor for each mode.
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

import torch_sim as ts
from ase.io import read

from torch_sim.models.fairchem import FairChemModel

def retain_uma_stress(calculator):
    """Preserve the native energy/force/stress model before compilation."""
    return calculator


# settings
INFERENCE_MODE = "turbo"
if INFERENCE_MODE not in {"compile", "turbo"}:
    raise ValueError(f"Unsupported UMA inference mode: {INFERENCE_MODE!r}")

MODEL_NAME = f"uma-m-omat-{INFERENCE_MODE}-stress"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-accelerated-stress"

CHAIN_LENGTH = 1              # Nose-Hoover chain settings, as in the ASE benchmark
CHAIN_STEPS = 1
SY_STEPS = 3
SEED = 42                     # Maxwell-Boltzmann velocity seed
STATE_DTYPE = torch.float32

device = torch.device("cuda")

# model
# fairchem's torch-sim wrapper has no inference_settings passthrough; it calls
# pretrained_mlip.get_predict_unit(...) internally, so inject the named mode at
# the factory.
from fairchem.core import pretrained_mlip
from fairchem.core.units.mlip_unit import InferenceSettings

FAIRCHEM_INFERENCE_SETTINGS = (
    InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=False,
        compile=True,
    )
    if INFERENCE_MODE == "compile"
    else "turbo"
)

_get_predict_unit = pretrained_mlip.get_predict_unit


def _get_predict_unit_with_mode(model_name, **kwargs):
    kwargs.setdefault("inference_settings", FAIRCHEM_INFERENCE_SETTINGS)
    return _get_predict_unit(model_name, **kwargs)


pretrained_mlip.get_predict_unit = _get_predict_unit_with_mode

model = FairChemModel(
    model="uma-m-1p1", task_name="omat", device=device, dtype=torch.float32,
    compute_stress=True,
)

# Remove stress tasks and strain derivatives before lazy compilation/merging.
retain_uma_stress(model)

# FairChem 2.22 prepares merged MOLE experts before its normal device move.
# TorchSim's first state is already on CUDA, so turbo must move the predictor
# early to keep embedding weights and indices on the same device during merging.
if INFERENCE_MODE == "turbo":
    model.predictor.move_to_device()


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

        atoms0 = read(init_file, index=0)                       # reference frame 0
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
                             "engine": f"torch-sim-0.6.1+{INFERENCE_MODE}", "seed": SEED})
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
