# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch-sim-atomistic==0.6.1",
#   "fairchem-core",
#   "ase>=3.26",
# ]
# ///
"""TorchSim NVT production MD — uma-m-omat.

Per-system MD parameters come from updated_configs/data/ref-trajs/md_metadata.json,
which records how each reference AIMD was run. The trajectory is saved as HDF5
with positions and velocities:

    <OUT_ROOT>/<system>/nvt_uma-m-omat-<mode>-force-only.h5

Run: uv run md_uma_m_omat_turbo.py
Standalone loader and MD loop; no base script is required.
"""

import csv
import json
import time
from pathlib import Path

import torch

# FairChem temporarily configures TF32 inside the UMA predictor for each mode.
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

import torch_sim as ts
from ase.io import read

from torch_sim.models.fairchem import FairChemModel

def disable_uma_stress(calculator):
    """Disable derivative stress and remove its predictor tasks before first use."""
    predictor = calculator.predictor
    if predictor.lazy_model_intialized:
        raise RuntimeError("Disable UMA stress before the first prediction/compilation")
    model = predictor.model.module
    config = model.backbone.regress_config
    if config.direct_forces or config.direct_stress or not config.forces or config.hessian:
        raise RuntimeError("Expected conservative UMA energy/forces without Hessians")
    if not config.stress:
        raise RuntimeError("Expected the original stress-enabled UMA model")
    tasks = {name: task for name, task in model.tasks.items() if task.property != "stress"}
    omat_properties = {task.property for task in tasks.values() if "omat" in task.datasets}
    if not {"energy", "forces"}.issubset(omat_properties):
        raise RuntimeError("UMA checkpoint is missing OMat energy/force tasks")

    # Task routing/normalization must agree with the outputs the heads produce.
    # Preserve the existing Task objects, including normalizers and atom references.
    from fairchem.core.models.base import _get_dataset_to_tasks_map

    model._tasks = tasks
    model._dataset_to_tasks = _get_dataset_to_tasks_map(tasks.values())
    model.backbone.validate_tasks(model._dataset_to_tasks)
    predictor.inference_settings.auto_add_default_untrained_tasks = False
    predictor.inference_settings.predict_untrained_stress = set()
    # EFS heads share this config with the backbone. Also check every head so a
    # package/API change cannot silently leave joint differentiation enabled.
    config.stress = False
    for module in model.modules():
        regression = getattr(module, "regress_config", None)
        if regression is not None and regression.stress:
            raise RuntimeError("UMA contains a head with an unshared stress configuration")
    calculator._compute_stress = False
    calculator.implemented_properties = ["energy", "forces"]
    return calculator


# settings
INFERENCE_MODE = "turbo"
if INFERENCE_MODE not in {"compile", "turbo"}:
    raise ValueError(f"Unsupported UMA inference mode: {INFERENCE_MODE!r}")

MODEL_NAME = f"uma-m-omat-{INFERENCE_MODE}-force-only"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-accelerated"

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
    compute_stress=False,
)

# Remove stress tasks and strain derivatives before lazy compilation/merging.
disable_uma_stress(model)

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
    if out_csv.exists():
        print(f"[{MODEL_NAME}] {name}: output exists, skipping")
        continue

    try:
        init_file = REPO / meta["initfile_path"]
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
        # A zero-byte timing CSV marks this model/system as failed.
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv.write_bytes(b"")
        except OSError as marker_error:
            print(f"[{MODEL_NAME}] {name}: could not write failure CSV: {marker_error}")

print(f"[{MODEL_NAME}] done.")
