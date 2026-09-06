# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch-sim-atomistic[fairchem]==0.6.1",
#   "fairchem-core==2.21.0",
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
"""Force-only TorchSim NVT MD for UMA-S-1p1, OMat task."""

import torch

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

"""Force-only inference for UMA 1p1 with fairchem-core 2.21.0."""


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


def load_force_only_uma(model_name, device):
    """Keep the standard loader, precision and inference settings; disable stress."""
    import torch
    from torch_sim.models.fairchem import FairChemModel

    calculator = FairChemModel(
        model=model_name, task_name="omat", device=device,
        dtype=torch.float32, compute_stress=False,
    )
    return disable_uma_stress(calculator)
"""Shared production-MD loop for TorchSim model runners."""

import csv
import json
import time
from pathlib import Path

import torch
import torch_sim as ts
from ase.io import read


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-matched"

CHAIN_LENGTH = 1
CHAIN_STEPS = 1
SY_STEPS = 3
SEED = 42
STATE_DTYPE = torch.float32


def run_torchsim_md(
    model_name: str,
    model,
    engine: str,
    *,
    require_stress_disabled: bool = True,
    warmup: bool = True,
    state_dtype: torch.dtype = STATE_DTYPE,
    validate=None,
) -> None:
    """Run the catalog's matched NVT workload for one TorchSim model."""
    metadata = json.loads(METADATA_FILE.read_text())
    device = torch.device("cuda")

    if require_stress_disabled and model.compute_stress:
        raise ValueError(f"{model_name} still reports compute_stress=True")

    for name, meta in metadata.items():
        init_file = REPO / meta["initfile_path"]
        if not init_file.is_file():
            print(f"[{model_name}] {name}: init file missing, skipping ({init_file})")
            continue

        out_dir = OUT_ROOT / name
        out_h5 = out_dir / f"nvt_{model_name}.h5"
        out_csv = out_dir / f"md_timing_{model_name}.csv"
        if out_csv.exists():
            print(f"[{model_name}] {name}: output exists, skipping")
            continue

        temp_k = float(meta["temperature"])
        dt_fs = float(meta["timestep"])
        tau_fs = float(meta["thermostat_coupling_constant"])
        thermostat = meta["thermostat_type"]
        stride = int(meta["position_print_stride"] or 1)
        trajectory_length_ps = float(meta["trajectory_length_ps"])
        n_steps = round(trajectory_length_ps * 1000.0 / dt_fs)

        if thermostat == "Langevin":
            integrator = ts.Integrator.nvt_langevin
            init_kwargs = {}
            step_kwargs = {"gamma": 1000.0 / tau_fs}
        elif thermostat == "Nose-Hoover":
            integrator = ts.Integrator.nvt_nose_hoover
            init_kwargs = {
                "tau": tau_fs / 1000.0,
                "chain_length": CHAIN_LENGTH,
                "chain_steps": CHAIN_STEPS,
                "sy_steps": SY_STEPS,
            }
            step_kwargs = {}
        else:
            integrator = ts.Integrator.nvt_vrescale
            init_kwargs = {}
            step_kwargs = {"tau": tau_fs / 1000.0}

        atoms0 = read(init_file, index=0)
        out_dir.mkdir(parents=True, exist_ok=True)

        state = ts.initialize_state(atoms0, device, state_dtype)
        state.rng = SEED

        # Uniform policy: exactly one force/energy evaluation on the initial
        # state, outside the timed region, without changing the MD state.
        if warmup:
            # A validation call already evaluates the production model once.
            # Use it as that system's warmup rather than evaluating twice.
            if validate is not None:
                validate(atoms0)
            else:
                model(state)
            torch.cuda.synchronize()

        print(
            f"[{model_name}] {name}: T={temp_k:.0f} K, dt={dt_fs} fs, "
            f"{thermostat} tau={tau_fs} fs, {n_steps} steps, "
            f"{len(atoms0)} atoms, stride {stride}"
        )
        torch.cuda.synchronize()
        start = time.perf_counter()

        ts.integrate(
            system=state,
            model=model,
            integrator=integrator,
            n_steps=n_steps,
            temperature=temp_k,
            timestep=dt_fs / 1000.0,
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
        elapsed = time.perf_counter() - start

        with out_csv.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "calculator", "system", "temperature_K", "n_steps",
                    "time_step_fs", "thermostat", "tau_fs", "record_interval",
                    "elapsed_seconds", "seconds_per_step", "engine", "seed",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "calculator": model_name,
                    "system": name,
                    "temperature_K": temp_k,
                    "n_steps": n_steps,
                    "time_step_fs": dt_fs,
                    "thermostat": thermostat,
                    "tau_fs": tau_fs,
                    "record_interval": stride,
                    "elapsed_seconds": f"{elapsed:.2f}",
                    "seconds_per_step": f"{elapsed / n_steps:.6f}",
                    "engine": engine,
                    "seed": SEED,
                }
            )
        print(
            f"[{model_name}] {name}: saved {out_h5} "
            f"({elapsed:.1f} s, {elapsed / n_steps * 1e3:.2f} ms/step)"
        )

    print(f"[{model_name}] done.")


def main():
    model = load_force_only_uma("uma-s-1p1", torch.device("cuda"))
    run_torchsim_md(
        "uma-s-omat-force-only", model,
        "torch-sim-0.6.1+fairchem-2.21.0-force-only",
    )


if __name__ == "__main__":
    main()
