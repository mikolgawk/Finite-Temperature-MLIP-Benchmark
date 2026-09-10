# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch-sim-atomistic[metatomic]==0.6.1",
#   "upet==0.2.6",
#   "metatomic-torchsim==0.1.3",  # 0.1.4 calls vesin's NeighborList(skin=...),
#                                 # a kwarg only added in vesin>=0.6, but 0.1.4
#                                 # itself pins vesin<0.6 - always TypeErrors.
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
"""Stress-enabled md_pet_oam_xl.py: record potential stress for every saved trajectory frame."""

FAILED_SYSTEMS = []

import csv
import json
import os
import time
from pathlib import Path

# Vesin reads this at import time.  PET's large cutoff produces 2,098
# neighbors/atom for dense periodic H at 1050 K, above Vesin's 1,000 default.
os.environ["VESIN_CUDA_MAX_PAIRS_PER_POINT"] = "4096"

import torch


def frame_stress(state, model):
    """Record one finite potential stress tensor per reported system."""
    stress = model(state)["stress"].detach()
    if stress.numel() != state.n_systems * 9 or not bool(torch.isfinite(stress).all()):
        raise ValueError("Model returned an invalid per-frame stress tensor")
    return stress.reshape(state.n_systems, 3, 3)


# no TF32, no autotuned kernels
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

import torch_sim as ts
from ase.io import read


def assert_eager_module(model):
    """Reject TorchScript and torch.compile modules, including nested modules."""
    import torch
    from torch._dynamo.eval_frame import OptimizedModule

    for name, module in model.named_modules():
        if isinstance(module, (torch.jit.ScriptModule, OptimizedModule)):
            raise RuntimeError(f"Compiled module in eager model: {name or '<root>'}")
        if getattr(module, "_compiled_call_impl", None) is not None:
            raise RuntimeError(f"Compiled call in eager model: {name or '<root>'}")


def load_eager_pet(*, model, size, version, checkpoint_path=None):
    """Follow UPET 0.2.6's loader but omit its final torch.jit.script call."""
    from upet._models import _get_upet_exported_atomistic_model
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    if checkpoint_path is None:
        filename = f"models/{model}-{size}-v{version}.ckpt"
        cached = try_to_load_from_cache("lab-cosmo/upet", filename)
        checkpoint_path = cached if isinstance(cached, str) else hf_hub_download(
            "lab-cosmo/upet", filename=filename
        )
    model = _get_upet_exported_atomistic_model(
        model=model, size=size, version=version, checkpoint_path=checkpoint_path
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert_eager_module(model)
    return model

import csv
import json
import time
from pathlib import Path

import torch
import torch_sim as ts
from ase.io import read


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-matched-stress"

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
    require_stress_enabled: bool = True,
    warmup: bool = True,
    state_dtype: torch.dtype = STATE_DTYPE,
    validate=None,
) -> None:
    """Run the catalog's matched NVT workload for one TorchSim model."""
    metadata = json.loads(METADATA_FILE.read_text())
    device = torch.device("cuda")

    if require_stress_enabled and not model.compute_stress:
        raise ValueError(f"{model_name} does not support stress")

    for name, meta in metadata.items():
        out_dir = OUT_ROOT / name
        out_h5 = out_dir / f"nvt_{model_name}.h5"
        out_csv = out_dir / f"md_timing_{model_name}.csv"
        if out_h5.is_file() and out_csv.is_file() and out_csv.stat().st_size > 0:
            print(f"[{model_name}] {name}: output exists, skipping")
            continue

        try:
            init_file = next((p for p in (REPO / meta["initfile_path"], REPO / "updated_configs" / meta["initfile_path"]) if p.is_file()), REPO / meta["initfile_path"])
            if not init_file.is_file():
                print(f"[{model_name}] {name}: init file missing, skipping ({init_file})")
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
                    "prop_calculators": {stride: {"stress": frame_stress}},
                    "metadata": {"stress_units": "eV/Angstrom^3",
                                 "stress_kind": "potential; tensile-positive; no kinetic term"},
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
        except Exception as exc:
            print(f"[{model_name}] {name}: FAILED ({type(exc).__name__}: {exc})")
            FAILED_SYSTEMS.append(name)
            # A zero-byte timing CSV marks this model/system as failed.
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                out_csv.write_bytes(b"")
            except OSError as marker_error:
                print(f"[{model_name}] {name}: could not write failure CSV: {marker_error}")
            continue

    print(f"[{model_name}] done.")

    if FAILED_SYSTEMS:

        raise RuntimeError(f"MD failed for {FAILED_SYSTEMS}")

from torch_sim.models.metatomic import MetatomicModel
import metatomic_torchsim._neighbors as metatomic_neighbors

def main():
    # settings
    MODEL_NAME = "pet-oam-xl-stress-eager"
    HERE = Path(__file__).resolve().parent
    REPO = HERE.parents[3]
    METADATA_FILE = REPO / "updated_configs" /  "data" / "ref-trajs" / "md_metadata.json"
    OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-matched-stress"

    CHAIN_LENGTH = 1              # Nose-Hoover chain settings, as in the ASE benchmark
    CHAIN_STEPS = 1
    SY_STEPS = 3
    SEED = 42                     # Maxwell-Boltzmann velocity seed
    STATE_DTYPE = torch.float32

    device = torch.device("cuda")

    # nvalchemiops' CUDA full-list implementation has a fixed per-atom neighbor
    # capacity (848 in the installed version).  Dense periodic H at 1050 K needs
    # 2,098 neighbors, so use metatomic-torchsim's Vesin fallback instead.
    # This must be set before MetatomicModel constructs its neighbor calculators.
    metatomic_neighbors.HAS_NVALCHEMIOPS = False

    # model
    # PET, xl size, OAM checkpoint (MP-consistent PBE), version pinned as in the ASE benchmark
    model = MetatomicModel(
        model=load_eager_pet(model="pet-oam", size="xl", version="1.0.0"),
        device=device, compute_stress=True,
    )

    run_torchsim_md(MODEL_NAME, model, "torch-sim-0.6.1+pet-eager-stress")


if __name__ == "__main__":
    main()
