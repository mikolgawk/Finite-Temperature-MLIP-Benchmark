# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch-sim-atomistic[nequip]==0.6.1",
#   "nequip>=0.19.1",
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
"""Standalone energy/force-only eager NVT MD using NequIP-OAM-L.

Run: uv run md_nequip_force_only.py
Uses the repository's MD metadata and initial structures; no local helper
scripts are required.
"""

import torch

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

import csv
import json
import time
from pathlib import Path

import torch_sim as ts
from ase.io import read

from nequip.data import AtomicDataDict
from nequip.model.modify_utils import modify
from torch_sim.models.nequip_framework import NequIPFrameworkModel


class PositionGradientOnly(torch.nn.Module):
    """Add forces to an energy-only NequIP graph without cell differentiation."""

    def __init__(self, energy_model: torch.nn.Module) -> None:
        super().__init__()
        self.energy_model = energy_model

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Detaching makes positions an independent differentiation variable and
        # deliberately leaves the cell outside the autograd graph.
        positions = data[AtomicDataDict.POSITIONS_KEY].detach().requires_grad_(True)
        data = dict(data)
        data[AtomicDataDict.POSITIONS_KEY] = positions

        output = self.energy_model(data)
        energy = output[AtomicDataDict.TOTAL_ENERGY_KEY]
        forces = -torch.autograd.grad(
            energy.sum(), positions, create_graph=self.training
        )[0]
        output[AtomicDataDict.FORCE_KEY] = forces
        return output


def load_force_only_nequip(
    model_source: str, device: torch.device | str
) -> NequIPFrameworkModel:
    """Load an OAM package and replace joint force/stress differentiation."""
    calculator = NequIPFrameworkModel._from_saved_model(
        model_path=model_source,
        device=device,
        chemical_species_to_atom_type_map=True,
        allow_tf32=False,
        compile_mode="eager",
    )

    # The packaged OAM model uses ForceStressOutput. Disable it so the wrapper
    # below performs one gradient with respect to positions and none with
    # respect to strain/cell displacement.
    joint_output_layers = [
        module
        for module in calculator.model.modules()
        if module.__class__.__name__ == "ForceStressOutput"
    ]
    if not joint_output_layers:
        raise RuntimeError("NequIP package has no ForceStressOutput layer to disable")

    try:
        energy_model = modify(
            calculator.model,
            [{"modifier": "disable_ForceStressOutput"}],
        )
    except RuntimeError:
        # Older packaged-code snapshots may not expose the public modifier even
        # though they use the same ForceStressOutput short-circuit flag.
        energy_model = calculator.model
        for layer in joint_output_layers:
            layer.do_derivatives = False

    if any(
        layer.do_derivatives
        for layer in energy_model.modules()
        if layer.__class__.__name__ == "ForceStressOutput"
    ):
        raise RuntimeError("failed to disable NequIP's joint force/stress layer")
    calculator.model = PositionGradientOnly(energy_model).to(device).eval()
    calculator.compute_stress = False
    calculator.compute_forces = True
    return calculator


def assert_eager_module(model):
    """Reject TorchScript and torch.compile modules, including nested modules."""
    import torch
    from torch._dynamo.eval_frame import OptimizedModule

    for name, module in model.named_modules():
        if isinstance(module, (torch.jit.ScriptModule, OptimizedModule)):
            raise RuntimeError(f"Compiled module in eager model: {name or '<root>'}")
        if getattr(module, "_compiled_call_impl", None) is not None:
            raise RuntimeError(f"Compiled call in eager model: {name or '<root>'}")


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
    device = torch.device("cuda")
    model = load_force_only_nequip(
        "nequip.net:mir-group/NequIP-OAM-L:0.1", device=device
    )
    assert_eager_module(model.model)
    run_torchsim_md(
        "nequip-oam-l-force-only", model, "torch-sim-0.6.1+nequip-eager-force-only"
    )


if __name__ == "__main__":
    main()
