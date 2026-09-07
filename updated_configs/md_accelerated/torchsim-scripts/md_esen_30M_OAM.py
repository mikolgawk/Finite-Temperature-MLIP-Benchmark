# /// script
# requires-python = ">=3.12,<3.13"  # torch 2.4.1+cu124 supplies wheels only through CPython 3.12
# dependencies = [
#   "torch-sim-atomistic==0.5.2",  # 0.6.1 needs torch>=2.8, fairchem 1.x pins <2.5;
#                                  # 0.5.2 has the same integrate API incl. nvt_vrescale
#   "fairchem-core==1.10.0",       # v1: only API that loads this legacy checkpoint
#   "torch==2.4.1",
#   "torch-scatter",               # runtime imports of fairchem v1, undeclared there
#   "torch-sparse",
#   "scipy<1.17",                # fairchem-core 1.10 imports the removed scipy.special.sph_harm
#   "ase>=3.26",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# explicit = true
#
# [[tool.uv.index]]
# name = "pyg-cu124"
# url = "https://data.pyg.org/whl/torch-2.4.0+cu124.html"
# format = "flat"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu124" }
# torch-scatter = { index = "pyg-cu124" }
# torch-sparse = { index = "pyg-cu124" }
# ///
"""Force-only compiled TorchSim MD for eSEN-30M-OAM; original checkpoint weights."""

from pathlib import Path

import torch

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

# Embedded implementation: this runner has no local helper imports.
def disable_legacy_stress(calculator, family: str) -> None:
    """Configure an eqV2 or eSEN FairChemV1Model for energy/forces only."""
    if family not in {"eqv2", "esen"}:
        raise ValueError(f"Unsupported legacy model family: {family}")
    trainer = calculator.trainer
    models = [m for m in trainer.model.modules() if hasattr(m, "output_heads")]
    if len(models) != 1:
        raise RuntimeError("Expected exactly one legacy Hydra model")
    model = models[0]
    heads = model.output_heads
    required = {"energy", "forces"}
    for outputs in (trainer.output_targets, trainer.config["outputs"], calculator.config["outputs"]):
        if not required.issubset(outputs):
            raise RuntimeError("Checkpoint does not expose energy and forces")

    if family == "eqv2":
        if set(heads) != {"energy", "forces", "stress"}:
            raise RuntimeError(f"Unexpected eqV2 heads: {list(heads)}")
        # Remove the actual network head, including its decomposed stress outputs.
        del heads["stress"]
    else:
        if set(heads) != {"mptrj"}:
            raise RuntimeError(f"Unexpected eSEN heads: {list(heads)}")
        backbone, head = model.backbone, heads["mptrj"]
        if head.__class__.__name__ != "MLP_EFS_Head":
            raise RuntimeError("Expected eSEN MLP_EFS_Head")
        for component in (backbone, head):
            if not component.regress_forces or not component.regress_stress:
                raise RuntimeError("Expected joint eSEN energy/force/stress model")
        if backbone.direct_forces:
            raise RuntimeError("Expected conservative eSEN backbone")
        # Both flags matter: the backbone constructs the strain graph, while
        # the head chooses positions-only versus joint position/strain gradients.
        backbone.regress_stress = False
        head.regress_stress = False

    # predict() and _forward() each consult a different target dictionary.
    # Whitelisting also removes eqV2's stress_isotropic/stress_anisotropic targets.
    trainer.output_targets = {k: v for k, v in trainer.output_targets.items() if k in required}
    trainer.config["outputs"] = {k: v for k, v in trainer.config["outputs"].items() if k in required}
    calculator.config["outputs"] = {k: v for k, v in calculator.config["outputs"].items() if k in required}
    calculator.implemented_properties = ["energy", "forces"]
    # The 0.5.2 legacy adapter exposes read-only public properties.
    calculator._compute_stress = False
    calculator._compute_forces = True


def load_force_only_legacy(checkpoint, family: str, device, seed: int = 42):
    """Load original weights with the existing legacy adapter, then disable stress."""
    from torch_sim.models.fairchem_legacy import FairChemV1Model

    calculator = FairChemV1Model(
        model=str(checkpoint), device=device, compute_stress=False,
        seed=seed, pbc=True, disable_amp=True,
    )
    disable_legacy_stress(calculator, family)
    return calculator

# Embedded implementation: this runner has no local helper imports.
import csv
import json
import time
from pathlib import Path

import torch
import torch_sim as ts
from ase.io import read


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-accelerated"

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
    checkpoint = Path(__file__).resolve().parents[2] / "data" / "models" / "esen_30m_oam.pt"
    model = load_force_only_legacy(checkpoint, "esen", torch.device("cuda"))
    # Compile after removing stress heads and strain derivatives.
    model.trainer.model = torch.compile(model.trainer.model, dynamic=True, fullgraph=False)
    run_torchsim_md(
        "eSEN-30M-OAM-force-only-compile", model,
        "torch-sim-0.5.2+fairchem-1.10.0-force-only-compile",
    )


if __name__ == "__main__":
    main()
