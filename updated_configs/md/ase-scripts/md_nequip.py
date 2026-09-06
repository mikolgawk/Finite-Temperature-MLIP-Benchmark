# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "h5py>=3.11", "numpy>=1.26",
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
"""Native ASE energy/force-only counterpart to torchsim-scripts/md_nequip.py.

Run: uv run ase-scripts/md_nequip.py
"""

"""Shared, system-independent ASE NVT driver for the model runners."""

import csv
import json
import time
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import torch
from ase import units
from ase.io import read
from ase.md import MDLogger
from ase.md.bussi import Bussi
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-matched"

CHAIN_LENGTH = 1
CHAIN_STEPS = 1
SEED = 42


torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False


class HDF5TrajectoryWriter:
    """Append ASE snapshots using the repository's TorchSim HDF5 schema."""

    def __init__(self, filename: Path, atoms, model_name: str):
        self.atoms = atoms
        self.file = h5py.File(filename, "w")
        self.file.attrs["program"] = "ASE"
        self.file.attrs["model"] = model_name
        self.file.attrs["engine"] = "ase"

        data = self.file.create_group("data")
        data.create_dataset(
            "atomic_numbers",
            data=atoms.get_atomic_numbers().astype(np.int32)[None, :],
        )
        data.create_dataset(
            "masses", data=atoms.get_masses().astype(np.float32)[None, :]
        )
        data.create_dataset("pbc", data=np.asarray(atoms.pbc, dtype=bool))

        n_atoms = len(atoms)
        self.positions = data.create_dataset(
            "positions", shape=(0, n_atoms, 3), maxshape=(None, n_atoms, 3),
            chunks=(1, n_atoms, 3), dtype=np.float32,
        )
        self.velocities = data.create_dataset(
            "velocities", shape=(0, n_atoms, 3), maxshape=(None, n_atoms, 3),
            chunks=(1, n_atoms, 3), dtype=np.float32,
        )
        self.cells = data.create_dataset(
            "cell", shape=(0, 3, 3), maxshape=(None, 3, 3),
            chunks=(1, 3, 3), dtype=np.float32,
        )

    def write(self) -> None:
        frame = self.positions.shape[0]
        self.positions.resize(frame + 1, axis=0)
        self.velocities.resize(frame + 1, axis=0)
        self.cells.resize(frame + 1, axis=0)
        self.positions[frame] = self.atoms.get_positions()
        # ASE velocities use Angstrom per ASE time unit; files use Angstrom/ps.
        self.velocities[frame] = self.atoms.get_velocities() * (1000.0 * units.fs)
        # TorchSim stores cell vectors as columns; ASE stores them as rows.
        self.cells[frame] = self.atoms.cell.array.T
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_integrator(atoms, thermostat: str, temperature_k: float,
                    timestep_fs: float, tau_fs: float,
                    rng: np.random.Generator):
    """Construct the ASE analogue of a metadata thermostat."""
    timestep = timestep_fs * units.fs
    if thermostat == "Langevin":
        return Langevin(
            atoms, timestep=timestep, temperature_K=temperature_k,
            friction=1.0 / (tau_fs * units.fs), rng=rng,
        )
    if thermostat == "Nose-Hoover":
        return NoseHooverChainNVT(
            atoms, timestep=timestep, temperature_K=temperature_k,
            tdamp=tau_fs * units.fs, tchain=CHAIN_LENGTH, tloop=CHAIN_STEPS,
        )
    if thermostat.lower().startswith("velocity"):
        return Bussi(
            atoms, timestep=timestep, temperature_K=temperature_k,
            taut=tau_fs * units.fs, rng=rng,
        )
    raise ValueError(f"unsupported thermostat type {thermostat!r}")


def run_ase_md(
    model_name: str,
    calculator_factory: Callable[[], object],
    engine: str,
    synchronize: Callable[[], None] = synchronize_cuda,
) -> None:
    """Run an ASE calculator over every system in the MD metadata."""
    metadata = json.loads(METADATA_FILE.read_text())
    calculator = calculator_factory()
    if "stress" in calculator.implemented_properties:
        raise RuntimeError("ASE production calculator must disable stress")
    run_name = f"{model_name}-ase"

    for name, meta in metadata.items():
        init_file = REPO / meta["initfile_path"]
        if not init_file.is_file():
            print(f"[{run_name}] {name}: init file missing, skipping ({init_file})")
            continue

        out_dir = OUT_ROOT / name
        out_h5 = out_dir / f"nvt_{run_name}.h5"
        temporary_h5 = out_h5.with_suffix(".h5.inprogress")
        out_log = out_dir / f"md_{run_name}.log"
        out_csv = out_dir / f"md_timing_{run_name}.csv"
        if out_h5.exists():
            print(f"[{run_name}] {name}: output exists, skipping")
            continue

        temperature_k = float(meta["temperature"])
        timestep_fs = float(meta["timestep"])
        tau_fs = float(meta["thermostat_coupling_constant"])
        thermostat = str(meta["thermostat_type"])
        stride = int(meta["position_print_stride"] or 1)
        energy_stride = int(meta["energy_print_stride"] or 1)
        n_steps = round(float(meta["trajectory_length_ps"]) * 1000.0 / timestep_fs)

        atoms = read(init_file, index=0)
        atoms.calc = calculator
        rng = np.random.default_rng(SEED)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, rng=rng)
        dynamics = make_integrator(
            atoms, thermostat, temperature_k, timestep_fs, tau_fs, rng
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        initial_energy = atoms.get_potential_energy()
        atoms.get_forces()
        print(
            f"[{run_name}] {name}: {len(atoms)} atoms, E0={initial_energy:.6f} eV, "
            f"T={temperature_k:g} K, dt={timestep_fs:g} fs, {thermostat} "
            f"tau={tau_fs:g} fs, {n_steps} steps, stride {stride}"
        )

        trajectory = HDF5TrajectoryWriter(temporary_h5, atoms, model_name)
        dynamics.attach(trajectory.write, interval=stride)
        dynamics.attach(
            MDLogger(
                dynamics, atoms, str(out_log), header=True, stress=False,
                peratom=False, mode="w",
            ),
            interval=energy_stride,
        )

        try:
            synchronize()
            start = time.perf_counter()
            dynamics.run(n_steps)
            synchronize()
            elapsed = time.perf_counter() - start
        finally:
            trajectory.close()

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
                    "calculator": model_name, "system": name,
                    "temperature_K": temperature_k, "n_steps": n_steps,
                    "time_step_fs": timestep_fs, "thermostat": thermostat,
                    "tau_fs": tau_fs, "record_interval": stride,
                    "elapsed_seconds": f"{elapsed:.2f}",
                    "seconds_per_step": f"{elapsed / n_steps:.6f}",
                    "engine": engine, "seed": SEED,
                }
            )

        temporary_h5.replace(out_h5)
        print(
            f"[{run_name}] {name}: saved {out_h5} "
            f"({elapsed:.1f} s, {elapsed / n_steps * 1e3:.2f} ms/step)"
        )

    print(f"[{run_name}] done.")


from nequip.data import AtomicDataDict
from nequip.integrations.ase import NequIPCalculator


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
) -> NequIPCalculator:
    """Load an OAM package and replace joint force/stress differentiation."""
    calculator = NequIPCalculator._from_saved_model(
        model_path=model_source,
        device=device,
        chemical_species_to_atom_type_map=True,
        allow_tf32=False,
        compile_mode="eager",
    )

    # Packaged models carry their own NequIP code. Older ForceStressOutput
    # implementations have no do_derivatives branch, so assigning that flag
    # silently leaves their joint backward pass active. Remove the wrapper
    # itself, retaining its energy function and every other model layer.
    removed = 0

    def remove_joint_output(module):
        nonlocal removed
        if module.__class__.__name__ == "ForceStressOutput":
            if not isinstance(getattr(module, "func", None), torch.nn.Module):
                raise RuntimeError("Unsupported NequIP ForceStressOutput structure")
            removed += 1
            return remove_joint_output(module.func)
        for name, child in list(module.named_children()):
            replacement = remove_joint_output(child)
            if replacement is not child:
                setattr(module, name, replacement)
        return module

    energy_model = remove_joint_output(calculator.model)
    if removed == 0:
        raise RuntimeError("NequIP package has no ForceStressOutput layer to remove")
    if any(layer.__class__.__name__ == "ForceStressOutput"
           for layer in energy_model.modules()):
        raise RuntimeError("failed to remove NequIP's joint force/stress layer")
    calculator.model = PositionGradientOnly(energy_model).to(device).eval()
    calculator.implemented_properties = ["energy", "free_energy", "forces"]
    return calculator

"""Uncompiled PyTorch loaders for the separate eager MD runners."""


def assert_eager_module(model):
    """Reject TorchScript and torch.compile modules, including nested modules."""
    import torch
    from torch._dynamo.eval_frame import OptimizedModule

    for name, module in model.named_modules():
        if isinstance(module, (torch.jit.ScriptModule, OptimizedModule)):
            raise RuntimeError(f"Compiled module in eager model: {name or '<root>'}")
        if getattr(module, "_compiled_call_impl", None) is not None:
            raise RuntimeError(f"Compiled call in eager model: {name or '<root>'}")



def make_calculator():
    calculator = load_force_only_nequip(
        "nequip.net:mir-group/NequIP-OAM-L:0.1", device="cuda",
    )
    assert_eager_module(calculator.model)
    return calculator

if __name__ == "__main__":
    run_ase_md("nequip-oam-l", make_calculator, "ase+nequip-eager-force-only")
