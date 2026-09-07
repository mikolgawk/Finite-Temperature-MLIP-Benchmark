# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "ase>=3.26",
#   "chgnet==0.4.2",
#   "torch-sim-atomistic==0.6.1",
#   "torch",
#   "tqdm>=4.66",
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
"""Standalone energy/force-only CHGNet NVT production MD.

CHGNet 0.4.2 exposes an ASE calculator, but does not expose a TorchSim model
adapter.  This runner therefore uses ASE's NVT integrators while writing the
same TorchSim HDF5 schema consumed by this repository's RDF analysis:

    <OUT_ROOT>/<system>/nvt_chgnet-force-only-compile.h5

Per-system MD settings come from ``md_metadata.json``.  CHGNet is fixed to
float32 by the package; TF32 and cuDNN autotuning are disabled to match the
precision policy of the other model runners.

Run:  uv run md_chgnet.py
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.io import read
from ase.md.bussi import Bussi
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from chgnet.model.dynamics import CHGNetCalculator
from chgnet.model.model import CHGNet
from torch_sim.trajectory import TorchSimTrajectory
from tqdm.auto import tqdm


# Match the deterministic floating-point settings used by the TorchSim scripts.
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False


MODEL_NAME = "chgnet-force-only-compile"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-accelerated"

CHAIN_LENGTH = 1
CHAIN_STEPS = 1
SEED = 42


class TorchSimHDF5Writer:
    """Write ASE snapshots in the HDF5 layout used by the TorchSim runners."""

    def __init__(self, filename: Path, atoms, model_name: str = MODEL_NAME):
        self.atoms = atoms
        self.trajectory = TorchSimTrajectory(
            filename, mode="w", metadata={"model": model_name, "engine": "ase"}
        )
        self.trajectory.write_arrays(
            {
                "atomic_numbers": atoms.get_atomic_numbers().astype(np.int32)[None, :],
                "masses": atoms.get_masses().astype(np.float32)[None, :],
            },
            [0],
        )
        self.trajectory.write_global_array("pbc", np.asarray(atoms.pbc, dtype=bool))

    def write(self, step: int) -> None:
        # TorchSim uses column-vector cells whereas ASE stores cell vectors as rows.
        # ASE velocities are Å per ASE time unit; convert them to Å / ps.
        self.trajectory.write_arrays(
            {
                "positions": atoms_positions(self.atoms)[None, :, :],
                "velocities": atoms_velocities_ps(self.atoms)[None, :, :],
                "cell": self.atoms.cell.array.T.astype(np.float32)[None, :, :],
            },
            [step],
        )

    def close(self) -> None:
        self.trajectory.close()


def atoms_positions(atoms) -> np.ndarray:
    return atoms.get_positions().astype(np.float32)


def atoms_velocities_ps(atoms) -> np.ndarray:
    return (atoms.get_velocities() * (1000.0 * units.fs)).astype(np.float32)


def make_integrator(atoms, thermostat: str, temperature_k: float, timestep_fs: float,
                    tau_fs: float | None, rng: np.random.Generator):
    """Create the closest ASE implementation of the reference thermostat."""
    timestep = timestep_fs * units.fs

    if thermostat == "Langevin":
        if tau_fs is None:
            raise ValueError("Langevin thermostat requires a coupling time")
        return Langevin(
            atoms, timestep=timestep, temperature_K=temperature_k,
            friction=1.0 / (tau_fs * units.fs), rng=rng,
        )

    if thermostat == "Nose-Hoover":
        if tau_fs is None:
            raise ValueError("Nose-Hoover thermostat requires a coupling time")
        return NoseHooverChainNVT(
            atoms, timestep=timestep, temperature_K=temperature_k,
            tdamp=tau_fs * units.fs, tchain=CHAIN_LENGTH, tloop=CHAIN_STEPS,
        )

    if thermostat.lower().startswith("velocity"):
        # The metadata does not specify a CSVR time constant for this system.
        # ASE's conventional default (100 integration steps) keeps the choice
        # explicit and uses the correct Bussi stochastic-rescaling ensemble.
        coupling_fs = tau_fs if tau_fs is not None else 100.0 * timestep_fs
        return Bussi(
            atoms, timestep=timestep, temperature_K=temperature_k,
            taut=coupling_fs * units.fs, rng=rng,
        )

    raise ValueError(f"unsupported thermostat type {thermostat!r}")


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class ForceOnlyCHGNetCalculator(CHGNetCalculator):
    """Request CHGNet energy and forces without its stress or magmom heads."""

    implemented_properties = ("energy", "free_energy", "forces")

    def calculate(self, atoms=None, properties=None, system_changes=None) -> None:
        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
            task="ef",
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CHGNet production runs require a CUDA-capable PyTorch device")

    device = "cuda:0"
    model_name = MODEL_NAME
    # Loading the model once reuses the same fp32 checkpoint across systems.
    potential = CHGNet.load(use_device=device)
    # predict_graph calls forward directly; compile that entry point.
    potential.forward = torch.compile(potential.forward, dynamic=True, fullgraph=False)
    calculator = ForceOnlyCHGNetCalculator(
        model=potential, use_device=device, on_isolated_atoms="error"
    )
    metadata = json.loads(METADATA_FILE.read_text())

    for name, meta in metadata.items():
        init_file = REPO / meta["initfile_path"]
        if not init_file.is_file():
            print(f"[{model_name}] {name}: init file missing, skipping ({init_file})")
            continue

        out_dir = OUT_ROOT / name
        out_h5 = out_dir / f"nvt_{model_name}.h5"
        temporary_h5 = out_h5.with_suffix(".h5.inprogress")
        out_csv = out_dir / f"md_timing_{model_name}.csv"
        if out_csv.exists():
            print(f"[{model_name}] {name}: output exists, skipping")
            continue

        temperature_k = float(meta["temperature"])
        timestep_fs = float(meta["timestep"])
        tau_value = meta["thermostat_coupling_constant"]
        tau_fs = float(tau_value) if tau_value is not None else None
        thermostat = meta["thermostat_type"]
        stride = int(meta["position_print_stride"] or 1)
        trajectory_length_ps = float(meta["trajectory_length_ps"])
        n_steps = round(trajectory_length_ps * 1000.0 / timestep_fs)
        engine = "ase+chgnet-0.4.2-force-only-compile"

        atoms = read(init_file, index=0)
        atoms.calc = calculator
        rng = np.random.default_rng(SEED)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, rng=rng)
        integrator = make_integrator(
            atoms, thermostat, temperature_k, timestep_fs, tau_fs, rng
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        # Uniform policy: one untimed energy/force evaluation on the initial state.
        atoms.get_potential_energy()
        atoms.get_forces()
        synchronize_cuda()
        # Do not let an interrupted run be mistaken for completed output.
        writer = TorchSimHDF5Writer(temporary_h5, atoms, model_name)
        # ASE calls observers once at step zero, matching TorchSim reporters.
        integrator.attach(lambda: writer.write(integrator.nsteps), interval=stride)

        tau_note = f"{tau_fs:g}" if tau_fs is not None else "default"
        print(
            f"[{model_name}] {name}: T={temperature_k:.0f} K, dt={timestep_fs:g} fs, "
            f"{thermostat} tau={tau_note} fs, {n_steps} steps, {len(atoms)} atoms, "
            f"stride {stride}"
        )

        progress = tqdm(
            total=n_steps,
            desc=f"[{model_name}] {name}",
            unit="step",
            dynamic_ncols=True,
        )

        def update_progress() -> None:
            # Observers also run once at step zero; updating by the delta keeps
            # that initial callback from advancing the bar.
            progress.update(integrator.nsteps - progress.n)

        integrator.attach(update_progress, interval=1)

        try:
            synchronize_cuda()
            start = time.perf_counter()
            integrator.run(n_steps)
            synchronize_cuda()
            elapsed = time.perf_counter() - start
        finally:
            progress.close()
            writer.close()

        with out_csv.open("w", newline="") as file:
            writer_csv = csv.DictWriter(
                file,
                fieldnames=[
                    "calculator", "system", "temperature_K", "n_steps",
                    "time_step_fs", "thermostat", "tau_fs", "record_interval",
                    "elapsed_seconds", "seconds_per_step", "engine", "seed",
                ],
            )
            writer_csv.writeheader()
            writer_csv.writerow(
                {
                    "calculator": model_name,
                    "system": name,
                    "temperature_K": temperature_k,
                    "n_steps": n_steps,
                    "time_step_fs": timestep_fs,
                    "thermostat": thermostat,
                    "tau_fs": tau_fs if tau_fs is not None else 100.0 * timestep_fs,
                    "record_interval": stride,
                    "elapsed_seconds": f"{elapsed:.2f}",
                    "seconds_per_step": f"{elapsed / n_steps:.6f}",
                    "engine": engine,
                    "seed": SEED,
                }
            )
        temporary_h5.replace(out_h5)
        print(
            f"[{model_name}] {name}: saved {out_h5} "
            f"({elapsed:.1f} s, {elapsed / n_steps * 1e3:.2f} ms/step)"
        )

    print(f"[{model_name}] done.")


if __name__ == "__main__":
    main()
