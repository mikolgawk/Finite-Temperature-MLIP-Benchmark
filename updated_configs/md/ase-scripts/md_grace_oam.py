# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "ase>=3.26", "h5py>=3.11", "numpy>=1.26,<2",
#   "tensorpotential==0.5.7.2", "nvidia-cuda-nvcc-cu12==12.8.*",
#   "tf-keras==2.19.*", "torch",
# ]
# [tool.uv]
# override-dependencies = ["tensorflow>=2.17,<2.20"]
# ///
"""Native ASE energy/force-only counterpart to torchsim-scripts/md_grace_oam.py.

Run: uv run ase-scripts/md_grace_oam.py
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
if os.environ["TF_ENABLE_ONEDNN_OPTS"] != "0":
    raise RuntimeError("GRACE requires TF_ENABLE_ONEDNN_OPTS=0")

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


from ase.calculators.calculator import Calculator, all_changes
from tensorpotential.calculator import TPCalculator

class ForceOnlyTPCalculator(TPCalculator):
    """Keep native TensorPotential neighbor construction, evaluate E/F only."""
    implemented_properties = ["energy", "free_energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.get_data(self.atoms)
        output = self.force_compute(self.data)
        energy = float(output["total_energy"].numpy().reshape(-1)[0])
        forces = output["total_f"].numpy()[:len(self.atoms)]
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}

"""Reconstruct GRACE Python instructions for genuine TensorFlow eager inference.

This does not invoke a SavedModel graph, tf.function, or XLA during inference.
GRACE-MP deliberately remains on its stress-pruned XLA SavedModel path because
its public artifact lacks the Python model definition required here.
"""

import os
from pathlib import Path


def load_eager_grace(family):
    import tensorflow as tf
    from tensorpotential.instructions.base import load_instructions
    from tensorpotential.tpmodel import TPModel

    if family != "grace_oam":
        raise ValueError("Only GRACE-OAM has the artifacts required for eager inference")
    artifact = "GRACE-2L-OMAT-large-ft-AM"
    cache = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache/grace"))
    root = cache / "checkpoints" / artifact
    potential = Path(os.environ.get("GRACE_OAM_POTENTIAL", root / "model-single.yaml"))
    checkpoint = Path(os.environ.get(
        "GRACE_OAM_CHECKPOINT", root / "checkpoint-single-fp32",
    ))
    if not potential.is_file() or not Path(str(checkpoint) + ".index").is_file():
        raise FileNotFoundError(
            f"True eager {artifact} needs its Python instruction YAML and matching "
            f"fp32 checkpoint. Set GRACE_OAM_POTENTIAL and GRACE_OAM_CHECKPOINT "
            f"(checkpoint prefix without .index). Looked for "
            f"{potential} and {checkpoint}.index. A SavedModel is not an eager fallback."
        )
    model = TPModel(load_instructions(str(potential)))
    model.build(tf.float32)
    status = tf.train.Checkpoint(model=model).read(str(checkpoint))
    status.assert_existing_objects_matched()
    status.expect_partial()  # Training step/optimizer are intentionally unused.

    def compute(input_data):
        from tensorpotential import constants as C
        from tensorpotential.tpmodel import execute_instructions

        if not tf.executing_eagerly():
            raise RuntimeError("GRACE eager inference was called inside a traced graph")
        data = dict(input_data)
        # Match TensorPotential's single-structure force convention exactly,
        # but never construct its virial reduction.
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(data[C.BOND_VECTOR])
            execute_instructions(data, model.instructions, training=False)
            atomic_energy = tf.reshape(data[C.PREDICT_ATOMIC_ENERGY], [-1, 1])
        pair_forces = -tape.gradient(atomic_energy, data[C.BOND_VECTOR])
        n_atoms = tf.reshape(data[C.N_ATOMS_BATCH_TOTAL], [])
        forces = tf.math.unsorted_segment_sum(
            pair_forces, data[C.BOND_IND_J], n_atoms
        ) - tf.math.unsorted_segment_sum(pair_forces, data[C.BOND_IND_I], n_atoms)
        return {
            "total_energy": tf.reduce_sum(atomic_energy, axis=0, keepdims=True),
            "total_f": forces,
        }

    return compute


def make_calculator():
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    model_path = REPO / "updated_configs/data/models/GRACE-2L-OMAT-large-ft-AM-fp32"
    if not model_path.is_dir():
        raise FileNotFoundError(f"Converted GRACE metadata artifact missing: {model_path}")
    calculator = ForceOnlyTPCalculator(str(model_path), float_dtype="float32")
    calculator.force_compute = load_eager_grace("grace_oam")
    calculator.compute_properties = ["energy", "forces"]
    return calculator

if __name__ == "__main__":
    run_ase_md("grace-oam", make_calculator, "ase+tensorpotential-force-only")
