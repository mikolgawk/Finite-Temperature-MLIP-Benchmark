"""Run the V2 reference-matched MD workload with a user-provided ASE model.

The user module must expose a zero-argument function that returns an ASE
``Calculator``.  For example::

    python md_custom_model.py \
        --model-name my-model \
        --factory /path/to/my_model.py:make_calculator

Model-specific dependencies belong in the user's environment (or in inline
``uv`` metadata in their model module).  This runner deliberately requests
energy and forces only; use a separate stress-enabled run for pressure.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import re
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any


SCRIPT = Path(__file__).resolve()
CUSTOM_ROOT = SCRIPT.parent.parent
REPO_ROOT = CUSTOM_ROOT.parent
DEFAULT_METADATA = REPO_ROOT / "paper_v2_configs/data/ref-trajs/md_metadata.json"
SEED = 42
CHAIN_LENGTH = 1
CHAIN_STEPS = 1
TIMING_FIELDS = (
    "calculator",
    "system",
    "temperature_K",
    "n_steps",
    "time_step_fs",
    "thermostat",
    "tau_fs",
    "record_interval",
    "elapsed_seconds",
    "seconds_per_step",
    "engine",
    "seed",
)


def parse_factory_spec(value: str) -> tuple[str, str]:
    """Split MODULE[:FUNCTION], defaulting to ``make_calculator``."""
    module, separator, function = value.rpartition(":")
    if not separator:
        return value, "make_calculator"
    if not module or not function:
        raise argparse.ArgumentTypeError(
            "factory must be MODULE[:FUNCTION] or /path/to/file.py[:FUNCTION]"
        )
    return module, function


def load_module(module_ref: str) -> ModuleType:
    """Import a dotted module or load a Python file without changing cwd."""
    path = Path(module_ref).expanduser()
    if path.suffix == ".py" or path.is_file():
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"model module does not exist: {path}")
        module_name = f"custom_ase_model_{abs(hash(path))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load model module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_ref)


def load_factory(value: str) -> Callable[[], Any]:
    module_ref, function_name = parse_factory_spec(value)
    module = load_module(module_ref)
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise TypeError(f"{value!r} does not identify a callable factory")
    return factory


def safe_model_name(value: str) -> str:
    """Keep model identifiers portable and unambiguous in output filenames."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", value):
        raise argparse.ArgumentTypeError(
            "model name must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_', '+', or '-'"
        )
    return value


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def parse_args(
    argv: list[str] | None = None, *, accelerated: bool = False
) -> argparse.Namespace:
    mode = "accelerated" if accelerated else "baseline"
    output_suffix = "-accelerated" if accelerated else ""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, type=safe_model_name)
    parser.add_argument(
        "--factory",
        required=True,
        help="dotted module or .py file, optionally followed by :FUNCTION",
    )
    parser.add_argument(
        "--engine-label",
        default=f"ase+custom-{mode}",
        help="dependency/backend description recorded in timing CSVs",
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--ref-root",
        type=Path,
        help="reference root containing <system>/traj.extxyz (default: metadata directory)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CUSTOM_ROOT / f"data/mlip-trajs-ase{output_suffix}",
    )
    parser.add_argument(
        "--system",
        action="append",
        dest="systems",
        help="run only this exact metadata key; repeat to select multiple systems",
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        help="cap each run for adapter smoke tests (not benchmark-comparable)",
    )
    parser.add_argument(
        "--calculator-per-system",
        action="store_true",
        help="construct a fresh calculator for every system",
    )
    parser.add_argument("--force", action="store_true", help="replace existing outputs")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first failed system instead of reporting all failures",
    )
    return parser.parse_args(argv)


def select_metadata(
    metadata: Mapping[str, Mapping[str, Any]], systems: list[str] | None
) -> list[tuple[str, Mapping[str, Any]]]:
    if not systems:
        return list(metadata.items())
    unknown = sorted(set(systems) - set(metadata))
    if unknown:
        raise ValueError(f"unknown --system value(s): {', '.join(unknown)}")
    requested = set(systems)
    return [(name, meta) for name, meta in metadata.items() if name in requested]


def resolve_init_file(
    system: str, meta: Mapping[str, Any], metadata_file: Path, ref_root: Path
) -> Path:
    configured = Path(str(meta["initfile_path"]))
    candidates = (
        ref_root / system / configured.name,
        metadata_file.parent.parent.parent / configured,
        REPO_ROOT / configured,
        configured,
    )
    return next(
        (path.resolve() for path in candidates if path.is_file()), candidates[0]
    )


def synchronize_accelerator() -> None:
    """Synchronize CUDA when the user model uses PyTorch; otherwise do nothing."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def configure_strict_torch_precision() -> None:
    """Apply the V2 strict-fp32 policy when the calculator uses PyTorch."""
    try:
        import torch
    except ImportError:
        return
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def make_integrator(
    atoms: Any,
    thermostat: str,
    temperature_k: float,
    timestep_fs: float,
    tau_fs: float,
    rng: Any,
) -> Any:
    from ase import units
    from ase.md.bussi import Bussi
    from ase.md.langevin import Langevin
    from ase.md.nose_hoover_chain import NoseHooverChainNVT

    timestep = timestep_fs * units.fs
    if thermostat == "Langevin":
        return Langevin(
            atoms,
            timestep=timestep,
            temperature_K=temperature_k,
            friction=1.0 / (tau_fs * units.fs),
            rng=rng,
        )
    if thermostat == "Nose-Hoover":
        return NoseHooverChainNVT(
            atoms,
            timestep=timestep,
            temperature_K=temperature_k,
            tdamp=tau_fs * units.fs,
            tchain=CHAIN_LENGTH,
            tloop=CHAIN_STEPS,
        )
    if thermostat.lower().startswith("velocity"):
        return Bussi(
            atoms,
            timestep=timestep,
            temperature_K=temperature_k,
            taut=tau_fs * units.fs,
            rng=rng,
        )
    raise ValueError(f"unsupported thermostat type {thermostat!r}")


class HDF5TrajectoryWriter:
    """Append ASE states using the V2 TorchSim-compatible HDF5 schema."""

    def __init__(
        self,
        filename: Path,
        atoms: Any,
        model_name: str,
        timestep_fs: float,
        record_interval: int,
    ) -> None:
        import h5py
        import numpy as np

        self.atoms = atoms
        self.file = h5py.File(filename, "w")
        self.file.attrs.update(
            program="ASE",
            model=model_name,
            engine="ase",
            timestep_fs=timestep_fs,
            record_interval=record_interval,
            first_step=0,
        )
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
        vector_options = {
            "shape": (0, n_atoms, 3),
            "maxshape": (None, n_atoms, 3),
            "chunks": (1, n_atoms, 3),
            "dtype": np.float32,
        }
        self.positions = data.create_dataset("positions", **vector_options)
        self.velocities = data.create_dataset("velocities", **vector_options)
        self.cells = data.create_dataset(
            "cell",
            shape=(0, 3, 3),
            maxshape=(None, 3, 3),
            chunks=(1, 3, 3),
            dtype=np.float32,
        )

    def write(self) -> None:
        from ase import units

        frame = self.positions.shape[0]
        for dataset in (self.positions, self.velocities, self.cells):
            dataset.resize(frame + 1, axis=0)
        self.positions[frame] = self.atoms.get_positions()
        # ASE velocity -> Angstrom/ps; TorchSim stores lattice vectors as columns.
        self.velocities[frame] = self.atoms.get_velocities() * (1000.0 * units.fs)
        self.cells[frame] = self.atoms.cell.array.T
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def validate_calculator(calculator: Any) -> None:
    from ase.calculators.calculator import Calculator

    if not isinstance(calculator, Calculator):
        raise TypeError("ASE factory must return an ase Calculator instance")
    properties = set(getattr(calculator, "implemented_properties", ()))
    missing = {"energy", "forces"} - properties
    if missing:
        raise ValueError(
            f"calculator is missing required properties: {sorted(missing)}"
        )
    if "stress" in properties:
        raise ValueError(
            "production calculator advertises stress; provide an energy/force-only "
            "calculator so pressure work is not included in MD timings"
        )


def prepare_outputs(
    out_dir: Path, run_name: str, force: bool
) -> tuple[Path, Path, Path, Path] | None:
    out_h5 = out_dir / f"nvt_{run_name}.h5"
    temporary_h5 = out_dir / f"nvt_{run_name}.inprogress.h5"
    out_log = out_dir / f"md_{run_name}.log"
    out_csv = out_dir / f"md_timing_{run_name}.csv"
    existing = [
        path for path in (out_h5, temporary_h5, out_log, out_csv) if path.exists()
    ]
    complete = (
        out_h5.is_file()
        and out_h5.stat().st_size > 0
        and out_csv.is_file()
        and out_csv.stat().st_size > 0
    )
    if complete and not force:
        print(f"[{run_name}] {out_dir.name}: complete output exists, skipping")
        return None
    if existing and not force:
        raise FileExistsError(
            f"incomplete output exists for {out_dir.name}; inspect it or rerun with --force"
        )
    if force:
        for path in existing:
            path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_h5, temporary_h5, out_log, out_csv


def run_system(
    *,
    name: str,
    meta: Mapping[str, Any],
    args: argparse.Namespace,
    calculator: Any,
    metadata_file: Path,
    ref_root: Path,
) -> None:
    import numpy as np
    from ase.io import read
    from ase.md import MDLogger
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    run_name = f"{args.model_name}-ase"
    out_dir = args.output_root / name
    outputs = prepare_outputs(out_dir, run_name, args.force)
    if outputs is None:
        return
    out_h5, temporary_h5, out_log, out_csv = outputs
    init_file = resolve_init_file(name, meta, metadata_file, ref_root)
    if not init_file.is_file():
        raise FileNotFoundError(f"initial structure not found: {init_file}")

    temperature_k = float(meta["temperature"])
    timestep_fs = float(meta["timestep"])
    tau_fs = float(meta["thermostat_coupling_constant"])
    thermostat = str(meta["thermostat_type"])
    stride = int(meta.get("position_print_stride") or 1)
    energy_stride = int(meta.get("energy_print_stride") or 1)
    full_steps = round(float(meta["trajectory_length_ps"]) * 1000.0 / timestep_fs)
    n_steps = min(full_steps, args.max_steps) if args.max_steps else full_steps

    atoms = read(init_file, index=0)
    atoms.calc = calculator
    rng = np.random.default_rng(SEED)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, rng=rng)
    dynamics = make_integrator(
        atoms, thermostat, temperature_k, timestep_fs, tau_fs, rng
    )

    # One explicit evaluation validates the adapter and keeps model warmup outside timing.
    initial_energy = float(atoms.get_potential_energy())
    initial_forces = np.asarray(atoms.get_forces(), dtype=float)
    if initial_forces.shape != (len(atoms), 3):
        raise ValueError(f"unexpected force shape {initial_forces.shape}")
    if not np.isfinite(initial_energy) or not np.isfinite(initial_forces).all():
        raise ValueError("model returned a non-finite initial energy or force")

    print(
        f"[{run_name}] {name}: {len(atoms)} atoms, E0={initial_energy:.6f} eV, "
        f"T={temperature_k:g} K, dt={timestep_fs:g} fs, {thermostat} "
        f"tau={tau_fs:g} fs, {n_steps}/{full_steps} steps, stride {stride}"
    )
    trajectory = HDF5TrajectoryWriter(
        temporary_h5, atoms, run_name, timestep_fs, stride
    )
    dynamics.attach(trajectory.write, interval=stride)
    dynamics.attach(
        MDLogger(
            dynamics,
            atoms,
            str(out_log),
            header=True,
            stress=False,
            peratom=False,
            mode="w",
        ),
        interval=energy_stride,
    )
    try:
        synchronize_accelerator()
        start = time.perf_counter()
        dynamics.run(n_steps)
        synchronize_accelerator()
        elapsed = time.perf_counter() - start
    finally:
        trajectory.close()

    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "calculator": run_name,
                "system": name,
                "temperature_K": temperature_k,
                "n_steps": n_steps,
                "time_step_fs": timestep_fs,
                "thermostat": thermostat,
                "tau_fs": tau_fs,
                "record_interval": stride,
                "elapsed_seconds": f"{elapsed:.6f}",
                "seconds_per_step": f"{elapsed / n_steps:.9f}",
                "engine": args.engine_label,
                "seed": SEED,
            }
        )
    temporary_h5.replace(out_h5)
    print(
        f"[{run_name}] {name}: saved {out_h5} ({elapsed / n_steps * 1e3:.3f} ms/step)"
    )


def main(argv: list[str] | None = None, *, accelerated: bool = False) -> int:
    args = parse_args(argv, accelerated=accelerated)
    args.metadata = args.metadata.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if not args.metadata.is_file():
        raise FileNotFoundError(f"metadata file not found: {args.metadata}")
    ref_root = (
        args.ref_root.expanduser().resolve()
        if args.ref_root is not None
        else args.metadata.parent
    )
    metadata = json.loads(args.metadata.read_text())
    selected = select_metadata(metadata, args.systems)
    configure_strict_torch_precision()
    factory = load_factory(args.factory)

    shared_calculator = None
    if not args.calculator_per_system:
        shared_calculator = factory()
        validate_calculator(shared_calculator)

    failures: list[tuple[str, Exception]] = []
    for name, meta in selected:
        try:
            calculator = shared_calculator
            if calculator is None:
                calculator = factory()
                validate_calculator(calculator)
            run_system(
                name=name,
                meta=meta,
                args=args,
                calculator=calculator,
                metadata_file=args.metadata,
                ref_root=ref_root,
            )
        except Exception as exc:
            failures.append((name, exc))
            print(
                f"[{args.model_name}-ase] {name}: FAILED ({type(exc).__name__}: {exc})"
            )
            if args.fail_fast:
                break

    if failures:
        print("\nFailed systems:")
        for name, exc in failures:
            print(f"- {name}: {type(exc).__name__}: {exc}")
        return 1
    print(f"[{args.model_name}-ase] completed {len(selected)} system(s).")
    return 0
