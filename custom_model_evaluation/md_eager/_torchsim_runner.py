"""Run the V2 reference-matched MD workload with a user-provided TorchSim model.

The user module must expose a zero-argument function that returns a
TorchSim-compatible callable model.  For example::

    python md_custom_model.py \
        --model-name my-model \
        --model-loader /path/to/my_model.py:make_model

The model receives a TorchSim state and must return total ``energy`` and
``forces`` tensors in eV and eV/Angstrom.  Production stress must be disabled.
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
SY_STEPS = 3
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


def parse_model_loader_spec(value: str) -> tuple[str, str]:
    """Split MODULE[:FUNCTION], defaulting to ``make_model``."""
    module, separator, function = value.rpartition(":")
    if not separator:
        return value, "make_model"
    if not module or not function:
        raise argparse.ArgumentTypeError(
            "model loader must be MODULE[:FUNCTION] or /path/to/file.py[:FUNCTION]"
        )
    return module, function


def load_module(module_ref: str) -> ModuleType:
    """Import a dotted module or load a Python file without changing cwd."""
    path = Path(module_ref).expanduser()
    if path.suffix == ".py" or path.is_file():
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"model module does not exist: {path}")
        module_name = f"custom_torchsim_model_{abs(hash(path))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load model module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_ref)


def load_model_loader(value: str) -> Callable[[], Any]:
    module_ref, function_name = parse_model_loader_spec(value)
    module = load_module(module_ref)
    model_loader = getattr(module, function_name, None)
    if not callable(model_loader):
        raise TypeError(f"{value!r} does not identify a callable model loader")
    return model_loader


def safe_model_name(value: str) -> str:
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
        "--model-loader",
        required=True,
        help="dotted module or .py file, optionally followed by :FUNCTION",
    )
    parser.add_argument(
        "--engine-label",
        default=f"torch-sim+custom-{mode}",
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
        default=CUSTOM_ROOT / f"data/mlip-trajs-torchsim{output_suffix}",
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="TorchSim state dtype",
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
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


def make_integrator(
    torch_sim: Any, thermostat: str, tau_fs: float
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    if thermostat == "Langevin":
        return torch_sim.Integrator.nvt_langevin, {}, {"gamma": 1000.0 / tau_fs}
    if thermostat == "Nose-Hoover":
        return (
            torch_sim.Integrator.nvt_nose_hoover,
            {
                "tau": tau_fs / 1000.0,
                "chain_length": CHAIN_LENGTH,
                "chain_steps": CHAIN_STEPS,
                "sy_steps": SY_STEPS,
            },
            {},
        )
    if thermostat.lower().startswith("velocity"):
        return torch_sim.Integrator.nvt_vrescale, {}, {"tau": tau_fs / 1000.0}
    raise ValueError(f"unsupported thermostat type {thermostat!r}")


def validate_model_configuration(model: Any, device: Any, dtype: Any) -> None:
    import torch

    if not callable(model):
        raise TypeError("TorchSim model loader must return a callable model")
    if not hasattr(model, "device") or not hasattr(model, "dtype"):
        raise TypeError(
            "TorchSim model must expose .device and .dtype attributes used by "
            "torch_sim.integrate"
        )
    model_device = torch.device(model.device)
    if model_device != device or model.dtype != dtype:
        raise ValueError(
            "runner/model device or dtype mismatch: "
            f"runner=({device}, {dtype}), model=({model_device}, {model.dtype}); "
            "change --device/--dtype or the model loader"
        )
    if bool(getattr(model, "compute_stress", False)):
        raise ValueError(
            "production TorchSim model has compute_stress=True; disable stress so "
            "pressure work is not included in MD timings"
        )


def validate_prediction(result: Any, n_atoms: int) -> None:
    import torch

    if not isinstance(result, Mapping):
        raise TypeError("TorchSim model must return a mapping")
    missing = {"energy", "forces"} - set(result)
    if missing:
        raise ValueError(f"model result is missing keys: {sorted(missing)}")
    energy = result["energy"]
    forces = result["forces"]
    if not isinstance(energy, torch.Tensor) or energy.numel() != 1:
        raise ValueError("result['energy'] must be a one-element torch.Tensor")
    if not isinstance(forces, torch.Tensor) or tuple(forces.shape) != (n_atoms, 3):
        raise ValueError(
            f"result['forces'] must have shape ({n_atoms}, 3), got "
            f"{getattr(forces, 'shape', None)}"
        )
    if not torch.isfinite(energy).all() or not torch.isfinite(forces).all():
        raise ValueError("model returned a non-finite initial energy or force")


def prepare_outputs(
    out_dir: Path, model_name: str, force: bool
) -> tuple[Path, Path, Path] | None:
    out_h5 = out_dir / f"nvt_{model_name}.h5"
    temporary_h5 = out_dir / f"nvt_{model_name}.inprogress.h5"
    out_csv = out_dir / f"md_timing_{model_name}.csv"
    existing = [path for path in (out_h5, temporary_h5, out_csv) if path.exists()]
    complete = (
        out_h5.is_file()
        and out_h5.stat().st_size > 0
        and out_csv.is_file()
        and out_csv.stat().st_size > 0
    )
    if complete and not force:
        print(f"[{model_name}] {out_dir.name}: complete output exists, skipping")
        return None
    if existing and not force:
        raise FileExistsError(
            f"incomplete output exists for {out_dir.name}; inspect it or rerun with --force"
        )
    if force:
        for path in existing:
            path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_h5, temporary_h5, out_csv


def annotate_and_validate_hdf5(
    path: Path, model_name: str, timestep_fs: float, record_interval: int
) -> None:
    """Check the analysis requirements before publishing the temporary trajectory."""
    import h5py

    with h5py.File(path, "r+") as handle:
        required = (
            "data/atomic_numbers",
            "data/masses",
            "data/positions",
            "data/cell",
            "data/pbc",
            "data/velocities",
        )
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(f"TorchSim trajectory is missing datasets: {missing}")
        frame_counts = {
            key: int(handle[f"data/{key}"].shape[0])
            for key in ("positions", "cell", "velocities")
        }
        if not frame_counts["positions"] or len(set(frame_counts.values())) != 1:
            raise ValueError(f"inconsistent trajectory frame counts: {frame_counts}")
        handle.attrs.update(
            model=model_name,
            engine="torchsim",
            timestep_fs=timestep_fs,
            record_interval=record_interval,
        )
        if "first_step" not in handle.attrs:
            handle.attrs["first_step"] = 0


def run_system(
    *,
    name: str,
    meta: Mapping[str, Any],
    args: argparse.Namespace,
    model: Any,
    metadata_file: Path,
    ref_root: Path,
    device: Any,
    dtype: Any,
) -> None:
    import torch
    import torch_sim as ts
    from ase.io import read

    out_dir = args.output_root / name
    outputs = prepare_outputs(out_dir, args.model_name, args.force)
    if outputs is None:
        return
    out_h5, temporary_h5, out_csv = outputs
    init_file = resolve_init_file(name, meta, metadata_file, ref_root)
    if not init_file.is_file():
        raise FileNotFoundError(f"initial structure not found: {init_file}")

    temperature_k = float(meta["temperature"])
    timestep_fs = float(meta["timestep"])
    tau_fs = float(meta["thermostat_coupling_constant"])
    thermostat = str(meta["thermostat_type"])
    stride = int(meta.get("position_print_stride") or 1)
    full_steps = round(float(meta["trajectory_length_ps"]) * 1000.0 / timestep_fs)
    n_steps = min(full_steps, args.max_steps) if args.max_steps else full_steps
    integrator, init_kwargs, step_kwargs = make_integrator(ts, thermostat, tau_fs)

    atoms = read(init_file, index=0)
    state = ts.initialize_state(atoms, device, dtype)
    state.rng = SEED
    if not args.no_warmup:
        result = model(state)
        validate_prediction(result, len(atoms))
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    print(
        f"[{args.model_name}] {name}: {len(atoms)} atoms, T={temperature_k:g} K, "
        f"dt={timestep_fs:g} fs, {thermostat} tau={tau_fs:g} fs, "
        f"{n_steps}/{full_steps} steps, stride {stride}"
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    ts.integrate(
        system=state,
        model=model,
        integrator=integrator,
        n_steps=n_steps,
        temperature=temperature_k,
        timestep=timestep_fs / 1000.0,
        init_kwargs=init_kwargs,
        trajectory_reporter={
            "filenames": [str(temporary_h5)],
            "state_frequency": stride,
            "state_kwargs": {"save_velocities": True, "save_forces": False},
        },
        pbar=not args.no_progress,
        **step_kwargs,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    annotate_and_validate_hdf5(temporary_h5, args.model_name, timestep_fs, stride)

    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "calculator": args.model_name,
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
        f"[{args.model_name}] {name}: saved {out_h5} ({elapsed / n_steps * 1e3:.3f} ms/step)"
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

    # Import heavyweight packages only after CLI/path validation.
    import torch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device is CUDA, but CUDA is not available")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dtype = getattr(torch, args.dtype)
    model = load_model_loader(args.model_loader)()
    validate_model_configuration(model, device, dtype)

    failures: list[tuple[str, Exception]] = []
    for name, meta in selected:
        try:
            run_system(
                name=name,
                meta=meta,
                args=args,
                model=model,
                metadata_file=args.metadata,
                ref_root=ref_root,
                device=device,
                dtype=dtype,
            )
        except Exception as exc:
            failures.append((name, exc))
            print(f"[{args.model_name}] {name}: FAILED ({type(exc).__name__}: {exc})")
            if args.fail_fast:
                break

    if failures:
        print("\nFailed systems:")
        for name, exc in failures:
            print(f"- {name}: {type(exc).__name__}: {exc}")
        return 1
    print(f"[{args.model_name}] completed {len(selected)} system(s).")
    return 0
