#!/usr/bin/env python3
"""Evaluate stresses on saved TorchSim HDF5 frames without running MD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


_ARGS = None
_SCRIPT = None
_BACKEND = None


def _pressure_root(script: Path) -> Path:
    return next(parent for parent in script.parents if parent.name == "pressures")


def early_cli(script, backend: str) -> None:
    """Parse lightweight CLI options before importing a model's dependencies."""
    global _ARGS, _SCRIPT, _BACKEND
    _SCRIPT = Path(script).resolve()
    _BACKEND = backend
    root = _pressure_root(_SCRIPT)
    data = root.parent / "data"
    accelerated = "md_accelerated" in _SCRIPT.parts
    family = "md_accelerated" if accelerated else "md"
    default_trajectories = data / (
        "mlip-trajs-torchsim-accelerated" if accelerated else "mlip-trajs-torchsim"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-dir", type=Path, default=default_trajectories)
    parser.add_argument("--ref-dir", type=Path, default=data / "ref-trajs")
    parser.add_argument("--metadata", type=Path, default=data / "ref-trajs" / "md_metadata.json")
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "results" / backend / family,
    )
    parser.add_argument(
        "--trajectory-model",
        help="Trajectory identifier after nvt_; normally inferred from the pressure model name.",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--debug", action="store_true")
    _ARGS = parser.parse_args()
    if _ARGS.max_frames is not None and _ARGS.max_frames < 1:
        parser.error("--max-frames must be positive")


def _trajectory_model(model_name: str) -> str:
    """Map stress-enabled evaluator names back to production trajectory names."""
    name = model_name
    for suffix in ("-compile-stress", "-turbo-stress", "-stress"):
        if name.endswith(suffix):
            replacement = suffix.removesuffix("-stress")
            name = name.removesuffix(suffix) + replacement
            break
    if name == "nequip-oam-l":
        return "nequip"
    if name == "mattersim-v1-5M-compile":
        return "mattersim-v1-5M"
    return name


def _stress_matrix(value):
    import numpy as np

    stress = np.asarray(value, dtype=float).squeeze()
    if stress.shape == (6,):
        xx, yy, zz, yz, xz, xy = stress
        stress = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    elif stress.size == 9:
        stress = stress.reshape(3, 3)
    if stress.shape != (3, 3) or not np.isfinite(stress).all():
        raise ValueError(f"invalid stress tensor with shape {stress.shape}")
    if not np.allclose(stress, stress.T, atol=1e-6, rtol=1e-5):
        raise ValueError("stress tensor is not symmetric")
    return stress


def _pressure_row(stress, system: str) -> dict[str, object]:
    import numpy as np

    matrix = _stress_matrix(stress)
    components = ("xx", "yy", "zz", "yz", "xz", "xy")
    indices = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    is_2d = system.split("_")[0] == "TiSe2"
    diagonal = np.diag(matrix)[:2] if is_2d else np.diag(matrix)
    return {
        **{
            f"stress_{component}_eV_A3": float(matrix[i, j])
            for component, (i, j) in zip(components, indices)
        },
        "pressure_GPa": -float(np.mean(diagonal)) * 160.21766208,
        "pressure_mode": "2d" if is_2d else "3d",
        "plane_used": "xy" if is_2d else "",
        "stress_kind": "potential",
    }


def _frames(path: Path, meta: dict[str, object], limit: int | None):
    """Yield ASE frames from the repository's TorchSim HDF5 schema."""
    import h5py
    import numpy as np
    from ase import Atoms

    with h5py.File(path, "r") as handle:
        data = handle["data"]
        positions = data["positions"]
        cells = data["cell"]
        numbers = np.asarray(data["atomic_numbers"][0], dtype=int)
        pbc = np.asarray(data["pbc"], dtype=bool)
        steps = (
            np.asarray(handle["steps/positions"]).reshape(-1)
            if "steps/positions" in handle
            else np.arange(len(positions)) * int(meta.get("position_print_stride") or 1)
        )
        count = min(len(positions), limit) if limit is not None else len(positions)
        if len(cells) != len(positions) or len(steps) != len(positions):
            raise ValueError("positions, cells, and recorded steps have different lengths")
        timestep = float(meta["timestep"])
        for index in range(count):
            # TorchSim serializes cell vectors as columns; ASE expects rows.
            atoms = Atoms(numbers=numbers, positions=positions[index], cell=cells[index].T, pbc=pbc)
            yield index, float(steps[index]), float(steps[index]) * timestep, atoms


def _reference_pressures(path: Path, meta: dict[str, object]):
    from pressure_pipeline import read_stress_frames

    return read_stress_frames(path, meta)


def _run(model_name: str, predict, engine: str) -> None:
    import numpy as np
    import pandas as pd
    args = _ARGS
    if args is None:
        raise RuntimeError("early_cli() must be called before run_pressure()")
    trajectory_model = args.trajectory_model or _trajectory_model(model_name)
    paths = sorted(args.traj_dir.rglob(f"nvt_{trajectory_model}.h5"))
    if not paths:
        raise SystemExit(
            f"No nvt_{trajectory_model}.h5 trajectories found under {args.traj_dir}"
        )
    metadata = json.loads(args.metadata.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{trajectory_model}_same-simulation-length_pressure_per_frame.csv"
    full_output = args.output_dir / f"{trajectory_model}_stress_per_frame.csv"
    summary_output = args.output_dir / f"{trajectory_model}_same-simulation-length_pressure_trajectory_summary.csv"
    failure_output = args.output_dir / f"{trajectory_model}_pressure_failures.json"
    if (output.exists() and full_output.exists() and summary_output.exists()
            and not failure_output.exists() and not args.force):
        print(f"{output} exists; use --force to recompute.")
        return

    full_rows, rows, reference_rows, summaries, failures = [], [], [], [], []
    for path in paths:
        system = path.parent.name
        if system.startswith("Pt111w24H2O_"):
            continue
        try:
            meta = metadata[system]
            reference_path = args.ref_dir / system / "traj.extxyz"
            reference = _reference_pressures(reference_path, meta)
            model_rows = []
            frame_iterator = _frames(path, meta, args.max_frames)
            total = args.max_frames
            calculator_state = None
            for index, step, time_fs, atoms in frame_iterator:
                try:
                    stress, calculator_state = predict(atoms, calculator_state)
                    model_rows.append({
                        "frame_index": index,
                        "step": step,
                        "time_fs": time_fs,
                        "trajectory_file": str(path.resolve()),
                        "system": system,
                        "calculator": model_name,
                        "engine": engine,
                        **_pressure_row(stress, system),
                    })
                except Exception as exc:
                    failures.append({"file": str(path), "frame": index, "error": str(exc)})
                    if args.debug:
                        import traceback
                        traceback.print_exc()
            if not model_rows:
                raise ValueError("no frames were evaluated successfully")
            full_rows.extend(model_rows)
            model = pd.DataFrame(model_rows)
            start = max(float(reference.time_fs.min()), float(model.time_fs.min()))
            end = min(float(reference.time_fs.max()), float(model.time_fs.max()))
            if end < start:
                raise ValueError("reference and model trajectories have no overlapping time")
            matched = model.loc[model.time_fs.between(start - 1e-8, end + 1e-8)].copy()
            matched_ref = reference.loc[
                reference.time_fs.between(start - 1e-8, end + 1e-8)
            ].copy()
            matched_ref["mlip_model"] = trajectory_model
            rows.extend(matched.to_dict("records"))
            reference_rows.extend(matched_ref.to_dict("records"))
            values = matched.pressure_GPa.to_numpy(dtype=float)
            reference_values = matched_ref.pressure_GPa.to_numpy(dtype=float)
            summaries.append({
                "system": system,
                "mlip_model": trajectory_model,
                "calculator": model_name,
                "engine": engine,
                "trajectory_file": str(path.resolve()),
                "mlip_frames_total": len(model),
                "mlip_frames_used": len(matched),
                "reference_frames_used": len(matched_ref),
                "matched_start_fs": start,
                "matched_end_fs": end,
                "pressure_mean_GPa": float(np.mean(values)),
                "pressure_std_GPa": float(np.std(values)),
                "pressure_min_GPa": float(np.min(values)),
                "pressure_max_GPa": float(np.max(values)),
                "pressure_ref_mean_GPa": float(np.mean(reference_values)),
                "signed_mean_error_GPa": float(np.mean(values) - np.mean(reference_values)),
                "absolute_mean_error_GPa": float(abs(np.mean(values) - np.mean(reference_values))),
            })
            print(f"{system}: evaluated {len(model)} frames; matched {len(matched)}")
        except Exception as exc:
            failures.append({"file": str(path), "error": str(exc)})
            print(f"FAILED {path}: {exc}")
            if args.debug:
                import traceback
                traceback.print_exc()

    if rows:
        pd.DataFrame(full_rows).to_csv(full_output, index=False)
        pd.DataFrame(rows).to_csv(output, index=False)
        pd.DataFrame(summaries).to_csv(summary_output, index=False)
        reference_dir = args.output_dir / "references"
        reference_dir.mkdir(exist_ok=True)
        pd.DataFrame(reference_rows).to_csv(reference_dir / f"{trajectory_model}.csv", index=False)
    failure_output.write_text(json.dumps(failures, indent=2) + "\n") if failures else failure_output.unlink(missing_ok=True)
    if failures:
        raise SystemExit(f"Pressure evaluation incomplete: {len(failures)} failures; see {failure_output}")
    if not rows:
        raise SystemExit("No pressure rows were produced")
    print(f"Saved {output}")
    print(f"Saved {full_output}")
    print(f"Saved {summary_output}")


def run_ase_pressure(model_name, calculator_factory, engine="ase", *, per_system_calculator=False, **_):
    """Evaluate ASE-calculator stress on saved frames, without integration."""
    shared_calculator = None

    def predict(atoms, calculator):
        import numpy as np

        nonlocal shared_calculator

        if per_system_calculator:
            if calculator is None:
                calculator = calculator_factory()
        else:
            if shared_calculator is None:
                shared_calculator = calculator_factory()
            calculator = shared_calculator
        frame = atoms.copy()
        frame.info["charge"] = int(frame.info.get("charge", 0))
        frame.info["spin"] = int(frame.info.get("spin", 0))
        frame.calc = calculator
        return np.asarray(frame.get_stress(voigt=False, include_ideal_gas=False)).copy(), calculator

    _run(model_name, predict, engine)


def run_torchsim_pressure(model_name, model, engine="torchsim", *, state_dtype=None, **_):
    """Evaluate a TorchSim model's stress on saved frames, without integration."""
    def predict(atoms, state):
        import numpy as np
        import torch
        import torch_sim as ts

        dtype = state_dtype if state_dtype is not None else torch.float32
        state = ts.initialize_state(atoms, torch.device("cuda"), dtype)
        result = model(state)
        if "stress" not in result:
            raise ValueError("model result does not contain stress")
        stress = result["stress"].detach().cpu().numpy()
        if stress.size != 9:
            raise ValueError(f"unexpected stress shape {stress.shape}")
        return np.asarray(stress).reshape(3, 3).copy(), state

    _run(model_name, predict, engine)
