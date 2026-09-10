#!/usr/bin/env python3
"""Extract stored stress, compare equal-time pressure distributions, and plot."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import h5py
from ase.io import iread
from get_model_pressure_errors import pressure_histogram_similarity, infer_system_type, write_metric_outputs

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SUFFIX = "_same-simulation-length_pressure_per_frame.csv"
REFERENCE_NAME = "reference_pressure_per_frame_same_simulation_length.csv"
EV_A3_TO_GPA = 160.21766208
COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")
INDICES = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def stress_matrix(value):
    a = np.asarray(value, dtype=float).squeeze()
    if a.shape == (6,):
        xx, yy, zz, yz, xz, xy = a
        a = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    elif a.size == 9:
        a = a.reshape(3, 3)
    if a.shape != (3, 3) or not np.isfinite(a).all():
        raise ValueError(f"Invalid stress tensor: {a}")
    if not np.allclose(a, a.T, atol=1e-6, rtol=1e-5):
        raise ValueError("Stress tensor is not symmetric")
    return a


def stress_row(stress, structure):
    a = stress_matrix(stress)
    inplane = structure.split("_")[0] == "TiSe2"
    pressure = -np.mean(np.diag(a)[:2] if inplane else np.diag(a)) * EV_A3_TO_GPA
    return {**{f"stress_{c}_eV_A3": a[i, j] for c, (i, j) in zip(COMPONENTS, INDICES)},
            "pressure_GPa": pressure, "pressure_mode": "2d" if inplane else "3d",
            "plane_used": "xy" if inplane else "", "stress_kind": "potential"}


def positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def read_stress_frames(path, meta, calculator=None, reference_first_step=0):
    """Read ASE extxyz or native ASE/TorchSim HDF5, preserving frame times."""
    path = Path(path)
    structure = path.parent.name
    dt = positive(meta["timestep"], "timestep")
    stride = positive(meta.get("position_print_stride") or 1, "position_print_stride")
    rows = []
    if path.suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as f:
            data = f["data"]
            n = len(data["positions"])
            if "stress" not in data:
                raise ValueError(f"{path}: no stored stress; run the stress-enabled MD runner")
            if len(data["stress"]) != n:
                raise ValueError(f"{path}: stress and positions have different frame counts")
            units = data["stress"].attrs.get("units", f.attrs.get("stress_units", "eV/Angstrom^3"))
            if isinstance(units, bytes): units = units.decode()
            if units not in {"eV/Angstrom^3", "eV/A^3"}:
                raise ValueError(f"{path}: unsupported stress units {units!r}")
            dt = positive(f.attrs.get("timestep_fs", dt), "timestep_fs")
            stride = positive(f.attrs.get("record_interval", stride), "record_interval")
            if "steps/positions" in f:
                steps = np.asarray(f["steps/positions"]).reshape(-1)
                if len(steps) != n:
                    raise ValueError(f"{path}: position step count does not match frames")
                if "steps/stress" not in f or not np.array_equal(steps, np.asarray(f["steps/stress"]).reshape(-1)):
                    raise ValueError(f"{path}: stress and position step numbers do not match")
            else:
                steps = float(f.attrs.get("first_step", 0)) + np.arange(n) * stride
            for i in range(n):
                rows.append({"frame_index": i, "step": steps[i], "time_fs": steps[i] * dt,
                             **stress_row(data["stress"][i], structure)})
    else:
        for i, atoms in enumerate(iread(path, index=":")):
            if calculator is not None:
                atoms.info["charge"] = int(atoms.info.get("charge", 0))
                atoms.info["spin"] = int(atoms.info.get("spin", 0))
                atoms.calc = calculator
            stress = atoms.get_stress(voigt=False, include_ideal_gas=False)
            step = atoms.info.get("step", reference_first_step + i * stride)
            time_fs = atoms.info.get("time_fs", float(step) * dt)
            rows.append({"frame_index": i, "step": step, "time_fs": time_fs,
                         **stress_row(stress, structure)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"{path}: empty trajectory")
    times = df.time_fs.to_numpy(dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"{path}: frame times must be finite and strictly increasing")
    df["trajectory_file"] = str(path.resolve())
    df["system"] = structure
    return df


def read_reference_csv(path, metadata, first_step=0):
    df = pd.read_csv(path).rename(columns={"pressure_ref_GPa": "pressure_GPa"})
    if "trajectory_file" not in df or "pressure_GPa" not in df:
        raise ValueError("Reference CSV needs trajectory_file and pressure_GPa (or pressure_ref_GPa)")
    df["system"] = df.trajectory_file.map(lambda p: str(p).replace(chr(92), "/").split("/")[-2])
    if "frame_index" not in df:
        raise ValueError("Reference CSV needs frame_index to align simulation times")
    df = df.drop_duplicates(["trajectory_file", "frame_index"])
    if "time_fs" not in df:
        df["time_fs"] = [
            (first_step + float(i) * positive(metadata[s].get("position_print_stride") or 1, "stride"))
            * positive(metadata[s]["timestep"], "timestep")
            for s, i in zip(df.system, df.frame_index)]
    if not np.isfinite(df[["pressure_GPa", "time_fs"]].to_numpy(dtype=float)).all():
        raise ValueError("Reference CSV contains non-finite pressure/time values")
    for structure, group in df.groupby("system"):
        if group.time_fs.duplicated().any():
            raise ValueError(f"Duplicate reference times for {structure}")
    return df.sort_values(["system", "time_fs"]).reset_index(drop=True)


def match_time_window(ref, model):
    start = max(float(ref.time_fs.min()), float(model.time_fs.min()))
    end = min(float(ref.time_fs.max()), float(model.time_fs.max()))
    if end < start:
        raise ValueError("Reference and MLIP have no overlapping simulation time")
    ref = ref.loc[ref.time_fs.between(start - 1e-8, end + 1e-8)].copy()
    model = model.loc[model.time_fs.between(start - 1e-8, end + 1e-8)].copy()
    if ref.empty or model.empty:
        raise ValueError("No saved frames in the common simulation-time window")
    return ref, model, start, end


def plot_outputs(pairs, matched, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    summary = pairs.groupby("mlip_model")[["absolute_mean_error_GPa", "pressure_similarity_percent"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(summary) * .3)))
    summary.absolute_mean_error_GPa.sort_values().plot.barh(ax=axes[0])
    axes[0].set_xlabel("Mean absolute error of trajectory means (GPa)")
    summary.pressure_similarity_percent.sort_values().plot.barh(ax=axes[1])
    axes[1].set_xlabel("Pressure distribution similarity (%)")
    axes[1].set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(plots / "pressure_model_comparison.png", dpi=180)
    fig.savefig(plots / "pressure_model_comparison.pdf")
    plt.close(fig)
    for model, structure, ref, mlip in matched:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        for frame, label in [(ref, "Reference"), (mlip, model)]:
            axes[0].plot(frame.time_fs / 1000, frame.pressure_GPa, label=label, alpha=.7, lw=.7)
        edges = np.histogram_bin_edges(np.r_[ref.pressure_GPa, mlip.pressure_GPa], bins=60)
        axes[1].hist(ref.pressure_GPa, bins=edges, density=True, histtype="step", label="Reference")
        axes[1].hist(mlip.pressure_GPa, bins=edges, density=True, histtype="step", label=model)
        axes[0].set(xlabel="Time (ps)", ylabel="Pressure (GPa)")
        axes[1].set(xlabel="Pressure (GPa)", ylabel="Probability density")
        axes[1].legend(fontsize=7)
        fig.suptitle(structure)
        fig.tight_layout()
        folder = plots / model
        folder.mkdir(exist_ok=True)
        fig.savefig(folder / f"{structure}.png", dpi=160)
        plt.close(fig)


def run_pipeline(args):
    metadata = json.loads(args.metadata.read_text())
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "references").mkdir(exist_ok=True)
    reference_csv = read_reference_csv(args.reference_file, metadata, args.reference_first_step) if args.reference_file else None
    roots = args.traj_dir or [DATA / "mlip-trajs-torchsim-matched-stress", DATA / "mlip-trajs-torchsim-accelerated-stress", DATA / "mlip-trajs-ase-accelerated-stress"]
    paths = sorted({p.resolve() for root in roots for p in root.rglob(f"{args.prefix or 'nvt_'}*") if p.suffix in {".h5", ".hdf5", ".extxyz", ".xyz"}})
    if args.model:
        paths = [p for p in paths if p.stem.removeprefix("nvt_").lower() == args.model.lower()]
    if not paths:
        raise FileNotFoundError("No matching MLIP trajectories found")
    pairs, matched, failures = [], [], []
    full_frames, used_frames, ref_frames, summaries = {}, {}, {}, {}
    refs = {}
    seen = set()
    calculator = None
    if args.recompute:
        if not args.model:
            raise ValueError("--recompute requires --model (an exact calculator catalog key)")
        catalog = json.loads((HERE / "model_calculators.json").read_text())
        entry = next(m for m in catalog["models"] if m["name"] == args.model)
        ns = {}
        for line in entry["imports"]: exec(line, ns)
        calculator = eval(entry["calculator_expr"], ns)
    for path in paths:
        model, structure = path.stem.removeprefix("nvt_").lower(), path.parent.name
        if structure.startswith("Pt111w24H2O_") and not args.include_interfaces:
            continue
        if (model, structure) in seen:
            raise ValueError(f"Ambiguous duplicate model/system {model}/{structure}; analyze the roots separately")
        seen.add((model, structure))
        try:
            meta = metadata[structure]
            mlip = read_stress_frames(path, meta, calculator)
            # Save every frame, even when the reference is shorter or unavailable.
            full_frames.setdefault(model, []).append(mlip)
            if structure not in refs:
                if reference_csv is not None:
                    ref = reference_csv.loc[reference_csv.system == structure].copy()
                    if ref.empty: raise ValueError(f"No reference CSV rows for {structure}")
                else:
                    folder = args.ref_dir / structure
                    candidates = [folder / "traj.extxyz"] + sorted(folder.glob("*.extxyz")) + sorted(folder.glob("*.xyz"))
                    refpath = next((p for p in candidates if p.is_file()), None)
                    if refpath is None: raise FileNotFoundError(f"No reference trajectory in {folder}")
                    ref = read_stress_frames(refpath, meta, reference_first_step=args.reference_first_step)
                refs[structure] = ref
            ref, use, start, end = match_time_window(refs[structure], mlip)
            ref["mlip_model"] = model
            use["matched_start_fs"], use["matched_end_fs"] = start, end
            use["reference_frames_used"] = len(ref)
            used_frames.setdefault(model, []).append(use)
            ref_frames.setdefault(model, []).append(ref)
            score = pressure_histogram_similarity(ref.pressure_GPa.to_numpy(), use.pressure_GPa.to_numpy(), args.bins)
            mean_ref, mean_model = ref.pressure_GPa.mean(), use.pressure_GPa.mean()
            row = dict(system=structure, system_type=infer_system_type(structure) or "other",
                       mlip_model=model, **score, n_ref_frames=len(ref), n_mlip_frames=len(use), bins=args.bins,
                       pressure_ref_mean_GPa=mean_ref, pressure_mean_GPa=mean_model,
                       signed_mean_error_GPa=mean_model-mean_ref, absolute_mean_error_GPa=abs(mean_model-mean_ref),
                       matched_start_fs=start, matched_end_fs=end,
                       reference_file=str(output / "references" / f"{model}.csv"),
                       model_file=str(output / f"{model}{SUFFIX}"))
            pairs.append(row)
            summaries.setdefault(model, []).append({**row, "trajectory_file": str(path), "mlip_frames_total": len(mlip),
                                                    "pressure_std_GPa": use.pressure_GPa.std(ddof=0)})
            matched.append((model, structure, ref, use))
            print(f"{model}/{structure}: saved {len(mlip)} stress frames, matched {len(use)} MLIP / {len(ref)} reference frames")
        except Exception as exc:
            failures.append(dict(trajectory_file=str(path), error=f"{type(exc).__name__}: {exc}"))
            print(f"FAILED {path}: {exc}")
    for model, frames in full_frames.items():
        pd.concat(frames, ignore_index=True).to_csv(output / f"{model}_stress_per_frame.csv", index=False)
    pd.DataFrame(failures, columns=["trajectory_file", "error"]).to_csv(output / "pressure_failures.csv", index=False)
    if not pairs:
        raise RuntimeError("No pressure comparisons produced; see pressure_failures.csv")
    for model in used_frames:
        pd.concat(used_frames[model], ignore_index=True).to_csv(output / f"{model}{SUFFIX}", index=False)
        pd.concat(ref_frames[model], ignore_index=True).to_csv(output / "references" / f"{model}.csv", index=False)
        pd.DataFrame(summaries[model]).to_csv(output / f"{model}_same-simulation-length_pressure_trajectory_summary.csv", index=False)
    pd.concat(list(refs.values()), ignore_index=True).to_csv(output / REFERENCE_NAME, index=False)
    pair_df = pd.DataFrame(pairs)
    comparison = pair_df.groupby("mlip_model", as_index=False).absolute_mean_error_GPa.mean().rename(columns={"mlip_model": "model", "absolute_mean_error_GPa": "error_GPa"})
    comparison.to_csv(output / "model_mean_pressure_comparison.csv", index=False)
    write_metric_outputs(pair_df, output / "pressure_pair_similarity_same_simulation_length.csv",
                         output / "pressure_system_model_mean_similarity_same_simulation_length.csv",
                         output / "model_pressure_error_metric.csv",
                         output / "pressure_model_system_type_mean_similarity_same_simulation_length.csv",
                         output / "model_mean_pressure_comparison.csv")
    if not args.no_plots:
        plot_outputs(pair_df, matched, output)
    if failures:
        raise RuntimeError(f"{len(failures)} trajectories failed; successful outputs saved, see pressure_failures.csv")
    return pair_df


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-dir", type=Path, action="append", help="MLIP root; repeat for multiple trees")
    parser.add_argument("--ref-dir", type=Path, default=DATA / "ref-trajs")
    parser.add_argument("--metadata", type=Path, default=DATA / "ref-trajs/md_metadata.json")
    parser.add_argument("--reference-file", type=Path, help="Optional per-frame reference pressure CSV in GPa")
    parser.add_argument("--reference-first-step", type=int, default=0, help="Step of first reference frame when no explicit time is stored")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME"), help="Exact trajectory identifier; default all models")
    parser.add_argument("--prefix", help="Trajectory filename prefix")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--include-interfaces", action="store_true")
    parser.add_argument("--recompute", action="store_true", help="Explicitly reevaluate extxyz frames with --model catalog calculator")
    args = parser.parse_args(argv)
    if args.bins < 2: parser.error("--bins must be >= 2")
    run_pipeline(args)


if __name__ == "__main__":
    main()
