#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "ase>=3.26",
#   "h5py>=3.11",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "scipy>=1.13",
# ]
# ///
"""Compute matched-length VDOS results for all ASE/TorchSim MD sources.

Reference positions are read from ``../data/ref-trajs/*/traj.extxyz`` and
their timesteps from ``../data/ref-trajs/md_metadata.json``. MLIP trajectories
use the HDF5 format produced by both the ASE and TorchSim MD runners. By
default, their stored velocities are used; ``--numerical-mlip-velocities``
instead differentiates their positions, as is necessarily done for references.

Every trajectory source is written to a separate subdirectory of ``results``.
Existing spectra are reused unless ``--overwrite`` is supplied, so interrupted
or expanded runs can be resumed cheaply.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from ase.io import iread
from scipy import signal


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

SOURCES = (
    "mlip-trajs-ase",
    "mlip-trajs-torchsim",
    "mlip-trajs-ase-accelerated",
    "mlip-trajs-torchsim-accelerated",
)

SPEED_OF_LIGHT_CM_S = 2.99792458e10
CM_INVERSE_TO_EV = 1.2398419843320026e-4

SYSTEM_TYPES = {
    "Pure metals": [
        "bulkAu_1500K_Kapil",
        "bulkAg_600K_Kapil",
        "bulkCu_1000K_Kapil",
    ],
    "Perovskites": ["CsSnI3_500K_Ivor_VASP", "MAPbBr3_300K_Ivor_VASP"],
    "Metal dichalcogenides": [
        "bulkMoS2_300K_NO-VdW_J.Kioseoglou_VASP",
        "TiSe2_400K_Ivor_VASP",
    ],
    "Metal alloys": [
        "bulkCuAu_500K-Artrith_VASP",
        "bulkCuZrAl_1500K_A.Wadowski-J.Schmidt_VASP",
        "bulkLiMgAlZnSn_600K_J_Schmidt_VASP",
        "bulkLiMgAlZnSn_900K_J_Schmidt_VASP",
        "bulkPt3Co_300K_J.Kioseoglou_VASP",
    ],
    "Molecular crystals": [
        "anthracene_293K_Sharma_S",
        "naphthalene_295K_Sharma_S",
        "pentacene_295K_Sharma_S",
        "picene_295K_Sharma_S",
        "tetracene_295K_Sharma_S",
    ],
    "Metal-water interfaces": ["Pt111w24H2O_380K_Heenen_VASP"],
    "Hydrogen": ["H_1050K_Rupp_QE"],
}
SYSTEM_TO_TYPE = {
    system: system_type
    for system_type, systems in SYSTEM_TYPES.items()
    for system in systems
}

PAIR_OUTPUT = "vdos_pair_errors_ev_normalized_same_simulation_length.csv"
SYSTEM_MODEL_OUTPUT = "vdos_system_model_mean_ev_normalized_same_simulation_length.csv"
MODEL_OUTPUT = "vdos_model_mean_ev_normalized_same_simulation_length.csv"
SCORES_OUTPUT = "vdos_similarity_scores_same_simulation_length.csv"
BY_TYPE_OUTPUT = "vdos_similarity_scores_by_system_type_same_simulation_length.csv"
SPECTRA_DIR = "vdos_same_simulation_length_saved"


def load_metadata(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Reference MD metadata not found: {path}")
    with path.open() as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return metadata


def discover_mlip_trajectories(base_dir: Path) -> dict[str, dict[str, Path]]:
    """Discover every ``<system>/nvt_<model>.h5`` trajectory."""
    trajectories: dict[str, dict[str, Path]] = {}
    for path in sorted(base_dir.glob("*/nvt_*.h5")):
        model = path.stem.removeprefix("nvt_")
        trajectories.setdefault(path.parent.name, {})[model] = path
    return trajectories


def load_reference_positions(path: Path) -> np.ndarray:
    positions = [
        np.asarray(atoms.positions, dtype=np.float64)
        for atoms in iread(path, index=":")
    ]
    if not positions:
        raise ValueError("trajectory contains no frames")
    return np.stack(positions)


def h5_frame_count(path: Path, *, numerical_velocities: bool) -> int:
    dataset = "data/positions" if numerical_velocities else "data/velocities"
    with h5py.File(path, "r") as h5:
        if dataset not in h5:
            raise ValueError(f"missing {dataset} dataset")
        return int(h5[dataset].shape[0])


def load_mlip_samples(
    path: Path,
    n_frames: int,
    *,
    numerical_velocities: bool,
) -> tuple[np.ndarray, bool]:
    dataset = "data/positions" if numerical_velocities else "data/velocities"
    with h5py.File(path, "r") as h5:
        if dataset not in h5:
            raise ValueError(f"missing {dataset} dataset")
        samples = np.asarray(h5[dataset][:n_frames], dtype=np.float64)
    if len(samples) != n_frames:
        raise ValueError(f"requested {n_frames} frames but loaded {len(samples)}")
    return samples, not numerical_velocities


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def compute_vdos(
    samples: np.ndarray,
    timestep_fs: float,
    *,
    samples_are_velocities: bool,
    pad_factor: int = 1,
    chunk_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return wavenumber and summed VDOS using the legacy Hann/VACF method.

    Columns are processed in chunks so large trajectories do not require a
    second trajectory-sized FFT work array. Autocorrelations are normalized
    separately for every Cartesian atom component before their spectra are
    summed, matching ``VDOS.py``.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[-1] != 3:
        raise ValueError(f"expected samples with shape (frames, atoms, 3), got {samples.shape}")
    if len(samples) < 3:
        raise ValueError("at least three frames are required for VDOS")
    if timestep_fs <= 0.0:
        raise ValueError("timestep must be positive")
    if pad_factor < 1:
        raise ValueError("pad factor must be positive")
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")

    n_frames = len(samples)
    timestep_s = timestep_fs * 1e-15
    n_fft = pad_factor * _next_power_of_two(n_frames)
    half = n_fft // 2
    window = signal.windows.hann(n_frames, sym=False)
    window /= np.mean(window)
    intensity = np.zeros(half, dtype=np.float64)
    flattened = samples.reshape(n_frames, -1)

    for start in range(0, flattened.shape[1], chunk_size):
        block = flattened[:, start : start + chunk_size]
        if not samples_are_velocities:
            block = np.gradient(block, timestep_s, axis=0)

        centered = block - np.mean(block, axis=0, keepdims=True)
        norms = np.sum(centered * centered, axis=0)
        valid = np.isfinite(norms) & (norms > 0.0)
        if not np.any(valid):
            continue

        centered = centered[:, valid]
        norms = norms[valid]
        acf = signal.fftconvolve(
            centered,
            centered[::-1],
            mode="full",
            axes=0,
        )[n_frames - 1 :] / norms
        transformed = np.fft.rfft(acf * window[:, None], n=n_fft, axis=0) / n_frames
        intensity += np.sum(np.abs(transformed[:half]) ** 2, axis=1)

    if not np.any(np.isfinite(intensity) & (intensity > 0.0)):
        raise ValueError("VDOS is identically zero")

    wavenumber = np.fft.fftfreq(n_fft, timestep_s * SPEED_OF_LIGHT_CM_S)[:half]
    return wavenumber, intensity


def integrate(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5))


def normalized_spectrum(
    wavenumber: np.ndarray,
    intensity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    energy_ev = np.asarray(wavenumber, dtype=float) * CM_INVERSE_TO_EV
    intensity = np.asarray(intensity, dtype=float)
    area = integrate(energy_ev, intensity)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("VDOS has non-positive area")
    return energy_ev, intensity / area


def vdos_similarity(
    reference: tuple[np.ndarray, np.ndarray],
    mlip: tuple[np.ndarray, np.ndarray],
    *,
    e_min: float | None = None,
    e_max: float | None = None,
) -> tuple[float, float]:
    """Return normalized-L1 VDOS error and similarity percentages."""
    ref_energy, ref_norm = normalized_spectrum(*reference)
    mlip_energy, mlip_norm = normalized_spectrum(*mlip)

    lower = max(float(ref_energy[0]), float(mlip_energy[0]))
    upper = min(float(ref_energy[-1]), float(mlip_energy[-1]))
    if e_min is not None:
        lower = max(lower, e_min)
    if e_max is not None:
        upper = min(upper, e_max)
    if lower >= upper:
        raise ValueError("reference and MLIP spectra have no requested energy overlap")

    mask = (ref_energy >= lower) & (ref_energy <= upper)
    energy = ref_energy[mask]
    if len(energy) < 2:
        raise ValueError("fewer than two reference points are in the energy overlap")
    ref_values = ref_norm[mask]
    mlip_values = np.interp(energy, mlip_energy, mlip_norm)
    numerator = integrate(energy, np.abs(ref_values - mlip_values))
    denominator = integrate(energy, ref_values) + integrate(energy, mlip_values)
    if denominator <= 0.0:
        raise ValueError("VDOS overlap has non-positive area")

    error_percent = float(np.clip(100.0 * numerator / denominator, 0.0, 100.0))
    return error_percent, 100.0 - error_percent


def save_spectrum(path: Path, spectrum: tuple[np.ndarray, np.ndarray]) -> None:
    energy_ev, normalized = normalized_spectrum(*spectrum)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack((spectrum[0], energy_ev, spectrum[1], normalized)),
        delimiter=",",
        header="wavenumber_cm-1,energy_eV,intensity_au,normalized_intensity_eV-1",
        comments="",
    )


def load_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] < 3:
        raise ValueError(f"invalid spectrum file: {path}")
    return values[:, 0], values[:, 2]


def spectrum_for_samples(
    path: Path,
    samples: np.ndarray,
    timestep_fs: float,
    *,
    samples_are_velocities: bool,
    pad_factor: int,
    chunk_size: int,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if path.is_file() and not overwrite:
        return load_spectrum(path)
    spectrum = compute_vdos(
        samples,
        timestep_fs,
        samples_are_velocities=samples_are_velocities,
        pad_factor=pad_factor,
        chunk_size=chunk_size,
    )
    save_spectrum(path, spectrum)
    return spectrum


def aggregate_and_save(pair_df: pd.DataFrame, results_dir: Path) -> None:
    pair_df = pair_df.sort_values(["model", "system"]).reset_index(drop=True)
    pair_df.to_csv(results_dir / PAIR_OUTPUT, index=False)

    system_model = (
        pair_df.groupby(["source", "model", "system", "system_type"], as_index=False)
        .agg(
            vdos_error_percent=("vdos_error_percent", "mean"),
            vdos_similarity_percent=("vdos_similarity_percent", "mean"),
            matched_frames=("matched_frames", "min"),
            timestep_fs=("timestep_fs", "first"),
        )
        .sort_values(["model", "system"])
    )
    system_model.to_csv(results_dir / SYSTEM_MODEL_OUTPUT, index=False)

    model_mean = (
        system_model.groupby(["source", "model"], as_index=False)
        .agg(
            vdos_error_percent=("vdos_error_percent", "mean"),
            vdos_similarity_percent=("vdos_similarity_percent", "mean"),
            n_systems=("system", "nunique"),
        )
        .sort_values("model")
    )
    model_mean.to_csv(results_dir / MODEL_OUTPUT, index=False)

    compatibility = model_mean.rename(
        columns={
            "model": "Calculator",
            "vdos_error_percent": "Mean VDOS Error [%]",
            "vdos_similarity_percent": "Mean Similarity Score [%]",
        }
    ).drop(columns=["source"])
    compatibility.to_csv(results_dir / SCORES_OUTPUT, index=False)

    by_type = system_model.pivot_table(
        index="model",
        columns="system_type",
        values="vdos_error_percent",
        aggfunc="mean",
    ).reindex(columns=SYSTEM_TYPES)
    by_type.index.name = "model"
    by_type.to_csv(results_dir / BY_TYPE_OUTPUT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute matched-length VDOS results for ASE/TorchSim trajectory sources."
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=SOURCES,
        dest="sources",
        help="trajectory source to process (repeatable; default: all four)",
    )
    parser.add_argument("--system", action="append", dest="systems", help="system filter (repeatable)")
    parser.add_argument("--model", action="append", dest="models", help="model filter (repeatable)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="trajectory data root")
    parser.add_argument("--metadata", type=Path, help="reference md_metadata.json")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="output root")
    parser.add_argument("--e-min", type=float, help="minimum energy in eV used for scoring")
    parser.add_argument("--e-max", type=float, help="maximum energy in eV used for scoring")
    parser.add_argument("--pad-factor", type=int, default=1, help="FFT zero-padding factor")
    parser.add_argument("--chunk-size", type=int, default=32, help="Cartesian signals per FFT chunk")
    parser.add_argument(
        "--numerical-mlip-velocities",
        action="store_true",
        help="differentiate MLIP positions instead of using stored velocities",
    )
    parser.add_argument("--overwrite", action="store_true", help="recompute existing spectra")
    parser.add_argument("--dry-run", action="store_true", help="show discovered inputs only")
    args = parser.parse_args()
    if args.e_min is not None and args.e_max is not None and args.e_min >= args.e_max:
        parser.error("--e-min must be smaller than --e-max")
    if args.pad_factor < 1:
        parser.error("--pad-factor must be positive")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    return args


def process_source(
    args: argparse.Namespace,
    source: str,
    *,
    data_dir: Path,
    reference_dir: Path,
    metadata: dict[str, dict],
) -> None:
    trajectory_dir = data_dir / source
    results_dir = args.results_dir.resolve() / source
    spectra_dir = results_dir / SPECTRA_DIR
    trajectories = discover_mlip_trajectories(trajectory_dir)

    selected_systems = set(args.systems) if args.systems else None
    selected_models = set(args.models) if args.models else None
    if selected_systems is not None:
        trajectories = {
            system: models
            for system, models in trajectories.items()
            if system in selected_systems
        }
    if selected_models is not None:
        trajectories = {
            system: {model: path for model, path in models.items() if model in selected_models}
            for system, models in trajectories.items()
        }
    trajectories = {system: models for system, models in trajectories.items() if models}

    model_names = sorted({model for models in trajectories.values() for model in models})
    print(f"\n=== Trajectory source: {trajectory_dir} ===")
    print(f"Results directory: {results_dir}")
    print(
        f"Found {sum(len(models) for models in trajectories.values())} trajectories "
        f"for {len(trajectories)} systems and {len(model_names)} models"
    )
    print(f"Models ({len(model_names)}): {', '.join(model_names)}")

    if args.dry_run:
        for system, models in sorted(trajectories.items()):
            ref_path = reference_dir / system / "traj.extxyz"
            status = "reference found" if ref_path.is_file() else "reference MISSING"
            print(f"{system}: {len(models)} MLIP trajectories ({status})")
        return
    if not trajectories:
        print(f"[SKIP] No matching trajectories found in {trajectory_dir}")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for system, models in sorted(trajectories.items()):
        print(f"\n=== System: {system} ===")
        ref_path = reference_dir / system / "traj.extxyz"
        if not ref_path.is_file():
            print(f"  [SKIP] Reference trajectory not found: {ref_path}")
            continue
        if system not in metadata:
            print(f"  [SKIP] No metadata entry for {system}")
            continue
        try:
            timestep_fs = float(metadata[system]["timestep"])
            reference_positions = load_reference_positions(ref_path)
        except Exception as exc:
            print(f"  [SKIP] Could not load reference trajectory: {exc}")
            continue

        n_ref = len(reference_positions)
        print(f"  Reference frames={n_ref}, timestep={timestep_fs:g} fs")
        ref_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def reference_spectrum(n_frames: int, model: str | None) -> tuple[np.ndarray, np.ndarray]:
            if n_frames == n_ref:
                output = spectra_dir / "reference" / f"{system}.csv"
            else:
                assert model is not None
                output = spectra_dir / "reference_matched" / model / f"{system}.csv"
            if n_frames in ref_cache:
                if not output.is_file():
                    save_spectrum(output, ref_cache[n_frames])
                return ref_cache[n_frames]
            spectrum = spectrum_for_samples(
                output,
                reference_positions[:n_frames],
                timestep_fs,
                samples_are_velocities=False,
                pad_factor=args.pad_factor,
                chunk_size=args.chunk_size,
                overwrite=args.overwrite,
            )
            ref_cache[n_frames] = spectrum
            return spectrum

        for model, mlip_path in sorted(models.items()):
            print(f"  Model: {model}")
            try:
                n_mlip = h5_frame_count(
                    mlip_path,
                    numerical_velocities=args.numerical_mlip_velocities,
                )
                n_matched = min(n_ref, n_mlip)
                if n_matched < 3:
                    raise ValueError("fewer than three common frames")
                print(
                    f"    matched frames={n_matched} | ref={n_matched}/{n_ref} | "
                    f"mlip={n_matched}/{n_mlip}"
                )
                ref_vdos = reference_spectrum(n_matched, model)
                mlip_samples, samples_are_velocities = load_mlip_samples(
                    mlip_path,
                    n_matched,
                    numerical_velocities=args.numerical_mlip_velocities,
                )
                mlip_output = spectra_dir / "mlip" / model / f"{system}.csv"
                mlip_vdos = spectrum_for_samples(
                    mlip_output,
                    mlip_samples,
                    timestep_fs,
                    samples_are_velocities=samples_are_velocities,
                    pad_factor=args.pad_factor,
                    chunk_size=args.chunk_size,
                    overwrite=args.overwrite,
                )
                error, similarity = vdos_similarity(
                    ref_vdos,
                    mlip_vdos,
                    e_min=args.e_min,
                    e_max=args.e_max,
                )
            except Exception as exc:
                print(f"    [SKIP] Could not compute/score VDOS: {exc}")
                continue

            print(f"    VDOS error={error:.6f} %, similarity={similarity:.6f} %")
            rows.append(
                {
                    "source": source,
                    "model": model,
                    "system": system,
                    "system_type": SYSTEM_TO_TYPE.get(system, "Other"),
                    "reference_frames": n_ref,
                    "mlip_frames": n_mlip,
                    "matched_frames": n_matched,
                    "timestep_fs": timestep_fs,
                    "vdos_error_percent": error,
                    "vdos_similarity_percent": similarity,
                    "reference_spectrum": str(
                        spectra_dir
                        / ("reference" if n_matched == n_ref else f"reference_matched/{model}")
                        / f"{system}.csv"
                    ),
                    "mlip_spectrum": str(mlip_output),
                }
            )

    if not rows:
        print(f"[WARN] No valid VDOS scores for {source}")
        return
    aggregate_and_save(pd.DataFrame(rows), results_dir)
    print(f"\nSaved {len(rows)} pair scores to {results_dir / PAIR_OUTPUT}")
    print(f"Saved model means to {results_dir / MODEL_OUTPUT}")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    reference_dir = data_dir / "ref-trajs"
    metadata_path = args.metadata.resolve() if args.metadata else reference_dir / "md_metadata.json"
    metadata = load_metadata(metadata_path)
    for source in dict.fromkeys(args.sources or SOURCES):
        process_source(
            args,
            source,
            data_dir=data_dir,
            reference_dir=reference_dir,
            metadata=metadata,
        )


if __name__ == "__main__":
    main()
