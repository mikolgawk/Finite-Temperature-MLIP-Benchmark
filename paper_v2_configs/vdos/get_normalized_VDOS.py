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
"""Compute matched-length, mass-weighted Welch VDOS and spectrum errors.

Both reference and MLIP velocities are central differences of minimum-image
unwrapped positions; the two endpoints are discarded. This prevents PBC jumps
and avoids comparing differentiated reference positions to stored MLIP
velocities. Only fixed-cell, fixed-PBC trajectories (NVT) are supported.

The default CLI estimator averages one-sided Hann-windowed periodograms of
overlapping velocity segments (Welch). Component powers are multiplied by
their atomic masses, summed, and finally normalized to unit area. A requested
physical window of 2 ps is shortened for short trajectories to provide at
least four 50%-overlapping segments. This reduces sampling variance without
squaring an already transformed autocorrelation. Actual segment duration,
count and frequency resolution are recorded; zero padding is not resolution.
The original corrected full-trajectory unweighted periodogram remains an
explicit diagnostic option and the backwards-compatible numerical API.

Each source has a separate results directory. Cache sidecars validate the
analyzed samples, input identity, method, timestep and FFT settings, as well
as the saved CSV content. Missing/failed MD and failed calculations retain
the benchmark's 100% error and 0% similarity policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metric_sources import cohort_model_path, model_trajectory_source, select_model_names
from md_success import (
    discover_md_trajectories,
    registered_md_failure_reason,
    torchsim_md_succeeded,
)
from system_filters import (
    DEFAULT_EXCLUDED_SYSTEM_TYPES,
    add_molecular_crystal_option,
    excluded_system_types,
)

import h5py
import numpy as np
import pandas as pd
from ase.io import iread
from ase.geometry import find_mic
from ase.data import atomic_masses
from scipy import signal


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

SOURCES = (
    "mlip-trajs-ase",
    "mlip-trajs-torchsim-eager",
    "mlip-trajs-ase-accelerated",
    "mlip-trajs-torchsim-accelerated",
)

SPEED_OF_LIGHT_CM_S = 2.99792458e10
CM_INVERSE_TO_EV = 1.2398419843320026e-4
METHOD_VERSION = "mass-welch-hann-central-mic-v2"
VELOCITY_HANDLING = "minimum-image-unwrapped-positions-central-difference-drop-endpoints"
DEFAULT_WINDOW_FS = 2000.0
DEFAULT_MIN_SEGMENTS = 4
DEFAULT_OVERLAP = 0.5
_CELL_RTOL = 2e-6
_CELL_ATOL = 2e-6
_TRAJECTORY_PROVENANCE: dict[str, dict] = {}
_TRAJECTORY_ATOMIC_NUMBERS: dict[str, np.ndarray] = {}

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
    """Discover ASE/TorchSim jobs, retaining timing-only failed runs."""
    return discover_md_trajectories(base_dir)


def _array_digest(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(values.dtype.str.encode())
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _record_geometry_provenance(path: Path, cells: np.ndarray, pbc: np.ndarray) -> None:
    _TRAJECTORY_PROVENANCE[str(path.resolve())] = {
        "cell_sha256": _array_digest(cells),
        "pbc_sha256": _array_digest(pbc),
    }


def trajectory_identity(path: Path) -> dict:
    """Input identity plus exact cell/PBC fingerprints of the loaded segment.

    The exact analyzed coordinate values are additionally hashed by the
    spectrum cache. Unrelated, unanalyzed HDF5 datasets need not be hashed.
    """
    path = path.resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        **_TRAJECTORY_PROVENANCE.get(str(path), {}),
    }


def unwrap_positions(
    positions: np.ndarray,
    cells: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    """Unwrap a fixed-cell trajectory using successive minimum-image steps.

    As for all nearest-image trajectory reconstruction, physical displacement
    between recorded frames must be smaller than the relevant half-box scale.
    Triclinic cells use ASE's general minimum-image algorithm, not fractional
    rounding (which is incorrect for sufficiently skew cells).
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3 or len(positions) < 1:
        raise ValueError("positions must have shape (frames, atoms, 3)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions contain nonfinite values")
    cells = np.asarray(cells, dtype=np.float64)
    if cells.shape == (3, 3):
        cell = cells
    elif cells.shape in ((1, 3, 3), (len(positions), 3, 3)):
        cell = cells[0]
        if not np.allclose(cells, cell, rtol=_CELL_RTOL, atol=_CELL_ATOL):
            raise ValueError("changing cells are unsupported: this analysis requires fixed-cell NVT")
    else:
        raise ValueError(f"cell shape {cells.shape} does not match positions")
    pbc = np.asarray(pbc, dtype=bool)
    if pbc.shape == (3,):
        periodic = pbc
    elif pbc.shape in ((1, 3), (len(positions), 3)):
        periodic = pbc[0]
        if not np.all(pbc == periodic):
            raise ValueError("changing PBC flags are unsupported")
    else:
        raise ValueError(f"PBC shape {pbc.shape} does not match positions")
    if not np.any(periodic):
        return positions.copy()
    if not np.all(np.isfinite(cell)) or abs(np.linalg.det(cell)) < 1e-12:
        raise ValueError("periodic trajectory has missing or singular cell")

    unwrapped = np.empty_like(positions)
    unwrapped[0] = positions[0]
    # A diagonal cell admits an inexpensive exact component-wise MIC. The
    # general path is chunked to bound ASE's temporary image arrays.
    diagonal = np.allclose(cell, np.diag(np.diag(cell)), rtol=0.0, atol=1e-12)
    block_frames = 128
    for start in range(1, len(positions), block_frames):
        stop = min(start + block_frames, len(positions))
        steps = positions[start:stop] - positions[start - 1 : stop - 1]
        if diagonal:
            for axis in np.flatnonzero(periodic):
                length = cell[axis, axis]
                steps[..., axis] -= np.rint(steps[..., axis] / length) * length
        else:
            steps, _ = find_mic(steps.reshape(-1, 3), cell, pbc=periodic)
            steps = steps.reshape(stop - start, positions.shape[1], 3)
        unwrapped[start:stop] = unwrapped[start - 1] + np.cumsum(steps, axis=0)
    return unwrapped


def load_reference_positions(path: Path) -> np.ndarray:
    positions, cells, pbcs = [], [], []
    numbers = None
    for atoms in iread(path, index=":"):
        current_numbers = atoms.get_atomic_numbers()
        if numbers is None:
            numbers = current_numbers.copy()
        elif not np.array_equal(current_numbers, numbers):
            raise ValueError("reference atomic identities/order change between frames")
        positions.append(np.asarray(atoms.positions, dtype=np.float64))
        cells.append(np.asarray(atoms.cell.array, dtype=np.float64))
        pbcs.append(np.asarray(atoms.pbc, dtype=bool))
    if not positions:
        raise ValueError("trajectory contains no frames")
    cells_array, pbc_array = np.stack(cells), np.stack(pbcs)
    _record_geometry_provenance(path, cells_array, pbc_array)
    _TRAJECTORY_ATOMIC_NUMBERS[str(path.resolve())] = numbers
    return unwrap_positions(np.stack(positions), cells_array, pbc_array)


def load_reference_atomic_numbers(path: Path) -> np.ndarray:
    """Return atom identities in coordinate order (cached while loading positions)."""
    key = str(path.resolve())
    if key not in _TRAJECTORY_ATOMIC_NUMBERS:
        try:
            atoms = next(iread(path, index="0"))
        except StopIteration as exc:
            raise ValueError("reference trajectory contains no frames") from exc
        return atoms.get_atomic_numbers()
    return _TRAJECTORY_ATOMIC_NUMBERS[key].copy()


def load_mlip_atomic_numbers(path: Path, n_atoms: int | None = None) -> np.ndarray:
    """Read and validate the recorded HDF5 atom identities, preserving order."""
    with h5py.File(path, "r") as handle:
        if "data/atomic_numbers" not in handle:
            raise ValueError("missing data/atomic_numbers dataset; atomic masses cannot be inferred safely")
        recorded = np.asarray(handle["data/atomic_numbers"])
    if recorded.ndim == 2 and len(recorded):
        numbers = recorded[0]
        if not np.all(recorded == numbers):
            raise ValueError("MLIP atomic identities/order change between recorded frames")
    elif recorded.ndim == 1:
        numbers = recorded
    else:
        raise ValueError(f"invalid atomic_numbers shape: {recorded.shape}")
    standard_atomic_masses(numbers)  # Validate integer chemical identities.
    if n_atoms is not None and len(numbers) != n_atoms:
        raise ValueError("atomic number count does not match coordinate atom count")
    return numbers.astype(np.int64)


def standard_atomic_masses(numbers: np.ndarray) -> np.ndarray:
    """Use the same ASE standard masses (amu) for reference and MLIP spectra.

    This explicitly defines a natural-isotope VDOS convention, avoiding mass
    unit differences in trajectory writers. Isotope-specific spectra would
    require passing a shared explicit mass array to the numerical API.
    """
    numbers = np.asarray(numbers)
    if (numbers.ndim != 1 or not len(numbers) or not np.all(np.isfinite(numbers))
            or not np.all(numbers == numbers.astype(np.int64))
            or np.any(numbers < 1) or np.any(numbers >= len(atomic_masses))):
        raise ValueError("atomic numbers must be a nonempty vector of valid chemical identities")
    masses = np.asarray(atomic_masses[numbers.astype(np.int64)], dtype=np.float64)
    if not np.all(np.isfinite(masses) & (masses > 0)):
        raise ValueError("atomic masses must be finite and positive")
    return masses


def h5_frame_count(path: Path, *, numerical_velocities: bool = True) -> int:
    dataset = "data/positions"
    with h5py.File(path, "r") as h5:
        if dataset not in h5:
            raise ValueError(f"missing {dataset} dataset")
        return int(h5[dataset].shape[0])


def load_mlip_samples(
    path: Path,
    n_frames: int,
    *,
    numerical_velocities: bool = True,
) -> tuple[np.ndarray, bool]:
    dataset = "data/positions"
    with h5py.File(path, "r") as h5:
        for required in (dataset, "data/cell", "data/pbc"):
            if required not in h5:
                raise ValueError(f"missing {required} dataset")
        samples = np.asarray(h5[dataset][:n_frames], dtype=np.float64)
        cell_dataset = h5["data/cell"]
        cells = np.asarray(cell_dataset[:] if cell_dataset.ndim == 2 else cell_dataset[:n_frames])
        pbc_dataset = h5["data/pbc"]
        pbcs = np.asarray(pbc_dataset[:] if pbc_dataset.ndim == 1 else pbc_dataset[:n_frames])
        if "steps/positions" in h5:
            steps = np.asarray(h5["steps/positions"][:n_frames])
            if len(steps) != len(samples) or not np.all(np.diff(steps) == 1):
                raise ValueError("positions must be recorded at every MD timestep")
    if len(samples) != n_frames:
        raise ValueError(f"requested {n_frames} frames but loaded {len(samples)}")
    _record_geometry_provenance(path, cells, pbcs)
    return unwrap_positions(samples, cells, pbcs), False


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def spectral_settings(
    n_velocity_frames: int,
    timestep_fs: float,
    *,
    estimator: str = "welch",
    window_fs: float = DEFAULT_WINDOW_FS,
    min_segments: int = DEFAULT_MIN_SEGMENTS,
    overlap: float = DEFAULT_OVERLAP,
    pad_factor: int = 1,
) -> dict:
    """Resolve the estimator's physical window and reproducible FFT settings.

    Both trajectories in a pair have the same matched sample count/timestep,
    so shortening a requested window cannot favour either side. For a Hann
    window, ``resolution_cm-1`` is the native 1/window frequency spacing;
    the noise-equivalent bandwidth is 1.5 times larger. Zero padding only
    changes the displayed FFT bin spacing.
    """
    if estimator not in {"welch", "periodogram"}:
        raise ValueError("estimator must be 'welch' or 'periodogram'")
    if n_velocity_frames < 3:
        raise ValueError("at least three velocity frames are required")
    if not np.isfinite(timestep_fs) or timestep_fs <= 0:
        raise ValueError("timestep must be positive")
    if not np.isfinite(window_fs) or window_fs <= 0:
        raise ValueError("window duration must be positive")
    if min_segments < 2 or int(min_segments) != min_segments:
        raise ValueError("minimum segment count must be an integer of at least two")
    if not np.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap must be between zero (inclusive) and one (exclusive)")
    if pad_factor < 1 or int(pad_factor) != pad_factor:
        raise ValueError("pad factor must be a positive integer")
    requested_nperseg = max(3, int(round(window_fs / timestep_fs)))
    if estimator == "periodogram":
        nperseg = n_velocity_frames
        noverlap = 0
        n_segments = 1
    else:
        # L + (K-1)*(L-floor(overlap*L)) <= N is monotone in L.
        # Binary search gives the largest window supporting K segments.
        low, high, maximum = 3, n_velocity_frames, 3
        while low <= high:
            candidate = (low + high) // 2
            candidate_step = candidate - int(np.floor(overlap * candidate))
            if candidate + (min_segments - 1) * candidate_step <= n_velocity_frames:
                maximum = candidate
                low = candidate + 1
            else:
                high = candidate - 1
        nperseg = min(requested_nperseg, maximum)
        noverlap = int(np.floor(overlap * nperseg))
        n_segments = 1 + (n_velocity_frames - nperseg) // (nperseg - noverlap)
    n_fft = int(pad_factor) * _next_power_of_two(nperseg)
    segment_duration_fs = nperseg * float(timestep_fs)
    resolution = 1.0 / (segment_duration_fs * 1e-15 * SPEED_OF_LIGHT_CM_S)
    return {
        "estimator": estimator,
        "requested_window_fs": float(window_fs),
        "requested_min_segments": int(min_segments),
        "requested_overlap_fraction": float(overlap),
        "requested_nperseg": requested_nperseg,
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "n_segments": int(n_segments),
        "minimum_segments_satisfied": bool(n_segments >= min_segments),
        "segment_duration_fs": segment_duration_fs,
        "window_shortened": bool(estimator == "welch" and nperseg < requested_nperseg),
        "resolution_cm-1": resolution,
        "hann_noise_equivalent_bandwidth_cm-1": 1.5 * resolution,
        "n_fft": n_fft,
        "bin_spacing_cm-1": 1.0 / (n_fft * timestep_fs * 1e-15 * SPEED_OF_LIGHT_CM_S),
        "averaging": "mean" if estimator == "welch" else "none",
    }


def _component_weights(n_atoms: int, weighting: str, masses: np.ndarray | None) -> np.ndarray:
    if weighting not in {"velocity", "mass"}:
        raise ValueError("weighting must be 'mass' or 'velocity'")
    if masses is not None:
        masses = np.asarray(masses, dtype=np.float64)
        if masses.shape != (n_atoms,) or not np.all(np.isfinite(masses) & (masses > 0)):
            raise ValueError("masses must contain one finite positive value per atom")
    if weighting == "mass":
        if masses is None:
            raise ValueError("mass-weighted VDOS requires explicit atomic masses")
        return np.repeat(masses, 3)
    return np.ones(n_atoms * 3)


def compute_vdos(
    samples: np.ndarray,
    timestep_fs: float,
    *,
    samples_are_velocities: bool,
    pad_factor: int = 1,
    chunk_size: int = 32,
    estimator: str = "periodogram",
    weighting: str = "velocity",
    masses: np.ndarray | None = None,
    window_fs: float = DEFAULT_WINDOW_FS,
    min_segments: int = DEFAULT_MIN_SEGMENTS,
    overlap: float = DEFAULT_OVERLAP,
) -> tuple[np.ndarray, np.ndarray]:
    """Return wavenumber and the summed one-sided velocity-power density.

    Position inputs must already be PBC-unwrapped. Central differences at
    interior frames provide a consistent derivative transfer function for
    reference and MLIP inputs. Velocity inputs support numerical validation;
    the analysis pipeline always uses positions for both trajectories. Legacy
    API defaults retain the unweighted periodogram; the CLI explicitly uses
    Welch with mass weighting. Mass weights apply to powers before summing,
    not to separately normalized atom/component spectra.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[-1] != 3:
        raise ValueError(f"expected samples with shape (frames, atoms, 3), got {samples.shape}")
    if len(samples) < 3:
        raise ValueError("at least three frames are required for VDOS")
    if not np.isfinite(timestep_fs) or timestep_fs <= 0.0:
        raise ValueError("timestep must be positive")
    if pad_factor < 1:
        raise ValueError("pad factor must be positive")
    if chunk_size < 1:
        raise ValueError("chunk size must be positive")

    if not np.all(np.isfinite(samples)):
        raise ValueError("samples contain nonfinite values")
    n_frames = len(samples) if samples_are_velocities else len(samples) - 2
    if n_frames < 3:
        raise ValueError("at least five position frames or three velocity frames are required")
    settings = spectral_settings(
        n_frames, timestep_fs, estimator=estimator, window_fs=window_fs,
        min_segments=min_segments, overlap=overlap, pad_factor=pad_factor,
    )
    timestep_s = timestep_fs * 1e-15
    n_fft = settings["n_fft"]
    intensity = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    flattened = samples.reshape(len(samples), -1)
    weights = _component_weights(samples.shape[1], weighting, masses)

    for start in range(0, flattened.shape[1], chunk_size):
        block = flattened[:, start : start + chunk_size]
        if not samples_are_velocities:
            # Angstrom/fs is consistent on both sides. The absolute scale
            # cancels in unit-area normalization of the summed spectrum.
            block = (block[2:] - block[:-2]) / (2.0 * timestep_fs)
        options = dict(fs=1.0 / timestep_s, window="hann", detrend="constant",
                       nfft=n_fft, return_onesided=True, scaling="density", axis=0)
        if estimator == "welch":
            frequencies, component_power = signal.welch(
                block, nperseg=settings["nperseg"], noverlap=settings["noverlap"],
                average="mean", **options,
            )
        else:
            frequencies, component_power = signal.periodogram(block, **options)
        intensity += np.sum(component_power * weights[None, start : start + chunk_size], axis=1)

    if not np.any(np.isfinite(intensity) & (intensity > 0.0)):
        raise ValueError("VDOS is identically zero")

    wavenumber = frequencies / SPEED_OF_LIGHT_CM_S
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
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savetxt(
            temporary_path,
            np.column_stack((spectrum[0], energy_ev, spectrum[1], normalized)),
            delimiter=",",
            header="wavenumber_cm-1,energy_eV,intensity_au,normalized_intensity_eV-1",
            comments="",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != 4 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid spectrum file: {path}")
    if not np.all(np.diff(values[:, 0]) > 0) or np.any(values[:, 2] < 0):
        raise ValueError(f"invalid frequency/power values in spectrum file: {path}")
    return values[:, 0], values[:, 2]


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary_path = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def spectrum_for_samples(
    path: Path,
    samples: np.ndarray,
    timestep_fs: float,
    *,
    samples_are_velocities: bool,
    pad_factor: int,
    chunk_size: int,
    overwrite: bool,
    input_identity: dict | None = None,
    estimator: str = "periodogram",
    weighting: str = "velocity",
    masses: np.ndarray | None = None,
    window_fs: float = DEFAULT_WINDOW_FS,
    min_segments: int = DEFAULT_MIN_SEGMENTS,
    overlap: float = DEFAULT_OVERLAP,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[-1] != 3:
        raise ValueError("samples must have shape (frames, atoms, 3)")
    _component_weights(samples.shape[1], weighting, masses)
    n_velocity_frames = len(samples) if samples_are_velocities else len(samples) - 2
    settings = spectral_settings(
        n_velocity_frames, timestep_fs, estimator=estimator, window_fs=window_fs,
        min_segments=min_segments, overlap=overlap, pad_factor=pad_factor,
    )
    signature = {
        "method_version": METHOD_VERSION,
        "estimator": f"one-sided-scipy-{estimator}-density",
        "window": "periodic-Hann",
        "detrend": "per-segment-per-component-constant",
        "weighting": weighting,
        "weighting_convention": "mass-times-component-power" if weighting == "mass" else "component-velocity-power",
        "normalization": "sum-weighted-component-powers-then-normalize-total-area",
        "atomic_masses_amu": None if masses is None else np.asarray(masses, dtype=float).tolist(),
        "masses_sha256": None if masses is None else _array_digest(np.asarray(masses, dtype=np.float64)),
        "spectral_settings": settings,
        "velocity_handling": "provided-velocities" if samples_are_velocities else VELOCITY_HANDLING,
        "timestep_fs": float(timestep_fs),
        "input_frames": len(samples),
        "velocity_frames": len(samples) if samples_are_velocities else len(samples) - 2,
        "sample_shape": list(samples.shape),
        "sample_sha256": _array_digest(samples),
        "pad_factor": int(pad_factor),
        "chunk_size": int(chunk_size),
        "input_identity": input_identity,
    }
    sidecar_path = path.with_suffix(path.suffix + ".json")
    if path.is_file() and sidecar_path.is_file() and not overwrite:
        try:
            with sidecar_path.open() as handle:
                cached = json.load(handle)
            if isinstance(cached, dict) and cached.get("signature") == signature and cached.get("spectrum_sha256") == _file_digest(path):
                return load_spectrum(path)
        except (OSError, ValueError, TypeError):
            # Interrupted, incompatible or corrupt caches are regenerated.
            pass
    spectrum = compute_vdos(
        samples,
        timestep_fs,
        samples_are_velocities=samples_are_velocities,
        pad_factor=pad_factor,
        chunk_size=chunk_size,
        estimator=estimator,
        weighting=weighting,
        masses=masses,
        window_fs=window_fs,
        min_segments=min_segments,
        overlap=overlap,
    )
    save_spectrum(path, spectrum)
    # CSV then sidecar are individually atomic. If interrupted between them,
    # their hash mismatch causes regeneration on the next invocation.
    _atomic_json(sidecar_path, {
        "signature": signature,
        "spectrum_sha256": _file_digest(path),
    })
    return spectrum


def aggregate_and_save(pair_df: pd.DataFrame, results_dir: Path, excluded_system_types: set[str] | None = frozenset(DEFAULT_EXCLUDED_SYSTEM_TYPES)) -> None:
    pair_df = pair_df.copy()
    pair_df["trajectory_source"] = [model_trajectory_source(source, model)
                                      for source, model in zip(pair_df["source"], pair_df["model"])]
    if excluded_system_types:
        excluded = {value.strip().lower() for value in excluded_system_types}
        pair_df = pair_df.loc[
            ~pair_df["system_type"].fillna("").str.lower().isin(excluded)
        ].copy()
    pair_df = pair_df.sort_values(["model", "system"]).reset_index(drop=True)
    pair_df.to_csv(results_dir / PAIR_OUTPUT, index=False)

    system_model = (
        pair_df.groupby(["source", "trajectory_source", "model", "system", "system_type"], as_index=False)
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
        system_model.groupby(["source", "trajectory_source", "model"], as_index=False)
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
    )
    included_types = [
        value for value in SYSTEM_TYPES if value.lower() not in (excluded_system_types or set())
    ]
    by_type = by_type.reindex(columns=included_types)
    by_type.index.name = "model"
    by_type.to_csv(results_dir / BY_TYPE_OUTPUT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute corrected matched-length velocity-power VDOS results for ASE/TorchSim sources."
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
    parser.add_argument("--exclude-system-type", action="append", dest="excluded_system_types", help="exclude this system type from computation (repeatable)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="trajectory data root")
    parser.add_argument("--metadata", type=Path, help="reference md_metadata.json")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="output root")
    parser.add_argument("--e-min", type=float, help="minimum energy in eV used for scoring")
    parser.add_argument("--e-max", type=float, help="maximum energy in eV used for scoring")
    parser.add_argument("--estimator", choices=("welch", "periodogram"), default="welch", help="spectral estimator (default: Hann Welch averaging)")
    parser.add_argument("--weighting", choices=("mass", "velocity"), default="mass", help="mass-weighted VDOS or unweighted velocity-power spectrum")
    parser.add_argument("--window-fs", type=float, default=DEFAULT_WINDOW_FS, help="requested Welch window duration in fs (default: 2000; shortened to retain the minimum segment count)")
    parser.add_argument("--min-segments", type=int, default=DEFAULT_MIN_SEGMENTS, help="minimum Welch segment count when trajectory length permits (default: 4)")
    parser.add_argument("--pad-factor", type=int, default=4, help="FFT zero-padding factor (default: 4; interpolates the frequency grid without increasing resolution)")
    parser.add_argument("--chunk-size", type=int, default=32, help="Cartesian signals per FFT chunk")
    parser.add_argument(
        "--numerical-mlip-velocities",
        action="store_true",
        help="compatibility flag: both reference and MLIP velocities are always derived from unwrapped positions",
    )
    parser.add_argument("--overwrite", action="store_true", help="recompute existing spectra")
    parser.add_argument("--dry-run", action="store_true", help="show discovered inputs only")
    add_molecular_crystal_option(parser)
    args = parser.parse_args()
    args.excluded_system_types = excluded_system_types(args)
    if args.e_min is not None and args.e_max is not None and args.e_min >= args.e_max:
        parser.error("--e-min must be smaller than --e-max")
    if args.pad_factor < 1:
        parser.error("--pad-factor must be positive")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if not np.isfinite(args.window_fs) or args.window_fs <= 0:
        parser.error("--window-fs must be finite and positive")
    if args.min_segments < 2:
        parser.error("--min-segments must be at least two")
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
    discovered_trajectories = discover_mlip_trajectories(trajectory_dir)
    all_discovered_model_names = {
        model for models in discovered_trajectories.values() for model in models
    }
    trajectories = {
        system: {
            model: path for model, path in models.items()
            if torchsim_md_succeeded(path)
        }
        for system, models in discovered_trajectories.items()
    }
    selected_systems = set(args.systems) if args.systems else None
    selected_models = select_model_names(all_discovered_model_names, set(args.models) if args.models else None)
    excluded_system_types = {
        value.strip().lower() for value in args.excluded_system_types or ()
    }
    excluded_systems = {
        system
        for system_type, systems in SYSTEM_TYPES.items()
        if system_type.lower() in excluded_system_types
        for system in systems
    }
    trajectories = {
        system: models for system, models in trajectories.items()
        if system not in excluded_systems
    }
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

    # Retain failed jobs in the expected model/system coverage.
    model_names = sorted(selected_models or all_discovered_model_names)
    expected_systems = (
        selected_systems - excluded_systems
        if selected_systems is not None
        else {
            path.parent.name
            for path in reference_dir.glob("*/traj.extxyz")
            if path.parent.name not in excluded_systems
        }
    )
    print(f"\n=== Trajectory source: {trajectory_dir} ===")
    print(f"Results directory: {results_dir}")
    print(
        f"Found {sum(len(models) for models in trajectories.values())} trajectories "
        f"for {len(expected_systems)} expected systems and {len(model_names)} models"
    )
    print(f"Models ({len(model_names)}): {', '.join(model_names)}")

    if args.dry_run:
        for system in sorted(expected_systems):
            models = trajectories.get(system, {})
            ref_path = reference_dir / system / "traj.extxyz"
            status = "reference found" if ref_path.is_file() else "reference MISSING"
            print(f"{system}: {len(models)} MLIP trajectories ({status})")
        return
    if not model_names:
        print(f"[SKIP] No matching trajectories found in {trajectory_dir}")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    estimator_options = {
        "estimator": getattr(args, "estimator", "welch"),
        "weighting": getattr(args, "weighting", "mass"),
        "window_fs": getattr(args, "window_fs", DEFAULT_WINDOW_FS),
        "min_segments": getattr(args, "min_segments", DEFAULT_MIN_SEGMENTS),
        "overlap": DEFAULT_OVERLAP,
    }

    def penalty_row(
        model: str,
        system: str,
        reason: str,
        *,
        timestep_fs: float = float("nan"),
        reference_frames: int = 0,
        mlip_frames: int = 0,
    ) -> dict:
        """Build an auditable 100% error row for unavailable VDOS data."""
        return {
            "source": source,
            "model": model,
            "system": system,
            "system_type": SYSTEM_TO_TYPE.get(system, "Other"),
            "reference_frames": reference_frames,
            "mlip_frames": mlip_frames,
            "matched_frames": 0,
            "timestep_fs": timestep_fs,
            "vdos_error_percent": 100.0,
            "vdos_similarity_percent": 0.0,
            "reference_spectrum": "",
            "mlip_spectrum": "",
            "failure_reason": reason,
            "estimator": estimator_options["estimator"],
            "weighting": estimator_options["weighting"],
            "requested_window_fs": estimator_options["window_fs"],
            "requested_min_segments": estimator_options["min_segments"],
        }

    for system in sorted(expected_systems):
        models = trajectories.get(system, {})
        eligible_models = model_names
        if not eligible_models:
            continue
        print(f"\n=== System: {system} ===")
        ref_path = reference_dir / system / "traj.extxyz"
        if not ref_path.is_file():
            reason = f"reference trajectory not found: {ref_path}"
            print(f"  [PENALTY] {reason}; assigning 100% error")
            rows.extend(penalty_row(model, system, reason) for model in eligible_models)
            continue
        if system not in metadata:
            reason = "reference metadata entry missing"
            print(f"  [PENALTY] {reason}; assigning 100% error")
            rows.extend(penalty_row(model, system, reason) for model in eligible_models)
            continue
        try:
            timestep_fs = float(metadata[system]["timestep"])
            reference_positions = load_reference_positions(ref_path)
            reference_numbers = load_reference_atomic_numbers(ref_path)
            masses = standard_atomic_masses(reference_numbers)
        except Exception as exc:
            reason = f"could not load reference trajectory: {exc}"
            print(f"  [PENALTY] {reason}; assigning 100% error")
            rows.extend(penalty_row(model, system, reason) for model in eligible_models)
            continue

        n_ref = len(reference_positions)
        print(f"  Reference frames={n_ref}, timestep={timestep_fs:g} fs")
        def reference_spectrum(n_frames: int, model: str | None) -> tuple[np.ndarray, np.ndarray]:
            if n_frames == n_ref:
                output = spectra_dir / "reference" / f"{system}.csv"
            else:
                assert model is not None
                output = spectra_dir / "reference_matched" / model / f"{system}.csv"
            return spectrum_for_samples(
                output,
                reference_positions[:n_frames],
                timestep_fs,
                samples_are_velocities=False,
                pad_factor=args.pad_factor,
                chunk_size=args.chunk_size,
                overwrite=args.overwrite,
                input_identity=trajectory_identity(ref_path),
                masses=masses,
                **estimator_options,
            )

        for model in eligible_models:
            print(f"  Model: {model}")
            mlip_path = models.get(model)
            if mlip_path is None:
                reason = (
                    registered_md_failure_reason(discovered_trajectories.get(system, {}).get(
                        model, cohort_model_path(trajectory_dir / system / f"nvt_{model}.h5", model)))
                ) or "MLIP trajectory missing or incomplete"
                print(f"    [PENALTY] {reason}; assigning 100% VDOS error")
                rows.append(
                    penalty_row(
                        model,
                        system,
                        reason,
                        timestep_fs=timestep_fs,
                        reference_frames=n_ref,
                    )
                )
                continue
            n_mlip = 0
            try:
                n_mlip = h5_frame_count(
                    mlip_path,
                )
                n_matched = min(n_ref, n_mlip)
                if n_matched < 5:
                    raise ValueError("fewer than five common position frames")
                print(
                    f"    matched frames={n_matched} | ref={n_matched}/{n_ref} | "
                    f"mlip={n_matched}/{n_mlip}"
                )
                mlip_samples, samples_are_velocities = load_mlip_samples(
                    mlip_path,
                    n_matched,
                )
                if mlip_samples.shape[1:] != reference_positions.shape[1:]:
                    raise ValueError("reference and MLIP trajectories have different atom counts")
                mlip_numbers = load_mlip_atomic_numbers(mlip_path, n_atoms=mlip_samples.shape[1])
                if not np.array_equal(mlip_numbers, reference_numbers):
                    raise ValueError("reference and MLIP atomic identities/order differ")
                ref_vdos = reference_spectrum(n_matched, model)
                mlip_output = spectra_dir / "mlip" / model / f"{system}.csv"
                mlip_vdos = spectrum_for_samples(
                    mlip_output,
                    mlip_samples,
                    timestep_fs,
                    samples_are_velocities=samples_are_velocities,
                    pad_factor=args.pad_factor,
                    chunk_size=args.chunk_size,
                    overwrite=args.overwrite,
                    input_identity=trajectory_identity(mlip_path),
                    masses=masses,
                    **estimator_options,
                )
                error, similarity = vdos_similarity(
                    ref_vdos,
                    mlip_vdos,
                    e_min=args.e_min,
                    e_max=args.e_max,
                )
            except Exception as exc:
                reason = f"could not compute/score VDOS: {exc}"
                print(f"    [PENALTY] {reason}; assigning 100% VDOS error")
                rows.append(
                    penalty_row(
                        model,
                        system,
                        reason,
                        timestep_fs=timestep_fs,
                        reference_frames=n_ref,
                        mlip_frames=n_mlip,
                    )
                )
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
                    "failure_reason": "",
                    "estimator": estimator_options["estimator"],
                    "weighting": estimator_options["weighting"],
                    **spectral_settings(
                        n_matched - 2, timestep_fs, estimator=estimator_options["estimator"],
                        window_fs=estimator_options["window_fs"],
                        min_segments=estimator_options["min_segments"],
                        overlap=estimator_options["overlap"], pad_factor=args.pad_factor,
                    ),
                }
            )

    if not rows:
        print(f"[WARN] No valid VDOS scores for {source}")
        return
    aggregate_and_save(pd.DataFrame(rows), results_dir, excluded_system_types)
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
