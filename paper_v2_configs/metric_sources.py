"""Shared source names for independently computed benchmark metrics."""
from pathlib import Path

SOURCE_DATASETS = {
    "mlip-trajs-ase": ("ase", "md_eager"),
    "mlip-trajs-torchsim-eager": ("torchsim", "md_eager"),
    "mlip-trajs-ase-accelerated": ("ase", "md_accelerated"),
    "mlip-trajs-torchsim-accelerated": ("torchsim", "md_accelerated"),
}
SOURCES = tuple(SOURCE_DATASETS)


def pressure_input_dir(root: Path, source: str) -> Path:
    """Prefer per-source raw data, with support for the evaluator's older layout."""
    canonical = root / source
    if any(canonical.glob("*_same-simulation-length_pressure_per_frame.csv")):
        return canonical
    backend, mode = SOURCE_DATASETS[source]
    current = root / backend / mode
    if any(current.glob("*_same-simulation-length_pressure_per_frame.csv")):
        return current
    # Early TorchSim evaluators omitted the backend directory.
    legacy = root / mode
    if backend == 'torchsim' and any(legacy.glob("*_same-simulation-length_pressure_per_frame.csv")):
        return legacy
    return current
