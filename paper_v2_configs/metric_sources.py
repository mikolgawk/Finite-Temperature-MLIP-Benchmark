"""Select the physical runs used in each paper benchmark cohort."""
from pathlib import Path

from model_display_names import normalize_display_key

SOURCE_DATASETS = {
    "mlip-trajs-ase": ("ase", "md_eager"),
    "mlip-trajs-torchsim-eager": ("torchsim", "md_eager"),
    "mlip-trajs-ase-accelerated": ("ase", "md_accelerated"),
    "mlip-trajs-torchsim-accelerated": ("torchsim", "md_accelerated"),
}
SOURCES = tuple(SOURCE_DATASETS)

TORCHSIM_EAGER = "mlip-trajs-torchsim-eager"
TORCHSIM_ACCELERATED = "mlip-trajs-torchsim-accelerated"
# Use eager runs for these models in the accelerated paper cohort.
ACCELERATED_EAGER_MODELS = frozenset({"esen-30m-oam", "eq-v2-m-omat"})


def model_trajectory_source(source: str, model: object) -> str:
    """Return the physical MD source, independently of the output cohort."""
    if source == TORCHSIM_ACCELERATED and normalize_display_key(model) in ACCELERATED_EAGER_MODELS:
        return TORCHSIM_EAGER
    return source


def source_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        if part in SOURCES:
            return part
    backend = "ase" if "ase" in path.parts or "e-f-predictions-ase" in path.parts else "torchsim"
    for part in reversed(path.parts):
        if part in {"md_accelerated", "md-accelerated", "md_eager"}:
            mode = "md_accelerated" if part != "md_eager" else part
            return next(source for source, dataset in SOURCE_DATASETS.items()
                        if dataset == (backend, mode))
    return None


def eager_counterpart(path: Path) -> Path:
    """Resolve canonical and evaluator-layout paths to the eager sibling."""
    return Path(*(TORCHSIM_EAGER if part == TORCHSIM_ACCELERATED else
                  "md_eager" if part in {"md_accelerated", "md-accelerated"} else part
                  for part in path.parts))


def cohort_model_path(path: Path, model: object) -> Path:
    source = source_from_path(path)
    if source is not None and model_trajectory_source(source, model) != source:
        return eager_counterpart(path)
    return path


def cohort_model_files(directory: Path, pattern: str, model_from_path, *, recursive: bool = False) -> list[Path]:
    """Select eager eSEN/EQ files and accelerated files for every other model.

    Never fall back to an accelerated eSEN/EQ run when its eager data is absent:
    missing eager evaluations retain the benchmark's normal failure policy.
    Returned paths keep their real provenance and model-matched references.
    """
    def scan(root):
        return root.rglob(pattern) if recursive else root.glob(pattern)
    files = sorted(scan(directory))
    if source_from_path(directory) != TORCHSIM_ACCELERATED:
        return files
    files = [path for path in files
             if normalize_display_key(model_from_path(path)) not in ACCELERATED_EAGER_MODELS]
    files.extend(path for path in scan(eager_counterpart(directory))
                 if normalize_display_key(model_from_path(path)) in ACCELERATED_EAGER_MODELS)
    return sorted(files)


def cohort_timing_files(directory: Path) -> list[Path]:
    return cohort_model_files(directory, "*/md_timing_*.csv",
                              lambda path: path.stem.removeprefix("md_timing_"))


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
