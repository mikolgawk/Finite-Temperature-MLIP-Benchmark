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
ASE_EAGER = "mlip-trajs-ase"
SOURCE_ALIASES = {"mlip-trajs-ase-eager": ASE_EAGER}
# Use eager runs for these models in the accelerated paper cohort.
ACCELERATED_EAGER_MODELS = frozenset({"esen-30m-oam", "eq-v2-m-omat"})
ACCELERATED_MODEL_SOURCES = {
    **dict.fromkeys(ACCELERATED_EAGER_MODELS, TORCHSIM_EAGER),
    "chgnet": ASE_EAGER,
}


def model_trajectory_source(source: str, model: object) -> str:
    """Return the physical MD source, independently of the output cohort."""
    if source == TORCHSIM_ACCELERATED:
        return ACCELERATED_MODEL_SOURCES.get(normalize_display_key(model), source)
    return source


def select_model_names(discovered: set[str], requested: set[str] | None) -> set[str] | None:
    """Match CLI model aliases while retaining missing models for failure penalties."""
    if requested is None:
        return None
    keys = {normalize_display_key(model) for model in requested}
    matches = {model for model in discovered if normalize_display_key(model) in keys}
    matched_keys = {normalize_display_key(model) for model in matches}
    return matches | {model for model in requested if normalize_display_key(model) not in matched_keys}


def source_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        if part in SOURCE_ALIASES:
            return SOURCE_ALIASES[part]
        if part in SOURCES:
            return part
    backend = "ase" if "ase" in path.parts or "e-f-predictions-ase" in path.parts else "torchsim"
    for part in reversed(path.parts):
        if part in {"md_accelerated", "md-accelerated", "md_eager"}:
            mode = "md_accelerated" if part != "md_eager" else part
            return next(source for source, dataset in SOURCE_DATASETS.items()
                        if dataset == (backend, mode))
    return None


def source_counterpart(path: Path, source: str) -> Path:
    """Resolve a physical source in canonical, backend, or evaluator layouts."""
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part in SOURCES or part in SOURCE_ALIASES:
            root = Path(*parts[:index])
            names = ("mlip-trajs-ase-eager", ASE_EAGER) if source == ASE_EAGER else (source,)
            candidates = [root / name for name in names]
            backend, mode = SOURCE_DATASETS[source]
            candidates.append(root / backend / mode)
            directory = next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])
            return directory / Path(*parts[index + 1:])
    backend, mode = SOURCE_DATASETS[source]
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] not in {"md_eager", "md_accelerated", "md-accelerated"}:
            continue
        parts[index] = mode
        if index and parts[index - 1] in {"ase", "torchsim"}:
            parts[index - 1] = backend
        else:
            prediction_index = next((i for i, part in enumerate(parts[:index])
                                     if part.startswith("e-f-predictions")), None)
            if prediction_index is not None:
                if backend == "ase":
                    parts[prediction_index] = "e-f-predictions-ase"
                elif parts[prediction_index] == "e-f-predictions-ase":
                    parts[prediction_index] = "e-f-predictions-torchsim"
            elif backend == "ase":
                parts.insert(index, backend)
        return Path(*parts)
    return path


def cohort_model_path(path: Path, model: object) -> Path:
    source = source_from_path(path)
    if source is not None and model_trajectory_source(source, model) != source:
        return source_counterpart(path, model_trajectory_source(source, model))
    return path


def cohort_trajectory_path(data_dir: Path, source: str, system: str, model: object) -> Path:
    """Find the actual HDF5 name, including ASE's CHGNet execution tags."""
    directory = cohort_model_path(data_dir / source, model) / system
    path = directory / f"nvt_{model}.h5"
    if path.is_file():
        return path
    matches = sorted(candidate for candidate in directory.glob("nvt_*.h5")
                     if normalize_display_key(candidate.stem.removeprefix("nvt_"))
                     == normalize_display_key(model))
    return next((candidate for candidate in matches if candidate.stem.endswith("-ase")),
                matches[0] if matches else path)


def cohort_model_files(directory: Path, pattern: str, model_from_path, *, recursive: bool = False) -> list[Path]:
    """Select eager TorchSim eSEN/EQ and ASE CHGNet for the accelerated cohort.

    Never fall back to an accelerated replacement run when eager data is absent:
    missing eager evaluations retain the benchmark's normal failure policy.
    Returned paths keep their real provenance and model-matched references.
    """
    def scan(root):
        return root.rglob(pattern) if recursive else root.glob(pattern)
    files = sorted(scan(directory))
    if source_from_path(directory) != TORCHSIM_ACCELERATED:
        return files
    files = [path for path in files
             if normalize_display_key(model_from_path(path)) not in ACCELERATED_MODEL_SOURCES]
    for source in sorted(set(ACCELERATED_MODEL_SOURCES.values())):
        files.extend(path for path in scan(source_counterpart(directory, source))
                     if ACCELERATED_MODEL_SOURCES.get(normalize_display_key(model_from_path(path))) == source)
    return sorted(files)


def cohort_timing_files(directory: Path) -> list[Path]:
    return cohort_model_files(directory, "*/md_timing_*.csv",
                              lambda path: path.stem.removeprefix("md_timing_"))


def pressure_input_dir(root: Path, source: str) -> Path:
    """Prefer per-source raw data, with support for the evaluator's older layout."""
    canonical = source_counterpart(root / source, source)
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
