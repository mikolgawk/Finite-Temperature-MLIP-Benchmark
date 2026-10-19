"""Identify completed HDF5 MD runs, excluding audited physical failures."""

import csv
from functools import lru_cache
from pathlib import Path

from metric_sources import (
    SOURCE_DATASETS, cohort_model_files, cohort_timing_files, source_counterpart, source_from_path,
)
from model_display_names import normalize_display_key


FAILED_MD_RUNS_FILE = Path(__file__).with_name("failed_md_runs.csv")


@lru_cache(maxsize=1)
def load_failed_md_runs() -> dict[tuple[str, str, str], str]:
    """Load audited failures keyed by exact source, system, and raw model name."""
    with FAILED_MD_RUNS_FILE.open(newline="") as handle:
        return {
            (row["source"], row["system"], row["model"]): row["failure_reason"]
            for row in csv.DictReader(handle)
        }


def registered_md_failure_reason(trajectory: Path) -> str | None:
    """Look up a trajectory or timing file without mixing trajectory sources."""
    sources = {source for source, _, _ in load_failed_md_runs()}
    source = next((part for part in reversed(trajectory.parts) if part in sources), None)
    source = source or source_from_path(trajectory)
    model = trajectory.stem.removeprefix("nvt_").removeprefix("md_timing_")
    return load_failed_md_runs().get((source, trajectory.parent.name, model))


def discover_torchsim_md_trajectories(source_dir: Path) -> dict[str, dict[str, Path]]:
    """Include failed jobs whose only remaining output is a timing marker."""
    trajectories: dict[str, dict[str, Path]] = {}
    for path in cohort_model_files(source_dir, "*/nvt_*.h5",
                                   lambda path: path.stem.removeprefix("nvt_")):
        model = path.stem.removeprefix("nvt_")
        trajectories.setdefault(path.parent.name, {})[model] = path
    for timing in cohort_timing_files(source_dir):
        model = timing.stem.removeprefix("md_timing_")
        trajectories.setdefault(timing.parent.name, {}).setdefault(
            model, timing.with_name(f"nvt_{model}.h5")
        )
    return trajectories


def discover_md_trajectories(source_dir: Path) -> dict[str, dict[str, Path]]:
    """Discover runs, collapsing physical ASE aliases to one model per system.

    ASE runners may leave an obsolete timing-only ``-ase`` alias beside a
    completed run without that tag. A completed run takes precedence over
    failed or incomplete aliases, preventing duplicate model penalties. Two
    completed aliases are ambiguous and require the inputs to be disambiguated.

    ASE keys are canonical model names, while paths retain the raw filenames
    needed for completion checks and audited-failure lookups. When all aliases
    failed, prefer an audited failure, then an existing partial trajectory, then
    a timing-only marker. TorchSim cohort discovery keeps its existing raw keys.
    """
    source = source_from_path(source_dir)
    physical_dir = source_counterpart(source_dir, source) if source is not None else source_dir
    trajectories = discover_torchsim_md_trajectories(physical_dir)
    if source is None or SOURCE_DATASETS[source][0] != "ase":
        return trajectories

    canonical: dict[str, dict[str, Path]] = {}
    for system, models in trajectories.items():
        aliases: dict[str, list[Path]] = {}
        for model, path in models.items():
            aliases.setdefault(normalize_display_key(model), []).append(path)
        canonical[system] = {}
        for model, paths in aliases.items():
            completed = [path for path in paths if torchsim_md_succeeded(path)]
            if len(completed) > 1:
                files = ", ".join(str(path) for path in sorted(completed))
                raise ValueError(
                    f"Ambiguous completed ASE MD runs for model {model!r}, "
                    f"system {system!r}, source {source!r}: {files}. "
                    "Keep one completed run per model/system or use separate source directories."
                )
            selected = completed or sorted(
                paths,
                key=lambda path: (
                    registered_md_failure_reason(path) is None,
                    not path.is_file(),
                    str(path),
                ),
            )
            canonical[system][model] = selected[0]
    return canonical


def torchsim_md_succeeded(trajectory: Path) -> bool:
    """Require both the trajectory and its completed, matching timing record.

    TorchSim and ASE MD runners leave an empty timing CSV when a system fails. A partial
    HDF5 file may also survive a failed run, so its presence alone is insufficient.
    Audited physical failures remain failed even when their timing CSV is complete.
    Physical ASE runs may use a final ``-ase`` filename tag that is absent from
    the timing calculator; all other model identifiers must match exactly.
    """
    if not trajectory.is_file() or not trajectory.name.startswith("nvt_"):
        return False
    if registered_md_failure_reason(trajectory) is not None:
        return False
    model = trajectory.stem.removeprefix("nvt_")
    timing = trajectory.with_name(f"md_timing_{model}.csv")
    source = source_from_path(trajectory)
    ase_source = source is not None and SOURCE_DATASETS[source][0] == "ase"

    def calculator_matches(calculator: str | None) -> bool:
        if calculator == model:
            return True
        return (
            ase_source
            and calculator is not None
            and calculator.removesuffix("-ase") == model.removesuffix("-ase")
        )

    try:
        with timing.open(newline="") as handle:
            records = csv.DictReader(handle)
            return any(
                calculator_matches(row.get("calculator"))
                and row.get("system") == trajectory.parent.name
                and float(row.get("n_steps") or 0) > 0
                for row in records
            )
    except (OSError, ValueError, csv.Error):
        return False
