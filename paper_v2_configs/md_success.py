"""Identify completed TorchSim MD runs, excluding audited physical failures."""

import csv
from functools import lru_cache
from pathlib import Path

from metric_sources import cohort_model_files, cohort_timing_files


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


def torchsim_md_succeeded(trajectory: Path) -> bool:
    """Require both the trajectory and its completed, matching timing record.

    TorchSim MD runners leave an empty timing CSV when a system fails. A partial
    HDF5 file may also survive a failed run, so its presence alone is insufficient.
    Audited physical failures remain failed even when their timing CSV is complete.
    """
    if not trajectory.is_file() or not trajectory.name.startswith("nvt_"):
        return False
    if registered_md_failure_reason(trajectory) is not None:
        return False
    model = trajectory.stem.removeprefix("nvt_")
    timing = trajectory.with_name(f"md_timing_{model}.csv")
    try:
        with timing.open(newline="") as handle:
            records = csv.DictReader(handle)
            return any(
                row.get("calculator") == model
                and row.get("system") == trajectory.parent.name
                and float(row.get("n_steps") or 0) > 0
                for row in records
            )
    except (OSError, ValueError, csv.Error):
        return False
