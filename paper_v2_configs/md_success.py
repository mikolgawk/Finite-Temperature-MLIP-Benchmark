"""Identify model/system pairs with a completed TorchSim MD run."""

import csv
from pathlib import Path


def discover_torchsim_md_trajectories(source_dir: Path) -> dict[str, dict[str, Path]]:
    """Include failed jobs whose only remaining output is a timing marker."""
    trajectories: dict[str, dict[str, Path]] = {}
    for path in sorted(source_dir.glob("*/nvt_*.h5")):
        model = path.stem.removeprefix("nvt_")
        trajectories.setdefault(path.parent.name, {})[model] = path
    for timing in sorted(source_dir.glob("*/md_timing_*.csv")):
        model = timing.stem.removeprefix("md_timing_")
        trajectories.setdefault(timing.parent.name, {}).setdefault(
            model, timing.with_name(f"nvt_{model}.h5")
        )
    return trajectories


def torchsim_md_succeeded(trajectory: Path) -> bool:
    """Require both the trajectory and its completed, matching timing record.

    TorchSim MD runners leave an empty timing CSV when a system fails. A partial
    HDF5 file may also survive a failed run, so its presence alone is insufficient.
    """
    if not trajectory.is_file() or not trajectory.name.startswith("nvt_"):
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
