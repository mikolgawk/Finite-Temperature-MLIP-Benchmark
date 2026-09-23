"""Identify model/system pairs with a completed TorchSim MD run."""

import csv
from pathlib import Path


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
