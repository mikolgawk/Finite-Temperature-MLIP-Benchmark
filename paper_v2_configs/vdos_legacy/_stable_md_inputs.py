"""Select each model's stable physical MD runs for correlation metric means."""

import csv
from pathlib import Path
import re

import numpy as np
import pandas as pd

from md_success import registered_md_failure_reason
from metric_sources import cohort_model_path, cohort_trajectory_path, source_from_path


def stable_md_run(trajectory: Path) -> bool:
    """Require completed MD and exclude audited failures, including ASE runs."""
    if not trajectory.is_file() or registered_md_failure_reason(trajectory) is not None:
        return False
    model = trajectory.stem.removeprefix("nvt_")
    physical_source = source_from_path(trajectory)
    is_ase = physical_source is not None and "ase" in physical_source
    try:
        with trajectory.with_name(f"md_timing_{model}.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1:
            return False
        row = rows[0]
        calculator = str(row.get("calculator", "")).strip()
        expected = model
        if is_ase:
            calculator = calculator.removesuffix("-ase")
            expected = expected.removesuffix("-ase")
        steps = float(row.get("n_steps") or 0)
        return (calculator == expected and row.get("system") == trajectory.parent.name
                and np.isfinite(steps) and steps > 0)
    except (OSError, ValueError, TypeError, csv.Error):
        return False


class StableMDFilter:
    """Share physical-run decisions across all four correlation metric inputs."""

    def __init__(self, data_dir: Path, source: str):
        self.data_dir = Path(data_dir)
        self.source = source
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Stable-only correlations require MD trajectories and timing records in {data_dir}. "
                "Use --md-data-dir to select their root directory."
            )
        self.systems: dict[Path, list[str]] = {}
        self.stability: dict[Path, bool] = {}

    def system_name(self, row: pd.Series, model: str) -> str:
        directory = cohort_model_path(self.data_dir / self.source, model)
        if directory not in self.systems:
            self.systems[directory] = sorted(path.name for path in directory.glob("*") if path.is_dir())
        systems = self.systems[directory]
        for column in ("trajectory", "trajectory_file", "System", "system", "structure", "reference_key"):
            value = row.get(column)
            if not isinstance(value, str) or not value.strip():
                continue
            path = Path(value.strip().replace(chr(92), "/"))
            candidate = path.parent.name if path.suffix.lower() in {
                ".extxyz", ".xyz", ".h5", ".hdf5", ".traj", ".csv",
            } else path.name
            if candidate in systems:
                return candidate
            matches = [name for name in systems if name.startswith(candidate + "_")]
            temperature = row.get("temperature_K")
            if len(matches) > 1 and pd.notna(temperature):
                prefix = f"{candidate}_{float(temperature):g}K"
                matches = [name for name in matches if name.startswith(prefix + "_")]
            if len(matches) == 1:
                return matches[0]
            # Full benchmark identifiers remain resolvable when a run is missing.
            if re.search(r"_\d+K_", candidate):
                return candidate
        raise ValueError(
            f"Cannot resolve MD system for model {model!r}; provide full system identifiers "
            "or reference trajectory paths in the per-system metric CSV."
        )

    def filter_rows(self, frame: pd.DataFrame, description: str) -> pd.DataFrame:
        """Keep stable systems, preserving genuine 100% scores and full IDs."""
        if "source" in frame:
            frame = frame.loc[frame["source"] == self.source].copy()
        if not any(column in frame for column in (
            "system", "System", "structure", "reference_key", "trajectory", "trajectory_file",
        )):
            raise ValueError(f"Stable-only correlations require per-system {description} rows, not model means.")
        model_column = next((column for column in ("calculator", "Calculator", "model", "mlip_model")
                             if column in frame), None)
        if model_column is None:
            raise ValueError(f"Missing model column in per-system {description} data")
        systems, keep = [], []
        for _, row in frame.iterrows():
            model = str(row[model_column])
            system = self.system_name(row, model)
            trajectory = cohort_trajectory_path(self.data_dir, self.source, system, model)
            if trajectory not in self.stability:
                self.stability[trajectory] = stable_md_run(trajectory)
            systems.append(system)
            keep.append(self.stability[trajectory])
        frame = frame.copy()
        frame["system"] = systems
        return frame.loc[np.asarray(keep, dtype=bool)].copy()
