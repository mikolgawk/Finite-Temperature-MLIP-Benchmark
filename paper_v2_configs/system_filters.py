"""Shared system selection for V2 metric computations and timing analysis."""

from pathlib import Path


MOLECULAR_CRYSTALS = frozenset(
    {
        "anthracene",
        "naphthalene",
        "pentacene",
        "picene",
        "tetracene",
    }
)
DEFAULT_EXCLUDED_SYSTEM_TYPES = ("molecular crystals",)


def is_molecular_crystal(system) -> bool:
    """Recognize short names, benchmark directories, and trajectory paths."""
    value = str(system).strip()
    if value.lower() == "molecular crystals":
        return True
    path = Path(value)
    name = (
        path.parent.name
        if path.suffix.lower()
        in {
            ".extxyz",
            ".xyz",
            ".h5",
            ".hdf5",
            ".traj",
            ".csv",
        }
        else path.name
    )
    return name.split("_", 1)[0].lower() in MOLECULAR_CRYSTALS


def include_system(system, include_molecular_crystals: bool = False) -> bool:
    return include_molecular_crystals or not is_molecular_crystal(system)


def add_molecular_crystal_option(parser) -> None:
    parser.add_argument(
        "--include-molecular-crystals",
        action="store_true",
        help="Include the five molecular crystals (excluded by default).",
    )


def excluded_system_types(args) -> set[str]:
    """Combine default exclusions with any explicitly excluded system types."""
    excluded = {
        value.strip().lower()
        for value in getattr(args, "excluded_system_types", None) or ()
    }
    if not getattr(args, "include_molecular_crystals", False):
        excluded.update(DEFAULT_EXCLUDED_SYSTEM_TYPES)
    return excluded


def filter_molecular_crystals(frame, include_molecular_crystals: bool = False):
    """Filter detailed metric tables, including older saved crystal results."""
    if include_molecular_crystals:
        return frame.copy()
    keep = frame.index.to_series().map(lambda _: True).astype(bool)
    for column in (
        "system",
        "System",
        "structure",
        "system_type",
        "sys_type",
        "trajectory",
        "trajectory_file",
    ):
        if column in frame:
            keep &= ~frame[column].map(is_molecular_crystal)
    return frame.loc[keep].copy()
