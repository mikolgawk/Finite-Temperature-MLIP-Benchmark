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
SYSTEM_COLUMNS = (
    "system", "System", "structure", "system_type", "sys_type",
    "trajectory", "trajectory_file",
)


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


def is_metal_water_interface(system) -> bool:
    """Recognize the Pt/water benchmark by system name, type, or trajectory path."""
    value = str(system).strip().lower().replace(chr(92), "/")
    if value == "metal-water interfaces":
        return True
    if "pt111w24h2o" not in value:
        return False
    path = Path(value)
    name = path.parent.name if path.suffix in {
        ".extxyz", ".xyz", ".h5", ".hdf5", ".traj", ".csv",
    } else path.name
    return name.split("_", 1)[0] == "pt111w24h2o"


def include_pressure_system(system, include_molecular_crystals: bool = False) -> bool:
    """Pressure always excludes Pt/water, even when crystals are included."""
    return not is_metal_water_interface(system) and include_system(
        system, include_molecular_crystals
    )


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
    for column in SYSTEM_COLUMNS:
        if column in frame:
            keep &= ~frame[column].map(is_molecular_crystal)
    return frame.loc[keep].copy()


def filter_pressure_systems(frame, include_molecular_crystals: bool = False):
    """Apply pressure exclusions to current and previously saved detailed tables."""
    frame = filter_molecular_crystals(frame, include_molecular_crystals)
    keep = frame.index.to_series().map(lambda _: True).astype(bool)
    for column in SYSTEM_COLUMNS:
        if column in frame:
            keep &= ~frame[column].map(is_metal_water_interface)
    return frame.loc[keep].copy()
