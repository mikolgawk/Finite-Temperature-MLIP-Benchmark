"""Shared paths and table loaders for the Pareto/correlation figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent

SOURCES = (
    "mlip-trajs-ase",
    "mlip-trajs-torchsim-eager",
    "mlip-trajs-ase-accelerated",
    "mlip-trajs-torchsim-accelerated",
)
DEFAULT_SOURCE = "mlip-trajs-torchsim-eager"

DEFAULT_PRESSURE_FILE = CONFIG_DIR / "pressures" / "results" / DEFAULT_SOURCE / "model_pressure_error_metric.csv"


def pressure_path(source: str) -> Path:
    return CONFIG_DIR / 'pressures' / 'results' / source / 'model_pressure_error_metric.csv'


def source_paths(source: str) -> tuple[Path, Path, Path]:
    """Return timing, RDF, and VDOS paths produced for *source*."""
    return (
        CONFIG_DIR / "data" / source,
        CONFIG_DIR / "rdfs" / "results" / source
        / "rdf_similarity_scores_same_simulation_length.csv",
        CONFIG_DIR / "vdos" / "results" / source
        / "vdos_model_mean_ev_normalized_same_simulation_length.csv",
    )


def source_pressure_provenance(source: str) -> tuple[str, str]:
    """Map a trajectory source to pressure metric backend and execution mode."""
    backend = "torchsim" if "torchsim" in source else "ase"
    mode = "md_accelerated" if source.endswith("-accelerated") else "md_eager"
    return backend, mode


def model_key(name: object) -> str:
    """Normalize execution-specific model names before joining metric tables."""
    key = str(name).strip().lower()
    key = key.replace("-force-only", "").replace("-stress", "")
    if key.endswith("-eager"):
        key = key.removesuffix("-eager")
    return {"nequip-oam-l": "nequip"}.get(key, key)


def values_to_percent(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values.dropna()
    if not finite.empty and float(finite.max()) <= 1.5:
        values = values * 100.0
    return values


def first_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def load_rdf(path: Path) -> pd.DataFrame:
    """Load a generated or legacy RDF model-summary table."""
    df = pd.read_csv(path)
    model_col = first_column(df, ("model", "Calculator", "calculator", "mlip_model"))
    error_col = first_column(
        df,
        ("rdf_error_percent", "Mean RDF Error [%]", "RDF Error [%]", "rdf_error_mean"),
    )
    similarity_col = first_column(
        df,
        (
            "rdf_similarity_percent",
            "Mean Similarity Score [%]",
            "RDF Similarity [%]",
            "rdf_similarity_mean",
        ),
    )
    if model_col is None or (error_col is None and similarity_col is None):
        raise ValueError(
            f"RDF file {path} needs a model column and an RDF error or similarity column."
        )

    out = pd.DataFrame({"model": df[model_col].map(model_key)})
    if error_col is not None:
        out["rdf_error_percent"] = values_to_percent(df[error_col])
        out["rdf_similarity_percent"] = 100.0 - out["rdf_error_percent"]
    else:
        out["rdf_similarity_percent"] = values_to_percent(df[similarity_col])
        out["rdf_error_percent"] = 100.0 - out["rdf_similarity_percent"]
    return out.groupby("model", as_index=False).mean(numeric_only=True)


def load_vdos(path: Path) -> pd.DataFrame:
    """Load a generated or legacy VDOS model-summary table."""
    df = pd.read_csv(path)
    model_col = first_column(df, ("model", "mlip_model", "Calculator", "calculator"))
    error_col = first_column(
        df,
        (
            "vdos_error_percent",
            "Mean VDOS Error [%]",
            "VDOS Error [%]",
            "final_mean_vdos_error",
        ),
    )
    similarity_col = first_column(
        df,
        (
            "vdos_similarity_percent",
            "Mean Similarity Score [%]",
            "VDOS Similarity [%]",
            "final_mean_vdos_similarity",
        ),
    )
    if model_col is None or (error_col is None and similarity_col is None):
        raise ValueError(
            f"VDOS file {path} needs a model column and a VDOS error or similarity column."
        )

    out = pd.DataFrame({"model": df[model_col].map(model_key)})
    if error_col is not None:
        out["vdos_error_percent"] = values_to_percent(df[error_col])
        out["vdos_similarity_percent"] = 100.0 - out["vdos_error_percent"]
    else:
        out["vdos_similarity_percent"] = values_to_percent(df[similarity_col])
        out["vdos_error_percent"] = 100.0 - out["vdos_similarity_percent"]
    return out.groupby("model", as_index=False).mean(numeric_only=True)


def load_pressure(
    path: Path,
    *,
    backend: str | None = None,
    mode: str | None = None,
) -> pd.DataFrame:
    """Load pressure errors, optionally selecting generated provenance columns."""
    df = pd.read_csv(path)
    for column, value in (("backend", backend), ("mode", mode)):
        if value is not None and column in df.columns:
            df = df.loc[df[column].astype(str).str.lower() == value.lower()].copy()
    if df.empty:
        raise ValueError(
            f"No pressure rows remain for backend={backend!r}, mode={mode!r} in {path}."
        )

    model_col = first_column(df, ("model", "mlip_model", "calculator"))
    error_col = first_column(
        df,
        (
            "final_mean_pressure_error_percent",
            "pressure_error_percent",
            "mean_pressure_error_percent",
            "Pressure Error [%]",
        ),
    )
    similarity_col = first_column(
        df,
        (
            "final_mean_pressure_similarity_percent",
            "pressure_similarity_percent",
            "mean_pressure_similarity_percent",
            "Pressure Similarity [%]",
            "final_mean_pressure_similarity",
        ),
    )
    if model_col is None or (error_col is None and similarity_col is None):
        raise ValueError(
            f"Pressure file {path} needs a model column and a pressure error or similarity column."
        )

    out = pd.DataFrame({"model": df[model_col].map(model_key)})
    if error_col is not None:
        out["pressure_error_percent"] = values_to_percent(df[error_col])
    else:
        out["pressure_error_percent"] = 100.0 - values_to_percent(df[similarity_col])
    return out.groupby("model", as_index=False).mean(numeric_only=True)
