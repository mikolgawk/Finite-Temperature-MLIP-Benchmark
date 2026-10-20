"""Shared helpers for comparing v2 RDF/VDOS errors with v1 results."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from model_display_names import display_model_name  # noqa: E402


SOURCE_DIRS = {
    "eager": "mlip-trajs-torchsim-eager",
    "accelerated": "mlip-trajs-torchsim-accelerated",
}

# These v2 systems have no legacy counterpart and are deliberately left out.
EXCLUDED_SYSTEMS = {
    "H_1050K_Rupp_QE",
    "Pt111w24H2O_380K_Heenen_VASP",
}

# Map execution-specific v2 trajectory names to the corresponding legacy model.
# None denotes a new v2 model that must not be included in this comparison.
V2_TO_LEGACY_MODEL: dict[str, str | None] = {
    "chgnet-force-only-ase": "chgnet",
    "eSEN-30M-OAM-force-only": "eSEN-30M-OAM",
    "eq-v2-M-omat-force-only": "eq-v2-M-omat",
    "grace-mp": "grace-mp",
    "grace-oam-force-only-eager": "grace-oam",
    "grace-oam-force-only-compiled": "grace-oam",
    "mace-mh-omat": "mace-mh-omat",
    "mace-mh-omat-compile": "mace-mh-omat",
    "mace-mp-0": "mace-mp-0",
    "mace-mp-0-compile": "mace-mp-0",
    "mace-mpa-0": "mace-mpa-0",
    "mace-mpa-0-compile": "mace-mpa-0",
    "mattersim-v1-5M-force-only": "mattersim-v1-5M",
    "mattersim-v1-5M-compile-force-only": "mattersim-v1-5M",
    # paper-v2 NequIP-L replaces the legacy NequIP-XL result.
    "nequip-oam-l-force-only": "nequip",
    "orb-v2-force-only-eager": "orb-v2",
    "orb-v2-force-only": "orb-v2",
    "orb-v3-omat-force-only-eager": None,
    "orb-v3-omat-force-only": None,
    "orb-v3-direct-omat-force-only-eager": None,
    "orb-v3-direct-omat-force-only": None,
    "pet-oam-xl-force-only-eager": "pet-oam-xl",
    "pet-oam-xl-force-only-torchscript": "pet-oam-xl",
    "pet-omat-xl-force-only-eager": None,
    "pet-omat-xl-force-only-torchscript": None,
    "uma-m-omat-force-only": "uma-m-omat",
    "uma-m-omat-compile-force-only": "uma-m-omat",
    "uma-m-omat-turbo-force-only": "uma-m-omat",
    "uma-s-omat-force-only": "uma-s-omat",
    "uma-s-omat-compile-force-only": "uma-s-omat",
    "uma-s-omat-turbo-force-only": "uma-s-omat",
}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    current_column: str
    legacy_column: str
    output_prefix: str
    legacy_scale: float
    current_loader: Callable[[Path], pd.DataFrame]
    legacy_loader: Callable[[Path], pd.DataFrame]


def _require_columns(data: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns - set(data.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")


def load_current_rdf(directory: Path) -> pd.DataFrame:
    prefix = "rdf_similarity_scores_same_simulation_length_"
    rows: list[pd.DataFrame] = []
    for path in sorted(directory.glob(f"{prefix}*.csv")):
        if path.name.endswith("by_system_type_same_simulation_length.csv"):
            continue
        model = path.stem.removeprefix(prefix)
        data = pd.read_csv(path)
        _require_columns(data, {"System", "RDF_Error"}, path)
        frame = data[["System", "RDF_Error"]].rename(
            columns={"System": "system", "RDF_Error": "current_error_percent"}
        )
        frame.insert(0, "current_model", model)
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No per-model RDF result files found in {directory}")
    return pd.concat(rows, ignore_index=True)


def load_legacy_rdf(directory: Path) -> pd.DataFrame:
    prefix = "rdf_similarity_scores_same_simulation_length_"
    rows: list[pd.DataFrame] = []
    for path in sorted(directory.glob(f"{prefix}*.csv")):
        if path.name.endswith("by_system_type_same_simulation_length.csv"):
            continue
        model = path.stem.removeprefix(prefix)
        data = pd.read_csv(path)
        _require_columns(data, {"System", "RDF_Error"}, path)
        frame = data[["System", "RDF_Error"]].rename(
            columns={"System": "system", "RDF_Error": "legacy_error"}
        )
        frame.insert(0, "legacy_model", model)
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No per-model RDF result files found in {directory}")
    return pd.concat(rows, ignore_index=True)


def load_current_vdos(directory: Path) -> pd.DataFrame:
    path = directory / "vdos_system_model_mean_ev_normalized_same_simulation_length.csv"
    data = pd.read_csv(path)
    _require_columns(data, {"model", "system", "vdos_error_percent"}, path)
    return data[["model", "system", "vdos_error_percent"]].rename(
        columns={"model": "current_model", "vdos_error_percent": "current_error_percent"}
    )


def load_legacy_vdos(directory: Path) -> pd.DataFrame:
    path = directory / "vdos_pair_errors_ev_normalized_same_simulation_length.csv"
    data = pd.read_csv(path)
    _require_columns(data, {"mlip_model", "system", "vdos_error"}, path)
    return data[["mlip_model", "system", "vdos_error"]].rename(
        columns={"mlip_model": "legacy_model", "vdos_error": "legacy_error"}
    )


RDF_SPEC = MetricSpec(
    name="RDF",
    current_column="current_error_percent",
    legacy_column="legacy_error_percent",
    output_prefix="rdf",
    legacy_scale=1.0,
    current_loader=load_current_rdf,
    legacy_loader=load_legacy_rdf,
)

VDOS_SPEC = MetricSpec(
    name="VDOS",
    current_column="current_error_percent",
    legacy_column="legacy_error_percent",
    output_prefix="vdos",
    legacy_scale=100.0,
    current_loader=load_current_vdos,
    legacy_loader=load_legacy_vdos,
)


def _validate_unique(data: pd.DataFrame, keys: list[str], label: str) -> None:
    duplicates = data[data.duplicated(keys, keep=False)]
    if not duplicates.empty:
        examples = duplicates[keys].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"Duplicate {label} rows for {keys}: {examples}")


def compare_source(
    spec: MetricSpec,
    source: str,
    current_dir: Path,
    legacy: pd.DataFrame,
    excluded_systems: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Align one v2 source with legacy rows and calculate error deltas."""
    current = spec.current_loader(current_dir).copy()
    _validate_unique(current, ["current_model", "system"], f"v2 {spec.name}")

    unknown_models = sorted(set(current["current_model"]) - set(V2_TO_LEGACY_MODEL))
    if unknown_models:
        raise ValueError(
            "No legacy-model mapping is defined for: " + ", ".join(unknown_models)
        )

    current["legacy_model"] = current["current_model"].map(V2_TO_LEGACY_MODEL)
    excluded_models = sorted(current.loc[current["legacy_model"].isna(), "current_model"].unique())
    current = current[current["legacy_model"].notna()].copy()

    requested_exclusions = set(excluded_systems or ())
    present_excluded_systems = sorted(set(current["system"]) & EXCLUDED_SYSTEMS)
    present_requested_exclusions = sorted(set(current["system"]) & requested_exclusions)
    current = current[
        ~current["system"].isin(EXCLUDED_SYSTEMS | requested_exclusions)
    ].copy()

    current_systems = set(current["system"])
    legacy_systems = set(legacy["system"])
    common_systems = sorted(current_systems & legacy_systems)
    if not common_systems:
        raise ValueError(f"No common systems found for {source} {spec.name}")

    current = current[current["system"].isin(common_systems)]
    relevant_legacy_models = set(current["legacy_model"])
    legacy_subset = legacy[
        legacy["system"].isin(common_systems)
        & legacy["legacy_model"].isin(relevant_legacy_models)
    ].copy()
    _validate_unique(legacy_subset, ["legacy_model", "system"], f"legacy {spec.name}")

    comparison = current.merge(
        legacy_subset,
        on=["legacy_model", "system"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    missing = comparison[comparison["_merge"] != "both"]
    if not missing.empty:
        examples = missing[["current_model", "legacy_model", "system"]].head(10)
        raise ValueError(
            "Missing legacy values for mapped v2 rows:\n" + examples.to_string(index=False)
        )
    comparison = comparison.drop(columns="_merge")
    comparison[spec.legacy_column] = comparison.pop("legacy_error") * spec.legacy_scale
    comparison["delta_percentage_points"] = (
        comparison[spec.current_column] - comparison[spec.legacy_column]
    )
    comparison["absolute_delta_percentage_points"] = comparison[
        "delta_percentage_points"
    ].abs()
    denominator = comparison[spec.legacy_column].replace(0.0, np.nan)
    comparison["relative_delta_percent"] = (
        100.0 * comparison["delta_percentage_points"] / denominator
    )
    comparison.insert(0, "source", source)
    comparison = comparison.sort_values(["current_model", "system"]).reset_index(drop=True)

    report: dict[str, object] = {
        "source": source,
        "excluded_new_models": excluded_models,
        "excluded_new_systems": present_excluded_systems,
        "excluded_requested_systems": present_requested_exclusions,
        "common_systems": common_systems,
        "legacy_only_systems": sorted(legacy_systems - current_systems),
        "n_comparison_rows": len(comparison),
    }
    return comparison, report


def _pearson(group: pd.DataFrame, spec: MetricSpec) -> float:
    if len(group) < 2:
        return float("nan")
    return float(group[spec.current_column].corr(group[spec.legacy_column]))


def model_summary(comparison: pd.DataFrame, spec: MetricSpec) -> pd.DataFrame:
    group_columns = ["source", "current_model", "legacy_model"]
    summary = (
        comparison.groupby(group_columns, sort=True)
        .agg(
            n_systems=("system", "nunique"),
            current_mean_error_percent=(spec.current_column, "mean"),
            legacy_mean_error_percent=(spec.legacy_column, "mean"),
            mean_delta_percentage_points=("delta_percentage_points", "mean"),
            mean_absolute_delta_percentage_points=(
                "absolute_delta_percentage_points",
                "mean",
            ),
            max_absolute_delta_percentage_points=(
                "absolute_delta_percentage_points",
                "max",
            ),
        )
        .reset_index()
    )
    rmses = (
        comparison.assign(squared_delta=comparison["delta_percentage_points"] ** 2)
        .groupby(group_columns)["squared_delta"]
        .mean()
        .pow(0.5)
        .rename("rmse_delta_percentage_points")
        .reset_index()
    )
    correlations = (
        comparison.groupby(group_columns, sort=True)
        .apply(lambda group: _pearson(group, spec), include_groups=False)
        .rename("pearson_r_across_systems")
        .reset_index()
    )
    return summary.merge(rmses, on=group_columns).merge(correlations, on=group_columns)


def system_summary(comparison: pd.DataFrame, spec: MetricSpec) -> pd.DataFrame:
    return (
        comparison.groupby(["source", "system"], sort=True)
        .agg(
            n_model_variants=("current_model", "nunique"),
            current_mean_error_percent=(spec.current_column, "mean"),
            legacy_mean_error_percent=(spec.legacy_column, "mean"),
            mean_delta_percentage_points=("delta_percentage_points", "mean"),
            mean_absolute_delta_percentage_points=(
                "absolute_delta_percentage_points",
                "mean",
            ),
            max_absolute_delta_percentage_points=(
                "absolute_delta_percentage_points",
                "max",
            ),
        )
        .reset_index()
    )


def save_plot(
    summary: pd.DataFrame,
    spec: MetricSpec,
    output: Path,
    excluded_systems: set[str] | None = None,
) -> None:
    """Plot mean v2-minus-v1 changes for each execution-specific model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = [name for name in SOURCE_DIRS if name in set(summary["source"])]
    fig, axes = plt.subplots(
        len(sources),
        1,
        figsize=(13.0, max(6.0, 0.42 * len(summary))),
        squeeze=False,
        layout="constrained",
    )
    for axis, source in zip(axes[:, 0], sources, strict=True):
        subset = summary[summary["source"] == source].sort_values(
            "mean_delta_percentage_points"
        )
        colors = np.where(subset["mean_delta_percentage_points"] >= 0, "#c44e52", "#4c72b0")
        labels = subset["current_model"].map(display_model_name)
        axis.barh(labels, subset["mean_delta_percentage_points"], color=colors)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_title(source.capitalize())
        axis.set_xlabel(f"Mean v2 - v1 {spec.name} error (percentage points)")
        axis.grid(axis="x", alpha=0.25)
    title = f"Mean v2 - v1 {spec.name} error by model"
    if excluded_systems:
        labels = ", ".join(sorted(system.split("_", maxsplit=1)[0] for system in excluded_systems))
        title += f" (excluding {labels})"
    fig.suptitle(title, fontsize=15, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def run_comparison(
    spec: MetricSpec,
    *,
    v2_results_root: Path,
    legacy_results_dir: Path,
    output_dir: Path,
    sources: list[str],
    make_plot: bool,
    excluded_systems: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run, save, and report a complete comparison for one metric."""
    if not legacy_results_dir.is_dir():
        raise FileNotFoundError(f"Legacy results directory not found: {legacy_results_dir}")
    legacy = spec.legacy_loader(legacy_results_dir).copy()
    legacy["legacy_error"] = pd.to_numeric(legacy["legacy_error"], errors="raise")
    _validate_unique(legacy, ["legacy_model", "system"], f"legacy {spec.name}")

    comparisons: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []
    for source in sources:
        source_dir = v2_results_root / SOURCE_DIRS[source]
        if not source_dir.is_dir():
            raise FileNotFoundError(f"V2 results directory not found: {source_dir}")
        comparison, report = compare_source(
            spec,
            source,
            source_dir,
            legacy,
            excluded_systems,
        )
        comparisons.append(comparison)
        reports.append(report)

    pairwise = pd.concat(comparisons, ignore_index=True)
    by_model = model_summary(pairwise, spec)
    by_system = system_summary(pairwise, spec)

    output_dir.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(output_dir / f"{spec.output_prefix}_pairwise_differences.csv", index=False)
    by_model.to_csv(output_dir / f"{spec.output_prefix}_model_summary.csv", index=False)
    by_system.to_csv(output_dir / f"{spec.output_prefix}_system_summary.csv", index=False)
    with (output_dir / f"{spec.output_prefix}_comparison_manifest.json").open("w") as handle:
        json.dump(
            {
                "metric": spec.name,
                "legacy_results_dir": str(legacy_results_dir.resolve()),
                "v2_results_root": str(v2_results_root.resolve()),
                "legacy_scale_to_percent": spec.legacy_scale,
                "analysis_excluded_systems": sorted(excluded_systems or ()),
                "sources": reports,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    if make_plot:
        save_plot(
            by_model,
            spec,
            output_dir / f"{spec.output_prefix}_mean_delta_by_model.pdf",
            excluded_systems,
        )

    return pairwise, by_model, by_system


def print_summary(spec: MetricSpec, pairwise: pd.DataFrame, by_model: pd.DataFrame) -> None:
    print(f"\n{spec.name} comparison: {len(pairwise)} aligned model/system rows")
    for source, group in by_model.groupby("source", sort=False):
        print(f"\n{source} ({len(group)} model variants)")
        display = group[
            [
                "current_model",
                "n_systems",
                "current_mean_error_percent",
                "legacy_mean_error_percent",
                "mean_delta_percentage_points",
                "mean_absolute_delta_percentage_points",
            ]
        ]
        print(display.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
