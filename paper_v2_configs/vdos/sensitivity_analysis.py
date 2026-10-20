#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["ase>=3.26", "h5py>=3.11", "numpy>=1.26", "pandas>=2.2", "scipy>=1.13"]
# ///
"""Measure Welch-window, weighting and frequency-grid sensitivity without plots.

The frozen first corrected analysis supplies the model/system population and
failure mask. Every successful trajectory is loaded once, then processed with
all settings. No variant is selected by how small its benchmark error becomes.
Overlapping Welch segments are correlated: setting-to-setting score changes
below are sensitivity diagnostics, not statistical confidence intervals.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from metric_sources import cohort_trajectory_path  # noqa: E402
from md_success import torchsim_md_succeeded  # noqa: E402
from model_display_names import normalize_display_key  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("sensitivity_vdos_core", HERE / "get_normalized_VDOS.py")
vdos = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vdos)

PAIR_OUTPUT = "vdos_pair_errors_ev_normalized_same_simulation_length.csv"
DEFAULT_SOURCE = "mlip-trajs-torchsim-accelerated"
DEFAULT_BASELINE = HERE / "archive/periodogram_velocity_v1/results"


@dataclass(frozen=True)
class Configuration:
    name: str
    window_fs: float = 2000.0
    min_segments: int = 4
    weighting: str = "mass"
    pad_factor: int = 4


CONFIGURATIONS = (
    Configuration("mass_2ps_min4_pad4"),
    Configuration("mass_1ps_min4_pad4", window_fs=1000.0),
    Configuration("mass_4ps_min4_pad4", window_fs=4000.0),
    Configuration("mass_2ps_min8_pad4", min_segments=8),
    Configuration("velocity_2ps_min4_pad4", weighting="velocity"),
    Configuration("mass_2ps_min4_pad8", pad_factor=8),
)
BASELINE_NAME = CONFIGURATIONS[0].name


def validate_population(pairs: pd.DataFrame) -> pd.DataFrame:
    """Validate the frozen population without inferring failures from scores."""
    required = {"source", "model", "system", "failure_reason", "vdos_error_percent", "matched_frames"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"Missing population columns: {sorted(missing)}")
    if pairs.empty or pairs.duplicated(["source", "model", "system"]).any():
        raise ValueError("Expected a nonempty, unique source/model/system population")
    pairs = pairs.copy()
    pairs["failure_reason"] = pairs.failure_reason.fillna("")
    failures = pairs.failure_reason.ne("")
    if not pairs.loc[failures, "vdos_error_percent"].eq(100.0).all():
        raise ValueError("Frozen failures must carry 100% penalties")
    return pairs


def aggregate_diagnostics(
    results: pd.DataFrame,
    *,
    configurations: tuple[Configuration, ...] = CONFIGURATIONS,
    baseline_name: str = BASELINE_NAME,
) -> tuple[pd.DataFrame, dict]:
    """Keep benchmark denominators fixed and quantify empirical sensitivity."""
    names = [config.name for config in configurations]
    if len(set(names)) != len(names) or baseline_name not in names:
        raise ValueError("Configurations need unique names and a baseline")
    keys = ["source", "model", "system"]
    if results.duplicated(keys + ["configuration"]).any():
        raise ValueError("Duplicate configuration/model/system scores")
    population = results.groupby(keys, sort=True).configuration.agg(set)
    if population.empty or not population.map(lambda values: values == set(names)).all():
        raise ValueError("Every model/system pair must have all configurations")
    errors = results.vdos_error_percent.to_numpy(float)
    if not np.isfinite(errors).all() or np.any((errors < 0) | (errors > 100)):
        raise ValueError("Errors must be finite percentages between 0 and 100")
    failure = results.failure_reason.fillna("").ne("")
    if not results.loc[failure, "vdos_error_percent"].eq(100).all():
        raise ValueError("Failure penalties changed between configurations")
    failed_consistency = results.assign(is_failure=failure).groupby(keys).is_failure.nunique()
    if not failed_consistency.eq(1).all():
        raise ValueError("Failure mask changed between configurations")

    models = results.assign(is_failure=failure).groupby(
        ["source", "configuration", "model"], as_index=False,
    ).agg(
        vdos_error_percent=("vdos_error_percent", "mean"),
        n_systems=("system", "nunique"),
        n_failure_penalties=("is_failure", "sum"),
    )
    models["rank"] = models.groupby(["source", "configuration"]).vdos_error_percent.rank(method="average")
    baseline = models.loc[models.configuration.eq(baseline_name)].set_index(["source", "model"])
    model_keys = pd.MultiIndex.from_frame(models[["source", "model"]])
    models["delta_from_baseline_pp"] = models.vdos_error_percent.to_numpy() - baseline.vdos_error_percent.reindex(model_keys).to_numpy()
    models["rank_change_from_baseline"] = models["rank"].to_numpy() - baseline["rank"].reindex(model_keys).to_numpy()

    comparisons = {}
    for name in names:
        current = models.loc[models.configuration.eq(name)].set_index(["source", "model"]).reindex(baseline.index)
        absolute = current.delta_from_baseline_pp.abs()
        coefficient = spearmanr(baseline.vdos_error_percent, current.vdos_error_percent).statistic
        comparisons[name] = {
            "rank_spearman_vs_baseline": float(coefficient) if np.isfinite(coefficient) else None,
            "median_absolute_model_error_change_pp": float(absolute.median()),
            "maximum_absolute_model_error_change_pp": float(absolute.max()),
            "mean_signed_model_error_change_pp": float(current.delta_from_baseline_pp.mean()),
            "maximum_absolute_rank_change": float(current.rank_change_from_baseline.abs().max()),
        }

    # Window sensitivity excludes the mass-convention and grid probes, which
    # answer different questions from the physical averaging bandwidth.
    window_names = [config.name for config in configurations if config.weighting == "mass" and config.pad_factor == 4]
    successful = results.loc[~failure]
    by_system = {}
    for system, values in successful.groupby("system", sort=True):
        score_grid = values.loc[values.configuration.isin(window_names)].pivot(
            index=["source", "model"], columns="configuration", values="vdos_error_percent",
        )
        spread = score_grid.max(axis=1) - score_grid.min(axis=1)
        details = {}
        for name, variant in values.groupby("configuration", sort=True):
            details[name] = {
                "minimum_segment_duration_fs": float(variant.segment_duration_fs.min()),
                "maximum_segment_duration_fs": float(variant.segment_duration_fs.max()),
                "minimum_n_segments": int(variant.n_segments.min()),
                "maximum_n_segments": int(variant.n_segments.max()),
                "minimum_resolution_cm_inverse": float(variant.resolution_cm_inverse.min()),
                "maximum_resolution_cm_inverse": float(variant.resolution_cm_inverse.max()),
            }
        by_system[system] = {
            "successful_models": int(values.model.nunique()),
            "median_window_score_range_pp": float(spread.median()),
            "maximum_window_score_range_pp": float(spread.max()),
            "settings": details,
        }
    summary = {
        "baseline_configuration": baseline_name,
        "configurations": [asdict(config) for config in configurations],
        "sources": sorted(results.source.unique()),
        "models": int(results[["source", "model"]].drop_duplicates().shape[0]),
        "systems": int(results.system.nunique()),
        "pairs_per_configuration": int(len(population)),
        "failure_penalties_per_configuration": int(failure.sum() // len(names)),
        "comparisons": comparisons,
        "systems_diagnostics": by_system,
        "interpretation": (
            "Identical settings are applied to matched reference/MLIP trajectories. "
            "Window and segment-count changes quantify estimator sensitivity; the weighting "
            "probe changes the observable definition. Zero-padding changes the numerical "
            "integration grid, not physical resolution. Segment overlap induces correlation. "
            "These diagnostics are not confidence intervals or proof of spectral convergence, "
            "and configurations are not selected to minimize benchmark error."
        ),
    }
    return models.sort_values(["source", "configuration", "model"]), summary


def run_sensitivity(
    *,
    source: str,
    data_dir: Path,
    baseline_results: Path,
    output_dir: Path,
    selected_systems: set[str] | None = None,
    selected_models: set[str] | None = None,
    chunk_size: int = 32,
) -> dict:
    pairs = validate_population(pd.read_csv(baseline_results / source / PAIR_OUTPUT))
    if selected_systems is not None:
        pairs = pairs.loc[pairs.system.isin(selected_systems)]
    if selected_models is not None:
        pairs = pairs.loc[pairs.model.isin(selected_models)]
    if pairs.empty:
        raise ValueError("Filters selected no model/system pairs")
    metadata = vdos.load_metadata(data_dir / "ref-trajs/md_metadata.json")
    rows = []
    for system, system_pairs in pairs.groupby("system", sort=True):
        print(f"System: {system} ({len(system_pairs)} models)", flush=True)
        successful = system_pairs.failure_reason.eq("")
        reference_positions, numbers, masses = None, None, None
        reference_cache = {}
        if successful.any():
            path = data_dir / "ref-trajs" / system / "traj.extxyz"
            reference_positions = vdos.load_reference_positions(path)
            numbers = vdos.load_reference_atomic_numbers(path)
            masses = vdos.standard_atomic_masses(numbers)
        for pair in system_pairs.itertuples(index=False):
            fixed = {
                "source": source, "model": pair.model, "system": system,
                "failure_reason": pair.failure_reason,
                "system_type": getattr(pair, "system_type", ""),
            }
            if pair.failure_reason:
                rows.extend({
                    **fixed, **asdict(config), "configuration": config.name,
                    "vdos_error_percent": 100.0, "matched_frames": 0,
                    "segment_duration_fs": np.nan, "n_segments": 0,
                    "resolution_cm_inverse": np.nan, "bin_spacing_cm_inverse": np.nan,
                } for config in CONFIGURATIONS)
                continue
            path = cohort_trajectory_path(data_dir, source, system, pair.model)
            if not torchsim_md_succeeded(path):
                raise ValueError(f"Frozen successful run is no longer successful: {path}")
            n_matched = min(len(reference_positions), vdos.h5_frame_count(path))
            if n_matched != pair.matched_frames:
                raise ValueError(f"Matched frame count changed for {system}/{pair.model}")
            samples, is_velocity = vdos.load_mlip_samples(path, n_matched)
            mlip_numbers = vdos.load_mlip_atomic_numbers(path, n_atoms=samples.shape[1])
            if samples.shape[1:] != reference_positions.shape[1:] or not np.array_equal(mlip_numbers, numbers):
                raise ValueError(f"Reference/MLIP atoms differ for {system}/{pair.model}")
            timestep = float(metadata[system]["timestep"])
            for config in CONFIGURATIONS:
                kwargs = {
                    "samples_are_velocities": False, "estimator": "welch",
                    "weighting": config.weighting, "masses": masses,
                    "window_fs": config.window_fs, "min_segments": config.min_segments,
                    "overlap": 0.5, "pad_factor": config.pad_factor, "chunk_size": chunk_size,
                }
                key = (config.name, n_matched)
                if key not in reference_cache:
                    reference_cache[key] = vdos.compute_vdos(reference_positions[:n_matched], timestep, **kwargs)
                kwargs["samples_are_velocities"] = is_velocity
                spectrum = vdos.compute_vdos(samples, timestep, **kwargs)
                error, _ = vdos.vdos_similarity(reference_cache[key], spectrum)
                settings = vdos.spectral_settings(
                    n_matched - 2, timestep, estimator="welch", window_fs=config.window_fs,
                    min_segments=config.min_segments, overlap=0.5, pad_factor=config.pad_factor,
                )
                rows.append({
                    **fixed, **asdict(config), "configuration": config.name,
                    "vdos_error_percent": error, "matched_frames": n_matched,
                    "segment_duration_fs": settings["segment_duration_fs"],
                    "n_segments": settings["n_segments"],
                    "resolution_cm_inverse": settings["resolution_cm-1"],
                    "bin_spacing_cm_inverse": settings["bin_spacing_cm-1"],
                })
            print(f"  {pair.model}: all {len(CONFIGURATIONS)} settings scored", flush=True)
    results = pd.DataFrame(rows).sort_values(["source", "model", "system", "configuration"])
    models, summary = aggregate_diagnostics(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "per_system.csv", index=False)
    models.to_csv(output_dir / "model_means.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "systems_diagnostics"}, indent=2), flush=True)
    return summary


def pareto_mask(times: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """Retain non-dominated configurations, including exact duplicate points."""
    times, errors = np.asarray(times, float), np.asarray(errors, float)
    if times.ndim != 1 or errors.shape != times.shape or not np.isfinite([times, errors]).all():
        raise ValueError("Pareto objectives must be equal-length finite vectors")
    no_worse = (times[None, :] <= times[:, None]) & (errors[None, :] <= errors[:, None])
    strictly_better = (times[None, :] < times[:, None]) | (errors[None, :] < errors[:, None])
    return ~(no_worse & strictly_better).any(axis=1)


def export_paper_impact(output_dir: Path, active_results: Path) -> dict:
    """Vary VDOS only in the current arithmetic RPV frontier/correlations."""
    models = pd.read_csv(output_dir / "model_means.csv")
    pairs = pd.read_csv(output_dir / "per_system.csv")
    source_names = models.source.unique()
    if len(source_names) != 1:
        raise ValueError("Paper-impact export expects one cohort")
    source = source_names[0]
    active_directory = active_results / source
    pareto_inputs = pd.read_csv(active_directory / "figure_7_metrics.csv")
    correlation_inputs = pd.read_csv(active_directory / "stable_correlation_metrics.csv")
    models["key"] = models.model.map(normalize_display_key)
    pareto_inputs["key"] = pareto_inputs.model.map(normalize_display_key)
    correlation_inputs["key"] = correlation_inputs.calculator.map(normalize_display_key)
    if pareto_inputs.key.duplicated().any() or correlation_inputs.key.duplicated().any():
        raise ValueError("Ambiguous normalized model keys in active plotting inputs")
    if set(models.key) != set(pareto_inputs.key) or set(models.key) != set(correlation_inputs.key):
        raise ValueError("Sensitivity and active plots contain different model populations")

    frontier_rows, correlation_rows, joined_rows = [], [], []
    for name in models.configuration.unique():
        variant = models.loc[models.configuration.eq(name)]
        joined = pareto_inputs.drop(columns=["VDOS Error [%]", "Combined Error [%]", "is_pareto"]).merge(
            variant[["key", "vdos_error_percent"]], on="key", validate="one_to_one",
        ).rename(columns={"vdos_error_percent": "VDOS Error [%]"})
        joined["Combined Error [%]"] = joined[["RDF Error [%]", "VDOS Error [%]", "Pressure Error [%]"]].mean(axis=1)
        joined["is_pareto"] = pareto_mask(joined.mean_time_per_step_ms, joined["Combined Error [%]"])
        joined["configuration"] = name
        if name == BASELINE_NAME:
            original = pareto_inputs.set_index("key").reindex(joined.key)
            if not np.allclose(joined["Combined Error [%]"], original["Combined Error [%]"], rtol=0, atol=1e-9):
                raise ValueError("Sensitivity baseline does not reproduce current combined scores")
            if not np.array_equal(joined.is_pareto.to_numpy(), original.is_pareto.to_numpy()):
                raise ValueError("Sensitivity baseline does not reproduce the current frontier")
        frontier_rows.append(joined.drop(columns="key"))

        current_pairs = pairs.loc[pairs.configuration.eq(name) & pairs.failure_reason.fillna("").eq("")].copy()
        current_pairs["key"] = current_pairs.model.map(normalize_display_key)
        stable_means = current_pairs.groupby("key", as_index=False).agg(
            vdos_error_percent=("vdos_error_percent", "mean"),
            n_stable_systems=("system", "nunique"),
        )
        correlations = correlation_inputs.drop(columns="vdos_error_percent").merge(stable_means, on="key", validate="one_to_one")
        if len(correlations) != len(correlation_inputs):
            raise ValueError("Stable-system filtering removed an active correlation model")
        correlations["configuration"] = name
        if name == BASELINE_NAME:
            original = correlation_inputs.set_index("key").reindex(correlations.key)
            if not np.allclose(correlations.vdos_error_percent, original.vdos_error_percent, rtol=0, atol=1e-9):
                raise ValueError("Sensitivity baseline does not reproduce stable VDOS means")
        joined_rows.append(correlations.drop(columns="key"))
        for x in ("force_rmse", "f1_score", "ksrme_score"):
            valid = correlations[[x, "vdos_error_percent"]].dropna()
            correlation_rows.append({
                "source": source, "configuration": name, "x_metric": x,
                "y_metric": "vdos_error_percent", "n_models": len(valid),
                "pearson_r": float(valid[x].corr(valid.vdos_error_percent)),
            })
    frontier_data = pd.concat(frontier_rows, ignore_index=True)
    coefficients = pd.DataFrame(correlation_rows)
    frontier_data.to_csv(output_dir / "combined_pareto.csv", index=False)
    pd.concat(joined_rows, ignore_index=True).to_csv(output_dir / "stable_correlation_metrics.csv", index=False)
    coefficients.to_csv(output_dir / "stable_correlation_coefficients.csv", index=False)

    baseline_frontier = set(frontier_data.loc[frontier_data.configuration.eq(BASELINE_NAME) & frontier_data.is_pareto, "model"])
    summary = {
        "source": source, "baseline_configuration": BASELINE_NAME,
        "fixed_inputs": "RDF, pressure, timing, stable-MD force RMSE and Matbench scores from active plotting inputs",
        "frontiers": {}, "stable_vdos_correlations": {},
        "interpretation": (
            "Only VDOS varies. Arithmetic RPV means include unchanged 100% failure penalties; "
            "stable-MD correlations exclude rows by failure_reason and retain genuine stable 100% scores. "
            "Frontier membership changes among near-dominated configurations are estimator-sensitive; "
            "they are not evidence for a statistically significant model difference."
        ),
    }
    for name, group in frontier_data.groupby("configuration", sort=False):
        current_frontier = set(group.loc[group.is_pareto, "model"])
        summary["frontiers"][name] = {
            "n_frontier_configurations": len(current_frontier),
            "models": sorted(current_frontier),
            "added_vs_baseline": sorted(current_frontier - baseline_frontier),
            "removed_vs_baseline": sorted(baseline_frontier - current_frontier),
        }
    for name, group in coefficients.groupby("configuration", sort=False):
        summary["stable_vdos_correlations"][name] = {
            row.x_metric: {"pearson_r": row.pearson_r, "n_models": row.n_models}
            for row in group.itertuples(index=False)
        }
    (output_dir / "paper_impact_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "data")
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/sensitivity")
    parser.add_argument("--system", action="append", dest="systems")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--impact-only", action="store_true", help="reuse sensitivity scores and export Pareto/correlation impact only")
    parser.add_argument("--active-results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if args.impact_only:
        export_paper_impact(args.output_dir.resolve(), args.active_results.resolve())
        return
    run_sensitivity(
        source=args.source, data_dir=args.data_dir.resolve(),
        baseline_results=args.baseline_results.resolve(), output_dir=args.output_dir.resolve(),
        selected_systems=set(args.systems) if args.systems else None,
        selected_models=set(args.models) if args.models else None, chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
