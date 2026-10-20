#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "adjustText>=1.3",
#   "matplotlib>=3.9",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "seaborn>=0.13",
# ]
# ///
"""Plot Pareto front for average RDF/VDOS/pressure error vs MD step time.

Objectives:
- minimize average RDF/VDOS/pressure error [%]
- minimize mean MD time per step [ms]

Definitions:
    E_RDF  = RDF error [%]
    E_VDOS = VDOS error [%]
    E_P    = pressure histogram error [%]

where E_P is obtained from the pressure histogram comparison.

The combined error is the arithmetic mean of the three errors:

    E_RPV = (E_RDF + E_VDOS + E_P) / 3

Figure 7 uses the TorchSim eager or accelerated paper cohort.
Use figure_SI_18.py for the accelerated TorchSim/ASE comparison.
Use --data-only to export the model metrics and frontier without plotting.
"""

from __future__ import annotations

import argparse
from _figure_data import pressure_path, require_vdos_metrics_file, vdos_metrics_path
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

import sys

PAPER_V2_CONFIG_DIR = Path(__file__).resolve().parents[1]
if str(PAPER_V2_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_V2_CONFIG_DIR))
from model_display_names import MODEL_DISPLAY_NAMES, display_model_name, normalize_display_key
from md_success import torchsim_md_succeeded
from metric_sources import cohort_timing_files
from system_filters import add_molecular_crystal_option, include_system, filter_pressure_systems
from correlation_labels import inward_label_offset, position_model_labels


FONT_SIZE = 6
LEGEND_FONT_SIZE = 6

plt.rcParams.update(
	{
		"lines.markersize": 4,
		"lines.linewidth": 1.5,
		"font.size": FONT_SIZE,
		"axes.labelsize": FONT_SIZE,
		"axes.titlesize": FONT_SIZE,
		"xtick.labelsize": FONT_SIZE,
		"ytick.labelsize": FONT_SIZE,
		"legend.fontsize": LEGEND_FONT_SIZE,
		"figure.titlesize": FONT_SIZE,
		"axes.grid": True,
		"grid.linewidth": 0.5,
		"grid.alpha": 1.0,
	}
)

palette = sns.color_palette("deep")


CALCULATOR_DISPLAY_NAMES = MODEL_DISPLAY_NAMES


def normalize_model_name(name: str) -> str:
	return str(name).strip().lower()


def metric_model_key(name: str) -> str:
	"""Normalize property/execution tags only when joining metric tables."""
	name = normalize_model_name(name)
	name = name.replace("-force-only", "").replace("-stress", "")
	if name.endswith("-eager"):
		name = name.removesuffix("-eager")
	return normalize_display_key(name)


def display_name(model: str) -> str:
    return display_model_name(model)


TIER_1 = ["chgnet", "mace-mp-0", "mace-mp-0-compile", "grace-mp"]
TIER_2 = ["mace-mpa-0", "mace-mpa-0-compile", "orb-v2"]
TIER_3 = ["mattersim-v1-5M", "grace-oam", "orb-v3-omat", "orb-v3-direct-omat", "eSEN-30M-OAM", "nequip", "eq-v2-M-omat", "pet-oam-xl", "pet-omat-xl", "grace-oam-compiled", "mattersim-v1-5M-compile", "pet-oam-xl-torchscript", "pet-omat-xl-torchscript"]
TIER_4 = ["mace-mh-omat", "mace-mh-omat-compile", "uma-s-omat", "uma-s-omat-compile", "uma-s-omat-turbo", "uma-m-omat", "uma-m-omat-compile", "uma-m-omat-turbo"]


TIER_COLORS = {
	"Tier 1": palette[2],
	"Tier 2": palette[1],
	"Tier 3": palette[3],
	"Tier 4": palette[0],
	"Other": "#757575",
}


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE = "mlip-trajs-torchsim-eager"
SOURCES = (
	"mlip-trajs-torchsim-eager",
	"mlip-trajs-torchsim-accelerated",
)


def source_input_paths(source: str) -> tuple[Path, Path, Path, str]:
	"""Return timing/RDF/VDOS inputs and pressure mode for a V2 source."""
	return (
		SCRIPTS_DIR / "data" / source,
		SCRIPTS_DIR / "rdfs" / "results" / source
		/ "rdf_similarity_scores_same_simulation_length.csv",
		vdos_metrics_path(source),
		"md_accelerated" if source.endswith("-accelerated") else "md_eager",
	)


def source_pressure_backend(source: str) -> str:
	return "torchsim" if "torchsim" in source else "ase"


DEFAULT_PRESSURE_METRICS_FILE = (
	SCRIPTS_DIR / "pressures" / "results" / "model_pressure_error_metric.csv"
)

DEFAULT_OUTPUT_FILE = (
	SCRIPT_DIR
	/ "plots"
	/ "figure_7.pdf"
)

DEFAULT_OUTPUT_CSV = (
	SCRIPT_DIR
	/ "results"
	/ "vdos_rdf_pressure_model_means_merged_average_error_same_simulation_length.csv"
)


def _pick_first_existing(columns: set[str], candidates: list[str]) -> str | None:
	for candidate in candidates:
		if candidate in columns:
			return candidate
	return None


def _as_percent(series: pd.Series) -> pd.Series:
	values = pd.to_numeric(series, errors="coerce")
	finite = values.dropna()
	if finite.empty:
		return values
	if float(finite.max()) <= 1.5:
		return values * 100.0
	return values


def _prepare_pressure_df(
	pressure_df: pd.DataFrame,
	pressure_scale_gpa: float,
	clip_pressure_error: bool,
	backend: str | None = None,
	mode: str | None = None,
) -> pd.DataFrame:
	pressure_df = filter_pressure_systems(pressure_df, include_molecular_crystals=True)
	if backend is not None and "backend" in pressure_df.columns:
		pressure_df = pressure_df.loc[
			pressure_df["backend"].astype(str).str.lower() == backend.lower()
		].copy()
	if mode is not None and "mode" in pressure_df.columns:
		pressure_df = pressure_df.loc[
			pressure_df["mode"].astype(str).str.lower() == mode.lower()
		].copy()
	if pressure_df.empty:
		raise ValueError(
			f"No pressure metrics remain after filtering for backend={backend!r}, mode={mode!r}."
		)

	model_col = _pick_first_existing(set(pressure_df.columns), ["model", "mlip_model", "calculator"])
	if model_col is None:
		raise ValueError("Pressure metrics file must contain one of: model, mlip_model, calculator.")

	pressure_columns = set(pressure_df.columns)

	precomputed_pressure_error_col = _pick_first_existing(
		pressure_columns,
		[
			"final_mean_pressure_error_percent",
			"pressure_error_percent",
			"mean_pressure_error_percent",
			"Pressure Error [%]",
			"pressure_error_%",
		],
	)

	precomputed_pressure_similarity_col = _pick_first_existing(
		pressure_columns,
		[
			"final_mean_pressure_similarity_percent",
			"pressure_similarity_percent",
			"mean_pressure_similarity_percent",
			"Pressure Similarity [%]",
			"final_mean_pressure_similarity",
			"pressure_similarity",
			"mean_pressure_similarity",
		],
	)

	pressure_mae_col = _pick_first_existing(
		pressure_columns,
		[
			"pressure_mae_GPa",
			"Pressure MAE [GPa]",
			"error_GPa",
		],
	)

	if (
		precomputed_pressure_error_col is None
		and precomputed_pressure_similarity_col is None
		and pressure_mae_col is None
	):
		raise ValueError(
			"Pressure metrics file must contain pressure similarity, pressure percentage error, "
			"or pressure MAE. Expected one of: final_mean_pressure_similarity_percent, "
			"pressure_similarity_percent, final_mean_pressure_error_percent, "
			"pressure_error_percent, pressure_mae_GPa, Pressure MAE [GPa], error_GPa."
		)

	cols = [model_col]
	for col in [pressure_mae_col, precomputed_pressure_error_col, precomputed_pressure_similarity_col]:
		if col is not None and col not in cols:
			cols.append(col)

	out = pressure_df[cols].copy()
	out["model"] = out[model_col].map(metric_model_key)

	if pressure_mae_col is not None:
		out["Pressure MAE [GPa]"] = pd.to_numeric(out[pressure_mae_col], errors="coerce")
	else:
		out["Pressure MAE [GPa]"] = np.nan

	if precomputed_pressure_error_col is not None:
		out["Pressure Error [%]"] = _as_percent(out[precomputed_pressure_error_col])
	else:
		out["Pressure Error [%]"] = np.nan

	if precomputed_pressure_similarity_col is not None:
		out["Pressure Similarity [%]"] = _as_percent(out[precomputed_pressure_similarity_col])
	else:
		out["Pressure Similarity [%]"] = np.nan

	if out["Pressure Error [%]"].isna().all() and not out["Pressure Similarity [%]"].isna().all():
		out["Pressure Error [%]"] = 100.0 - out["Pressure Similarity [%]"]

	if out["Pressure Error [%]"].isna().all():
		if pressure_scale_gpa <= 0.0:
			raise ValueError("--pressure-scale-gpa must be positive.")
		out["Pressure Error [%]"] = 100.0 * out["Pressure MAE [GPa]"] / pressure_scale_gpa

	if clip_pressure_error:
		out["Pressure Error [%]"] = out["Pressure Error [%]"].clip(lower=0.0, upper=100.0)
	else:
		out["Pressure Error [%]"] = out["Pressure Error [%]"].clip(lower=0.0)

	out["Pressure Similarity [%]"] = out["Pressure Similarity [%]"].fillna(
		100.0 - out["Pressure Error [%]"]
	)

	if clip_pressure_error:
		out["Pressure Similarity [%]"] = out["Pressure Similarity [%]"].clip(lower=0.0, upper=100.0)
	else:
		out["Pressure Similarity [%]"] = out["Pressure Similarity [%]"].clip(upper=100.0)

	out["Pressure Score [%]"] = out["Pressure Similarity [%]"]

	if clip_pressure_error:
		out["Pressure Score [%]"] = out["Pressure Score [%]"].clip(lower=0.0, upper=100.0)

	out = out.dropna(subset=["model", "Pressure Error [%]"]).copy()

	return out[
		[
			"model",
			"Pressure MAE [GPa]",
			"Pressure Similarity [%]",
			"Pressure Error [%]",
			"Pressure Score [%]",
		]
	]


def compute_combined_error(df: pd.DataFrame) -> pd.Series:
	return (
		df["RDF Error [%]"].astype(float)
		+ df["VDOS Error [%]"].astype(float)
		+ df["Pressure Error [%]"].astype(float)
	) / 3.0


def load_model_avg_timings(timings_dir: Path, include_molecular_crystals: bool = False) -> pd.DataFrame:
	"""Average completed MD timings over systems, retaining native provenance."""
	if not timings_dir.is_dir():
		raise FileNotFoundError(f"Timing directory does not exist: {timings_dir}")
	model_timings: dict[str, list[float]] = {}
	paths = cohort_timing_files(timings_dir)
	all_systems = {path.parent for path in paths if include_system(path.parent.name, include_molecular_crystals)}
	for system_dir in sorted(all_systems):
		for csv_path in paths:
			if csv_path.parent != system_dir:
				continue
			trajectory = csv_path.with_name(f"nvt_{csv_path.stem.removeprefix('md_timing_')}.h5")
			if not torchsim_md_succeeded(trajectory):
				continue
			model_name = metric_model_key(csv_path.stem.removeprefix("md_timing_"))
			try:
				df = pd.read_csv(csv_path)
				if len(df) != 1:
					continue
				sps = float(df["seconds_per_step"].iloc[0])
				if not np.isfinite(sps) or sps <= 0:
					continue
				model_timings.setdefault(model_name, []).append(sps)
			except Exception:
				pass
	rows = [
		{"model": m, "mean_time_per_step_ms": float(np.mean(vals)) * 1000}
		for m, vals in model_timings.items()
	]
	if not rows:
		raise ValueError(f"No valid md_timing_*.csv files found under {timings_dir}")
	return pd.DataFrame(rows)


def apply_matched_runner_timings(
	frame: pd.DataFrame, *, timings_dirs: dict[str, Path],
	include_molecular_crystals: bool = False,
) -> pd.DataFrame:
	"""Pair shared configurations' timing workloads without changing their errors.

	Reuse SI17's native-run validation and PET execution metadata. Configurations
	shown for only one runner keep their ordinary cohort timing; shared ones must
	have at least one successful, matching workload in both runners.
	"""
	from figure_SI_17 import (
		ASE_SOURCE, TORCHSIM_SOURCE, compare_timings, load_native_timings, timing_model_key,
	)

	sources = (TORCHSIM_SOURCE, ASE_SOURCE)
	if set(frame["dataset"]) != set(sources):
		raise ValueError("Matched runner timings require both accelerated TorchSim and ASE cohorts")
	observations = {
		source: load_native_timings(timings_dirs[source], source, include_molecular_crystals)[0]
		for source in sources
	}
	keys = pd.Series(index=frame.index, dtype=object)
	for source in sources:
		aliases: dict[str, set[str]] = {}
		for row in observations[source].itertuples():
			aliases.setdefault(metric_model_key(row.raw_model), set()).add(row.model)
		for index, row in frame.loc[frame["dataset"].eq(source)].iterrows():
			candidates = aliases.get(metric_model_key(row["model"]), {timing_model_key(row["model"])})
			if len(candidates) != 1:
				raise ValueError(f"Ambiguous timing configuration for {source}: {row['model']}")
			keys.at[index] = next(iter(candidates))
	shared = set(keys[frame["dataset"].eq(TORCHSIM_SOURCE)]) & set(keys[frame["dataset"].eq(ASE_SOURCE)])
	paired, summary = compare_timings(observations[TORCHSIM_SOURCE], observations[ASE_SOURCE])
	summary = summary.set_index("model")
	matched_systems = paired.groupby("model")["system"].agg(lambda values: ";".join(sorted(values)))
	out = frame.copy()
	out["n_timing_systems"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
	out["timing_systems"] = ""
	out["timing_basis"] = "runner-specific"
	keep = pd.Series(True, index=out.index)
	for index, row in out.iterrows():
		key = keys.at[index]
		if key not in shared:
			continue
		if key not in summary.index:
			keep.at[index] = False
			continue
		column = "torchsim_mean_ms_per_step" if row["dataset"] == TORCHSIM_SOURCE else "ase_mean_ms_per_step"
		out.at[index, "mean_time_per_step_ms"] = summary.at[key, column]
		out.at[index, "n_timing_systems"] = summary.at[key, "n_common_systems"]
		out.at[index, "timing_systems"] = matched_systems.at[key]
		out.at[index, "timing_basis"] = "matched systems"
	dropped = sorted(set(keys[~keep]))
	if dropped:
		print("[INFO] Excluding shared configurations without matching successful MD workloads: " + ", ".join(dropped))
	return out.loc[keep].reset_index(drop=True)


def exclude_model_configurations(
	frame: pd.DataFrame, excluded_models_by_source: dict[str, set[str]],
) -> pd.DataFrame:
	"""Apply a figure's source-specific exclusions before finding its frontier."""
	keep = pd.Series(True, index=frame.index)
	models = frame["model"].map(metric_model_key)
	for source, excluded in excluded_models_by_source.items():
		keys = {metric_model_key(model) for model in excluded}
		keep &= ~(frame["dataset"].eq(source) & models.isin(keys))
	return frame.loc[keep].reset_index(drop=True)


def load_rdf_vdos_metrics(
	rdf_metrics_file: Path,
	vdos_metrics_file: Path,
) -> pd.DataFrame:
	"""Merge the source-specific RDF and VDOS outputs produced by V2."""
	require_vdos_metrics_file(vdos_metrics_file)
	rdf_df = pd.read_csv(rdf_metrics_file)
	vdos_df = pd.read_csv(vdos_metrics_file)

	rdf_model_col = _pick_first_existing(
		set(rdf_df.columns), ["model", "Calculator", "calculator"]
	)
	rdf_error_col = _pick_first_existing(
		set(rdf_df.columns),
		["rdf_error_percent", "RDF Error [%]", "Mean RDF Error [%]", "rdf_error", "RDF_Error"],
	)
	rdf_similarity_col = _pick_first_existing(
		set(rdf_df.columns),
		["rdf_similarity_percent", "RDF Similarity [%]", "rdf_similarity"],
	)
	if rdf_model_col is None or (rdf_error_col is None and rdf_similarity_col is None):
		raise ValueError(
			f"RDF metrics file {rdf_metrics_file} needs a model column and an RDF error/similarity column."
		)

	vdos_model_col = _pick_first_existing(
		set(vdos_df.columns), ["model", "Calculator", "calculator"]
	)
	vdos_error_col = _pick_first_existing(
		set(vdos_df.columns),
		["vdos_error_percent", "VDOS Error [%]", "Mean VDOS Error [%]", "vdos_error"],
	)
	if vdos_model_col is None or vdos_error_col is None:
		raise ValueError(
			f"VDOS metrics file {vdos_metrics_file} needs a model column and a VDOS error column."
		)

	rdf = pd.DataFrame({"model": rdf_df[rdf_model_col].map(metric_model_key)})
	if rdf_error_col is not None:
		rdf["RDF Error [%]"] = _as_percent(rdf_df[rdf_error_col])
	else:
		rdf["RDF Error [%]"] = 100.0 - _as_percent(rdf_df[rdf_similarity_col])
	rdf["RDF Similarity [%]"] = 100.0 - rdf["RDF Error [%]"]
	rdf = rdf.groupby("model", as_index=False).mean(numeric_only=True)

	vdos = pd.DataFrame(
		{
			"model": vdos_df[vdos_model_col].map(metric_model_key),
			"VDOS Error [%]": _as_percent(vdos_df[vdos_error_col]),
		}
	).groupby("model", as_index=False).mean(numeric_only=True)

	merged = rdf.merge(vdos, on="model", how="inner")
	if merged.empty:
		raise ValueError(
			f"No overlapping models between {rdf_metrics_file} and {vdos_metrics_file}."
		)
	return merged


def load_and_merge(
	timings_dir: Path,
	pressure_metrics_file: Path,
	pressure_scale_gpa: float,
	clip_pressure_error: bool,
	dataset_label: str | None = None,
	combined_metrics_file: Path | None = None,
	rdf_metrics_file: Path | None = None,
	vdos_metrics_file: Path | None = None,
	pressure_backend: str | None = None,
	pressure_mode: str | None = None,
	include_molecular_crystals: bool = False,
) -> pd.DataFrame:
	timings_df = load_model_avg_timings(timings_dir, include_molecular_crystals)
	if combined_metrics_file is not None:
		combined_df = pd.read_csv(combined_metrics_file)
	else:
		if rdf_metrics_file is None or vdos_metrics_file is None:
			raise ValueError(
				"Supply --combined-metrics-file or both --rdf-metrics-file and --vdos-metrics-file."
			)
		combined_df = load_rdf_vdos_metrics(rdf_metrics_file, vdos_metrics_file)
	pressure_df = pd.read_csv(pressure_metrics_file)

	combined_needed = {"model"}
	missing_combined = combined_needed - set(combined_df.columns)
	if missing_combined:
		raise ValueError(f"Missing columns in combined metrics file: {sorted(missing_combined)}")

	combined_columns = set(combined_df.columns)

	rdf_similarity_col = _pick_first_existing(
		combined_columns,
		[
			"rdf_similarity_percent",
			"RDF Similarity [%]",
			"rdf_similarity",
			"rdf_score_percent",
			"RDF Score [%]",
		],
	)

	rdf_error_col = _pick_first_existing(
		combined_columns,
		[
			"rdf_error_percent",
			"RDF Error [%]",
			"rdf_error",
			"rdf_rmse",
		],
	)

	vdos_error_col = _pick_first_existing(
		combined_columns,
		[
			"vdos_error_percent",
			"VDOS Error [%]",
			"vdos_error",
		],
	)

	if rdf_similarity_col is None and rdf_error_col is None:
		raise ValueError(
			"Missing RDF similarity/error column in combined metrics file. Expected one of: "
			"rdf_similarity_percent, RDF Similarity [%], rdf_similarity, rdf_score_percent, "
			"RDF Score [%], rdf_error_percent, RDF Error [%], rdf_error, rdf_rmse"
		)

	if vdos_error_col is None:
		raise ValueError(
			"Missing VDOS error column in combined metrics file. Expected one of: "
			"vdos_error_percent, VDOS Error [%], vdos_error"
		)

	combined_keep_cols = ["model"]
	for col in [rdf_similarity_col, rdf_error_col, vdos_error_col]:
		if col is not None and col not in combined_keep_cols:
			combined_keep_cols.append(col)

	combined_small = combined_df[combined_keep_cols].copy()
	combined_small["model"] = combined_small["model"].map(metric_model_key)
	combined_small = combined_small.groupby("model", as_index=False).mean(numeric_only=True)

	pressure_small = _prepare_pressure_df(
		pressure_df=pressure_df,
		pressure_scale_gpa=pressure_scale_gpa,
		clip_pressure_error=clip_pressure_error,
		backend=pressure_backend,
		mode=pressure_mode,
	)
	for metric, available_models in (
		("RDF/VDOS", set(combined_small["model"])),
		("pressure", set(pressure_small["model"])),
	):
		missing_models = set(timings_df["model"]) - available_models
		if missing_models:
			names = ", ".join(display_name(model) for model in sorted(missing_models))
			print(f"[INFO] Excluding timed models without {metric} metrics ({dataset_label}): {names}")

	merged = pd.merge(
		timings_df,
		combined_small,
		on="model",
		how="inner",
	)

	merged = pd.merge(
		merged,
		pressure_small,
		on="model",
		how="inner",
	)

	merged["mean_time_per_step_ms"] = pd.to_numeric(merged["mean_time_per_step_ms"], errors="coerce")

	if rdf_similarity_col is not None:
		merged["RDF Similarity [%]"] = _as_percent(merged[rdf_similarity_col])
	else:
		merged["RDF Similarity [%]"] = 100.0 - _as_percent(merged[rdf_error_col])

	if rdf_error_col is not None:
		merged["RDF Error [%]"] = _as_percent(merged[rdf_error_col])
	else:
		merged["RDF Error [%]"] = 100.0 - merged["RDF Similarity [%]"]

	merged["VDOS Error [%]"] = _as_percent(merged[vdos_error_col])

	merged = merged.dropna(
		subset=[
			"mean_time_per_step_ms",
			"RDF Error [%]",
			"VDOS Error [%]",
			"Pressure Error [%]",
		]
	).copy()

	if merged.empty:
		raise ValueError(
			"No overlapping models between timings, RDF/VDOS, and pressure metrics files."
		)

	merged["Combined Error [%]"] = compute_combined_error(merged)
	if dataset_label is not None:
		merged["dataset"] = dataset_label

	for _, row in merged.iterrows():
		print(
			f"{display_name(str(row['model']))}: "
			f"RDF Similarity = {row['RDF Similarity [%]']:.2f}%, "
			f"RDF Error = {row['RDF Error [%]']:.2f}%, "
				f"VDOS Error = {row['VDOS Error [%]']:.2f}%, "
			f"Pressure Similarity = {row['Pressure Similarity [%]']:.2f}%, "
			f"Pressure Error = {row['Pressure Error [%]']:.2f}%, "
				f"Combined Error = {row['Combined Error [%]']:.2f}%, "
			f"Time per step = {row['mean_time_per_step_ms']:.2f} ms"
		)

	return merged


def is_pareto_optimal(df: pd.DataFrame) -> np.ndarray:
	"""Return mask for non-dominated points for objectives: min time, min combined error."""
	times = df["mean_time_per_step_ms"].to_numpy()
	combined_error = df["Combined Error [%]"].to_numpy()

	n = len(df)
	pareto = np.ones(n, dtype=bool)

	for i in range(n):
		for j in range(n):
			if i == j:
				continue

			better_or_equal_time = times[j] <= times[i]
			better_or_equal_error = combined_error[j] <= combined_error[i]
			strictly_better_one = (
				(times[j] < times[i])
				or
				(combined_error[j] < combined_error[i])
			)

			if better_or_equal_time and better_or_equal_error and strictly_better_one:
				pareto[i] = False
				break

	return pareto


def mark_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
	"""Mark dominance separately for each MD source in a comparison."""
	result = df.copy()
	result["is_pareto"] = False
	groups = result.groupby("dataset", sort=False) if "dataset" in result else [(None, result)]
	for _, rows in groups:
		result.loc[rows.index, "is_pareto"] = is_pareto_optimal(rows)
	return result


def model_tier(model_name: str) -> str:
	model_name = normalize_model_name(model_name)

	if model_name in map(normalize_model_name, TIER_1):
		return "Tier 1"

	if model_name in map(normalize_model_name, TIER_2):
		return "Tier 2"

	if model_name in map(normalize_model_name, TIER_3):
		return "Tier 3"

	if model_name in map(normalize_model_name, TIER_4):
		return "Tier 4"

	return "Other"


def source_label(source: str) -> str:
	return {
		"mlip-trajs-torchsim-accelerated": "TorchSim",
		"mlip-trajs-ase-accelerated": "ASE",
		"mlip-trajs-torchsim-eager": "TorchSim eager",
		"mlip-trajs-ase": "ASE eager",
	}.get(source, str(source))


def comparison_layout(sources: list[str], layout: str | None = None) -> str:
	if layout is not None:
		return layout
	accelerated_pair = {"mlip-trajs-torchsim-accelerated", "mlip-trajs-ase-accelerated"}
	return "overlay" if set(sources) == accelerated_pair else "panels"


def plot_pareto(
	df: pd.DataFrame, output_file: Path, *, layout: str | None = None,
	font_size: float = FONT_SIZE, label_distance_points: int = 6,
) -> None:
	all_df = df.copy()
	if "dataset" not in all_df.columns:
		all_df["dataset"] = "dataset"
	all_df["tier"] = all_df["model"].map(model_tier)
	datasets = all_df["dataset"].drop_duplicates().tolist()
	overlay = comparison_layout(datasets, layout) == "overlay" and len(datasets) > 1
	n_panels = 1 if overlay else len(datasets)
	y_values = all_df["Combined Error [%]"].to_numpy()
	y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))
	y_padding = max(0.75, (y_max - y_min) * 0.08)
	y_bottom, y_top = y_min - y_padding, y_max + y_padding

	fig, axes = plt.subplots(
		1,
		n_panels,
		figsize=(3.53 * (2 if overlay else 1.5 * n_panels), 3.53 * 1.5),
		sharey=True,
		squeeze=False,
	)
	axes = axes[0]

	for panel_index, dataset_name in enumerate(datasets):
		ax = axes[0] if overlay else axes[panel_index]
		dataset_df = all_df[all_df["dataset"] == dataset_name]
		pareto_df = dataset_df[is_pareto_optimal(dataset_df)].sort_values("mean_time_per_step_ms")
		style_index = {"mlip-trajs-torchsim-accelerated": 0, "mlip-trajs-ase-accelerated": 1}.get(dataset_name, panel_index)
		marker = ("o", "^", "s", "D")[style_index % 4] if overlay else "o"
		linestyle = ("-", "--", ":", "-.")[style_index % 4] if overlay else "-"

		for tier_name in ["Tier 1", "Tier 2", "Tier 3", "Tier 4", "Other"]:
			tier_df = dataset_df[dataset_df["tier"] == tier_name]
			if tier_df.empty:
				continue
			ax.scatter(
				tier_df["mean_time_per_step_ms"],
				tier_df["Combined Error [%]"],
				color=TIER_COLORS[tier_name],
				marker=marker,
				alpha=0.9,
				s=26,
				label=tier_name,
				zorder=2,
			)

		ax.scatter(
			pareto_df["mean_time_per_step_ms"],
			pareto_df["Combined Error [%]"],
			facecolors="none",
			edgecolors="black",
			marker=marker,
			alpha=0.95,
			s=58,
			linewidths=1.0,
			label="Pareto-optimal",
			zorder=4,
		)
		ax.plot(
			pareto_df["mean_time_per_step_ms"],
			pareto_df["Combined Error [%]"],
			color="black",
			linestyle=linestyle,
			linewidth=1.2,
			alpha=0.9,
			zorder=3,
		)

		x_vals = dataset_df["mean_time_per_step_ms"].to_numpy()
		y_vals = dataset_df["Combined Error [%]"].to_numpy()
		axis_times = all_df["mean_time_per_step_ms"].to_numpy() if overlay else x_vals
		ax.set_xlim(left=0, right=max(1.0, float(np.nanmax(axis_times)) * 1.1))
		ax.set_ylim(bottom=y_bottom, top=y_top)
		for xi, yi, model_name in zip(x_vals, y_vals, dataset_df["model"].values):
			offset, horizontal, vertical = inward_label_offset(ax, float(xi))
			label = ax.annotate(
				display_name(str(model_name)),
				xy=(xi, yi),
				xytext=offset,
				textcoords="offset points",
				fontsize=font_size,
				alpha=0.8,
				ha=horizontal,
				va=vertical,
				bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.55),
			)
			label._model_point_label = True

		ax.set_xlabel(r"Mean time per step [ms]", fontsize=font_size)
		ax.tick_params(axis="both", which="both", labelsize=font_size)
		ax.xaxis.get_offset_text().set_fontsize(font_size)
		ax.yaxis.get_offset_text().set_fontsize(font_size)
		if len(datasets) > 1 and not overlay:
			ax.text(
				0.02, 0.98, f"{chr(97 + panel_index)}) {source_label(dataset_name)}",
				transform=ax.transAxes, ha="left", va="top", fontsize=font_size,
			)
		ax.grid(True, linestyle="--", alpha=0.4)
		if not overlay:
			ax.legend(loc="best", frameon=True, fontsize=font_size)

	if overlay:
		handles = [
			Line2D([], [], color=TIER_COLORS[tier], marker="o", linestyle="none", label=tier)
			for tier in TIER_COLORS if tier in set(all_df["tier"])
		]
		for index, dataset_name in enumerate(datasets):
			style_index = {"mlip-trajs-torchsim-accelerated": 0, "mlip-trajs-ase-accelerated": 1}.get(dataset_name, index)
			handles.append(Line2D(
				[], [], color="black", marker=("o", "^", "s", "D")[style_index % 4],
				linestyle=("-", "--", ":", "-.")[style_index % 4], markerfacecolor="none",
				label=f"{source_label(dataset_name)} (Pareto frontier)",
			))
		axes[0].legend(handles=handles, loc="best", frameon=True, fontsize=font_size)

	axes[0].set_ylabel(r"$\bar{E}_{RPV}$ [%]", fontsize=font_size)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	position_model_labels(
		fig, axes, max_offset_points=label_distance_points,
		max_distance_points=label_distance_points,
	)
	fig.savefig(output_file, bbox_inches="tight", pad_inches=0.02)
	plt.close(fig)

	pareto_count = sum(len(dataset_df[is_pareto_optimal(dataset_df)]) for dataset_df in (all_df[all_df["dataset"] == name] for name in datasets))
	print(f"Saved: {output_file}")
	print(f"Models plotted: {len(all_df)}")
	print(f"Pareto-optimal models: {pareto_count}")


def run_pareto_figure(
	*, figure_name: str, allowed_sources: tuple[str, ...],
	default_sources: tuple[str, ...], required_sources: tuple[str, ...] | None = None,
	argv: list[str] | None = None, font_size: float = FONT_SIZE,
	label_distance_points: int = 6,
	matched_runner_timings: bool = False,
	excluded_models_by_source: dict[str, set[str]] | None = None,
) -> None:
	"""Share metric loading, independent frontiers and styling across Figures 7/SI18."""
	parser = argparse.ArgumentParser(
		description=(
			"Plot Pareto front of average RDF/VDOS/pressure error "
			"vs mean MD step time."
		)
	)

	parser.add_argument(
		"--source",
		choices=allowed_sources,
		action="append",
		help=f"Trajectory source (default: {', '.join(default_sources)}).",
	)
	parser.add_argument(
		"--layout", choices=("overlay", "panels"), default=None,
		help="Comparison layout (default: overlay for accelerated TorchSim/ASE, panels otherwise).",
	)

	parser.add_argument(
		"--timings-dir",
		default=None,
		help="Directory containing per-system md_timing_*.csv files (default: data/<source>).",
	)

	parser.add_argument(
		"--combined-metrics-file",
		default=None,
		help=(
			"Optional pre-merged RDF/VDOS CSV. When omitted, the V2 RDF and VDOS "
			"outputs are merged directly."
		),
	)

	parser.add_argument(
		"--rdf-metrics-file",
		default=None,
		help="RDF model metrics CSV (default: rdfs/results/<source>/...).",
	)

	parser.add_argument(
		"--vdos-metrics-file",
		default=None,
		help="VDOS model metrics CSV for one source (default: vdos/results/<source>/...).",
	)
	parser.add_argument(
		"--vdos-results-dir", type=Path, default=None,
		help="VDOS results root for all selected sources (default: vdos/results; select vdos_legacy/results explicitly for a legacy comparison).",
	)

	parser.add_argument(
		"--pressure-metrics-file",
		action="append",
		help=(
			"CSV with pressure histogram similarity/error by model. Expected columns include "
			"model or mlip_model plus final_mean_pressure_similarity_percent, "
			"pressure_similarity_percent, final_mean_pressure_error_percent, or "
			"pressure_error_percent. MAE columns are still accepted as a fallback."
		),
	)

	parser.add_argument(
		"--pressure-backend",
		default=None,
		help="Pressure backend to select (default: inferred from --source).",
	)

	parser.add_argument(
		"--pressure-mode",
		default=None,
		help="Pressure mode to select (default: md_eager or md_accelerated from --source).",
	)

	parser.add_argument(
		"--pressure-scale-gpa",
		type=float,
		default=10.0,
		help=(
			"Pressure MAE scale P0 in GPa. "
			"Default: 10 GPa corresponds to 100 percent pressure error."
		),
	)

	parser.add_argument(
		"--no-pressure-clip",
		action="store_true",
		help=(
			"Do not clip pressure error to 100 percent. "
			"By default, pressure error is clipped to [0, 100]."
		),
	)

	parser.add_argument(
		"--output-file",
		default=None,
		help=f"Output plot path (default: plots/<source or comparison>/{figure_name}.pdf).",
	)

	parser.add_argument(
		"--output-csv",
		default=None,
		help=f"Model metrics CSV (default: results/<source or comparison>/{figure_name}_metrics.csv).",
	)
	parser.add_argument(
		"--frontier-csv", default=None,
		help=f"Pareto-optimal rows CSV (default: {figure_name}_frontier.csv beside the metrics CSV).",
	)
	parser.add_argument(
		"--data-only", action="store_true",
		help="Export model metrics and the Pareto frontier without creating a figure.",
	)

	add_molecular_crystal_option(parser)
	args = parser.parse_args(argv)
	sources = list(dict.fromkeys(args.source or default_sources))
	if required_sources is not None and set(sources) != set(required_sources):
		parser.error(f"{figure_name} requires both sources: {', '.join(required_sources)}")
	if args.vdos_metrics_file is not None and args.vdos_results_dir is not None:
		parser.error("--vdos-metrics-file and --vdos-results-dir cannot be used together")
	if len(sources) > 1 and any(
		value is not None
		for value in [args.timings_dir, args.combined_metrics_file, args.rdf_metrics_file, args.vdos_metrics_file]
	):
		parser.error("custom timing/RDF/VDOS inputs are only supported for one source")
	if args.pressure_metrics_file and len(args.pressure_metrics_file) not in {1, len(sources)}:
		parser.error("--pressure-metrics-file must be supplied once or once per --source")

	merged_frames = []
	timings_dirs = {}
	for index, source in enumerate(sources):
		default_timings, default_rdf, default_vdos, default_pressure_mode = source_input_paths(source)
		timings_dirs[source] = Path(args.timings_dir) if args.timings_dir else default_timings
		vdos_file = (
			Path(args.vdos_metrics_file) if args.vdos_metrics_file
			else vdos_metrics_path(source, args.vdos_results_dir) if args.vdos_results_dir is not None
			else default_vdos
		)
		if not args.combined_metrics_file:
			require_vdos_metrics_file(vdos_file)
		pressure_file = (
			Path(args.pressure_metrics_file[index if len(args.pressure_metrics_file or []) > 1 else 0])
			if args.pressure_metrics_file
			else pressure_path(source)
		)
		merged_frames.append(
			load_and_merge(
				timings_dir=Path(args.timings_dir) if args.timings_dir else default_timings,
				combined_metrics_file=Path(args.combined_metrics_file) if args.combined_metrics_file else None,
				rdf_metrics_file=Path(args.rdf_metrics_file) if args.rdf_metrics_file else default_rdf,
				vdos_metrics_file=vdos_file,
				pressure_metrics_file=pressure_file,
				pressure_scale_gpa=args.pressure_scale_gpa,
				clip_pressure_error=not args.no_pressure_clip,
				pressure_backend=args.pressure_backend or source_pressure_backend(source),
				pressure_mode=args.pressure_mode or default_pressure_mode,
				dataset_label=source,
				include_molecular_crystals=args.include_molecular_crystals,
			)
		)
	merged = pd.concat(merged_frames, ignore_index=True)
	if excluded_models_by_source:
		merged = exclude_model_configurations(merged, excluded_models_by_source)
	if matched_runner_timings:
		merged = apply_matched_runner_timings(
			merged, timings_dirs=timings_dirs,
			include_molecular_crystals=args.include_molecular_crystals,
		)
	merged = mark_pareto_frontier(merged)

	accelerated_pair = {"mlip-trajs-torchsim-accelerated", "mlip-trajs-ase-accelerated"}
	output_group = "torchsim_vs_ase_accelerated" if set(sources) == accelerated_pair else "__".join(sources)
	output_csv = Path(args.output_csv) if args.output_csv else SCRIPT_DIR / "results" / output_group / f"{figure_name}_metrics.csv"
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	merged.to_csv(output_csv, index=False)
	print(f"Saved merged RDF/VDOS/pressure average-error metrics: {output_csv}")
	frontier = merged.loc[merged["is_pareto"]].sort_values(["dataset", "mean_time_per_step_ms"])
	frontier_csv = Path(args.frontier_csv) if args.frontier_csv else output_csv.with_name(f"{figure_name}_frontier.csv")
	frontier_csv.parent.mkdir(parents=True, exist_ok=True)
	frontier.to_csv(frontier_csv, index=False)
	print(f"Saved Pareto frontier: {frontier_csv}")
	print(frontier[["dataset", "model", "mean_time_per_step_ms", "Combined Error [%]"]].to_string(index=False))
	if args.data_only:
		return

	output_file = Path(args.output_file) if args.output_file else SCRIPT_DIR / "plots" / output_group / f"{figure_name}.pdf"
	plot_pareto(
		merged, output_file, layout=comparison_layout(sources, args.layout),
		font_size=font_size, label_distance_points=label_distance_points,
	)


def main() -> None:
	run_pareto_figure(
		figure_name="figure_7", allowed_sources=SOURCES,
		default_sources=(DEFAULT_SOURCE,),
	)


if __name__ == "__main__":
	main()
