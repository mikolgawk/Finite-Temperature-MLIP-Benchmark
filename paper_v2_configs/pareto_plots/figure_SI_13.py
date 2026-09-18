#!/usr/bin/env python3
"""Plot Pareto front for combined RDF/VDOS/pressure error vs MD step time.

Objectives:
- minimize combined RDF/VDOS/pressure error [%]
- minimize mean MD time per step [ms]

Definitions:
    E_RDF  = RDF error [%]
    E_VDOS = VDOS error [%]
    E_P    = pressure histogram error [%] = 100 * (1 - S_P)

where S_P is the pressure histogram similarity score computed from
area-normalized reference and MLIP pressure histograms.

The combined error is the Lehmer mean of the three errors:

    L_RPV = (E_RDF^2 + E_P^2 + E_VDOS^2) / (E_RDF + E_P + E_VDOS)

This penalizes the largest error component more strongly than a simple arithmetic mean.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _figure_data import DEFAULT_PRESSURE_FILE, DEFAULT_SOURCE, SOURCES
from figure_7 import (
	load_and_merge as load_average_inputs,
	load_model_avg_timings as load_repo_timings,
	source_input_paths,
	source_pressure_backend,
)

import sys

PAPER_V2_CONFIG_DIR = Path(__file__).resolve().parents[1]
if str(PAPER_V2_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_V2_CONFIG_DIR))
from model_display_names import MODEL_DISPLAY_NAMES, display_model_name

try:
	from adjustText import adjust_text
except Exception:
	adjust_text = None


FONT_SIZE = 6

plt.rcParams.update(
	{
		"lines.markersize": 4,
		"lines.linewidth": 1.5,
		"font.size": FONT_SIZE,
		"axes.labelsize": FONT_SIZE,
		"axes.titlesize": FONT_SIZE,
		"xtick.labelsize": FONT_SIZE,
		"ytick.labelsize": FONT_SIZE,
		"legend.fontsize": FONT_SIZE,
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


def display_name(model: str) -> str:
    return display_model_name(model)


TIER_1 = ["chgnet", "mace-mp-0", "mace-mp-0-compile", "grace-mp"]
TIER_2 = ["mace-mpa-0", "mace-mpa-0-compile", "orb-v2"]
TIER_3 = ["mattersim-v1-5m", "grace-oam", "orb-v3", "esen-30m-oam", "nequip", "eq-v2-m-omat", "pet-oam-xl", "pet-omat-xl", "grace-oam-compiled", "mattersim-v1-5m-compile", "pet-oam-xl-torchscript", "pet-omat-xl-torchscript"]
TIER_4 = ["mace-mh-omat", "mace-mh-omat-compile", "uma-s-omat", "uma-s-omat-compile", "uma-s-omat-turbo", "uma-m-omat", "uma-m-omat-compile", "uma-m-omat-turbo"]


TIER_COLORS = {
	"Tier 1": palette[2],
	"Tier 2": palette[1],
	"Tier 3": palette[3],
	"Tier 4": palette[0],
	"Other": "#757575",
}


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRESSURE_METRICS_FILE = DEFAULT_PRESSURE_FILE

DEFAULT_OUTPUT_FILE = (
	SCRIPT_DIR / "plots" / "plot_pareto_combined_vdos_rdf_pressure_same_length.pdf"
)

DEFAULT_OUTPUT_CSV = (
	SCRIPT_DIR / "results" / "vdos_rdf_pressure_model_means_merged_same_simulation_length.csv"
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
) -> pd.DataFrame:
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
	out["model"] = out[model_col].map(normalize_model_name)

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
	error_cols = ["RDF Error [%]", "VDOS Error [%]", "Pressure Error [%]"]

	errors = df[error_cols].astype(float)
	err_sum = errors.sum(axis=1).to_numpy()
	err_sq_sum = (errors**2).sum(axis=1).to_numpy()

	combined_error = np.divide(
		err_sq_sum,
		err_sum,
		out=np.zeros_like(err_sum, dtype=float),
		where=err_sum > 0.0,
	)

	return pd.Series(combined_error, index=df.index)


def load_model_avg_timings(timings_dir: Path) -> pd.DataFrame:
	"""Return model-level mean MD step timings from repository timing CSVs."""
	return load_repo_timings(timings_dir)


def load_and_merge(
	timings_dir: Path,
	pressure_metrics_file: Path,
	pressure_scale_gpa: float,
	clip_pressure_error: bool,
	combined_metrics_file: Path | None = None,
	rdf_metrics_file: Path | None = None,
	vdos_metrics_file: Path | None = None,
	pressure_backend: str | None = None,
	pressure_mode: str | None = None,
) -> pd.DataFrame:
	merged = load_average_inputs(
		timings_dir=timings_dir,
		combined_metrics_file=combined_metrics_file,
		rdf_metrics_file=rdf_metrics_file,
		vdos_metrics_file=vdos_metrics_file,
		pressure_metrics_file=pressure_metrics_file,
		pressure_scale_gpa=pressure_scale_gpa,
		clip_pressure_error=clip_pressure_error,
		pressure_backend=pressure_backend,
		pressure_mode=pressure_mode,
	)
	merged["Combined Error [%]"] = compute_combined_error(merged)
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


def model_tier(model_name: str) -> str:
	model_name = normalize_model_name(model_name)

	if model_name in TIER_1:
		return "Tier 1"

	if model_name in TIER_2:
		return "Tier 2"

	if model_name in TIER_3:
		return "Tier 3"

	if model_name in TIER_4:
		return "Tier 4"

	return "Other"


def plot_pareto(df: pd.DataFrame, output_file: Path) -> None:
	pareto_mask = is_pareto_optimal(df)

	all_df = df.copy()
	all_df["tier"] = all_df["model"].map(model_tier)

	pareto_df = df[pareto_mask].copy().sort_values("mean_time_per_step_ms")
	pareto_df["tier"] = pareto_df["model"].map(model_tier)

	fig, ax = plt.subplots(figsize=(3.53 * 1.5, 3.53 * 1.5))

	for tier_name in ["Tier 1", "Tier 2", "Tier 3", "Tier 4", "Other"]:
		tier_df = all_df[all_df["tier"] == tier_name]

		if tier_df.empty:
			continue

		ax.scatter(
			tier_df["mean_time_per_step_ms"],
			tier_df["Combined Error [%]"],
			color=TIER_COLORS[tier_name],
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
		linewidth=1.2,
		alpha=0.9,
		zorder=3,
	)

	label_texts = []
	x_vals = all_df["mean_time_per_step_ms"].to_numpy()
	y_vals = all_df["Combined Error [%]"].to_numpy()

	for xi, yi, model_name in zip(x_vals, y_vals, all_df["model"].values):
		txt = ax.annotate(
			display_name(str(model_name)),
			xy=(xi, yi),
			xytext=(3, 3),
			textcoords="offset points",
			fontsize=FONT_SIZE,
			alpha=0.8,
			ha="left",
			va="bottom",
			rotation=0,
			bbox=dict(
				boxstyle="round,pad=0.08",
				facecolor="white",
				edgecolor="none",
				alpha=0.55,
			),
		)
		label_texts.append(txt)

	if adjust_text is not None and label_texts:
		try:
			adjust_text(
				label_texts,
				ax=ax,
				x=x_vals,
				y=y_vals,
				avoid_self=True,
				only_move={"points": "xy", "text": "xy"},
				force_text=(1.2, 1.4),
				force_points=(0.8, 1.0),
				expand_points=(1.3, 1.4),
				expand_text=(1.2, 1.3),
				lim=400,
				arrowprops=dict(
					arrowstyle="-",
					color="0.5",
					lw=0.4,
					alpha=0.45,
				),
			)
		except Exception:
			pass

	# ax.set_xlabel("Mean time per step [ms]")
	ax.set_xlabel(r"Mean time per step [ms]")
	ax.set_ylabel(r"$L_{\mathrm{RPV}}$ error [%]")
	ax.set_xlim(right=590)
	ax.grid(True, linestyle="--", alpha=0.4)
	ax.legend(loc="best", frameon=True)

	output_file.parent.mkdir(parents=True, exist_ok=True)

	plt.tight_layout()
	plt.savefig(output_file, bbox_inches="tight", pad_inches=0.02)
	plt.close(fig)

	print(f"Saved: {output_file}")
	print(f"Models plotted: {len(all_df)}")
	print(f"Pareto-optimal models: {len(pareto_df)}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Plot Pareto front of combined RDF/VDOS/pressure Lehmer-mean error "
			"vs mean MD step time."
		)
	)
	parser.add_argument(
		"--source",
		choices=SOURCES,
		default=DEFAULT_SOURCE,
		help="Trajectory source used to derive default timing, RDF, and VDOS inputs.",
	)

	parser.add_argument(
		"--timings-dir",
		default=None,
		help="Directory containing per-system md_timing_*.csv files (default: data/<source>).",
	)

	parser.add_argument(
		"--combined-metrics-file",
		default=None,
		help="Optional pre-merged RDF/VDOS CSV; generated source tables are used by default.",
	)

	parser.add_argument(
		"--rdf-metrics-file",
		default=None,
		help="RDF model metrics CSV (default: rdfs/results/<source>/...).",
	)

	parser.add_argument(
		"--vdos-metrics-file",
		default=None,
		help="VDOS model metrics CSV (default: vdos/results/<source>/...).",
	)

	parser.add_argument(
		"--pressure-backend",
		default=None,
		help="Pressure backend to select (default: inferred from --source).",
	)

	parser.add_argument(
		"--pressure-mode",
		default=None,
		help="Pressure mode to select (default: inferred from --source).",
	)

	parser.add_argument(
		"--pressure-metrics-file",
		default=str(DEFAULT_PRESSURE_METRICS_FILE),
		help=(
			"CSV with pressure histogram similarity/error by model. Expected columns include "
			"model or mlip_model plus final_mean_pressure_similarity_percent, "
			"pressure_similarity_percent, final_mean_pressure_error_percent, or "
			"pressure_error_percent. MAE columns are still accepted as a fallback."
		),
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
		default=str(DEFAULT_OUTPUT_FILE),
		help="Output plot path.",
	)

	parser.add_argument(
		"--output-csv",
		default=str(DEFAULT_OUTPUT_CSV),
		help="Output CSV containing merged RDF, VDOS, pressure, and combined errors.",
	)

	args = parser.parse_args()
	default_timings, default_rdf, default_vdos, default_pressure_mode = source_input_paths(
		args.source
	)

	merged = load_and_merge(
		timings_dir=Path(args.timings_dir) if args.timings_dir else default_timings,
		combined_metrics_file=(
			Path(args.combined_metrics_file) if args.combined_metrics_file else None
		),
		rdf_metrics_file=(
			Path(args.rdf_metrics_file) if args.rdf_metrics_file else default_rdf
		),
		vdos_metrics_file=(
			Path(args.vdos_metrics_file) if args.vdos_metrics_file else default_vdos
		),
		pressure_metrics_file=Path(args.pressure_metrics_file),
		pressure_scale_gpa=args.pressure_scale_gpa,
		clip_pressure_error=not args.no_pressure_clip,
		pressure_backend=args.pressure_backend or source_pressure_backend(args.source),
		pressure_mode=args.pressure_mode or default_pressure_mode,
	)

	output_csv = Path(args.output_csv)
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	merged.to_csv(output_csv, index=False)
	print(f"Saved merged RDF/VDOS/pressure metrics: {output_csv}")

	plot_pareto(merged, Path(args.output_file))


if __name__ == "__main__":
	main()
