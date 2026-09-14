#!/usr/bin/env python3
"""Plot standard and accelerated TorchSim MD timings for each MLIP model.

Each ``md_timing_<model>.csv`` in either timing directory contributes one
observation: its recorded ``seconds_per_step`` converted to milliseconds.
Invalid rows are skipped with a warning, including rows whose calculator does
not match the filename or whose elapsed time is inconsistent with
``n_steps * seconds_per_step``.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TIMINGS_DIR = SCRIPT_DIR.parent / "data" / "mlip-trajs-torchsim"
DEFAULT_ACCELERATED_TIMINGS_DIR = (
	SCRIPT_DIR.parent / "data" / "mlip-trajs-torchsim-accelerated"
)
DEFAULT_OUTPUT_PNG = SCRIPT_DIR / "plots" / "model_timings.png"
DEFAULT_OUTPUT_PDF = SCRIPT_DIR / "plots" / "model_timings.pdf"
DEFAULT_SUMMARY_CSV = SCRIPT_DIR / "results" / "model_timings_summary.csv"
DEFAULT_OBSERVATIONS_CSV = SCRIPT_DIR / "results" / "model_timings_observations.csv"


DISPLAY_NAMES = {
	"esen-30m-oam": "eSEN-30M-OAM",
	"grace-mp": "GRACE-2L-MPtrj",
	"grace-oam": "GRACE-2L-OAM",
	"mace-mh-omat": "MACE-MH-1-OMAT",
	"mace-mp-0": "MACE-MP-0",
	"mace-mpa-0": "MACE-MPA-0",
	"mattersim-v1-5m": "MatterSim-v1.0.0-5M",
	"nequip": "NequIP-OAM-XL",
	"orb-v2": "ORB-v2",
	"orb-v3": "ORB-v3 conservative",
	"orb-v3-direct": "ORB-v3 direct",
	"pet-oam-xl": "PET-OAM-XL",
	"pet-omat-xl": "PET-OMAT-XL",
	"uma-m-omat": "UMA-M-P1",
	"uma-s-omat": "UMA-S-P1",
}


REQUIRED_COLUMNS = {
	"calculator",
	"system",
	"n_steps",
	"elapsed_seconds",
	"seconds_per_step",
}


def normalize_name(value: object) -> str:
	return str(value).strip().lower()


def display_name(model: str) -> str:
	return DISPLAY_NAMES.get(model, model)


def accelerated_display_name(model: str) -> str:
	"""Return a distinct plot label for an accelerated model variant."""
	for variant in ("compile", "turbo"):
		suffix = f"-{variant}"
		if model.endswith(suffix):
			return f"{display_name(model.removesuffix(suffix))} {variant}"
	return f"{display_name(model)} accelerated"


def read_timing_files(
	timings_dir: Path, timing_mode: str = "Standard"
) -> tuple[pd.DataFrame, list[str]]:
	"""Load and validate one timing observation from every timing CSV."""
	if not timings_dir.is_dir():
		raise FileNotFoundError(f"Timing directory does not exist: {timings_dir}")

	paths = sorted(timings_dir.glob("*/md_timing_*.csv"))
	if not paths:
		raise FileNotFoundError(f"No md_timing_*.csv files found below {timings_dir}")

	rows: list[dict[str, object]] = []
	skipped: list[str] = []

	for csv_path in paths:
		try:
			df = pd.read_csv(csv_path)
		except Exception as exc:
			skipped.append(f"{csv_path}: could not read CSV ({exc})")
			continue

		missing = REQUIRED_COLUMNS - set(df.columns)
		if missing:
			skipped.append(f"{csv_path}: missing columns {sorted(missing)}")
			continue
		if len(df) != 1:
			skipped.append(f"{csv_path}: expected one row, found {len(df)}")
			continue

		row = df.iloc[0]
		filename_model = normalize_name(csv_path.stem.removeprefix("md_timing_"))
		calculator = normalize_name(row["calculator"])
		csv_system = str(row["system"]).strip()
		if calculator != filename_model:
			skipped.append(
				f"{csv_path}: calculator '{calculator}' does not match filename "
				f"model '{filename_model}'"
			)
			continue
		if csv_system != csv_path.parent.name:
			skipped.append(
				f"{csv_path}: system '{csv_system}' does not match parent directory "
				f"'{csv_path.parent.name}'"
			)
			continue

		try:
			n_steps = float(row["n_steps"])
			elapsed_seconds = float(row["elapsed_seconds"])
			seconds_per_step = float(row["seconds_per_step"])
		except (TypeError, ValueError) as exc:
			skipped.append(f"{csv_path}: non-numeric timing value ({exc})")
			continue

		values = np.array([n_steps, elapsed_seconds, seconds_per_step], dtype=float)
		if not np.isfinite(values).all() or (values <= 0).any():
			skipped.append(f"{csv_path}: timing values must be finite and positive")
			continue

		expected_seconds_per_step = elapsed_seconds / n_steps
		if not np.isclose(seconds_per_step, expected_seconds_per_step, rtol=0.02, atol=1e-9):
			skipped.append(
				f"{csv_path}: seconds_per_step={seconds_per_step:g} is inconsistent "
				f"with elapsed_seconds/n_steps={expected_seconds_per_step:g}"
			)
			continue

		rows.append(
			{
				"model": filename_model,
				"model_display": display_name(filename_model),
				"timing_mode": timing_mode,
				"system": csv_system,
				"temperature_K": pd.to_numeric(row.get("temperature_K"), errors="coerce"),
				"n_steps": int(n_steps),
				"elapsed_seconds": elapsed_seconds,
				"seconds_per_step": seconds_per_step,
				"milliseconds_per_step": seconds_per_step * 1000.0,
				"engine": str(row.get("engine", "")),
				"source_file": str(csv_path),
			}
		)

	if not rows:
		raise ValueError("No valid timing observations were found")

	return pd.DataFrame(rows), skipped


def select_accelerated_timings(observations: pd.DataFrame) -> pd.DataFrame:
	"""Use compiled cuEquivariance timings, rather than cuEq-only, for MACE."""
	selected = observations.copy()
	models = selected["model"].astype(str)
	engines = selected["engine"].astype(str).str.lower()
	is_mace = models.str.startswith("mace-")
	is_compiled_mace = is_mace & engines.str.contains("cueq") & engines.str.contains(
		"compile"
	)
	selected = selected.loc[~is_mace | is_compiled_mace].copy()

	compiled_mace_rows = selected["model"].astype(str).str.startswith("mace-")
	selected.loc[compiled_mace_rows, "model"] = selected.loc[
		compiled_mace_rows, "model"
	].str.removesuffix("-compile")
	selected.loc[compiled_mace_rows, "model_display"] = (
		selected.loc[compiled_mace_rows, "model"].map(display_name) + " compile"
	)
	selected.loc[compiled_mace_rows, "timing_mode"] = "Compile"

	accelerated_rows = ~compiled_mace_rows
	selected.loc[accelerated_rows, "model_display"] = selected.loc[
		accelerated_rows, "model"
	].map(accelerated_display_name)
	return selected


def summarize_timings(observations: pd.DataFrame) -> pd.DataFrame:
	"""Return per-model and timing-mode statistics, ordered by median runtime."""
	grouped = observations.groupby(["model", "model_display", "timing_mode"])[
		"milliseconds_per_step"
	]
	summary = grouped.agg(
		n_systems="count",
		median_ms_per_step="median",
		mean_ms_per_step="mean",
		std_ms_per_step="std",
		min_ms_per_step="min",
		max_ms_per_step="max",
	).reset_index()

	geometric_means = grouped.apply(
		lambda values: float(np.exp(np.mean(np.log(values.to_numpy(dtype=float)))))
	).rename("geometric_mean_ms_per_step")
	summary = summary.merge(
		geometric_means.reset_index(),
		on=["model", "model_display", "timing_mode"],
		how="left",
	)
	model_order = (
		observations.groupby("model_display")["milliseconds_per_step"]
		.median()
		.sort_values()
		.index
	)
	summary["model_order"] = pd.Categorical(
		summary["model_display"], categories=model_order, ordered=True
	)
	summary["timing_mode_order"] = pd.Categorical(
		summary["timing_mode"],
		categories=["Standard", "Compile", "Accelerated"],
		ordered=True,
	)
	return summary.sort_values(
		["model_order", "timing_mode_order"], ignore_index=True
	).drop(columns=["model_order", "timing_mode_order"])


def plot_timings(
	observations: pd.DataFrame,
	summary: pd.DataFrame,
	output_png: Path,
	output_pdf: Path,
	skipped_count: int,
) -> None:
	"""Draw standard/accelerated box plots with system timings overlaid."""
	sns.set_theme(style="ticks", context="paper")
	plt.rcParams.update(
		{
			"font.size": 8,
			"axes.labelsize": 9,
			"axes.titlesize": 11,
			"xtick.labelsize": 8,
			"ytick.labelsize": 8,
			"pdf.fonttype": 42,
			"ps.fonttype": 42,
		}
	)

	model_order = summary["model_display"].drop_duplicates().tolist()
	plot_observations = observations.copy()
	plot_observations["plot_mode"] = np.where(
		plot_observations["timing_mode"].eq("Standard"),
		"Standard",
		"Accelerated",
	)
	mode_order = ["Standard", "Accelerated"]
	palette = {
		"Standard": "#345995",
		"Accelerated": "#e07a3f",
	}
	counts = observations.groupby("model_display").size()
	y_labels = [f"{name}  ($n$={int(counts[name])})" for name in model_order]

	fig_height = max(4.8, 0.38 * len(model_order) + 1.4)
	fig, ax = plt.subplots(figsize=(7.2, fig_height))

	sns.boxplot(
		data=plot_observations,
		x="milliseconds_per_step",
		y="model_display",
		hue="plot_mode",
		hue_order=mode_order,
		order=model_order,
		orient="h",
		dodge=False,
		width=0.55,
		showfliers=False,
		palette=palette,
		boxprops={"alpha": 0.35},
		whiskerprops={"linewidth": 1.0},
		capprops={"linewidth": 1.0},
		medianprops={"color": "#18181b", "linewidth": 1.8},
		ax=ax,
	)
	np.random.seed(42)
	sns.stripplot(
		data=plot_observations,
		x="milliseconds_per_step",
		y="model_display",
		hue="plot_mode",
		hue_order=mode_order,
		order=model_order,
		orient="h",
		palette=palette,
		alpha=0.68,
		size=3.5,
		jitter=0.16,
		dodge=False,
		legend=False,
		ax=ax,
	)

	ax.set_xscale("log")
	ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
	ax.set_xlabel("MD time per step [ms] (log scale)")
	ax.set_ylabel("")
	ax.set_title(
		"Standard and accelerated TorchSim model timings",
		loc="left",
		weight="bold",
		pad=24,
	)
	valid_count = len(observations)
	subtitle = f"Points: individual systems; boxes: interquartile range; dark lines: medians ({valid_count} valid CSVs"
	if skipped_count:
		subtitle += f", {skipped_count} invalid skipped"
	subtitle += ")"
	ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
	ax.grid(axis="x", which="major", linestyle="--", linewidth=0.6, alpha=0.55)
	ax.grid(axis="x", which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
	ax.grid(axis="y", visible=False)
	ax.tick_params(axis="y", length=0)
	ax.legend(title="Timing mode", loc="upper right", frameon=True)
	sns.despine(ax=ax, left=True)

	fig.tight_layout()
	for output_path in (output_png, output_pdf):
		output_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
	plt.close(fig)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--timings-dir",
		type=Path,
		default=DEFAULT_TIMINGS_DIR,
		help="Directory containing per-system md_timing_<model>.csv files.",
	)
	parser.add_argument(
		"--accelerated-timings-dir",
		type=Path,
		default=DEFAULT_ACCELERATED_TIMINGS_DIR,
		help="Directory containing accelerated per-system timing CSV files.",
	)
	parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
	parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
	parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
	parser.add_argument("--observations-csv", type=Path, default=DEFAULT_OBSERVATIONS_CSV)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	standard_observations, standard_skipped = read_timing_files(
		args.timings_dir, timing_mode="Standard"
	)
	accelerated_observations, accelerated_skipped = read_timing_files(
		args.accelerated_timings_dir, timing_mode="Accelerated"
	)
	accelerated_observations = select_accelerated_timings(accelerated_observations)
	observations = pd.concat(
		[standard_observations, accelerated_observations], ignore_index=True
	)
	skipped = standard_skipped + accelerated_skipped
	summary = summarize_timings(observations)

	for message in skipped:
		warnings.warn(message, stacklevel=1)

	args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
	args.observations_csv.parent.mkdir(parents=True, exist_ok=True)
	summary.to_csv(args.summary_csv, index=False)
	observations.sort_values(["model", "timing_mode", "system"]).to_csv(
		args.observations_csv, index=False
	)

	plot_timings(
		observations=observations,
		summary=summary,
		output_png=args.output_png,
		output_pdf=args.output_pdf,
		skipped_count=len(skipped),
	)

	print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
	print(f"\nSaved plot: {args.output_png}")
	print(f"Saved plot: {args.output_pdf}")
	print(f"Saved summary: {args.summary_csv}")
	print(f"Saved observations: {args.observations_csv}")


if __name__ == "__main__":
	main()
