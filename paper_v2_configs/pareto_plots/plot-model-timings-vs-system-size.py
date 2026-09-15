#!/usr/bin/env python3
"""Plot eager and accelerated MD scaling in separate figures.

Standard and accelerated timings are validated using ``plot-model-timings.py``.
Time-averaged graph sizes are computed from uniformly sampled HDF5 trajectory
frames using the model's neighbor-list convention. Each panel shows one model;
dashed lines are log-log least-squares (power-law) scaling guides.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import warnings
from pathlib import Path
from types import ModuleType

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from vesin import NeighborList


SCRIPT_DIR = Path(__file__).resolve().parent
TIMING_SCRIPT = SCRIPT_DIR / "plot-model-timings.py"
DEFAULT_TIMINGS_DIR = SCRIPT_DIR.parent / "data" / "mlip-trajs-torchsim-eager"
DEFAULT_ACCELERATED_TIMINGS_DIR = (
	SCRIPT_DIR.parent / "data" / "mlip-trajs-torchsim-accelerated"
)
DEFAULT_EAGER_OUTPUT_PNG = (
	SCRIPT_DIR / "plots" / "model_timings_vs_system_size_eager.png"
)
DEFAULT_EAGER_OUTPUT_PDF = (
	SCRIPT_DIR / "plots" / "model_timings_vs_system_size_eager.pdf"
)
DEFAULT_ACCELERATED_OUTPUT_PNG = (
	SCRIPT_DIR / "plots" / "model_timings_vs_system_size_accelerated.png"
)
DEFAULT_ACCELERATED_OUTPUT_PDF = (
	SCRIPT_DIR / "plots" / "model_timings_vs_system_size_accelerated.pdf"
)
DEFAULT_RESULTS_CSV = (
	SCRIPT_DIR / "results" / "model_timings_vs_system_size_observations.csv"
)

DEFAULT_GRAPH_MODELS = ("orb-v2", "mace-mp-0")
DEFAULT_GRAPH_SAMPLES = 200
MODEL_GRAPH_CONFIG = {
	"orb-v2": {"cutoff_A": 10.0, "max_neighbors": 20},
	"mace-mp-0": {"cutoff_A": 6.0, "max_neighbors": None},
}
X_AXIS_CONFIG = {
	"edges": (
		"mean_n_edges",
		"Time-averaged number of directed graph edges (log scale)",
		"graph size",
	),
	"edge-density": (
		"mean_edges_per_atom",
		"Time-averaged directed edges per atom (log scale)",
		"edge density",
	),
	"atoms": (
		"n_atoms",
		"System size [number of atoms] (log scale)",
		"system size",
	),
}
MODE_STYLES = {
	"Standard": {"color": "#345995", "marker": "o"},
	"Accelerated": {"color": "#e07a3f", "marker": "^"},
}


def load_timing_module() -> ModuleType:
	"""Load the sibling timing script so both plots share CSV validation."""
	spec = importlib.util.spec_from_file_location("plot_model_timings", TIMING_SCRIPT)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load timing functions from {TIMING_SCRIPT}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def read_graph_metrics(
	trajectory_path: Path, model: str, graph_samples: int
) -> dict[str, int | float]:
	"""Calculate model-specific graph statistics over sampled trajectory frames."""
	if model not in MODEL_GRAPH_CONFIG:
		raise ValueError(f"no graph configuration is defined for model '{model}'")
	config = MODEL_GRAPH_CONFIG[model]

	with h5py.File(trajectory_path, "r") as handle:
		required = (
			"data/atomic_numbers",
			"data/positions",
			"data/cell",
			"data/pbc",
		)
		missing = [dataset for dataset in required if dataset not in handle]
		if missing:
			raise KeyError(f"missing datasets {missing}")
		atomic_numbers = np.asarray(handle["data/atomic_numbers"][0])
		positions = handle["data/positions"]
		cells = handle["data/cell"]
		pbc = np.asarray(handle["data/pbc"][:], dtype=bool)

		n_atoms = len(atomic_numbers)
		n_frames = positions.shape[0]
		if n_atoms < 1 or positions.shape[1:] != (n_atoms, 3):
			raise ValueError(
				f"invalid structure shapes: Z={atomic_numbers.shape}, "
				f"positions={positions.shape}"
			)
		if cells.shape != (n_frames, 3, 3):
			raise ValueError(
				f"cell shape {cells.shape} does not match {n_frames} trajectory frames"
			)
		if not (pbc.all() or (~pbc).all()):
			raise ValueError("mixed periodic boundary conditions are not supported")
		if graph_samples == 0 or graph_samples >= n_frames:
			frame_indices = np.arange(n_frames)
		else:
			frame_indices = np.unique(
				np.linspace(0, n_frames - 1, graph_samples, dtype=int)
			)

		neighbor_calculator = NeighborList(
			cutoff=float(config["cutoff_A"]), full_list=True
		)
		edge_counts = np.empty(len(frame_indices), dtype=np.int64)
		max_neighbors = config["max_neighbors"]
		for output_index, frame_index in enumerate(frame_indices):
			senders = neighbor_calculator.compute(
				positions[frame_index],
				cells[frame_index],
				periodic=bool(pbc.all()),
				quantities="i",
				copy=False,
			)[0]
			if max_neighbors is None:
				edge_counts[output_index] = len(senders)
			else:
				neighbors_per_atom = np.bincount(senders, minlength=n_atoms)
				edge_counts[output_index] = np.minimum(
					neighbors_per_atom, max_neighbors
				).sum()

	return {
		"n_atoms": n_atoms,
		"initial_n_edges": int(edge_counts[0]),
		"mean_n_edges": float(edge_counts.mean()),
		"std_n_edges": float(edge_counts.std(ddof=0)),
		"min_n_edges": int(edge_counts.min()),
		"max_n_edges": int(edge_counts.max()),
		"mean_edges_per_atom": float(edge_counts.mean() / n_atoms),
		"graph_frames_sampled": len(frame_indices),
		"graph_frames_total": n_frames,
	}


def read_atom_count(trajectory_path: Path) -> int:
	"""Read and validate an atom count from a TorchSim HDF5 trajectory."""
	with h5py.File(trajectory_path, "r") as handle:
		if "data/atomic_numbers" not in handle:
			raise KeyError("missing dataset 'data/atomic_numbers'")
		shape = handle["data/atomic_numbers"].shape
		if not shape or shape[-1] < 1:
			raise ValueError(f"invalid atomic-number dataset shape {shape}")
		return int(shape[-1])


def add_atom_counts(
	observations: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
	"""Add atom counts without requiring model-specific graph metadata."""
	rows: list[dict[str, object]] = []
	skipped: list[str] = []
	count_cache: dict[Path, int] = {}

	for row in observations.to_dict(orient="records"):
		timing_path = Path(str(row["source_file"]))
		trajectory_path = timing_path.with_name(f"nvt_{row['model']}.h5")
		if not trajectory_path.is_file():
			fallback_paths = sorted(timing_path.parent.glob("nvt_*.h5"))
			if not fallback_paths:
				skipped.append(
					f"{timing_path}: no trajectory found from which to read atom count"
				)
				continue
			trajectory_path = fallback_paths[0]

		try:
			if trajectory_path not in count_cache:
				count_cache[trajectory_path] = read_atom_count(trajectory_path)
		except (OSError, KeyError, ValueError) as exc:
			skipped.append(f"{trajectory_path}: could not read atom count ({exc})")
			continue

		row["n_atoms"] = count_cache[trajectory_path]
		row["structure_source_file"] = str(trajectory_path)
		rows.append(row)

	if not rows:
		raise ValueError("No timing observations had a readable atom count")
	result = pd.DataFrame(rows)
	atom_counts_per_system = result.groupby("system")["n_atoms"].nunique()
	inconsistent = atom_counts_per_system[atom_counts_per_system > 1]
	if not inconsistent.empty:
		raise ValueError(
			"Inconsistent atom counts across runs for systems: "
			+ ", ".join(inconsistent.index)
		)
	return result, skipped


def add_graph_metrics(
	observations: pd.DataFrame, graph_samples: int
) -> tuple[pd.DataFrame, list[str]]:
	"""Add atom and time-averaged graph sizes for every timing observation."""
	rows: list[dict[str, object]] = []
	skipped: list[str] = []
	metric_cache: dict[tuple[Path, str], dict[str, int | float]] = {}

	for row in observations.to_dict(orient="records"):
		timing_path = Path(str(row["source_file"]))
		trajectory_path = timing_path.with_name(f"nvt_{row['model']}.h5")
		if not trajectory_path.is_file():
			fallback_paths = sorted(timing_path.parent.glob("nvt_*.h5"))
			if not fallback_paths:
				skipped.append(
					f"{timing_path}: no trajectory found from which to calculate graph size"
				)
				continue
			trajectory_path = fallback_paths[0]

		try:
			cache_key = (trajectory_path, str(row["model"]))
			if cache_key not in metric_cache:
				metric_cache[cache_key] = read_graph_metrics(
					trajectory_path, str(row["model"]), graph_samples
				)
		except (OSError, KeyError, ValueError) as exc:
			skipped.append(f"{trajectory_path}: could not calculate graph size ({exc})")
			continue

		graph_config = MODEL_GRAPH_CONFIG[str(row["model"])]
		row.update(metric_cache[cache_key])
		row["edge_cutoff_A"] = graph_config["cutoff_A"]
		row["max_neighbors"] = graph_config["max_neighbors"]
		row["structure_source_file"] = str(trajectory_path)
		rows.append(row)

	if not rows:
		raise ValueError("No timing observations had readable graph metrics")

	result = pd.DataFrame(rows)
	atom_counts_per_system = result.groupby("system")["n_atoms"].nunique()
	inconsistent = atom_counts_per_system[atom_counts_per_system > 1]
	if not inconsistent.empty:
		raise ValueError(
			"Inconsistent atom counts across runs for systems: "
			+ ", ".join(inconsistent.index)
		)
	return result, skipped


def fit_power_law(
	values: pd.DataFrame, x_column: str
) -> tuple[np.ndarray, np.ndarray] | None:
	"""Fit y = a*x**b in log space and return coordinates for a guide line."""
	x = values[x_column].to_numpy(dtype=float)
	y = values["milliseconds_per_step"].to_numpy(dtype=float)
	if len(np.unique(x)) < 3 or len(x) < 3:
		return None

	slope, intercept = np.polyfit(np.log10(x), np.log10(y), deg=1)
	x_fit = np.geomspace(x.min(), x.max(), 100)
	y_fit = 10 ** (intercept + slope * np.log10(x_fit))
	return x_fit, y_fit


def model_order_by_median(observations: pd.DataFrame) -> list[str]:
	"""Order model display names by median runtime."""
	return (
		observations.groupby("model_display")["milliseconds_per_step"]
		.median()
		.sort_values()
		.index.tolist()
	)


def plot_timings_vs_size(
	observations: pd.DataFrame,
	output_png: Path,
	output_pdf: Path,
	skipped_count: int,
	x_axis: str,
	mode_label: str,
	color: str,
	marker: str,
) -> None:
	"""Draw one scaling panel per model for a single MD execution mode."""
	sns.set_theme(style="ticks", context="paper")
	plt.rcParams.update(
		{
			"font.size": 8,
			"axes.labelsize": 9,
			"axes.titlesize": 9,
			"xtick.labelsize": 7,
			"ytick.labelsize": 7,
			"pdf.fonttype": 42,
			"ps.fonttype": 42,
		}
	)

	x_column, x_label, title_metric = X_AXIS_CONFIG[x_axis]
	model_order = model_order_by_median(observations)
	n_cols = min(4, len(model_order))
	n_rows = math.ceil(len(model_order) / n_cols)
	fig, axes = plt.subplots(
		n_rows,
		n_cols,
		figsize=(max(8.5, 2.8 * n_cols), max(5.2, 2.55 * n_rows + 1.1)),
		sharex=True,
		sharey=True,
		squeeze=False,
	)

	for ax, model_display in zip(axes.flat, model_order):
		model_values = observations[observations["model_display"] == model_display]
		model = str(model_values["model"].iloc[0])
		panel_heading = model_display
		if x_axis != "atoms":
			graph_config = MODEL_GRAPH_CONFIG[model]
			graph_description = f"{graph_config['cutoff_A']:g} Å"
			if graph_config["max_neighbors"] is not None:
				graph_description += f", max {graph_config['max_neighbors']} neighbors"
			panel_heading += f" — {graph_description}"
		ax.scatter(
			model_values[x_column],
			model_values["milliseconds_per_step"],
			s=24,
			color=color,
			marker=marker,
			alpha=0.72,
			edgecolors="white",
			linewidths=0.35,
			zorder=3,
		)
		fit = fit_power_law(model_values, x_column)
		if fit is not None:
			ax.plot(
				fit[0],
				fit[1],
				color=color,
				linestyle="--",
				linewidth=1.2,
				alpha=0.85,
				zorder=2,
			)

		ax.set_xscale("log")
		ax.set_yscale("log")
		ax.set_title(
			f"{panel_heading}\n$n$={len(model_values)}",
			loc="left",
			fontweight="bold",
			pad=5,
		)
		ax.grid(which="major", linestyle="--", linewidth=0.55, alpha=0.5)
		ax.grid(which="minor", linestyle=":", linewidth=0.35, alpha=0.25)
		sns.despine(ax=ax)

	for ax in axes.flat[len(model_order) :]:
		ax.set_visible(False)

	fig.supxlabel(x_label, y=0.035)
	fig.supylabel("MD time per step [ms] (log scale)", x=0.025)
	fig.suptitle(
		f"{mode_label} TorchSim timing scaling with {title_metric}",
		x=0.055,
		y=0.985,
		ha="left",
		fontsize=13,
		fontweight="bold",
	)
	if x_axis == "atoms":
		subtitle = f"Points: systems; dashed: fits ($n$={len(observations)}"
	else:
		max_graph_frames = int(observations["graph_frames_sampled"].max())
		subtitle = (
			f"Graph mean: ≤{max_graph_frames} uniform frames; "
			f"dashed: fits ($n$={len(observations)}"
		)
	if skipped_count:
		subtitle += f"; {skipped_count} invalid skipped"
	subtitle += ")"
	fig.text(0.055, 0.955, subtitle, ha="left", va="top", fontsize=8)

	panel_top = 0.78 if n_rows == 1 else 0.90
	fig.subplots_adjust(
		left=0.105,
		right=0.97,
		bottom=0.12 if n_rows == 1 else 0.09,
		top=panel_top,
		hspace=0.50,
		wspace=0.20,
	)
	for output_path in (output_png, output_pdf):
		output_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
	plt.close(fig)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--timings-dir",
		type=Path,
		default=DEFAULT_TIMINGS_DIR,
		help="Directory containing standard per-system timing CSV and HDF5 files.",
	)
	parser.add_argument(
		"--accelerated-timings-dir",
		type=Path,
		default=DEFAULT_ACCELERATED_TIMINGS_DIR,
		help="Directory containing accelerated per-system timing CSV and HDF5 files.",
	)
	parser.add_argument(
		"--models",
		nargs="+",
		default=None,
		help=(
			"Model names to plot. Defaults to all models for the atom axis and "
			"orb-v2/mace-mp-0 for graph axes."
		),
	)
	parser.add_argument(
		"--x-axis",
		choices=tuple(X_AXIS_CONFIG),
		default="atoms",
		help="Horizontal metric to plot (default: atoms).",
	)
	parser.add_argument(
		"--graph-samples",
		type=int,
		default=DEFAULT_GRAPH_SAMPLES,
		help=(
			"Number of uniformly spaced trajectory frames used for graph averages; "
			"use 0 for every recorded frame (default: 200)."
		),
	)
	parser.add_argument(
		"--eager-output-png", type=Path, default=DEFAULT_EAGER_OUTPUT_PNG
	)
	parser.add_argument(
		"--eager-output-pdf", type=Path, default=DEFAULT_EAGER_OUTPUT_PDF
	)
	parser.add_argument(
		"--accelerated-output-png",
		type=Path,
		default=DEFAULT_ACCELERATED_OUTPUT_PNG,
	)
	parser.add_argument(
		"--accelerated-output-pdf",
		type=Path,
		default=DEFAULT_ACCELERATED_OUTPUT_PDF,
	)
	parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.graph_samples < 0:
		raise ValueError("--graph-samples must be non-negative")
	timing_module = load_timing_module()
	standard, standard_skipped = timing_module.read_timing_files(
		args.timings_dir, timing_mode="Standard"
	)
	accelerated, accelerated_skipped = timing_module.read_timing_files(
		args.accelerated_timings_dir, timing_mode="Accelerated"
	)
	observations = pd.concat([standard, accelerated], ignore_index=True)
	if args.models:
		selected_models: set[str] | None = {
			timing_module.normalize_name(model) for model in args.models
		}
	elif args.x_axis == "atoms":
		selected_models = None
	else:
		selected_models = set(DEFAULT_GRAPH_MODELS)
	if selected_models is not None:
		observations = observations[observations["model"].isin(selected_models)].copy()
	if observations.empty:
		requested = ", ".join(args.models or [])
		raise ValueError(f"None of the requested models were found: {requested}")
	if args.x_axis == "atoms":
		observations, metric_skipped = add_atom_counts(observations)
	else:
		assert selected_models is not None
		unsupported_models = selected_models - MODEL_GRAPH_CONFIG.keys()
		if unsupported_models:
			raise ValueError(
				"No graph configuration is defined for: "
				+ ", ".join(sorted(unsupported_models))
			)
		observations, metric_skipped = add_graph_metrics(
			observations, graph_samples=args.graph_samples
		)
	timing_skipped = standard_skipped + accelerated_skipped
	if selected_models is None:
		selected_timing_skipped = timing_skipped
	else:
		selected_timing_skipped = [
			message
			for message in timing_skipped
			if any(f"md_timing_{model}.csv" in message for model in selected_models)
		]
	skipped = selected_timing_skipped + metric_skipped
	eager_skipped_count = sum(
		str(args.timings_dir) in message for message in skipped
	)
	accelerated_skipped_count = sum(
		str(args.accelerated_timings_dir) in message for message in skipped
	)

	for message in skipped:
		warnings.warn(message, stacklevel=1)

	args.results_csv.parent.mkdir(parents=True, exist_ok=True)
	x_column = X_AXIS_CONFIG[args.x_axis][0]
	observations.sort_values(["model", "timing_mode", x_column, "system"]).to_csv(
		args.results_csv, index=False
	)
	for mode_observations, output_png, output_pdf, mode_label, style, mode_skipped in (
		(
			observations[observations["timing_mode"] == "Standard"],
			args.eager_output_png,
			args.eager_output_pdf,
			"Eager MD",
			MODE_STYLES["Standard"],
			eager_skipped_count,
		),
		(
			observations[observations["timing_mode"] == "Accelerated"],
			args.accelerated_output_png,
			args.accelerated_output_pdf,
			"Accelerated MD",
			MODE_STYLES["Accelerated"],
			accelerated_skipped_count,
		),
	):
		if mode_observations.empty:
			warnings.warn(f"No {mode_label.lower()} observations to plot", stacklevel=1)
			continue
		plot_timings_vs_size(
			mode_observations,
			output_png=output_png,
			output_pdf=output_pdf,
			skipped_count=mode_skipped,
			x_axis=args.x_axis,
			mode_label=mode_label,
			color=style["color"],
			marker=style["marker"],
		)

	aggregations = {
		"n_observations": ("milliseconds_per_step", "size"),
		"min_atoms": ("n_atoms", "min"),
		"max_atoms": ("n_atoms", "max"),
	}
	if "mean_n_edges" in observations:
		aggregations.update(
			{
				"min_mean_edges": ("mean_n_edges", "min"),
				"max_mean_edges": ("mean_n_edges", "max"),
			}
		)
	print(
		observations.groupby(["model_display", "timing_mode"])
		.agg(**aggregations)
		.to_string()
	)
	print(f"\nSaved eager plot: {args.eager_output_png}")
	print(f"Saved eager plot: {args.eager_output_pdf}")
	print(f"Saved accelerated plot: {args.accelerated_output_png}")
	print(f"Saved accelerated plot: {args.accelerated_output_pdf}")
	print(f"Saved observations: {args.results_csv}")


if __name__ == "__main__":
	main()
