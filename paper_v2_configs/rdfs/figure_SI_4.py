#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "numpy>=1.26", "pandas>=2.2", "seaborn>=0.13"]
# ///
"""Plot the metal-alloy RDF tier panels for bulk Pt3Co at 300 K.

Use existing source-specific RDF scores and saved curves. Model labels and tier
means describe Pt3Co alone. The curve selection, colors, fonts, and full-error
legend policy follow Figure 3. No RDF computations are performed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import fig_3 as rdf


HERE = Path(__file__).resolve().parent
SYSTEM = "bulkPt3Co_300K_J.Kioseoglou_VASP"
DEFAULT_SOURCE = "mlip-trajs-torchsim-accelerated"
# Match one system column in Figure 3, preserving its text-to-panel scale.
FIGURE_SIZE = (3.53 * 1.05, 3.53 * 1.65)


@dataclass
class TierPanel:
    label: str
    color: object
    mean_error: float
    curves: list[tuple[str, tuple[np.ndarray, np.ndarray], str]]
    legend_only: list[str]


def load_panel_data(
    rdf_csv_dir: Path, rdf_save_dir: Path,
) -> tuple[tuple[np.ndarray, np.ndarray], list[TierPanel]]:
    """Select Pt3Co data without creating a figure or altering metric scores."""
    if not rdf_csv_dir.is_dir():
        raise FileNotFoundError(f"RDF result directory does not exist: {rdf_csv_dir}")
    errors = rdf.find_model_system_errors(SYSTEM, rdf_csv_dir)
    if not any(np.isfinite(error) for error in errors.values()):
        raise ValueError(f"No per-model RDF scores found for {SYSTEM} in {rdf_csv_dir}")
    saved = str(rdf_save_dir)
    reference = rdf.load_saved_rdf(SYSTEM, None, saved)
    if reference is None:
        raise FileNotFoundError(f"Reference RDF for {SYSTEM} is missing or unreadable in {rdf_save_dir}")
    panels = []
    for label, models, color in rdf.TIER_DEFS:
        best, worst = rdf.pick_best_worst_models(SYSTEM, models, errors, saved)
        curves = []
        for model, linestyle in ((best, "-"), (worst, "--")):
            if model is None or (linestyle == "--" and worst == best):
                continue
            curve = rdf.load_saved_rdf(SYSTEM, model, saved)
            if curve is not None:
                curves.append((rdf.format_model_label(model, errors), curve, linestyle))
        legend_only = [rdf.format_model_label(model, errors)
                       for model in rdf.full_error_models_for_legend(SYSTEM, models, errors, saved)]
        panels.append(TierPanel(label, color, rdf.tier_mean_rdf_error(models, errors), curves, legend_only))
    return reference, panels


def plot_panel(
    reference: tuple[np.ndarray, np.ndarray], panels: list[TierPanel], output: Path,
) -> None:
    """Draw the standalone Pt3Co column with Figure 3's RDF panel style."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    # Apply the shared RDF style locally so other imports cannot alter its fonts.
    with plt.rc_context(rdf.RDF_STYLE):
        fig = plt.figure(figsize=FIGURE_SIZE, layout="constrained")
        grid = GridSpec(len(panels), 1, figure=fig, height_ratios=[1] * len(panels), hspace=0.05)
        axes = []
        for index, panel in enumerate(panels):
            ax = fig.add_subplot(grid[index], sharex=axes[0] if axes else None)
            axes.append(ax)
            ax.plot(*reference, color="black", linewidth=2.0, label="Reference")
            for label, curve, linestyle in panel.curves:
                ax.plot(*curve, color=panel.color, linewidth=1.6 if linestyle == "-" else 1.4,
                        alpha=0.9, linestyle=linestyle, label=label)
            if index == 0:
                ax.set_title(rdf.build_system_title("Metal alloys", rdf.format_system_name(SYSTEM)))
            ax.set_ylabel("g(r)")
            ax.text(1.02, 0.5, panel.label, transform=ax.transAxes, rotation=90,
                    ha="left", va="center", fontsize=rdf.FONT_SIZE)
            ax.set_xlim(left=0)
            if index == len(panels) - 1:
                ax.set_xlabel("r [Å]")
            else:
                ax.tick_params(labelbottom=False)
            ax.grid()
            handles, labels = ax.get_legend_handles_labels()
            for label in panel.legend_only:
                handles.append(Line2D([], [], color="none", linestyle="none"))
                labels.append(label)
            if np.isfinite(panel.mean_error):
                handles.append(Line2D([], [], color="none", linestyle="none"))
                labels.append(f"{panel.label} mean RDF error: {panel.mean_error:.1f}%")
            ax.legend(handles, labels, loc="upper right", fontsize=rdf.LEGEND_FONT_SIZE,
                      handlelength=1.0, handletextpad=0.3, borderpad=0.2,
                      labelspacing=0.2, borderaxespad=0.2)
        output.parent.mkdir(parents=True, exist_ok=True)
        rdf.fit_subplot_legends(fig, grid, first_panel_row=0)
        fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=rdf.SOURCES, default=DEFAULT_SOURCE)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results",
                        help="Root directory containing per-source RDF results.")
    parser.add_argument("--plots-dir", type=Path, default=HERE / "plots")
    parser.add_argument("--rdf-csv-dir", type=Path,
                        help="Override the source's per-model score directory.")
    parser.add_argument("--rdf-save-dir", type=Path,
                        help="Override the source's saved RDF curve directory.")
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--data-only", action="store_true",
                        help="Report selected curves and Pt3Co tier means without creating a figure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = args.rdf_csv_dir or args.results_dir / args.source
    saved = args.rdf_save_dir or scores / "rdf_same_simulation_length_saved"
    output = args.output_file or args.plots_dir / args.source / "figure_SI_4.pdf"
    reference, panels = load_panel_data(scores, saved)
    print(f"Source: {args.source}; system: {SYSTEM}")
    for panel in panels:
        selected = ", ".join(label for label, _, _ in panel.curves) or "reference only"
        print(f"{panel.label}: Pt3Co mean RDF error {panel.mean_error:.3f}%; curves: {selected}")
    if not args.data_only:
        plot_panel(reference, panels, output)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
