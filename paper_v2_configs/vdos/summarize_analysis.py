#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=1.26", "pandas>=2.2", "matplotlib>=3.9", "seaborn>=0.13"]
# ///
"""Export correlation inputs, comparisons and a readable refined-VDOS report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

import figure_5 as panels
import figure_6 as correlations


HERE = Path(__file__).resolve().parent
SOURCES = ("mlip-trajs-torchsim-eager", "mlip-trajs-torchsim-accelerated")
MODEL_FILE = "vdos_model_mean_ev_normalized_same_simulation_length.csv"
PAIR_FILE = "vdos_pair_errors_ev_normalized_same_simulation_length.csv"


def ase_comparison_report(results: Path) -> list[str]:
    """Report the optional native ASE calculation and its complete-metric subset."""
    source = "mlip-trajs-ase-accelerated"
    path = results / source / PAIR_FILE
    if not path.is_file():
        return []
    pairs = pd.read_csv(path)
    failed = pairs.failure_reason.fillna("").ne("")
    lines = ["", "## ASE accelerated VDOS and Figure SI18", "",
             f"Native ASE accelerated VDOS has been recomputed with the same refined "
             f"mass-weighted Welch settings: {pairs.model.nunique()} configurations, "
             f"{pairs.system.nunique()} systems and {len(pairs)} expected model/system pairs. "
             f"There are {(~failed).sum()} successful pairs and {failed.sum()} missing, "
             "incomplete or failed pairs carrying 100% error. These results do not "
             "introduce eager substitutions into the native ASE cohort.", ""]
    comparison = results / "torchsim_vs_ase_accelerated/figure_SI_18_metrics.csv"
    if comparison.is_file():
        metrics = pd.read_csv(comparison)
        lines += ["Figure SI18 uses refined VDOS for both runners, with the existing "
                  "runner-specific RDF, pressure and timing inputs. Each runner has an "
                  "independent Pareto frontier.", "",
                  "| Runner/cohort | Plotted configurations | Pareto configurations |",
                  "|---|---:|---:|"]
        for runner, frame in metrics.groupby("dataset", sort=True):
            lines.append(f"| {runner} | {len(frame)} | {frame.is_pareto.sum()} |")
        included = metrics.loc[metrics.dataset.eq(source), "model"].map(panels.normalize_model_name)
        absent = sorted(set(pairs.model.map(panels.normalize_model_name)) - set(included))
        if absent:
            lines += ["", "ASE configurations without a full set of RDF, pressure and timing "
                      "inputs are excluded from the combined plot: "
                      + ", ".join(panels.display_name(model) for model in absent) + "."]
        lines += ["", "The TorchSim cohort retains its documented eager EQ/eSEN and "
                  "ASE-eager CHGNet substitutions. Missing native ASE metric inputs "
                  "are not replaced with TorchSim values. The separate plotting command "
                  "is `uv run --script paper_v2_configs/vdos/figure_SI_18.py`.", ""]
    return lines


def sensitivity_report(results: Path) -> list[str]:
    """Describe completed diagnostics and verify their baseline matches the run."""
    path = results / "sensitivity/summary.json"
    if not path.is_file():
        return ["Sensitivity diagnostics have not yet been generated.", ""]
    summary = json.loads(path.read_text())
    pairs = pd.read_csv(path.parent / "per_system.csv")
    baseline = pairs.loc[pairs.configuration.eq(summary["baseline_configuration"])]
    keys = ["source", "model", "system"]
    active = pd.concat([pd.read_csv(results / source / PAIR_FILE)
                        for source in summary["sources"]], ignore_index=True)
    joined = active.merge(baseline, on=keys, suffixes=("_active", "_sensitivity"),
                          validate="one_to_one", how="outer", indicator=True)
    if not joined["_merge"].eq("both").all() or not np.allclose(
        joined.vdos_error_percent_active, joined.vdos_error_percent_sensitivity,
        rtol=0, atol=1e-9,
    ):
        raise ValueError("Sensitivity baseline does not reproduce the active model/system scores")
    labels = {
        "mass_2ps_min4_pad4": "Default: mass, 2 ps, minimum 4 segments",
        "mass_1ps_min4_pad4": "Mass, 1 ps, minimum 4 segments",
        "mass_4ps_min4_pad4": "Mass, 4 ps, minimum 4 segments",
        "mass_2ps_min8_pad4": "Mass, 2 ps, minimum 8 segments",
        "velocity_2ps_min4_pad4": "Unweighted, 2 ps, minimum 4 segments",
        "mass_2ps_min4_pad8": "Default with eightfold FFT padding",
    }
    lines = [
        f"The full {', '.join(summary['sources'])} cohort was scored under all six "
        f"settings: {summary['models']} model configurations, {summary['systems']} systems "
        f"and {summary['pairs_per_configuration']} pairs per setting, including "
        f"{summary['failure_penalties_per_configuration']} unchanged failure penalties. "
        "The diagnostic baseline reproduces the active per-system scores to 1e-9 percentage points.", "",
        "Changes below refer to each model's failure-penalized benchmark mean; "
        "all scores are compared with the default.", "",
        "| Setting | Median absolute change (pp) | Maximum absolute change (pp) | Spearman rank correlation | Largest rank change |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in summary["comparisons"].items():
        lines.append(
            f"| {labels.get(name, name)} | {item['median_absolute_model_error_change_pp']:.3f} | "
            f"{item['maximum_absolute_model_error_change_pp']:.3f} | "
            f"{item['rank_spearman_vs_baseline']:.4f} | {item['maximum_absolute_rank_change']:.0f} |"
        )
    windows = [summary["comparisons"][name] for name in (
        "mass_1ps_min4_pad4", "mass_4ps_min4_pad4", "mass_2ps_min8_pad4",
    )]
    lines += ["", f"Across the tested mass-weighted window settings, model-mean rank "
              f"correlations are at least {min(item['rank_spearman_vs_baseline'] for item in windows):.4f}; "
              f"the largest model-mean change is "
              f"{max(item['maximum_absolute_model_error_change_pp'] for item in windows):.3f} pp. "
              "This supports stability of the overall accelerated VDOS ranking across these "
              "settings, rather than convergence of every system-specific error. The 2 ps, "
              "four-segment default is retained as a compromise between averaging and "
              "frequency resolution; settings were not selected to minimize benchmark errors."]
    lines += ["", "| System | Median score range across mass-weighted window settings (pp) | Maximum range (pp) |",
              "|---|---:|---:|"]
    for system, item in summary["systems_diagnostics"].items():
        lines.append(f"| {system} | {item['median_window_score_range_pp']:.3f} | "
                     f"{item['maximum_window_score_range_pp']:.3f} |")
    lines += ["", "The system ranges use successful pairs only and vary the 1/2/4 ps target "
              "and the minimum of four/eight segments. Short records may share identical "
              "effective windows for several requested targets. The unweighted probe changes "
              "the observable; the padding probe changes the integration grid, not physical "
              "resolution. These diagnostics measure method sensitivity, not confidence "
              "intervals or convergence with MD trajectory length.", ""]
    impact_path = path.parent / "paper_impact_summary.json"
    if impact_path.is_file():
        impact = json.loads(impact_path.read_text())
        frontiers = list(impact["frontiers"].values())
        if all(not item["added_vs_baseline"] and not item["removed_vs_baseline"] for item in frontiers):
            lines += [f"All {len(frontiers)} settings retain the same "
                      f"{frontiers[0]['n_frontier_configurations']} accelerated combined RPV "
                      "Pareto configurations, with RDF, pressure and timings held fixed.", ""]
        lines += ["Stable-MD VDOS correlation sensitivity:", "",
                  "| Setting | Force RMSE r (n) | F1 r (n) | kappa SRME r (n) |",
                  "|---|---:|---:|---:|"]
        for name, item in impact["stable_vdos_correlations"].items():
            values = [item[key] for key in ("force_rmse", "f1_score", "ksrme_score")]
            lines.append(f"| {labels.get(name, name)} | "
                         + " | ".join(f"{value['pearson_r']:.4f} ({value['n_models']})" for value in values) + " |")
        lines += ["", "The correlation directions and relative strengths remain qualitatively "
                  "consistent under these choices. These are correlations across models with "
                  "model-specific stable-system subsets, not significance tests or a guarantee "
                  "of predictive accuracy for a particular system.", ""]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--output-file", type=Path, default=HERE / "analysis_summary.md")
    args = parser.parse_args()
    results = args.results_dir.resolve()
    for baseline, output in (
        (HERE.parent / "vdos_legacy/results", results / "comparison"),
        (HERE / "archive/periodogram_velocity_v1/results", results / "comparison_periodogram"),
    ):
        subprocess.run([
            sys.executable, str(HERE / "compare_results.py"),
            "--original-results", str(baseline), "--updated-results", str(results),
            "--output-dir", str(output),
        ], check=True, stdout=subprocess.DEVNULL)
    summary = json.loads((results / "comparison/summary.json").read_text())
    periodogram_summary = json.loads((results / "comparison_periodogram/summary.json").read_text())
    lines = [
        "# Refined VDOS analysis", "",
        "The current results use mass-weighted Hann-windowed Welch spectra, "
        "50% overlap, a 2 ps target window, at least four overlapping segments "
        "and fourfold FFT padding. Reference and MLIP velocities share the "
        "same unwrapping and central-difference preprocessing. The displayed "
        "spectra are the spectra used for scoring.", "",
        "## Coverage and model errors", "",
        "| Cohort | Models/configurations | Systems | Pairs | Failure penalties | Legacy median | First corrected periodogram median | Refined median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source in SOURCES:
        item = summary[source]
        lines.append(
            f"| {source.removeprefix('mlip-trajs-torchsim-')} | {item['models']} | "
            f"{item['systems']} | {item['model_system_pairs']} | {item['failure_penalties']} | "
            f"{item['original_median_model_error_percent']:.3f}% | "
            f"{periodogram_summary[source]['original_median_model_error_percent']:.3f}% | "
            f"{item['updated_median_model_error_percent']:.3f}% |"
        )
    lines += [
        "", "Model/system coverage, physical source mappings and failure "
        "classification are unchanged. Failed pairs retain 100% error. "
        "Molecular crystals remain excluded; hydrogen and Pt/H2O remain in VDOS.", "",
        "## Refined tier medians", "",
        "| Cohort | Tier 1 | Tier 2 | Tier 3 | Tier 4 |", "|---|---:|---:|---:|---:|",
    ]
    coefficient_rows = {}
    for source in SOURCES:
        directory = results / source
        means = pd.read_csv(directory / MODEL_FILE)
        means["key"] = means.model.map(panels.normalize_model_name)
        medians = [means.loc[means.key.isin([panels.normalize_model_name(m) for m in models]),
                             "vdos_error_percent"].median()
                   for _, models, _ in panels.TIER_PANELS]
        lines.append("| " + source.removeprefix("mlip-trajs-torchsim-") + " | "
                     + " | ".join(f"{value:.3f}%" for value in medians) + " |")
        inputs = correlations.parse_args([
            "--source", source, "--vdos-model-means-file", str(directory / MODEL_FILE),
        ])
        joined = correlations.load_joined_data(inputs)
        joined.to_csv(directory / "stable_correlation_metrics.csv", index=False)
        coefficients = []
        for y, _ in correlations.Y_METRICS:
            for x, _ in correlations.X_METRICS:
                paired = joined[[x, y]].dropna()
                coefficients.append({"x_metric": x, "y_metric": y, "n_models": len(paired),
                                     "pearson_r": paired[x].corr(paired[y])})
        coefficients = pd.DataFrame(coefficients)
        coefficients.to_csv(directory / "stable_correlation_coefficients.csv", index=False)
        coefficient_rows[source] = coefficients
        (directory / "figure_6_outlier_labels.json").write_text(
            json.dumps(correlations.outlier_label_summary(joined, 2.5, None), indent=2) + "\n"
        )
        pairs = pd.read_csv(directory / PAIR_FILE)
        resolution_columns = ["system", "timestep_fs", "segment_duration_fs", "n_segments",
                              "resolution_cm-1", "hann_noise_equivalent_bandwidth_cm-1",
                              "n_fft", "bin_spacing_cm-1"]
        pairs.loc[pairs.failure_reason.fillna("").eq(""), resolution_columns].drop_duplicates().to_csv(
            directory / "spectral_resolution.csv", index=False,
        )
    lines += ["", "## Stable-MD VDOS correlations", "",
              "| Cohort | Force RMSE | F1 | kappa SRME |", "|---|---:|---:|---:|"]
    for source, frame in coefficient_rows.items():
        frame = frame.loc[frame.y_metric.eq("vdos_error_percent")].set_index("x_metric")
        values = [frame.loc[key, "pearson_r"] for key in ("force_rmse", "f1_score", "ksrme_score")]
        lines.append("| " + source.removeprefix("mlip-trajs-torchsim-") + " | "
                     + " | ".join(f"{value:.6f}" for value in values) + " |")
    lines += ["", "Figure 6 and SI7--9 use each model's stable-system subset. "
              "SI11/SI12 and combined Pareto figures use failure-penalized benchmark means.", "",
              "## Combined Pareto frontier", ""]
    for source in SOURCES:
        frontier = pd.read_csv(results / source / "figure_7_frontier.csv")
        lines.append(f"- {source.removeprefix('mlip-trajs-torchsim-')} ({len(frontier)} configurations): "
                     + ", ".join(panels.display_name(model) for model in frontier.model) + ".")
    lines += ase_comparison_report(results)
    resolution = pd.read_csv(results / SOURCES[1] / "spectral_resolution.csv")
    lines += ["", "## Actual accelerated spectral resolution", "",
              "Native spacing is 1/window duration; the Hann noise-equivalent bandwidth "
              "is wider. Neither quantity improves with zero padding.", "",
              "| System | Window (fs) | Segments | Native spacing (cm^-1) | Hann bandwidth (cm^-1) |",
              "|---|---:|---:|---:|---:|"]
    for system, frame in resolution.groupby("system", sort=True):
        def value_range(column: str, digits: int) -> str:
            lower, upper = frame[column].min(), frame[column].max()
            return (f"{lower:.{digits}f}" if lower == upper
                    else f"{lower:.{digits}f}--{upper:.{digits}f}")
        lines.append(f"| {system} | {value_range('segment_duration_fs', 1)} | "
                     f"{value_range('n_segments', 0)} | {value_range('resolution_cm-1', 3)} | "
                     f"{value_range('hann_noise_equivalent_bandwidth_cm-1', 3)} |")
    lines += ["", "## Interpretation and sensitivity", "",
              "The default has a 2 ps target, not a universal 2 ps effective window. "
              "Hydrogen and CuAu require shorter windows and therefore have coarser resolution. "
              "Overlapping segments are not independent trajectories. Finite-length uncertainty "
              "remains, and close model differences should be interpreted with the sensitivity results.", "",
              "Comparisons to the legacy and first corrected periodogram runs are in "
              "results/comparison/ and results/comparison_periodogram/. The mass weighting changes "
              "the observable for mixed-element systems; differences are not solely due to averaging.", "",
              "Requested 1/2/4 ps windows, stronger short-record averaging, unweighted spectra and "
              "a denser FFT grid are assessed in results/sensitivity/.", ""]
    lines += sensitivity_report(results)
    lines += ["Lower refined errors do not demonstrate improved model accuracy: the spectral "
              "estimator, bandwidth and mass convention changed. Close rankings and membership "
              "of the combined Pareto frontier should be assessed under those choices before "
              "revising claims in the paper.", "",
              "No MD integrations or model inference were rerun. The original vdos directory "
              "is retained for comparison. See README.md and methods_vdos.tex for methods.", ""]
    args.output_file.write_text("\n".join(lines))
    print(f"Saved {args.output_file}")


if __name__ == "__main__":
    main()
