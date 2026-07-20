#!/usr/bin/env python3
"""Run the matched-length VDOS batch, then normalize and plot the results.

This combines two previously separate steps into a single command:

1. `run_vdos_hann_batch_same_simulation_length.py`: for each MLIP trajectory,
   truncate MLIP/reference trajectories to a matched number of post-stride
   frames and run `VDOS.py` on both, producing:
   - MLIP:      <output-root>/mlip/<system_dir>/<nvt_stem>_vdos_hann.dat
   - Reference: <output-root>/reference_matched/<system_dir>/
                traj_match_<nvt_stem>_vdos_hann.dat

2. `normalize_and_plot_vdos_matched_lengths.py`: convert those .dat files to
   an eV axis, normalize by area, compute similarity scores, save CSVs, and
   generate the combined tier/model VDOS panel figure.

Use --skip-batch to only (re-)run the normalize/plot step against an
existing `--output-root`, or --skip-plot to only run the batch VDOS step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_vdos_hann_batch_same_simulation_length as batch_mod
import normalize_and_plot_vdos_matched_lengths as plot_mod


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Run matched-length-of-simulation VDOS batch processing and then "
            "normalize/plot the resulting spectra in one step."
        )
    )

    batch_group = parser.add_argument_group("VDOS batch options")
    batch_group.add_argument(
        "--ref-root",
        type=Path,
        default="/share/rcif2/mjgawkowski/phd_mlip_matbench_benchmarks_from_hypatia/ref-trajs",
        help="Reference trajectories root",
    )
    batch_group.add_argument(
        "--mlip-root",
        type=Path,
        default="/share/rcif2/mjgawkowski/phd_mlip_matbench_benchmarks_from_hypatia/mlip-trajs",
        help="MLIP trajectories root",
    )
    batch_group.add_argument(
        "--ref-settings",
        type=Path,
        default=script_dir / "vdos_settings_ref.csv",
        help="CSV file with reference VDOS settings",
    )
    batch_group.add_argument(
        "--mlip-settings",
        type=Path,
        default=script_dir / "vdos_settings_mlip.csv",
        help="CSV file with MLIP VDOS settings",
    )
    batch_group.add_argument(
        "--vdos-script",
        type=Path,
        default=script_dir / "VDOS.py",
        help="Path to VDOS.py",
    )
    batch_group.add_argument(
        "--output-root",
        type=Path,
        default=script_dir / "vdos_results_hann_same_simulation_length_new_test",
        help="Output directory for .dat files, and root for the normalize/plot step",
    )
    batch_group.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter to run get_VDOS_padding_hann.py",
    )
    batch_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .dat outputs",
    )
    batch_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print batch actions without running VDOS (implies --skip-plot)",
    )
    batch_group.add_argument(
        "--skip-batch",
        action="store_true",
        help="Skip the VDOS batch step and only normalize/plot existing --output-root data",
    )

    plot_group = parser.add_argument_group("Normalize/plot options")
    plot_group.add_argument(
        "--mlip-subdir",
        default="mlip",
        help="MLIP VDOS subdirectory name under --output-root",
    )
    plot_group.add_argument(
        "--ref-subdir",
        default="reference_matched",
        help="Reference VDOS subdirectory name under --output-root",
    )
    plot_group.add_argument(
        "--e-min",
        type=float,
        default=None,
        help="Optional minimum energy in eV used for similarity scoring",
    )
    plot_group.add_argument(
        "--e-max",
        type=float,
        default=None,
        help="Optional maximum energy in eV used for similarity scoring",
    )
    plot_group.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable area normalization before similarity scoring",
    )
    plot_group.add_argument(
        "--max-mlip-per-system",
        type=int,
        default=0,
        help="Limit MLIP trajectories/files per system for both steps (0 means no limit)",
    )
    plot_group.add_argument(
        "--pair-output",
        type=Path,
        default=script_dir / "vdos_pair_errors_ev_normalized_same_simulation_length.csv",
        help="Output CSV for pairwise normalized VDOS similarity rows",
    )
    plot_group.add_argument(
        "--system-model-mean-output",
        type=Path,
        default=script_dir / "vdos_system_model_mean_ev_normalized_same_simulation_length.csv",
        help="Output CSV for mean score per (system, model)",
    )
    plot_group.add_argument(
        "--model-mean-output",
        type=Path,
        default=script_dir / "vdos_model_mean_ev_normalized_same_simulation_length.csv",
        help="Output CSV for mean score per model",
    )
    plot_group.add_argument(
        "--plot-output",
        type=Path,
        default=script_dir / "plot_vdos_panel_combined_same_simulation_length.pdf",
        help="Output path for combined plot figure",
    )
    plot_group.add_argument(
        "--no-plot",
        action="store_true",
        help="Only compute CSV outputs; skip plot generation",
    )
    plot_group.add_argument(
        "--show",
        action="store_true",
        help="Show plot window in addition to saving file",
    )
    plot_group.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip the normalize/plot step and only run the VDOS batch step",
    )

    return parser.parse_args()


def run_batch_step(args: argparse.Namespace, output_root: Path) -> None:
    ref_root = args.ref_root.resolve()
    mlip_root = args.mlip_root.resolve()
    ref_settings_path = args.ref_settings.resolve()
    mlip_settings_path = args.mlip_settings.resolve()
    vdos_script = args.vdos_script.resolve()
    python_executable = args.python.resolve()

    batch_mod.ensure_file_exists(vdos_script, "VDOS script")
    batch_mod.ensure_file_exists(ref_settings_path, "Reference settings CSV")
    batch_mod.ensure_file_exists(mlip_settings_path, "MLIP settings CSV")
    batch_mod.ensure_dir_exists(ref_root, "Reference root")
    batch_mod.ensure_dir_exists(mlip_root, "MLIP root")

    ref_settings = batch_mod.load_settings(ref_settings_path)
    mlip_settings = batch_mod.load_settings(mlip_settings_path)

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    print("=== Step 1/2: VDOS batch (matched simulation lengths) ===")
    print("Configuration")
    print(f"- VDOS script:    {vdos_script}")
    print(f"- Python:         {python_executable}")
    print(f"- Ref root:       {ref_root}")
    print(f"- MLIP root:      {mlip_root}")
    print(f"- Output root:    {output_root}")
    print(f"- Dry run:        {args.dry_run}")
    print(f"- Overwrite:      {args.overwrite}")

    success, failure = batch_mod.process_all_systems(
        ref_settings=ref_settings,
        mlip_settings=mlip_settings,
        ref_root=ref_root,
        mlip_root=mlip_root,
        output_root=output_root,
        python_executable=python_executable,
        vdos_script=vdos_script,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        max_mlip_per_system=args.max_mlip_per_system,
    )

    print("\nBatch step summary")
    print(f"- Successful runs: {success}")
    print(f"- Failed/skipped:  {failure}")

    if failure:
        raise SystemExit(
            f"Error: VDOS batch step reported {failure} failure(s); "
            "not proceeding to normalize/plot step."
        )


def run_plot_step(args: argparse.Namespace, output_root: Path) -> None:
    mlip_root = (output_root / args.mlip_subdir).resolve()
    ref_root = (output_root / args.ref_subdir).resolve()

    pair_output = args.pair_output.resolve()
    system_model_mean_output = args.system_model_mean_output.resolve()
    model_mean_output = args.model_mean_output.resolve()
    plot_output = args.plot_output.resolve()

    if not mlip_root.is_dir():
        raise NotADirectoryError(f"MLIP directory not found: {mlip_root}")
    if not ref_root.is_dir():
        raise NotADirectoryError(f"Reference directory not found: {ref_root}")
    if args.e_min is not None and args.e_max is not None and args.e_min >= args.e_max:
        raise ValueError("--e-min must be smaller than --e-max")

    print("\n=== Step 2/2: Normalize VDOS and plot ===")
    print("Configuration")
    print(f"- VDOS root:      {output_root}")
    print(f"- MLIP dir:       {mlip_root}")
    print(f"- Ref dir:        {ref_root}")
    print(f"- Normalize area: {not args.no_normalize}")
    print(f"- e_min (eV):     {args.e_min}")
    print(f"- e_max (eV):     {args.e_max}")
    print(f"- Pair output:    {pair_output}")
    print(f"- Model output:   {model_mean_output}")
    print(f"- Plot output:    {plot_output}")

    pair_df, systems_count, total_pairs = plot_mod.compute_pairwise_scores(
        mlip_root=mlip_root,
        ref_root=ref_root,
        e_min=args.e_min,
        e_max=args.e_max,
        normalize=not args.no_normalize,
        max_mlip_per_system=args.max_mlip_per_system,
    )

    system_model_mean_df, model_mean_df = plot_mod.build_aggregate_tables(pair_df)

    pair_output.parent.mkdir(parents=True, exist_ok=True)
    system_model_mean_output.parent.mkdir(parents=True, exist_ok=True)
    model_mean_output.parent.mkdir(parents=True, exist_ok=True)

    pair_df.to_csv(pair_output, index=False)
    system_model_mean_df.to_csv(system_model_mean_output, index=False)
    model_mean_df.to_csv(model_mean_output, index=False)

    if not args.no_plot:
        plot_mod.plot_combined(
            norm_df=pair_df,
            model_means_df=model_mean_df,
            output_file=plot_output,
            show=args.show,
        )
        print(f"Saved combined VDOS panel plot to {plot_output}")

    print("\n================ Plot step summary ================")
    print(f"Systems matched: {systems_count}")
    print(f"Pairs attempted: {total_pairs}")
    print(f"Pairs scored:    {len(pair_df)}")
    print(f"Saved pairwise scores: {pair_output}")
    print(f"Saved system-model means: {system_model_mean_output}")
    print(f"Saved model means: {model_mean_output}")


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()

    if args.skip_batch and args.skip_plot:
        raise SystemExit("Error: --skip-batch and --skip-plot cannot both be set.")

    if not args.skip_batch:
        run_batch_step(args, output_root)
    else:
        print("=== Step 1/2: VDOS batch skipped (--skip-batch) ===")

    if args.dry_run:
        print(
            "\nDry run requested for the batch step; skipping normalize/plot "
            "step since no .dat files were produced."
        )
        return 0

    if not args.skip_plot:
        run_plot_step(args, output_root)
    else:
        print("\n=== Step 2/2: Normalize/plot skipped (--skip-plot) ===")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
