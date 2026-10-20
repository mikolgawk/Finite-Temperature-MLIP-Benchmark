"""Check corrected VDOS path isolation and unchanged stable-only correlations."""

from pathlib import Path
import contextlib
import io
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import matplotlib.pyplot as plt

try:
    from . import compare_results, figure_5, figure_6, figure_SI_7_8_9, figure_SI_16, figure_SI_18, plot_all, plot_dependent, run_vdos_pipeline, summarize_analysis
except ImportError:
    import compare_results
    import figure_5
    import figure_6
    import figure_SI_7_8_9
    import figure_SI_16
    import figure_SI_18
    import plot_all
    import plot_dependent
    import run_vdos_pipeline
    import summarize_analysis

import md_success


EAGER = "mlip-trajs-torchsim-eager"
ACCELERATED = "mlip-trajs-torchsim-accelerated"
ASE_SOURCES = ("mlip-trajs-ase", "mlip-trajs-ase-accelerated")
CU = "bulkCu_1000K_Kapil"
AG = "bulkAg_600K_Kapil"
MODEL = "mace-mp-0"


class UpdatedPlotPipelineTest(unittest.TestCase):
    def test_default_comparison_uses_legacy_folder_instead_of_active_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            active = root / "vdos"
            with (
                patch.object(compare_results, "HERE", active),
                patch.object(sys, "argv", ["compare_results.py", "--source", EAGER]),
                patch.object(compare_results.pd, "read_csv", side_effect=RuntimeError("stop before loading")) as load,
                self.assertRaisesRegex(RuntimeError, "stop before loading"),
            ):
                compare_results.main()
            load.assert_called_once_with(root / "vdos_legacy/results" / EAGER / compare_results.MODELS)

    def test_summary_uses_distinct_legacy_and_archived_comparison_baselines(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            active = root / "vdos"
            results = root / "chosen-results"
            with (
                patch.object(summarize_analysis, "HERE", active),
                patch.object(sys, "argv", ["summarize_analysis.py", "--results-dir", str(results)]),
                patch.object(summarize_analysis.subprocess, "run", side_effect=[
                    None, RuntimeError("stop after routing"),
                ]) as run,
                self.assertRaisesRegex(RuntimeError, "stop after routing"),
            ):
                summarize_analysis.main()
            self.assertEqual(run.call_count, 2)
            for call, baseline, output in zip(run.call_args_list, (
                root / "vdos_legacy/results", active / "archive/periodogram_velocity_v1/results",
            ), ("comparison", "comparison_periodogram"), strict=True):
                command = call.args[0]
                self.assertEqual(Path(command[command.index("--original-results") + 1]), baseline)
                self.assertEqual(Path(command[command.index("--updated-results") + 1]), results)
                self.assertEqual(Path(command[command.index("--output-dir") + 1]), results / output)
            self.assertEqual(list(root.iterdir()), [])

    def test_shared_spectrum_scale_fits_visible_curves_without_legacy_caps(self):
        fig, axes = plt.subplots(4, 1)
        self.addCleanup(plt.close, fig)
        for ax in axes:
            ax.plot([0, 0.01, 0.02, 0.03], [0, 5, 1, 999])
            ax.set_xlim(0, 0.02)
        axes[2].plot([0, 0.01, 0.02], [0, 15, 0])
        axes[3].plot([0, 0.01, 0.02], [float("nan"), 2, 0])
        limits = figure_5.shared_visible_spectrum_ylim(list(axes))
        self.assertEqual(limits, (0.0, 16.5))
        for ax in axes:
            ax.set_ylim(*limits)
            self.assertEqual(ax.get_ylim(), (0.0, 16.5))

    def test_corrected_vdos_defaults_keep_original_non_vdos_inputs(self):
        config = plot_all.HERE.parent
        for source in (EAGER, ACCELERATED):
            force, rdf, vdos = figure_SI_7_8_9.source_paths(source)
            self.assertEqual(force.parent, config / "e_f_rmses/results" / source)
            self.assertEqual(rdf.parent, config / "rdfs/results" / source)
            self.assertEqual(vdos.parent, plot_all.HERE / "results" / source)
            timings, pareto_vdos = figure_SI_16.source_input_paths(source)
            self.assertEqual(timings, config / "data" / source)
            self.assertEqual(pareto_vdos, vdos)
        self.assertEqual(figure_5.SCRIPT_DIR, plot_all.HERE)
        self.assertEqual(figure_6.load_correlation_data, figure_SI_7_8_9.load_joined_data)
        self.assertEqual(
            figure_SI_7_8_9.correlation_output_path("figure_6", False).parent,
            plot_all.HERE / "plots",
        )

    def test_every_plot_receives_explicit_corrected_inputs_and_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(
                results_dir=root / "custom_results", plots_dir=root / "custom_plots",
                pressure_file=None,
                f1_file=plot_all.CONFIG_DIR / "data/matbench-scores/f1-scores.csv",
                ksrme_file=plot_all.CONFIG_DIR / "data/matbench-scores/ksrme-scores.csv",
                include_molecular_crystals=False,
            )
            commands = {Path(command[2]).name: command for _, command in plot_all.plot_stages(args, EAGER)}
            for script, option in (
                ("plot_vdos_results.py", "--model-means-file"),
                ("figure_5.py", "--model-mean-file"),
                ("figure_6.py", "--vdos-model-means-file"),
                ("figure_SI_7_8_9.py", "--vdos-model-means-file"),
                ("figure_SI_16.py", "--vdos-scores-file"),
            ):
                command = commands[script]
                self.assertEqual(Path(command[2]).parent, plot_all.HERE)
                self.assertEqual(
                    Path(command[command.index(option) + 1]),
                    args.results_dir / EAGER / plot_all.MODEL_MEANS,
                )
                for index, value in enumerate(command):
                    if value.endswith("output-file"):
                        self.assertEqual(Path(command[index + 1]).parent, args.plots_dir / EAGER)
                self.assertNotIn("--exclude-hydrogen", command)
                self.assertNotIn("--exclude-pt-water", command)
            pairs = commands["figure_5.py"]
            self.assertEqual(
                Path(pairs[pairs.index("--normalized-pairs-file") + 1]),
                args.results_dir / EAGER / plot_all.PAIR_ERRORS,
            )
            dependent = commands["plot_dependent.py"]
            self.assertEqual(Path(dependent[dependent.index("--results-dir") + 1]), args.results_dir)
            self.assertEqual(Path(dependent[dependent.index("--plots-dir") + 1]), args.plots_dir)

    def test_native_ase_stages_keep_supported_figures_and_input_overrides(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(
                results_dir=root / "results", plots_dir=root / "plots",
                pressure_file=root / "pressure.csv", f1_file=root / "f1.csv",
                ksrme_file=root / "ksrme.csv", include_molecular_crystals=True,
            )
            expected_scripts = {
                "plot_vdos_results.py", "figure_5.py", "figure_6.py",
                "figure_SI_7_8_9.py", "figure_SI_16.py",
            }
            for source in ASE_SOURCES:
                with self.subTest(source=source):
                    commands = {
                        Path(command[2]).name: command
                        for _, command in plot_all.plot_stages(args, source)
                    }
                    self.assertEqual(set(commands), expected_scripts)
                    for script, command in commands.items():
                        if script != "plot_vdos_results.py":
                            self.assertEqual(command[command.index("--source") + 1], source)
                        for index, value in enumerate(command):
                            if value.endswith("output-file"):
                                self.assertEqual(
                                    Path(command[index + 1]).parent, args.plots_dir / source,
                                )
                        self.assertEqual(
                            "--include-molecular-crystals" in command,
                            script in {"figure_5.py", "figure_SI_16.py"},
                        )
                    for script in ("figure_6.py", "figure_SI_7_8_9.py"):
                        command = commands[script]
                        for option, path in (
                            ("--pressure-file", args.pressure_file),
                            ("--f1-file", args.f1_file),
                            ("--ksrme-file", args.ksrme_file),
                            ("--vdos-model-means-file", args.results_dir / source / plot_all.MODEL_MEANS),
                        ):
                            self.assertEqual(Path(command[command.index(option) + 1]), path)
            self.assertEqual(list(root.iterdir()), [])

    def test_explicit_native_ase_pipeline_plot_command_skips_torchsim_only_stages(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            command = run_vdos_pipeline.plot_command(
                list(ASE_SOURCES), root / "results", root / "plots",
                include_molecular_crystals=True,
            )
            result = subprocess.run(
                [*command, "--dry-run"], check=True, text=True, capture_output=True,
            )
            commands = [
                shlex.split(line.removeprefix("$ "))
                for line in result.stdout.splitlines() if line.startswith("$ ")
            ]
            self.assertEqual(len(commands), 10)
            self.assertNotIn("plot_dependent.py", [Path(command[2]).name for command in commands])
            for source in ASE_SOURCES:
                self.assertIn(f"[SKIP] TorchSim-only dependent plot stage for {source}", result.stdout)
                source_commands = [
                    command for command in commands
                    if "--source" in command and command[command.index("--source") + 1] == source
                ]
                self.assertEqual(len(source_commands), 4)
            self.assertEqual(list(root.iterdir()), [])

    def test_mixed_runner_dry_run_keeps_dependent_stages_on_torchsim_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            command = run_vdos_pipeline.plot_command(
                [EAGER, ASE_SOURCES[1], ACCELERATED, ASE_SOURCES[1]],
                root / "results", root / "plots",
            )
            result = subprocess.run(
                [*command, "--dry-run"], check=True, text=True, capture_output=True,
            )
            commands = [
                shlex.split(line.removeprefix("$ "))
                for line in result.stdout.splitlines() if line.startswith("$ ")
            ]
            self.assertEqual(len(commands), 17)
            dependent_sources = [
                command[command.index("--source") + 1]
                for command in commands if Path(command[2]).name == "plot_dependent.py"
            ]
            self.assertEqual(dependent_sources, [EAGER, ACCELERATED])
            self.assertEqual(list(root.iterdir()), [])

    def test_default_computation_sources_are_available_torchsim_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for source in (*run_vdos_pipeline.DEFAULT_SOURCES, "mlip-trajs-ase-accelerated"):
                path = root / source / CU / f"nvt_{MODEL}.h5"
                path.parent.mkdir(parents=True)
                path.touch()
            self.assertEqual(run_vdos_pipeline.available_sources(root), [EAGER, ACCELERATED])
        command = run_vdos_pipeline.plot_command([EAGER, ACCELERATED], Path("/tmp/results"), Path("/tmp/plots"))
        self.assertEqual(Path(command[2]), plot_all.HERE / "plot_all.py")
        self.assertEqual(command.count("--source"), 2)

    def test_pipeline_forwards_refined_defaults_to_isolated_computation(self):
        with patch.object(sys, "argv", ["run_vdos_pipeline.py"]):
            args = run_vdos_pipeline.parse_args()
        command = run_vdos_pipeline.computation_command(args, [ACCELERATED])
        self.assertEqual(Path(command[2]), plot_all.HERE / "get_normalized_VDOS.py")
        for option, expected in (
            ("--estimator", "welch"), ("--weighting", "mass"),
            ("--window-fs", "2000.0"), ("--min-segments", "4"),
            ("--pad-factor", "4"),
        ):
            self.assertEqual(command[command.index(option) + 1], expected)
        self.assertEqual(
            Path(command[command.index("--results-dir") + 1]),
            plot_all.HERE / "results",
        )

    def test_pipeline_forwards_explicit_diagnostic_and_sensitivity_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            argv = [
                "run_vdos_pipeline.py", "--estimator", "periodogram",
                "--weighting", "velocity", "--window-fs", "1000",
                "--min-segments", "8", "--results-dir", str(root / "results"),
                "--plots-dir", str(root / "plots"), "--compute-only", "--dry-run",
            ]
            with patch.object(sys, "argv", argv):
                args = run_vdos_pipeline.parse_args()
            command = run_vdos_pipeline.computation_command(args, [EAGER])
            for option, expected in (
                ("--estimator", "periodogram"), ("--weighting", "velocity"),
                ("--window-fs", "1000.0"), ("--min-segments", "8"),
                ("--results-dir", str(root / "results")),
            ):
                self.assertEqual(command[command.index(option) + 1], expected)
            self.assertIn("--dry-run", command)
            self.assertEqual(list(root.iterdir()), [])

    def test_pipeline_rejects_invalid_spectral_options(self):
        for option, value in (
            ("--estimator", "unknown"), ("--weighting", "unknown"),
            ("--window-fs", "0"), ("--window-fs", "-1"),
            ("--window-fs", "nan"), ("--window-fs", "inf"),
            ("--min-segments", "1"), ("--min-segments", "0"),
        ):
            with self.subTest(option=option, value=value):
                with patch.object(sys, "argv", ["run_vdos_pipeline.py", option, value]):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            run_vdos_pipeline.parse_args()
                self.assertEqual(raised.exception.code, 2)

    def test_si18_routes_both_refined_sources_and_retains_runner_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for source in figure_SI_18.SOURCES:
                path = root / source / figure_SI_18.MODEL_FILE
                path.parent.mkdir(parents=True)
                path.touch()

            def dataset(**kwargs):
                source = kwargs["dataset_label"]
                return pd.DataFrame({
                    "dataset": [source], "model": [MODEL],
                    "mean_time_per_step_ms": [1 if source == ACCELERATED else 2],
                    "Combined Error [%]": [10 if source == ACCELERATED else 20],
                })

            with (patch.object(figure_SI_18.pareto, "load_and_merge", side_effect=dataset) as load,
                  patch.object(figure_SI_18.pareto, "apply_matched_runner_timings",
                               side_effect=lambda frame, **kwargs: frame) as matched):
                combined = figure_SI_18.load_comparison(root, include_molecular_crystals=True)
            self.assertEqual(len(load.call_args_list), 2)
            matched.assert_called_once()
            self.assertTrue(matched.call_args.kwargs["include_molecular_crystals"])
            self.assertEqual(matched.call_args.kwargs["timings_dirs"], {
                source: figure_SI_18.pareto.source_input_paths(source)[0]
                for source in figure_SI_18.SOURCES
            })
            for source, call in zip(figure_SI_18.SOURCES, load.call_args_list, strict=True):
                timings, rdf, _, pressure_mode = figure_SI_18.pareto.source_input_paths(source)
                self.assertEqual(call.kwargs["vdos_metrics_file"], root / source / figure_SI_18.MODEL_FILE)
                self.assertEqual(call.kwargs["timings_dir"], timings)
                self.assertEqual(call.kwargs["rdf_metrics_file"], rdf)
                self.assertEqual(call.kwargs["pressure_metrics_file"], figure_SI_18.pareto.pressure_path(source))
                self.assertEqual(call.kwargs["pressure_mode"], pressure_mode)
                self.assertEqual(call.kwargs["pressure_backend"], "torchsim" if source == ACCELERATED else "ase")
                self.assertIsNone(call.kwargs["combined_metrics_file"])
                self.assertTrue(call.kwargs["include_molecular_crystals"])
            # The ASE point is dominated across runners, but must remain on
            # its own runner's independent frontier.
            self.assertTrue(combined["is_pareto"].all())

    def test_si18_missing_refined_source_fails_before_any_legacy_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            refined = root / "results" / ACCELERATED / figure_SI_18.MODEL_FILE
            refined.parent.mkdir(parents=True)
            refined.touch()
            with patch.object(figure_SI_18.pareto, "load_and_merge") as load:
                with self.assertRaisesRegex(FileNotFoundError, "mlip-trajs-ase-accelerated"):
                    figure_SI_18.load_comparison(root / "results")
            load.assert_not_called()
            self.assertEqual(sorted(root.rglob("*")), [root / "results", refined.parent, refined])

    def test_si18_data_only_keeps_all_exports_in_refined_custom_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            merged = pd.DataFrame({
                "dataset": list(figure_SI_18.SOURCES), "model": [MODEL, MODEL],
                "mean_time_per_step_ms": [1, 2], "Combined Error [%]": [10, 20],
                "is_pareto": [True, True],
            })
            with patch.object(figure_SI_18, "load_comparison", return_value=merged) as load:
                with patch.object(figure_SI_18.pareto, "plot_pareto") as plot:
                    with contextlib.redirect_stdout(io.StringIO()):
                        figure_SI_18.main([
                            "--results-dir", str(root / "results"),
                            "--plots-dir", str(root / "plots"), "--data-only",
                        ])
            load.assert_called_once_with(root / "results", include_molecular_crystals=False)
            plot.assert_not_called()
            output = root / "results" / figure_SI_18.OUTPUT_GROUP
            pd.testing.assert_frame_equal(pd.read_csv(output / "figure_SI_18_metrics.csv"), merged)
            pd.testing.assert_frame_equal(pd.read_csv(output / "figure_SI_18_frontier.csv"), merged.sort_values(["dataset", "mean_time_per_step_ms"]).reset_index(drop=True))
            self.assertFalse((root / "plots").exists())

    def test_si18_keeps_font_size_with_more_room_for_nearby_model_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            merged = pd.DataFrame({
                "dataset": list(figure_SI_18.SOURCES), "model": [MODEL, MODEL],
                "mean_time_per_step_ms": [1, 2], "Combined Error [%]": [10, 20],
                "is_pareto": [True, True],
            })
            with patch.object(figure_SI_18, "load_comparison", return_value=merged):
                with patch.object(figure_SI_18.pareto, "plot_pareto") as plot:
                    with contextlib.redirect_stdout(io.StringIO()):
                        figure_SI_18.main([
                            "--results-dir", str(root / "results"),
                            "--plots-dir", str(root / "plots"),
                        ])
            plot.assert_called_once()
            self.assertEqual(plot.call_args.kwargs, {
                "layout": "overlay", "font_size": 8, "label_distance_points": 24,
            })
            self.assertEqual(plot.call_args.args[1], root / "plots" / figure_SI_18.OUTPUT_GROUP / "figure_SI_18.pdf")
            self.assertFalse((root / "plots").exists())

    def test_dependent_pipeline_adds_si18_once_only_when_both_refined_sources_exist(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            results = root / "results"
            plots = root / "plots"
            torchsim = results / ACCELERATED / plot_dependent.MODEL_FILE
            torchsim.parent.mkdir(parents=True)
            torchsim.touch()
            before = plot_dependent.commands(ACCELERATED, results, plots)
            self.assertNotIn("figure_SI_18.py", [Path(command[2]).name for command in before])
            ase = results / "mlip-trajs-ase-accelerated" / plot_dependent.MODEL_FILE
            ase.parent.mkdir(parents=True)
            ase.touch()
            stages = [
                *plot_dependent.commands(EAGER, results, plots),
                *plot_dependent.commands(ACCELERATED, results, plots),
            ]
            comparison = [command for command in stages if Path(command[2]).name == "figure_SI_18.py"]
            self.assertEqual(len(comparison), 1)
            self.assertEqual(comparison[0][comparison[0].index("--results-dir") + 1], str(results))
            self.assertEqual(comparison[0][comparison[0].index("--plots-dir") + 1], str(plots))
            self.assertEqual(len(before), 4)
            self.assertEqual(len(stages), 9)

    def test_dry_run_plots_never_create_output_or_select_extra_sets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = subprocess.run(
                [sys.executable, str(plot_all.HERE / "plot_all.py"), "--dry-run",
                 "--results-dir", str(root / "results"), "--plots-dir", str(root / "plots")],
                check=True, text=True, capture_output=True,
            )
            self.assertIn(EAGER, result.stdout)
            self.assertIn(ACCELERATED, result.stdout)
            self.assertNotIn("--exclude-hydrogen", result.stdout)
            self.assertNotIn("--exclude-pt-water", result.stdout)
            self.assertEqual(list(root.iterdir()), [])

    def test_stable_correlations_preserve_genuine_100_percent_and_exclude_failed_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data"
            for system in (CU, AG):
                path = data / EAGER / system
                path.mkdir(parents=True)
                (path / f"nvt_{MODEL}.h5").touch()
                (path / f"md_timing_{MODEL}.csv").write_text(
                    f"calculator,system,n_steps\n{MODEL},{system},100\n"
                )
            args = SimpleNamespace(
                source=EAGER, md_data_dir=data,
                force_rmse_input=root / "force.csv", rdf_file=root / "rdf.csv",
                pressure_file=root / "pressure.csv", pressure_backend=None, pressure_mode=None,
                vdos_model_means_file=root / "vdos.csv", f1_file=root / "f1.csv",
                ksrme_file=root / "ksrme.csv", exclude_hydrogen_force_rmse=False,
                exclude_pt_water=False,
            )
            for filename, column, values in (
                ("force.csv", "force_rmse", [2, 900]),
                ("rdf.csv", "rdf_error_percent", [100, 100]),
                ("pressure.csv", "pressure_error_percent", [30, 100]),
                ("vdos.csv", "vdos_error_percent", [100, 100]),
            ):
                pd.DataFrame({
                    "calculator": [MODEL, MODEL], "system": [CU, AG],
                    "source": [EAGER, EAGER], column: values,
                }).to_csv(root / filename, index=False)
            for filename, column in (("f1.csv", "f1_score"), ("ksrme.csv", "ksrme_score")):
                pd.DataFrame({"calculator": [MODEL], column: [0.8]}).to_csv(root / filename, index=False)
            original = args.vdos_model_means_file.read_bytes()
            with patch.object(md_success, "load_failed_md_runs", return_value={(EAGER, AG, MODEL): "unstable"}):
                main = figure_6.load_joined_data(args)
                si = figure_SI_7_8_9.load_joined_data(args)
            pd.testing.assert_frame_equal(main, si)
            self.assertEqual(main.loc[0, ["force_rmse", "rdf_error_percent", "pressure_error_percent", "vdos_error_percent"]].tolist(), [2, 100, 30, 100])
            self.assertEqual(args.vdos_model_means_file.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
