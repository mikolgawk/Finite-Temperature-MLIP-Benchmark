"""Check native ASE Pareto inputs and CSV exports without creating figures."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import figure_7
import figure_SI_18


ASE = "mlip-trajs-ase-accelerated"
TORCHSIM = "mlip-trajs-torchsim-accelerated"
METAL = "bulkCu_1000K_Kapil"
CRYSTAL = "naphthalene_295K_Sharma_S"


class Figure7ASETests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.failures = {}
        patcher = patch("md_success.load_failed_md_runs", return_value=self.failures)
        patcher.start()
        self.addCleanup(patcher.stop)

    def add_run(self, source=ASE, model="orb-v2-force-only-ase", system=METAL,
                seconds=0.001, row_fields=None, **fields):
        directory = self.root / "data" / source / system
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"md_timing_{model}.csv"
        row = {
            "calculator": model.removesuffix("-ase") if "ase" in source else model,
            "system": system, "n_steps": 100,
            "seconds_per_step": seconds, "elapsed_seconds": 100 * seconds,
        }
        row.update(fields)
        row.update(row_fields or {})
        pd.DataFrame([row]).to_csv(path, index=False)
        path.with_name(f"nvt_{model}.h5").touch()
        return path

    def write_metrics(self, source, models, errors):
        directory = self.root / "metrics" / source
        directory.mkdir(parents=True, exist_ok=True)
        rdf, vdos, pressure = (directory / name for name in ("rdf.csv", "vdos.csv", "pressure.csv"))
        pd.DataFrame({"Calculator": models, "rdf_error_percent": errors}).to_csv(rdf, index=False)
        pd.DataFrame({"model": models, "vdos_error_percent": errors}).to_csv(vdos, index=False)
        pd.DataFrame({
            "model": models, "pressure_error_percent": errors,
            "backend": "ase" if "ase" in source else "torchsim",
            "mode": "md_accelerated",
        }).to_csv(pressure, index=False)
        return rdf, vdos, pressure

    def merge(self, source, metrics):
        rdf, vdos, pressure = metrics
        with contextlib.redirect_stdout(io.StringIO()):
            return figure_7.load_and_merge(
                timings_dir=self.root / "data" / source,
                pressure_metrics_file=pressure, pressure_scale_gpa=10,
                clip_pressure_error=True, dataset_label=source,
                rdf_metrics_file=rdf, vdos_metrics_file=vdos,
                pressure_backend="ase" if "ase" in source else "torchsim",
                pressure_mode="md_accelerated",
            )

    def test_native_ase_aliases_join_all_metrics_and_preserve_execution_variants(self):
        self.add_run(model="orb-v2-force-only-ase", seconds=0.001)
        self.add_run(model="uma-s-omat-compile-force-only-ase", seconds=0.002)
        self.add_run(model="uma-s-omat-turbo-force-only-ase", seconds=0.003)
        metrics = self.write_metrics(
            ASE, ["orb-v2-stress-ase", "uma-s-omat-compile", "uma-s-omat-turbo"], [30, 20, 10]
        )
        frame = self.merge(ASE, metrics).set_index("model")
        self.assertEqual(set(frame.index), {"orb-v2", "uma-s-omat-compile", "uma-s-omat-turbo"})
        for model, expected_ms, expected_error in (
            ("orb-v2", 1, 30), ("uma-s-omat-compile", 2, 20), ("uma-s-omat-turbo", 3, 10)
        ):
            self.assertEqual(frame.loc[model, "mean_time_per_step_ms"], expected_ms)
            self.assertEqual(frame.loc[model, "Combined Error [%]"], expected_error)
        self.assertEqual(set(frame.dataset), {ASE})

    def test_failed_incomplete_mismatched_and_crystal_timings_are_excluded(self):
        self.add_run()
        self.add_run(system=CRYSTAL)
        missing = self.add_run(system="missing-trajectory")
        missing.with_name("nvt_orb-v2-force-only-ase.h5").unlink()
        partial = self.add_run(system="in-progress")
        partial.write_text("calculator,system,n_steps,seconds_per_step\n")
        duplicate = self.add_run(system="ambiguous-completion")
        duplicate_rows = pd.read_csv(duplicate)
        pd.concat([duplicate_rows, duplicate_rows], ignore_index=True).to_csv(duplicate, index=False)
        for index, invalid in enumerate((
            {"n_steps": 0}, {"seconds_per_step": 0},
            {"seconds_per_step": -1}, {"seconds_per_step": np.inf},
            {"seconds_per_step": np.nan}, {"calculator": "other-model"},
            {"system": "wrong-system"},
        )):
            self.add_run(system=f"invalid-{index}", row_fields=invalid)
        self.add_run(system="audited-failure")
        self.failures[(ASE, "audited-failure", "orb-v2-force-only-ase")] = "Unphysical completed run"
        frame = figure_7.load_model_avg_timings(self.root / "data" / ASE)
        self.assertEqual(frame.model.tolist(), ["orb-v2"])
        self.assertEqual(frame.mean_time_per_step_ms.tolist(), [1.0])
        self.assertTrue(np.isfinite(frame.mean_time_per_step_ms).all())

    def test_timing_means_weight_systems_equally_and_allow_explicit_crystals(self):
        self.add_run(seconds=0.001, n_steps=10, elapsed_seconds=0.01)
        self.add_run(system="bulkPt3Co_300K", seconds=0.003,
                     n_steps=1000, elapsed_seconds=3)
        self.add_run(system=CRYSTAL, seconds=0.005)
        directory = self.root / "data" / ASE
        normal = figure_7.load_model_avg_timings(directory)
        included = figure_7.load_model_avg_timings(directory, include_molecular_crystals=True)
        self.assertEqual(len(normal), 1)
        self.assertEqual(len(included), 1)
        self.assertAlmostEqual(normal.mean_time_per_step_ms.iloc[0], 2.0)
        self.assertAlmostEqual(included.mean_time_per_step_ms.iloc[0], 3.0)

    def test_native_ase_requires_all_three_errors_and_never_borrows_torchsim_or_eager(self):
        for raw in ("orb-v2-force-only-ase", "mace-mp-0-compile-ase", "nequip-force-only-ase"):
            self.add_run(model=raw)
        self.add_run(TORCHSIM, "uma-s-omat-turbo")
        self.add_run("mlip-trajs-ase-eager", "chgnet-ase")
        models = ["orb-v2", "mace-mp-0-compile", "nequip", "uma-s-omat-turbo", "chgnet"]
        metrics = self.write_metrics(ASE, models, [10, 20, 30, 40, 50])
        rdf, vdos, pressure = metrics
        pd.read_csv(rdf).query("Calculator != 'mace-mp-0-compile'").to_csv(rdf, index=False)
        pd.read_csv(vdos).query("model != 'nequip'").to_csv(vdos, index=False)
        frame = self.merge(ASE, metrics)
        self.assertEqual(frame.model.tolist(), ["orb-v2"])
        # Even when TorchSim has a pressure entry, it cannot fill an absent ASE entry.
        pressure_frame = pd.read_csv(pressure)
        pressure_frame.loc[pressure_frame.model == "orb-v2", "backend"] = "torchsim"
        pressure_frame.to_csv(pressure, index=False)
        with self.assertRaisesRegex(ValueError, "No overlapping models"):
            self.merge(ASE, metrics)

    def run_data_only(self, sources, inputs, **extra_paths):
        output = extra_paths.get("output", self.root / "no-figure.pdf")
        metrics_csv = extra_paths.get("metrics_csv", self.root / "merged.csv")
        frontier_csv = extra_paths.get("frontier_csv", self.root / "frontier.csv")
        argv = ["figure_SI_18.py", "--data-only", "--output-file", str(output),
                "--output-csv", str(metrics_csv), "--frontier-csv", str(frontier_csv)]
        for source in sources:
            argv.extend(["--source", source])

        def source_paths(source):
            rdf, vdos, _ = inputs[source]
            return self.root / "data" / source, rdf, vdos, "md_accelerated"

        with (patch.object(sys, "argv", argv),
              patch.object(figure_7, "source_input_paths", side_effect=source_paths),
              patch.object(figure_7, "pressure_path", side_effect=lambda source: inputs[source][2]),
              patch.object(figure_7, "plot_pareto", side_effect=AssertionError("Must not plot")) as plot,
              contextlib.redirect_stdout(io.StringIO())):
            if len(sources) == 1:
                # Exercise reusable native ASE loading without exposing a single-runner SI18 CLI.
                figure_7.run_pareto_figure(
                    figure_name="figure_SI_18", allowed_sources=(TORCHSIM, ASE),
                    default_sources=tuple(sources),
                )
            else:
                figure_SI_18.main()
        plot.assert_not_called()
        self.assertFalse(output.exists())
        return pd.read_csv(metrics_csv), pd.read_csv(frontier_csv)

    def test_data_only_cli_exports_frontier_flags_in_increasing_time_order(self):
        for raw, seconds in (("orb-v2-force-only-ase", 0.001),
                             ("mace-mp-0-compile-ase", 0.002),
                             ("nequip-force-only-ase", 0.003)):
            self.add_run(model=raw, seconds=seconds)
        inputs = {ASE: self.write_metrics(ASE, ["orb-v2", "mace-mp-0-compile", "nequip"], [40, 20, 30])}
        frame, frontier = self.run_data_only([ASE], inputs)
        self.assertEqual(frame.set_index("model").is_pareto.to_dict(),
                         {"orb-v2": True, "mace-mp-0-compile": True, "nequip": False})
        self.assertEqual(frontier.model.tolist(), ["orb-v2", "mace-mp-0-compile"])
        self.assertEqual(frontier.mean_time_per_step_ms.tolist(), [1, 2])
        self.assertTrue(frontier.is_pareto.all())
        self.assertEqual(set(frame.dataset), {ASE})

    def test_mixed_accelerated_sources_keep_separate_frontiers(self):
        self.add_run(model="orb-v2-force-only-ase", seconds=0.01)
        self.add_run(model="mace-mp-0-compile-ase", seconds=0.02)
        self.add_run(TORCHSIM, "orb-v2-force-only", seconds=0.001)
        self.add_run(TORCHSIM, "mace-mp-0-compile", seconds=0.002)
        models = ["orb-v2", "mace-mp-0-compile"]
        inputs = {ASE: self.write_metrics(ASE, models, [40, 20]),
                  TORCHSIM: self.write_metrics(TORCHSIM, models, [10, 5])}
        frame, frontier = self.run_data_only([ASE, TORCHSIM], inputs)
        self.assertEqual(set(frame.dataset), {ASE, TORCHSIM})
        # Both ASE points are globally dominated by TorchSim, but belong to their own frontier.
        self.assertTrue(frame.is_pareto.all())
        self.assertEqual(frontier.groupby("dataset").size().to_dict(), {ASE: 2, TORCHSIM: 2})

    def test_si_18_omits_only_ase_uma_turbo_and_recomputes_its_frontier(self):
        models = ["orb-v2", "uma-s-omat-turbo", "uma-m-omat-turbo"]
        for source in (TORCHSIM, ASE):
            for model, seconds in zip(models, (0.003, 0.001, 0.002)):
                suffix = "-force-only-ase" if source == ASE else "-force-only"
                self.add_run(source, model + suffix, seconds=seconds)
        inputs = {
            TORCHSIM: self.write_metrics(TORCHSIM, models, [30, 10, 5]),
            ASE: self.write_metrics(ASE, models, [40, 20, 15]),
        }
        original_inputs = {path: path.read_bytes()
                           for paths in inputs.values() for path in paths}

        frame, frontier = self.run_data_only([TORCHSIM, ASE], inputs)

        self.assertEqual(frame.loc[frame.dataset.eq(ASE), "model"].tolist(), ["orb-v2"])
        self.assertEqual(set(frame.loc[frame.dataset.eq(TORCHSIM), "model"]), set(models))
        self.assertEqual(frontier.loc[frontier.dataset.eq(ASE), "model"].tolist(), ["orb-v2"])
        self.assertEqual(set(frontier.loc[frontier.dataset.eq(TORCHSIM), "model"]),
                         {"uma-s-omat-turbo", "uma-m-omat-turbo"})
        # Orb was dominated by both ASE UMA configurations before filtering;
        # the ASE frontier must be recalculated after those rows are removed.
        ase_orb = frame.loc[frame.dataset.eq(ASE)].iloc[0]
        self.assertTrue(ase_orb.is_pareto)
        self.assertEqual(ase_orb["Combined Error [%]"], 40)
        for path, contents in original_inputs.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_figure_7_defaults_to_a_single_torchsim_source(self):
        source = figure_7.DEFAULT_SOURCE
        self.assertIn("torchsim", source)
        self.add_run(source, "orb-v2-force-only")
        rdf, vdos, pressure = self.write_metrics(source, ["orb-v2"], [30])
        script_dir = self.root / "pareto_plots"
        argv = ["figure_7.py", "--data-only"]
        paths = (self.root / "data" / source, rdf, vdos, "md_accelerated")
        with (patch.object(sys, "argv", argv),
              patch.object(figure_7, "SCRIPT_DIR", script_dir),
              patch.object(figure_7, "source_input_paths", return_value=paths) as source_inputs,
              patch.object(figure_7, "pressure_path", return_value=pressure),
              patch.object(figure_7, "plot_pareto", side_effect=AssertionError("Must not plot")) as plot,
              contextlib.redirect_stdout(io.StringIO())):
            figure_7.main()
        plot.assert_not_called()
        source_inputs.assert_called_once_with(source)
        directory = script_dir / "results" / source
        metrics = pd.read_csv(directory / "figure_7_metrics.csv")
        self.assertEqual(metrics.model.tolist(), ["orb-v2"])
        self.assertEqual(set(metrics.dataset), {source})
        self.assertEqual(pd.read_csv(directory / "figure_7_frontier.csv").model.tolist(), ["orb-v2"])
        self.assertFalse((script_dir / "plots").exists())

    def test_si_18_defaults_to_both_runners_and_preserves_figure_7_outputs(self):
        self.add_run(model="orb-v2-force-only-ase", seconds=0.01)
        self.add_run(TORCHSIM, "orb-v2-force-only", seconds=0.001)
        inputs = {ASE: self.write_metrics(ASE, ["orb-v2"], [30]),
                  TORCHSIM: self.write_metrics(TORCHSIM, ["orb-v2"], [10])}
        script_dir = self.root / "pareto_plots"
        figure7_output = script_dir / "results" / TORCHSIM / "figure_7_metrics.csv"
        figure7_output.parent.mkdir(parents=True)
        figure7_output.write_text("existing Figure 7 results\n")
        argv = ["figure_SI_18.py", "--data-only"]

        def source_paths(source):
            rdf, vdos, _ = inputs[source]
            return self.root / "data" / source, rdf, vdos, "md_accelerated"

        with (patch.object(sys, "argv", argv),
              patch.object(figure_7, "SCRIPT_DIR", script_dir),
              patch.object(figure_7, "source_input_paths", side_effect=source_paths),
              patch.object(figure_7, "pressure_path", side_effect=lambda source: inputs[source][2]),
              patch.object(figure_7, "plot_pareto", side_effect=AssertionError("Must not plot")) as plot,
              contextlib.redirect_stdout(io.StringIO())):
            figure_SI_18.main()
        plot.assert_not_called()
        directory = script_dir / "results" / "torchsim_vs_ase_accelerated"
        metrics = pd.read_csv(directory / "figure_SI_18_metrics.csv")
        frontier = pd.read_csv(directory / "figure_SI_18_frontier.csv")
        self.assertEqual(set(metrics.dataset), {TORCHSIM, ASE})
        self.assertTrue(metrics.is_pareto.all())
        self.assertEqual(len(frontier), 2)
        self.assertEqual(figure7_output.read_text(), "existing Figure 7 results\n")
        self.assertFalse((directory / "figure_7_metrics.csv").exists())
        self.assertFalse((directory / "figure_7_frontier.csv").exists())
        self.assertFalse((script_dir / "plots").exists())

    def test_si_18_data_only_respects_custom_output_paths(self):
        self.add_run(model="orb-v2-force-only-ase", seconds=0.01)
        self.add_run(TORCHSIM, "orb-v2-force-only", seconds=0.001)
        inputs = {ASE: self.write_metrics(ASE, ["orb-v2"], [30]),
                  TORCHSIM: self.write_metrics(TORCHSIM, ["orb-v2"], [10])}
        directory = self.root / "custom results"
        metrics, frontier = self.run_data_only(
            [TORCHSIM, ASE], inputs, output=self.root / "plots" / "custom.pdf",
            metrics_csv=directory / "metrics.csv", frontier_csv=directory / "front.csv",
        )
        self.assertEqual(len(metrics), 2)
        self.assertEqual(len(frontier), 2)
        self.assertTrue((directory / "metrics.csv").is_file())
        self.assertTrue((directory / "front.csv").is_file())
        self.assertFalse((self.root / "plots").exists())

    def test_figure_7_cli_rejects_ase_sources_before_loading_or_plotting(self):
        for source in (ASE, "mlip-trajs-ase"):
            with self.subTest(source=source):
                argv = ["figure_7.py", "--source", source, "--data-only"]
                with (patch.object(sys, "argv", argv),
                      patch.object(figure_7, "source_input_paths") as inputs,
                      patch.object(figure_7, "plot_pareto") as plot,
                      contextlib.redirect_stderr(io.StringIO()),
                      self.assertRaises(SystemExit) as error):
                    figure_7.main()
                self.assertEqual(error.exception.code, 2)
                inputs.assert_not_called()
                plot.assert_not_called()

    def test_si_18_cli_requires_both_accelerated_runners(self):
        for source in (TORCHSIM, ASE):
            with self.subTest(source=source):
                argv = ["figure_SI_18.py", "--source", source, "--data-only"]
                with (patch.object(sys, "argv", argv),
                      patch.object(figure_7, "source_input_paths") as inputs,
                      patch.object(figure_7, "plot_pareto") as plot,
                      contextlib.redirect_stderr(io.StringIO()),
                      self.assertRaises(SystemExit) as error):
                    figure_SI_18.main()
                self.assertEqual(error.exception.code, 2)
                inputs.assert_not_called()
                plot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
