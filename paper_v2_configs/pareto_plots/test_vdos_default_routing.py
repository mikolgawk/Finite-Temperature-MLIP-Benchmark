"""Keep paper defaults on refined VDOS and legacy comparisons explicit."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import _figure_data as data
import figure_7
import figure_SI_18
import plot_all


TORCHSIM = "mlip-trajs-torchsim-accelerated"
ASE = "mlip-trajs-ase-accelerated"


class RefinedVDOSDefaultsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.refined = self.root / "vdos/results"
        self.legacy = self.root / "vdos_legacy/results"
        for source, refined_error, legacy_error in ((TORCHSIM, 10, 80), (ASE, 20, 90)):
            for directory, error in ((self.refined, refined_error), (self.legacy, legacy_error)):
                path = data.vdos_metrics_path(source, directory)
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"model": ["orb-v2"], "vdos_error_percent": [error]}).to_csv(path, index=False)

    def export(self, *, comparison=False, extra_args=()):
        output = self.root / "exports" / ("si18.csv" if comparison else "fig7.csv")
        argv = ["--data-only", "--output-csv", str(output), *extra_args]
        if not comparison:
            argv.extend(("--source", TORCHSIM))

        def load(**kwargs):
            source = kwargs["dataset_label"]
            errors = pd.read_csv(kwargs["vdos_metrics_file"])
            return pd.DataFrame({
                "model": ["orb-v2"], "dataset": [source],
                "mean_time_per_step_ms": [1 if source == TORCHSIM else 2],
                "Combined Error [%]": errors["vdos_error_percent"],
            })

        with (patch.object(data, "DEFAULT_VDOS_RESULTS_DIR", self.refined),
              patch.object(figure_7, "SCRIPTS_DIR", self.root),
              patch.object(figure_7, "load_and_merge", side_effect=load) as merge,
              patch.object(figure_7, "apply_matched_runner_timings",
                           side_effect=lambda frame, **kwargs: frame),
              patch.object(figure_7, "plot_pareto", side_effect=AssertionError("Must not create figures")),
              contextlib.redirect_stdout(io.StringIO())):
            if comparison:
                figure_SI_18.main(argv)
            else:
                with patch.object(sys, "argv", ["figure_7.py", *argv]):
                    figure_7.main()
        return pd.read_csv(output), merge.call_args_list

    def test_shared_paths_and_direct_figures_default_to_refined_scores(self):
        with patch.object(data, "DEFAULT_VDOS_RESULTS_DIR", self.refined):
            for source in (TORCHSIM, ASE):
                chosen = data.vdos_metrics_path(source, self.refined)
                self.assertEqual(data.source_paths(source)[2], chosen)
                self.assertEqual(figure_7.source_input_paths(source)[2], chosen)
        for comparison, expected in ((False, [10]), (True, [10, 20])):
            with self.subTest(comparison=comparison):
                table, calls = self.export(comparison=comparison)
                self.assertEqual(table["Combined Error [%]"].tolist(), expected)
                self.assertTrue(all(call.kwargs["vdos_metrics_file"].is_relative_to(self.refined)
                                    for call in calls))
                self.assertTrue(table.is_pareto.all())

    def test_direct_si18_accepts_an_explicit_legacy_root_for_both_runners(self):
        table, calls = self.export(comparison=True, extra_args=("--vdos-results-dir", str(self.legacy)))
        self.assertEqual(table["Combined Error [%]"].tolist(), [80, 90])
        self.assertEqual([call.kwargs["vdos_metrics_file"] for call in calls],
                         [data.vdos_metrics_path(source, self.legacy) for source in (TORCHSIM, ASE)])

    def test_missing_default_fails_despite_existing_legacy_and_writes_nothing(self):
        missing = data.vdos_metrics_path(TORCHSIM, self.refined)
        missing.unlink()
        self.assertTrue(data.vdos_metrics_path(TORCHSIM, self.legacy).is_file())
        with self.assertRaisesRegex(FileNotFoundError, "Legacy VDOS is never used as a fallback"):
            self.export()
        self.assertFalse((self.root / "exports").exists())
        with self.assertRaisesRegex(FileNotFoundError, "vdos/run_vdos_pipeline.py"):
            data.load_vdos(missing)

    def test_source_discovery_uses_the_selected_root_without_legacy_fallback(self):
        for source in (TORCHSIM, ASE):
            rdf = self.root / "rdfs/results" / source / "rdf_similarity_scores_same_simulation_length.csv"
            rdf.parent.mkdir(parents=True)
            rdf.touch()
        data.vdos_metrics_path(ASE, self.refined).unlink()
        with patch.object(plot_all, "CONFIG_DIR", self.root):
            self.assertEqual(plot_all.available_sources(self.refined), [TORCHSIM])
            self.assertEqual(set(plot_all.available_sources(self.legacy)), {TORCHSIM, ASE})

    def test_pipeline_rejects_missing_refined_metrics_before_running_any_stage(self):
        missing = data.vdos_metrics_path(TORCHSIM, self.refined)
        missing.unlink()
        argv = ["plot_all.py", "--source", TORCHSIM, "--vdos-results-dir", str(self.refined),
                "--plots-dir", str(self.root / "plots"), "--results-dir", str(self.root / "exports")]
        with (patch.object(sys, "argv", argv), patch.object(plot_all, "run_stage") as run,
              self.assertRaisesRegex(FileNotFoundError, "--compute-only --source " + TORCHSIM)):
            plot_all.main()
        run.assert_not_called()
        self.assertFalse((self.root / "plots").exists())
        self.assertFalse((self.root / "exports").exists())


if __name__ == "__main__":
    unittest.main()
