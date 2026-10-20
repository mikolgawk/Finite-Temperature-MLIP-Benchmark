"""Population, penalty and sensitivity-summary regressions."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


_SPEC = importlib.util.spec_from_file_location("updated_vdos_sensitivity", Path(__file__).with_name("sensitivity_analysis.py"))
sensitivity = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = sensitivity
_SPEC.loader.exec_module(sensitivity)


def example_results():
    rows = []
    for config in sensitivity.CONFIGURATIONS:
        change = 2.0 if config.name == "mass_1ps_min4_pad4" else 0.0
        for model, stable_error in (("A", 10.0), ("B", 30.0)):
            for system in ("stable", "other"):
                failed = model == "A" and system == "other"
                # A genuine stable 100% score must remain distinguishable from
                # a failed run carrying the same numerical penalty.
                error = 100.0 if failed or (model == "B" and system == "other") else stable_error + change
                rows.append({
                    "source": "cohort", "model": model, "system": system,
                    "configuration": config.name, "vdos_error_percent": error,
                    "failure_reason": "audited failure" if failed else "",
                    "matched_frames": 0 if failed else 1000,
                    "segment_duration_fs": np.nan if failed else config.window_fs,
                    "n_segments": 0 if failed else 4,
                    "resolution_cm_inverse": np.nan if failed else 20.,
                })
    return pd.DataFrame(rows)


class SensitivityPopulationTest(unittest.TestCase):
    def test_all_failures_kept_in_equal_system_model_means(self):
        models, summary = sensitivity.aggregate_diagnostics(example_results())
        baseline = models.loc[models.configuration.eq(sensitivity.BASELINE_NAME)].set_index("model")
        self.assertEqual(baseline.loc["A", "vdos_error_percent"], 55.0)
        self.assertEqual(baseline.loc["B", "vdos_error_percent"], 65.0)
        self.assertEqual(baseline.loc["A", "n_failure_penalties"], 1)
        self.assertEqual(baseline.loc["B", "n_failure_penalties"], 0)
        self.assertTrue(models.n_systems.eq(2).all())
        self.assertEqual(summary["pairs_per_configuration"], 4)
        self.assertEqual(summary["failure_penalties_per_configuration"], 1)
        self.assertEqual(summary["systems_diagnostics"]["other"]["successful_models"], 1)

    def test_window_changes_are_reported_as_percentage_points_and_ranks(self):
        _, summary = sensitivity.aggregate_diagnostics(example_results())
        comparison = summary["comparisons"]["mass_1ps_min4_pad4"]
        self.assertAlmostEqual(comparison["rank_spearman_vs_baseline"], 1.0)
        self.assertEqual(comparison["maximum_absolute_model_error_change_pp"], 1.0)
        self.assertEqual(comparison["mean_signed_model_error_change_pp"], 1.0)
        self.assertEqual(summary["systems_diagnostics"]["stable"]["maximum_window_score_range_pp"], 2.0)

    def test_missing_duplicate_and_changed_failure_configurations_are_rejected(self):
        frame = example_results()
        changed = frame.copy()
        failed = changed.failure_reason.ne("")
        changed.loc[changed.index[failed][0], "failure_reason"] = ""
        for invalid in (frame.iloc[1:], pd.concat([frame, frame.iloc[:1]]), changed):
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                sensitivity.aggregate_diagnostics(invalid)

    def test_failure_penalty_cannot_silently_change(self):
        frame = example_results()
        frame.loc[frame.failure_reason.ne(""), "vdos_error_percent"] = 99.0
        with self.assertRaisesRegex(ValueError, "Failure penalties"):
            sensitivity.aggregate_diagnostics(frame)

    def test_frozen_population_does_not_infer_failure_from_100_percent(self):
        frame = example_results().loc[lambda rows: rows.configuration.eq(sensitivity.BASELINE_NAME)]
        validated = sensitivity.validate_population(frame)
        self.assertEqual(validated.failure_reason.ne("").sum(), 1)
        self.assertEqual(validated.vdos_error_percent.eq(100).sum(), 2)

    def test_pareto_retains_duplicate_points_but_excludes_dominated_ones(self):
        actual = sensitivity.pareto_mask([1, 1, 2, 3, 2], [5, 5, 4, 6, 5])
        np.testing.assert_array_equal(actual, [True, True, True, False, False])


if __name__ == "__main__":
    unittest.main()
