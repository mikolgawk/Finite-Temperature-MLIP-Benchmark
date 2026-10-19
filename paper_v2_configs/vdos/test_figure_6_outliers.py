"""Check all qualifying Figure 6 labels without constructing any figures."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import figure_6


class Figure6OutlierTests(unittest.TestCase):
    def setUp(self):
        self.names = {f"outlier-{index}" for index in range(4)}
        x = np.r_[np.linspace(-1., 1., 80), [-.9, -.3, .3, .9]]
        deviations = np.r_[.05 * np.sin(np.arange(80)), [20., 21., 22., 23.]]
        self.frame = pd.DataFrame({
            "calculator": [f"ordinary-{index}" for index in range(80)] + sorted(self.names),
            "force_rmse": x,
            "rdf_error_percent": 2. * x + deviations,
        })

    def select(self, frame=None, max_labels=None):
        return figure_6.select_outlier_labels(
            self.frame if frame is None else frame, "force_rmse", "rdf_error_percent",
            threshold=2.5, max_labels=max_labels,
        )

    def test_default_keeps_all_qualifying_models_and_no_ordinary_points(self):
        self.assertEqual(self.select(), self.names)
        self.assertEqual(self.select(self.frame.sample(frac=1, random_state=42)), self.names)
        self.assertEqual(self.select(self.frame.iloc[:80]), set())

    def test_explicit_limits_remain_available(self):
        self.assertEqual(self.select(max_labels=3), {"outlier-1", "outlier-2", "outlier-3"})
        self.assertEqual(self.select(max_labels=1), {"outlier-3"})
        self.assertEqual(self.select(max_labels=0), set())

    def test_scores_match_independent_deleted_prediction_residuals(self):
        x = self.frame.force_rmse.to_numpy()
        y = self.frame.rdf_error_percent.to_numpy()
        design = np.column_stack([np.ones(len(x)), x])
        expected = []
        for index in range(len(x)):
            keep = np.arange(len(x)) != index
            deleted_design, deleted_y = design[keep], y[keep]
            coefficients = np.linalg.lstsq(deleted_design, deleted_y, rcond=None)[0]
            deleted_errors = deleted_y - deleted_design @ coefficients
            variance = np.dot(deleted_errors, deleted_errors) / (len(x) - 3)
            prediction_variance = variance * (
                1. + design[index] @ np.linalg.inv(deleted_design.T @ deleted_design) @ design[index]
            )
            expected.append(abs(y[index] - design[index] @ coefficients) / np.sqrt(prediction_variance))
        np.testing.assert_allclose(figure_6.panel_outlier_scores(x, y), expected, rtol=1e-11, atol=1e-12)

    def test_scores_do_not_depend_on_units_or_offsets(self):
        x = self.frame.force_rmse.to_numpy()
        y = self.frame.rdf_error_percent.to_numpy()
        scores = figure_6.panel_outlier_scores(x, y)
        transformed = figure_6.panel_outlier_scores(1.e6 * x + 2.e8, -3.1 * y + 1.e4)
        np.testing.assert_allclose(transformed, scores, rtol=1e-9, atol=1e-9)

    def test_extreme_coordinates_on_the_trend_are_not_outliers(self):
        x = np.r_[np.linspace(-1., 1., 80), 100.]
        y = 2. * x + np.r_[.05 * np.sin(np.arange(80)), 0.]
        frame = pd.DataFrame({
            "calculator": [f"ordinary-{index}" for index in range(80)] + ["on-trend-extreme"],
            "force_rmse": x,
            "rdf_error_percent": y,
        })
        self.assertEqual(self.select(frame), set())

    def test_undefined_and_exact_fits_do_not_create_outlier_labels(self):
        cases = [
            (np.array([]), np.array([])),
            (np.arange(3.), np.array([0., 100., 0.])),
            (np.ones(8), np.arange(8.)),
            (np.arange(8.), np.ones(8)),
            (np.array([-1., 0., 1., 1.e6]), np.array([7., 8., 9., 1.e6 + 8.])),
        ]
        for x, y in cases:
            with self.subTest(x=x, y=y):
                np.testing.assert_array_equal(figure_6.panel_outlier_scores(x, y), np.zeros(len(x)))
        # Removing the sole distinct x coordinate leaves no estimable slope.
        scores = figure_6.panel_outlier_scores(
            np.array([0., 0., 0., 0., 0., 1.]),
            np.array([0., 1., -1., 2., -2., 100.]),
        )
        self.assertEqual(scores[-1], 0.)
        self.assertFalse(np.isnan(scores).any())

    def test_lone_departure_from_an_otherwise_exact_line_is_labeled(self):
        x = np.linspace(-1., 1., 20)
        y = 2. * x + 4.
        y[9] += 2.
        scores = figure_6.panel_outlier_scores(x, y)
        self.assertTrue(np.isinf(scores[9]))
        self.assertEqual(set(np.flatnonzero(scores >= 2.5)), {9})

    def test_small_real_residuals_are_preserved(self):
        frame = self.frame.copy()
        x = frame.force_rmse.to_numpy()
        deviations = frame.rdf_error_percent.to_numpy() - 2. * x
        frame["rdf_error_percent"] = 2. * x + deviations * 1.e-8
        self.assertEqual(self.select(frame), self.names)

    def test_nonfinite_pairs_and_missing_names_do_not_change_labels(self):
        invalid = pd.DataFrame({
            "calculator": ["infinite-x", "infinite-y", "missing", "nonnumeric", None],
            "force_rmse": [np.inf, 999., np.nan, "bad", 999.],
            "rdf_error_percent": [0., -np.inf, 0., 0., 0.],
        })
        combined = pd.concat([self.frame, invalid], ignore_index=True)
        self.assertEqual(self.select(combined), self.names)
        filtered = figure_6.finite_panel_data(combined, "force_rmse", "rdf_error_percent")
        pd.testing.assert_frame_equal(filtered, self.frame)
        self.assertEqual(self.select(invalid), set())

    def test_every_selected_name_reaches_the_annotation_call(self):
        axes = Mock()
        axes.get_xlim.return_value = (-1., 10.)
        axes.annotate.side_effect = lambda *args, **kwargs: SimpleNamespace()
        sub = self.frame.assign(x_value=self.frame.force_rmse, y_value=self.frame.rdf_error_percent)
        figure_6.add_point_labels(axes, sub, self.select(), figure_6.OUTLIER_LABEL_FONT_SIZE)
        self.assertEqual({call.args[0] for call in axes.annotate.call_args_list}, self.names)
        self.assertEqual(axes.annotate.call_count, 4)
        for call in axes.annotate.call_args_list:
            self.assertNotIn("arrowprops", call.kwargs)

    def test_cli_defaults_to_no_cap_and_data_only_reports_all_names(self):
        with patch.object(sys, "argv", ["figure_6.py"]):
            self.assertIsNone(figure_6.parse_args().max_labels_per_panel)
        with patch.object(sys, "argv", ["figure_6.py", "--max-labels-per-panel", "3"]):
            self.assertEqual(figure_6.parse_args().max_labels_per_panel, 3)
        frame = self.frame.assign(
            f1_score=self.frame.force_rmse, ksrme_score=self.frame.force_rmse,
            pressure_error_percent=self.frame.rdf_error_percent,
            vdos_error_percent=self.frame.rdf_error_percent,
        )
        summary = figure_6.outlier_label_summary(frame, threshold=2.5)
        self.assertEqual(len(summary), 9)
        self.assertTrue(all(set(names) == self.names for names in summary.values()))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused.pdf"
            argv = ["figure_6.py", "--data-only", "--exclude-hydrogen", "--output-file", str(output)]
            with (patch.object(sys, "argv", argv),
                  patch.object(figure_6, "load_joined_data", return_value=frame),
                  patch.object(figure_6, "render_figure", side_effect=AssertionError("Must not plot")) as render,
                  redirect_stdout(io.StringIO()) as report):
                figure_6.main()
            render.assert_not_called()
            self.assertFalse(output.exists())
            for name in self.names:
                self.assertEqual(report.getvalue().count(name), 9)


if __name__ == "__main__":
    unittest.main()
