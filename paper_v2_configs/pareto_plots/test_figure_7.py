"""Regression checks for misplaced Pareto labels and excessive axis ranges."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matplotlib.figure import Figure
from matplotlib.text import Annotation
import numpy as np
import pandas as pd

import figure_7


class Figure7Tests(unittest.TestCase):
    def capture_plot(self, frame):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused.pdf"
            with (
                patch.object(Figure, "savefig", autospec=True) as save,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                figure_7.plot_pareto(frame, output)
            self.assertFalse(output.exists())
            save.assert_called_once()
            return save.call_args.args[0]

    def test_labels_stay_beside_their_points_in_single_and_comparison_plots(self):
        eager = pd.DataFrame({
            "model": ["orb-v2", "mace-mp-0", "nequip", "uma-s-omat"],
            "mean_time_per_step_ms": [10.0, 25.0, 80.0, 120.0],
            "Combined Error [%]": [29.0, 32.0, 25.0, 23.0],
            "dataset": "eager",
        })
        accelerated = eager.assign(
            mean_time_per_step_ms=eager.mean_time_per_step_ms / 2,
            dataset="accelerated",
        )
        for frame in (eager, pd.concat([eager, accelerated], ignore_index=True)):
            with self.subTest(datasets=frame.dataset.unique().tolist()):
                original = frame.copy(deep=True)
                fig = self.capture_plot(frame)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                self.assertEqual(len(fig.axes), frame.dataset.nunique())
                for ax, (_, rows) in zip(fig.axes, frame.groupby("dataset", sort=False)):
                    labels = [text for text in ax.texts if isinstance(text, Annotation)]
                    expected = {
                        figure_7.display_name(row.model):
                        (row.mean_time_per_step_ms, row["Combined Error [%]"])
                        for _, row in rows.iterrows()
                    }
                    self.assertEqual({label.get_text() for label in labels}, set(expected))
                    bounds = ax.get_window_extent(renderer)
                    for label in labels:
                        self.assertIsNone(label.arrow_patch)
                        self.assertEqual(tuple(label.xy), expected[label.get_text()])
                        box = label.get_window_extent(renderer)
                        point = ax.transData.transform(label.xy)
                        nearest = np.clip(point, [box.x0, box.y0], [box.x1, box.y1])
                        self.assertLessEqual(np.linalg.norm(point - nearest), 6 * fig.dpi / 72)
                        self.assertGreaterEqual(box.x0, bounds.x0 - 0.5)
                        self.assertGreaterEqual(box.y0, bounds.y0 - 0.5)
                        self.assertLessEqual(box.x1, bounds.x1 + 0.5)
                        self.assertLessEqual(box.y1, bounds.y1 + 0.5)
                    maximum_time = rows.mean_time_per_step_ms.max()
                    self.assertGreater(ax.get_xlim()[1], maximum_time)
                    self.assertLess(ax.get_xlim()[1], 2 * maximum_time)
                pd.testing.assert_frame_equal(frame, original)

    def test_equal_errors_have_finite_limits_and_visible_labels(self):
        frame = pd.DataFrame({
            "model": ["orb-v2", "mace-mp-0"],
            "mean_time_per_step_ms": [0.1, 0.2],
            "Combined Error [%]": [100.0, 100.0],
        })
        fig = self.capture_plot(frame)
        ax = fig.axes[0]
        self.assertTrue(np.isfinite(ax.get_ylim()).all())
        self.assertLess(ax.get_ylim()[0], 100.0)
        self.assertGreater(ax.get_ylim()[1], 100.0)
        self.assertEqual(len(ax.texts), 2)


if __name__ == "__main__":
    unittest.main()
