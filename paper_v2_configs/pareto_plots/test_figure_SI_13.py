"""Ensure the SI Pareto plot keeps slow models and labels visible."""

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

import figure_SI_13


class FigureSI13Tests(unittest.TestCase):
    def test_all_models_and_nearby_labels_remain_inside_the_axes(self):
        for times in ([10.0, 25.0, 80.0, 1115.0], [5.0, 12.0, 40.0, 376.0]):
            with self.subTest(times=times), tempfile.TemporaryDirectory() as directory:
                frame = pd.DataFrame({
                    "model": ["orb-v2", "mace-mp-0", "nequip", "uma-m-omat"],
                    "mean_time_per_step_ms": times,
                    "Combined Error [%]": [29.0, 32.0, 25.0, 23.0],
                })
                original = frame.copy(deep=True)
                output = Path(directory) / "unused.pdf"
                with (
                    patch.object(Figure, "savefig", autospec=True) as save,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    figure_SI_13.plot_pareto(frame, output)
                self.assertFalse(output.exists())
                save.assert_called_once()
                fig = save.call_args.args[0]
                fig.canvas.draw()
                ax = fig.axes[0]
                renderer = fig.canvas.get_renderer()
                bounds = ax.get_window_extent(renderer)
                labels = [text for text in ax.texts if isinstance(text, Annotation)]
                expected = {
                    figure_SI_13.display_name(row.model):
                    (row.mean_time_per_step_ms, row["Combined Error [%]"])
                    for _, row in frame.iterrows()
                }
                self.assertEqual({label.get_text() for label in labels}, set(expected))
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
                self.assertGreater(ax.get_xlim()[1], max(times))
                self.assertLess(ax.get_xlim()[1], 2 * max(times))
                self.assertEqual(ax.get_title(), "")
                pd.testing.assert_frame_equal(frame, original)


if __name__ == "__main__":
    unittest.main()
