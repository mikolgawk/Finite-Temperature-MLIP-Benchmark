"""Validate overlay commands using fake objects; never construct a figure."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import figure_7


TORCHSIM = "mlip-trajs-torchsim-accelerated"
ASE = "mlip-trajs-ase-accelerated"


class Figure7OverlayTests(unittest.TestCase):
    def frame(self, order=(TORCHSIM, ASE)):
        values = {
            TORCHSIM: ([1.0, 2.0, 3.0], [10.0, 5.0, 7.0]),
            ASE: ([10.0, 20.0, 30.0], [40.0, 20.0, 30.0]),
        }
        return pd.concat([
            pd.DataFrame({
                "model": ["orb-v2", "mace-mp-0-compile", "nequip"],
                "mean_time_per_step_ms": values[source][0],
                "Combined Error [%]": values[source][1], "dataset": source,
            })
            for source in order
        ], ignore_index=True)

    def simulate_commands(self, frame, *, layout=None):
        """Record commands on mocks; all Matplotlib creation/drawing is bypassed."""
        n_axes = 2 if layout == "panels" else 1
        axes = [MagicMock(name=f"fake_axis_{index}") for index in range(n_axes)]
        annotations = []

        def annotate(*args, **kwargs):
            annotation = MagicMock(name="fake_annotation")
            annotations.append(annotation)
            return annotation

        for axis in axes:
            axis.annotate.side_effect = annotate
        figure = MagicMock(name="fake_figure")
        axes_array = np.empty((1, n_axes), dtype=object)
        for index, axis in enumerate(axes):
            axes_array[0, index] = axis
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "no-figure.pdf"
            with (patch.object(figure_7.plt, "subplots", return_value=(figure, axes_array)) as subplots,
                  patch.object(figure_7.plt, "close") as close,
                  patch.object(figure_7, "Line2D") as legend_handle,
                  patch.object(figure_7, "inward_label_offset", return_value=((3, 3), "left", "bottom")),
                  patch.object(figure_7, "position_model_labels") as labels,
                  contextlib.redirect_stdout(io.StringIO())):
                figure_7.plot_pareto(frame, output, layout=layout)
            self.assertFalse(output.exists())
        close.assert_called_once_with(figure)
        labels.assert_called_once()
        self.assertIs(labels.call_args.args[0], figure)
        self.assertTrue(all(left is right for left, right in zip(labels.call_args.args[1], axes)))
        figure.savefig.assert_called_once()
        figure.suptitle.assert_not_called()
        for axis in axes:
            axis.set_title.assert_not_called()
        self.assertTrue(all(annotation._model_point_label is True for annotation in annotations))
        return axes, subplots, legend_handle

    def test_layout_defaults_overlay_only_for_the_accelerated_runner_pair(self):
        self.assertEqual(figure_7.comparison_layout([TORCHSIM, ASE]), "overlay")
        self.assertEqual(figure_7.comparison_layout([ASE, TORCHSIM]), "overlay")
        for sources in ([TORCHSIM], [ASE], ["mlip-trajs-torchsim-eager", TORCHSIM],
                        [TORCHSIM, ASE, "mlip-trajs-torchsim-eager"]):
            with self.subTest(sources=sources):
                self.assertEqual(figure_7.comparison_layout(sources), "panels")
        self.assertEqual(figure_7.comparison_layout([TORCHSIM, ASE], "panels"), "panels")
        self.assertEqual(figure_7.comparison_layout(["mlip-trajs-torchsim-eager", TORCHSIM], "overlay"), "overlay")

    def test_overlay_has_independent_frontiers_on_one_axis_with_stable_runner_styles(self):
        expected = {
            TORCHSIM: ([1.0, 2.0], [10.0, 5.0], "o", "-"),
            ASE: ([10.0, 20.0], [40.0, 20.0], "^", "--"),
        }
        for order in ((TORCHSIM, ASE), (ASE, TORCHSIM)):
            with self.subTest(order=order):
                frame = self.frame(order)
                original = frame.copy(deep=True)
                axes, subplots, handles = self.simulate_commands(frame)
                self.assertEqual(subplots.call_args.args, (1, 1))
                self.assertEqual(len(axes), 1)
                axis = axes[0]
                self.assertEqual(len(axis.plot.call_args_list), 2)
                self.assertEqual(len(axis.scatter.call_args_list), 8)
                for source, line in zip(order, axis.plot.call_args_list):
                    times, errors, _, linestyle = expected[source]
                    np.testing.assert_allclose(line.args[0], times)
                    np.testing.assert_allclose(line.args[1], errors)
                    self.assertEqual(line.kwargs["linestyle"], linestyle)
                for index, source in enumerate(order):
                    marker = expected[source][2]
                    scatters = axis.scatter.call_args_list[index * 4:(index + 1) * 4]
                    for scatter in scatters:
                        self.assertEqual(scatter.kwargs["marker"], marker)
                        if scatter.kwargs["label"] != "Pareto-optimal":
                            self.assertEqual(scatter.kwargs["color"],
                                             figure_7.TIER_COLORS[scatter.kwargs["label"]])
                    self.assertEqual(scatters[-1].kwargs["facecolors"], "none")
                self.assertEqual(axis.annotate.call_count, 6)
                self.assertTrue(all("arrowprops" not in call.kwargs
                                    for call in axis.annotate.call_args_list))
                self.assertEqual(axis.legend.call_count, 1)
                legend_labels = [call.kwargs["label"] for call in handles.call_args_list]
                self.assertIn("TorchSim (Pareto frontier)", legend_labels)
                self.assertIn("ASE (Pareto frontier)", legend_labels)
                pd.testing.assert_frame_equal(frame, original)

    def test_explicit_panels_keep_the_two_runners_on_separate_axes(self):
        axes, subplots, handles = self.simulate_commands(self.frame(), layout="panels")
        self.assertEqual(subplots.call_args.args, (1, 2))
        self.assertEqual(len(axes), 2)
        for axis, expected_times in zip(axes, ([1, 2], [10, 20])):
            axis.plot.assert_called_once()
            np.testing.assert_allclose(axis.plot.call_args.args[0], expected_times)
            self.assertEqual(axis.plot.call_args.kwargs["linestyle"], "-")
            axis.legend.assert_called_once()
            self.assertEqual(axis.annotate.call_count, 3)
        handles.assert_not_called()


if __name__ == "__main__":
    unittest.main()
