"""Axis-break regressions: readable distributions without changing metrics."""

from __future__ import annotations

import contextlib
import io
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_4
import figure_SI_4
from pressure_axis_breaks import (
    distribution_axis_windows,
    histogram_segments,
    mae_axis_windows,
)


class PressureAxisBreaksTest(unittest.TestCase):
    def test_ordinary_data_keep_one_linear_axis(self):
        reference = np.linspace(-1, 1, 101)
        prediction = reference + 0.2
        self.assertEqual(len(distribution_axis_windows([reference, prediction], [reference])), 1)
        self.assertEqual(len(mae_axis_windows([0.5, 1.0, 2.0])), 1)

    def test_mae_breaks_retain_all_errors_and_medians(self):
        values = np.array([0.7, 1, 3, 5, 40, 60, 100, 1e6, 1e10])
        windows = mae_axis_windows(values)
        self.assertGreater(len(windows), 1)
        self.assertLess(windows[0][1], 10)
        for value in values:
            self.assertTrue(any(lo <= value <= hi for lo, hi in windows))

    def test_signed_outliers_keep_reference_range_readable(self):
        reference = np.linspace(-1, 1, 1001)
        prediction = np.r_[reference + 0.1, -1e7, 1e11]
        original = prediction.copy()
        windows = distribution_axis_windows([reference, prediction], [reference])
        self.assertEqual(len(windows), 3)
        core = next((lo, hi) for lo, hi in windows if lo <= 0 <= hi)
        self.assertLess(core[1] - core[0], 5)
        for value in [reference.min(), reference.max(), prediction.min(), prediction.max()]:
            self.assertTrue(any(lo <= value <= hi for lo, hi in windows))
        np.testing.assert_array_equal(prediction, original)

    def test_histogram_density_uses_full_frame_count(self):
        reference = np.linspace(-1, 1, 1001)
        values = np.r_[reference, -1e7, 1e11]
        windows = distribution_axis_windows([reference, values], [reference])
        segments = histogram_segments(values, windows, 80)
        masses = [np.sum(density * np.diff(edges)) for density, edges in segments]
        core_index = next(i for i, (lo, hi) in enumerate(windows) if lo <= 0 <= hi)
        self.assertAlmostEqual(masses[core_index], len(reference) / len(values))
        self.assertAlmostEqual(sum(masses), 1.0)

    def test_pooled_hydrogen_reference_gets_a_separate_range(self):
        ordinary = np.linspace(-1, 5, 1001)
        hydrogen = np.linspace(140, 145, 51)
        reference = np.r_[ordinary, hydrogen]
        values = np.r_[ordinary + 0.1, hydrogen - 3, 1e11]
        windows = distribution_axis_windows([reference, values], [reference])
        low = next(window for window in windows if window[0] <= 0 <= window[1])
        high = next(window for window in windows if window[0] <= 142 <= window[1])
        self.assertNotEqual(low, high)
        self.assertLess(low[1], 30)
        self.assertTrue(all(any(lo <= value <= hi for lo, hi in windows)
                            for value in reference))

    def test_extreme_tail_does_not_flatten_violin_core(self):
        reference = np.random.default_rng(17).normal(0, 1, 5000)
        values = np.r_[reference, 1e11]
        windows = distribution_axis_windows([reference, values], [reference])
        profiles = figure_SI_4._violin_profile(values, windows)
        core_index = next(i for i, (lo, hi) in enumerate(windows) if lo <= 0 <= hi)
        y, density = profiles[core_index]
        self.assertAlmostEqual(float(density.max()), 1.0)
        self.assertLess(abs(y[np.argmax(density)]), 0.25)
        self.assertEqual(values[-1], 1e11)

    def test_constant_violin_is_supported(self):
        profiles = figure_SI_4._violin_profile(np.ones(20), [(0, 2)])
        fig, ax = plt.subplots()
        try:
            figure_SI_4._half_violin(ax, 0, profiles[0], "left", "gray", 0.65, 0.42)
            self.assertEqual(len(ax.collections), 1)
        finally:
            plt.close(fig)

    def test_bar_rendering_keeps_original_maes_and_tier_medians(self):
        ranking = pd.DataFrame({
            "model": ["mace-mp-0", "mace-mpa-0", "orb-v2", "grace-oam", "nequip", "uma-s-omat"],
            "error_GPa": [1e6, 100, 3, 1, 1e10, 5],
        })
        original = ranking.copy(deep=True)
        fig, ax = plt.subplots()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                figure_4.draw_overall_pressure_mae_plot(ax, ranking, "(a)")
            self.assertGreater(len(fig.axes), 1)
            self.assertLess(fig.axes[0].get_ylim()[1], 10)
            for segment in fig.axes:
                np.testing.assert_array_equal(
                    [bar.get_height() for bar in segment.patches], ranking.error_GPa
                )
            for median in [1e6, 51.5, (1 + 1e10) / 2, 5]:
                self.assertTrue(any(lo <= median <= hi for lo, hi in
                                    (axis.get_ylim() for axis in fig.axes)))
            pd.testing.assert_frame_equal(ranking, original)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
