"""Check pressure overview annotations without constructing figures."""

from types import SimpleNamespace
import unittest

from pressure_panel_style import annotate_tier_medians


class RecordingAxes:
    """Record annotation coordinates without constructing a Matplotlib figure."""

    def __init__(self, limits, height_points=120):
        self.limits = limits
        self.figure = SimpleNamespace(get_figheight=lambda: 12.5)
        self.position = SimpleNamespace(height=height_points / (12.5 * 72))
        self.texts = []
        self.connectors = []
        self.transform = object()

    def get_position(self):
        return self.position

    def get_ylim(self):
        return self.limits

    def set_ylim(self, lo, hi=None):
        if hi is None:
            lo, hi = lo
        self.limits = (lo, hi)

    def get_xaxis_transform(self):
        return self.transform

    def text(self, x, y, label, **kwargs):
        self.texts.append((x, y, label, kwargs))

    def vlines(self, x, start, stop, **kwargs):
        self.connectors.append((x, start, stop))


class PressurePanelStyleTest(unittest.TestCase):
    def test_headings_are_inside_and_median_connectors_reach_true_medians(self):
        ax = RecordingAxes((0, 125))
        tiers = [('Tier 1', 0, 2, 10, 'green'), ('Tier 2', 2, 4, 100, 'orange')]
        annotate_tier_medians([ax], tiers, maximum=100, font_size=10, format_value=str)
        self.assertEqual([text[2] for text in ax.texts], ['Tier 1', '10', 'Tier 2', '100'])
        for x, y, _, style in ax.texts:
            self.assertTrue(0 <= x <= 3)
            self.assertTrue(0 < y < 1)
            self.assertIs(style['transform'], ax.transform)
            self.assertEqual(style['va'], 'top')
        self.assertEqual([(x, start) for x, start, _ in ax.connectors], [(0.5, 10), (2.5, 100)])
        self.assertTrue(all(stop <= ax.get_ylim()[1] for _, _, stop in ax.connectors))
        median_label_y = ax.texts[1][1] * ax.get_ylim()[1]
        self.assertTrue(all(stop < median_label_y for _, _, stop in ax.connectors))
        self.assertLess(100 / ax.get_ylim()[1], ax.texts[1][1])

    def test_broken_axis_connectors_skip_gaps_and_preserve_large_medians(self):
        axes = [RecordingAxes((0, 10)), RecordingAxes((90, 120)),
                RecordingAxes((900, 1200), height_points=40)]
        tiers = [('Tier 1', 0, 2, 5, 'green'), ('Tier 2', 2, 4, 1000, 'orange')]
        annotate_tier_medians(axes, tiers, maximum=1000, font_size=10, format_value=str)
        self.assertEqual(axes[0].get_ylim(), (0, 10))
        self.assertEqual(axes[1].get_ylim(), (90, 120))
        self.assertGreater(axes[-1].get_ylim()[1], 1000)
        self.assertEqual(axes[0].connectors, [(0.5, 5, 10)])
        self.assertEqual(axes[1].connectors, [(0.5, 90, 120)])
        self.assertEqual([x for x, _, _ in axes[-1].connectors], [0.5, 2.5])
        self.assertEqual(axes[-1].connectors[1][1], 1000)
        self.assertTrue(all(0 < y < 1 for _, y, _, _ in axes[-1].texts))
        for ax in axes:
            lo, hi = ax.get_ylim()
            self.assertTrue(all(lo <= start < stop <= hi for _, start, stop in ax.connectors))

    def test_zero_errors_have_finite_axis_limits_and_visible_connectors(self):
        ax = RecordingAxes((0, 1))
        annotate_tier_medians([ax], [('Tier 1', 0, 1, 0, 'green')],
                              maximum=0, font_size=10, format_value=str)
        self.assertEqual(ax.get_ylim(), (0, 1))
        self.assertEqual(ax.texts[1][2], '0')
        self.assertGreater(ax.connectors[0][2], 0)

if __name__ == '__main__':
    unittest.main()
