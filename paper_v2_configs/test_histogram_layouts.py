"""Check bar spacing and subplot layout without writing figures or metrics."""

import contextlib
import io
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'e_f_rmses'), str(HERE / 'rdfs')]
import _plot_inputs
import fig_3


class HistogramLayoutTests(unittest.TestCase):
    def run_rmse_plot(self, script, frame):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'mean_metrics_by_system_type_and_model.csv').touch()
            with (
                patch.object(_plot_inputs, 'plot_args', return_value=(root, root)),
                patch.object(_plot_inputs, 'read_metric_csv', return_value=frame.copy()),
                patch.object(pd.DataFrame, 'to_csv'),
                patch.object(Figure, 'savefig', autospec=True) as save,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                runpy.run_path(str(HERE / 'e_f_rmses' / script), run_name='__main__')
            save.assert_called_once()
            self.assertFalse(list(root.glob('*.pdf')))
            fig = save.call_args.args[0]
            fig.canvas.draw()
            return fig

    def test_rmse_overviews_pack_present_models_and_skip_empty_tier_headers(self):
        frame = pd.DataFrame({
            'calculator': ['mace-mp-0', 'orb-v3-omat', 'custom-model'],
            'system': ['bulkCu_1000K_Kapil'] * 3,
            'energy_rmse': [0.05, 0.02, 0.07],
            'force_rmse': [0.1, 0.2, 0.3],
        })
        for script in ('figure_2.py', 'figure_SI_4.py'):
            with self.subTest(script=script):
                fig = self.run_rmse_plot(script, frame)
                for ax, metric in zip(fig.axes, ('energy_rmse', 'force_rmse')):
                    centers = [bar.get_x() + bar.get_width() / 2 for bar in ax.patches]
                    np.testing.assert_allclose(centers, [0, 1, 2])
                    np.testing.assert_allclose([bar.get_height() for bar in ax.patches], frame[metric])
                    headers = [text.get_text() for text in ax.texts
                               if text.get_text().startswith('Tier')]
                    self.assertEqual(headers, ['Tier 1', 'Tier 3'])

    def test_rmse_category_panels_have_consistent_widths_and_centered_last_rows(self):
        types = ['pure metals', 'perovskites', 'metal dichalcogenides',
                 'metal alloys', 'metal-water interfaces', 'hydrogen', 'molecular crystals']
        widths = {}
        for script in ('figure_SI_2.py', 'figure_SI_3.py'):
            for count in (5, 6, 7):
                with self.subTest(script=script, categories=count):
                    frame = pd.DataFrame({
                        'system_type': types[:count], 'calculator': ['mace-mp-0'] * count,
                        'energy_rmse': np.linspace(0.01, 0.05, count),
                        'force_rmse': np.linspace(0.1, 0.5, count),
                    })
                    fig = self.run_rmse_plot(script, frame)
                    self.assertEqual(len(fig.axes), count)
                    self.assertEqual([bool(ax.get_ylabel()) for ax in fig.axes],
                                     [index % 3 == 0 for index in range(count)])
                    specs = [ax.get_subplotspec() for ax in fig.axes]
                    self.assertTrue(all(spec.colspan.stop - spec.colspan.start == 2
                                        for spec in specs))
                    last_row = [spec for spec in specs if spec.rowspan.start == specs[-1].rowspan.start]
                    self.assertEqual(last_row[0].colspan.start + last_row[-1].colspan.stop, 6)
                    widths[(script, count)] = fig.get_size_inches()[0]
        self.assertEqual(len(set(widths.values())), 1)

    def test_rdf_overview_uses_present_models_and_retains_full_error_values(self):
        frame = pd.DataFrame({'Calculator': ['mace-mp-0', 'orb-v3-omat', 'custom-model'],
                              'RDF Error [%]': [100.0, 20.0, 30.0]})
        fig, ax = plt.subplots(figsize=(10, 3))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fig_3.draw_overall_error_plot(ax, frame)
            np.testing.assert_allclose([bar.get_x() + bar.get_width() / 2
                                        for bar in ax.patches], [0, 1, 2])
            np.testing.assert_allclose([bar.get_height() for bar in ax.patches], [100, 20, 30])
            headers = [text.get_text() for text in ax.texts if text.get_text().startswith('Tier')]
            self.assertEqual(headers, ['Tier 1', 'Tier 3'])
        finally:
            plt.close(fig)

    def test_rdf_tick_labels_clear_the_system_headings(self):
        frame = pd.DataFrame({'Calculator': ['mace-mp-0', 'orb-v3-omat', 'custom-model'],
                              'RDF Error [%]': [100.0, 20.0, 30.0]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(fig_3, 'load_overall_scores', return_value=frame),
                patch.object(fig_3, 'select_system_with_worst_mean_rdf_error',
                             side_effect=lambda systems, *args: systems[0]),
                patch.object(fig_3, 'find_model_error_means', return_value={}),
                patch.object(fig_3, 'find_model_system_errors', return_value={}),
                patch.object(fig_3, 'pick_best_worst_models', return_value=(None, None)),
                patch.object(fig_3, 'load_saved_rdf', return_value=(np.array([0., 1., 2.]),
                                                                 np.array([0., 1., 0.]))),
                patch.object(Figure, 'savefig', autospec=True) as save,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                fig_3.plot_combined(root, root, '', '', str(root), str(root / 'unused.pdf'))
            fig = save.call_args.args[0]
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            overview = fig.axes[-1]
            labels = overview.get_xticklabels() + [overview.xaxis.label]
            label_bottom = min(label.get_window_extent(renderer).y0 for label in labels)
            heading_top = max(ax.title.get_window_extent(renderer).y1
                              for ax in fig.axes if ax.get_title())
            self.assertGreater(label_bottom, heading_top)


if __name__ == '__main__':
    unittest.main()
