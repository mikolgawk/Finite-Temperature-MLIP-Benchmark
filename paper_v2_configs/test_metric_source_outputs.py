"""Regression tests for models with identical names across execution sources."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'e_f_rmses'))
sys.path.insert(0, str(HERE / 'pressures'))
import compute_mean_rmses_by_system_type as rmse
import get_model_pressure_errors as pressure
from metric_sources import SOURCE_DATASETS


class SourceOutputTests(unittest.TestCase):
    def test_rmse_means_are_separate_for_all_four_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for value, (source, (backend, mode)) in enumerate(SOURCE_DATASETS.items(), 1):
                input_dir = root / ('e-f-predictions' if backend == 'torchsim' else 'e-f-predictions-ase') / mode.replace('md_accelerated', 'md-accelerated')
                input_dir.mkdir(parents=True)
                pd.DataFrame({'system': ['bulkCu'], 'trajectory': ['/ref/bulkCu_300K/traj.extxyz'],
                              'energy_rmse': [value], 'force_rmse': [value * 10]}).to_csv(
                    input_dir / 'rmse-results-all_shared.csv', index=False)
            argv = ['rmse', '--data-dir', str(root / 'e-f-predictions'), '--results-dir', str(root / 'results')]
            with patch.object(sys, 'argv', argv), patch.object(rmse, 'torchsim_md_succeeded', return_value=True), contextlib.redirect_stdout(io.StringIO()):
                rmse.main()
            for value, source in enumerate(SOURCE_DATASETS, 1):
                directory = root / 'results' / source
                means = pd.read_csv(directory / 'mean_metrics_by_system_type_and_model.csv')
                self.assertEqual(means['energy_rmse'].tolist(), [value])
                self.assertEqual(means['force_rmse'].tolist(), [value * 10])
                self.assertEqual(pd.read_csv(directory / 'rmse_per_system.csv')['source'].tolist(), [source])
            self.assertFalse((root / 'results/mean_metrics_by_system_type_and_model.csv').exists())
            selected = next(iter(SOURCE_DATASETS))
            argv += ['--source', selected, '--results-dir', str(root / 'selected')]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                rmse.main()
            self.assertEqual([p.name for p in (root / 'selected').iterdir()], [selected])

    def test_pressure_mae_and_similarity_outputs_are_separate_and_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for value, (source, (backend, mode)) in enumerate(SOURCE_DATASETS.items(), 1):
                directory = root / 'inputs' / backend / mode
                (directory / 'references').mkdir(parents=True)
                frames = pd.DataFrame({'trajectory_file': ['/ref/bulkCu_300K/traj.extxyz'] * 3,
                                       'frame_index': [0, 1, 2], 'pressure_GPa': [1., 2., 3.]})
                frames.to_csv(directory / ('shared' + pressure.DEFAULT_MODEL_FILE_SUFFIX), index=False)
                frames.to_csv(directory / 'references/shared.csv', index=False)
                pd.DataFrame({'system': ['bulkCu_300K'], 'absolute_mean_error_GPa': [value]}).to_csv(
                    directory / ('shared' + pressure.TRAJECTORY_SUMMARY_SUFFIX), index=False)
            argv = ['pressure', '--pressures-dir', str(root / 'inputs'), '--results-dir', str(root / 'results')]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                pressure.main()
            for value, (source, (backend, mode)) in enumerate(SOURCE_DATASETS.items(), 1):
                directory = root / 'results' / source
                metric = pd.read_csv(directory / pressure.DEFAULT_OUTPUT_FILE.name)
                self.assertEqual(metric['pressure_mae_GPa'].tolist(), [value])
                self.assertEqual(metric['backend'].tolist(), [backend])
                self.assertEqual(metric['mode'].tolist(), [mode])
                self.assertEqual(metric['final_mean_pressure_similarity'].tolist(), [1.])
                self.assertEqual(pd.read_csv(directory / 'model_mean_pressure_comparison.csv')['error_GPa'].tolist(), [value])
            self.assertFalse((root / 'results' / pressure.DEFAULT_OUTPUT_FILE.name).exists())
            # Older TorchSim results omit the backend directory.
            (root / 'inputs/torchsim/md_eager').rename(root / 'inputs/md_eager')
            source = 'mlip-trajs-torchsim-eager'
            summary = root / 'inputs/md_eager' / ('shared' + pressure.TRAJECTORY_SUMMARY_SUFFIX)
            pd.DataFrame({'system': ['bulkCu_300K'], 'absolute_mean_error_GPa': [99]}).to_csv(summary, index=False)
            with patch.object(sys, 'argv', argv + ['--source', source]), contextlib.redirect_stdout(io.StringIO()):
                pressure.main()
            metric = pd.read_csv(root / 'results' / source / pressure.DEFAULT_OUTPUT_FILE.name)
            self.assertEqual(metric['pressure_mae_GPa'].tolist(), [99])
            # A model present in only one source must not fail on other sources.
            unique = root / 'inputs/md_eager' / ('only-eager' + pressure.DEFAULT_MODEL_FILE_SUFFIX)
            unique.write_text((root / 'inputs/md_eager' / ('shared' + pressure.DEFAULT_MODEL_FILE_SUFFIX)).read_text())
            with patch.object(sys, 'argv', argv + ['--model', 'only-eager']), patch.object(pressure, 'compute_pressure_metric') as compute:
                pressure.main()
            compute.assert_called_once()
            self.assertEqual(compute.call_args.kwargs['source'], source)


if __name__ == '__main__':
    unittest.main()
