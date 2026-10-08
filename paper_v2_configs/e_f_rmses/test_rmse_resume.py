"""Exercise both v2 RMSE backends with real ASE references and fake predictions."""

import csv
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from ase import Atoms
from ase.io import write


HERE = Path(__file__).resolve().parent


def load_backend(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ASE = load_backend('resume_test_ase_rmse', 'rmse_ase_scripts/ase_rmse.py')
TORCHSIM = load_backend('resume_test_torchsim_rmse', 'rmse_torchsim_scripts/torchsim_rmse.py')


class RmseResumeTests(unittest.TestCase):
    backends = (ASE, TORCHSIM)

    @contextmanager
    def fixture(self, backend):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(ref_dir=root / 'ref-trajs', output_dir=root / 'results',
                                   md_dir=root / 'md', raw_energies=True, max_frames=None,
                                   force=False, no_progress=True, debug=False)
            names = ('bulkCu_300K_test', 'bulkCu_600K_test', 'bulkZn_300K_test')
            paths = []
            for name in names:
                path = args.ref_dir / name / 'traj.extxyz'
                path.parent.mkdir(parents=True)
                frames = []
                for index in range(2):
                    atoms = Atoms('Cu', positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
                    atoms.info.update(REF_energy=0.0, fixture=name, frame_index=index)
                    atoms.arrays['REF_forces'] = np.zeros((1, 3))
                    frames.append(atoms)
                write(path, frames)
                paths.append(path)
                md = args.md_dir / name
                md.mkdir(parents=True)
                (md / 'nvt_test-model.h5').touch()
                (md / 'md_timing_test-model.csv').write_text(
                    f'calculator,system,n_steps\ntest-model,{name},100\n'
                )
            output = args.output_dir / 'rmse-results-all_test-model.csv'
            yield SimpleNamespace(backend=backend, args=args, names=names, paths=paths,
                                  output=output, failures=output.with_suffix('.failures.json'))

    def run_backend(self, fixture, *, failed_frames=(), energy_error=1.0, interrupt=None,
                    calls=None):
        calls = [] if calls is None else calls

        def predict(model, atoms, *unused):
            label = (atoms.info['fixture'], atoms.info['frame_index'])
            calls.append(label)
            if label == interrupt:
                raise KeyboardInterrupt
            if label in failed_frames:
                raise RuntimeError('prediction failed')
            return energy_error, np.full((len(atoms), 3), 0.25)

        with patch.object(fixture.backend, '_ARGS', fixture.args), \
                patch.object(fixture.backend, 'predict', side_effect=predict), \
                redirect_stdout(io.StringIO()):
            if fixture.backend is ASE:
                fixture.backend.run_rmse('test-model', Mock(return_value=object()))
            else:
                fixture.backend.run_rmse('test-model', object())
        return calls

    def rows(self, fixture):
        with fixture.output.open(newline='') as handle:
            return list(csv.DictReader(handle))

    def fail_partial_and_full(self, fixture):
        failed = [(fixture.names[1], 1), (fixture.names[2], 0), (fixture.names[2], 1)]
        with self.assertRaisesRegex(SystemExit, 'Evaluation incomplete'):
            self.run_backend(fixture, failed_frames=failed)

    def test_retry_only_failed_trajectories_and_replace_partial_rows(self):
        for backend in self.backends:
            with self.subTest(backend=backend.__name__), self.fixture(backend) as fixture:
                self.fail_partial_and_full(fixture)
                before = self.rows(fixture)
                self.assertEqual(len(before), 2)
                self.assertEqual(before[1]['n_evaluated_frames'], '1')

                calls = self.run_backend(fixture, energy_error=2.0)

                self.assertEqual(calls, [(name, index) for name in fixture.names[1:]
                                         for index in range(2)])
                after = self.rows(fixture)
                self.assertEqual(len(after), 3)
                self.assertEqual(after[0], before[0])
                self.assertEqual([row['n_evaluated_frames'] for row in after], ['2'] * 3)
                self.assertEqual([float(row['energy_rmse']) for row in after], [1, 2, 2])
                self.assertEqual(len({row['trajectory'] for row in after}), 3)
                self.assertFalse(fixture.failures.exists())
                saved = fixture.output.read_bytes()
                self.assertEqual(self.run_backend(fixture), [])
                self.assertEqual(fixture.output.read_bytes(), saved)

    def test_retry_failure_keeps_completed_and_previous_partial_results(self):
        for backend in self.backends:
            with self.subTest(backend=backend.__name__), self.fixture(backend) as fixture:
                self.fail_partial_and_full(fixture)
                before = self.rows(fixture)
                failed = [(fixture.names[1], index) for index in range(2)]
                with self.assertRaisesRegex(SystemExit, 'Evaluation incomplete'):
                    self.run_backend(fixture, failed_frames=failed, energy_error=2.0)

                after = self.rows(fixture)
                self.assertEqual(after[:2], before)
                self.assertEqual(len(after), 3)
                remaining = json.loads(fixture.failures.read_text())
                self.assertEqual({failure['file'] for failure in remaining}, {str(fixture.paths[1])})
                calls = self.run_backend(fixture)
                self.assertEqual(calls, [(fixture.names[1], index) for index in range(2)])
                self.assertEqual(len(self.rows(fixture)), 3)
                self.assertFalse(fixture.failures.exists())

    def test_checkpoint_resumes_after_interrupt(self):
        for backend in self.backends:
            with self.subTest(backend=backend.__name__), self.fixture(backend) as fixture:
                with self.assertRaises(KeyboardInterrupt):
                    self.run_backend(fixture, interrupt=(fixture.names[1], 0))
                self.assertEqual(len(self.rows(fixture)), 1)

                calls = self.run_backend(fixture)

                self.assertEqual(calls, [(name, index) for name in fixture.names[1:]
                                         for index in range(2)])
                self.assertEqual(len(self.rows(fixture)), 3)
                self.assertFalse(fixture.failures.exists())

    def test_without_sidecar_computes_only_missing_trajectories(self):
        for backend in self.backends:
            with self.subTest(backend=backend.__name__), self.fixture(backend) as fixture:
                ref_root = fixture.args.ref_dir
                fixture.args.ref_dir = fixture.paths[0].parent
                self.run_backend(fixture)
                before = self.rows(fixture)
                self.assertFalse(fixture.failures.exists())
                fixture.args.ref_dir = ref_root

                calls = self.run_backend(fixture)

                self.assertEqual(calls, [(name, index) for name in fixture.names[1:]
                                         for index in range(2)])
                self.assertEqual(self.rows(fixture)[0], before[0])
                self.assertEqual(len(self.rows(fixture)), 3)

    def test_interrupted_csv_save_keeps_successful_retry_pending(self):
        for backend in self.backends:
            with self.subTest(backend=backend.__name__), self.fixture(backend) as fixture:
                self.fail_partial_and_full(fixture)
                saved = fixture.output.read_bytes()
                replace = Path.replace

                def interrupt_save(source, target):
                    if source == fixture.output.with_suffix('.csv.tmp'):
                        raise OSError('checkpoint interrupted')
                    return replace(source, target)

                with patch.object(Path, 'replace', autospec=True, side_effect=interrupt_save), \
                        self.assertRaisesRegex(OSError, 'checkpoint interrupted'):
                    self.run_backend(fixture)

                self.assertEqual(fixture.output.read_bytes(), saved)
                calls = self.run_backend(fixture)
                self.assertEqual(calls, [(name, index) for name in fixture.names[1:]
                                         for index in range(2)])
                self.assertEqual(len(self.rows(fixture)), 3)
                self.assertFalse(fixture.failures.exists())

    def test_sidecar_limits_retries_to_listed_trajectories(self):
        for backend in self.backends:
            with self.subTest(backend=backend.__name__), self.fixture(backend) as fixture:
                ref_root = fixture.args.ref_dir
                fixture.args.ref_dir = fixture.paths[0].parent
                self.run_backend(fixture)
                fixture.args.ref_dir = ref_root
                fixture.failures.write_text(json.dumps([
                    {'file': str(fixture.paths[1]), 'error': 'previous failure'}
                ]))

                calls = self.run_backend(fixture)

                self.assertEqual(calls, [(fixture.names[1], index) for index in range(2)])
                self.assertEqual(len(self.rows(fixture)), 2)
                self.assertFalse(fixture.failures.exists())

    def test_resume_matches_paths_from_before_directory_rename(self):
        for backend in self.backends:
            with self.subTest(backend=backend.__name__), self.fixture(backend) as fixture:
                self.fail_partial_and_full(fixture)
                rows = self.rows(fixture)
                failures = json.loads(fixture.failures.read_text())
                for row in rows:
                    path = Path(row['trajectory'])
                    row['trajectory'] = str(Path('/old/updated_configs/ref-trajs') / path.parent.name / path.name)
                for failure in failures:
                    path = Path(failure['file'])
                    failure['file'] = str(Path('/old/updated_configs/ref-trajs') / path.parent.name / path.name)
                with fixture.output.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                fixture.failures.write_text(json.dumps(failures))

                calls = self.run_backend(fixture)

                self.assertEqual(len(calls), 4)
                self.assertEqual(self.rows(fixture)[0], rows[0])
                self.assertEqual(len(self.rows(fixture)), 3)
                self.assertEqual(self.run_backend(fixture), [])

    def test_force_recomputes_every_eligible_trajectory(self):
        for backend in self.backends:
            with self.subTest(backend=backend.__name__), self.fixture(backend) as fixture:
                self.fail_partial_and_full(fixture)
                fixture.args.force = True

                calls = self.run_backend(fixture, energy_error=3.0)

                self.assertEqual(len(calls), 6)
                self.assertEqual(len(self.rows(fixture)), 3)
                self.assertEqual([float(row['energy_rmse']) for row in self.rows(fixture)], [3] * 3)
                self.assertFalse(fixture.failures.exists())

    def test_unavailable_failed_trajectory_preserves_outputs_and_marker(self):
        for backend in self.backends:
            with self.subTest(backend=backend.__name__), self.fixture(backend) as fixture:
                failed = [(fixture.names[2], index) for index in range(2)]
                with self.assertRaises(SystemExit):
                    self.run_backend(fixture, failed_frames=failed)
                saved = fixture.output.read_bytes()
                marker = fixture.failures.read_bytes()
                if backend is ASE:
                    fixture.paths[2].unlink()
                else:
                    (fixture.args.md_dir / fixture.names[2] / 'md_timing_test-model.csv').unlink()

                calls = []
                with self.assertRaises(SystemExit):
                    self.run_backend(fixture, calls=calls)

                self.assertEqual(calls, [])
                self.assertEqual(fixture.output.read_bytes(), saved)
                self.assertEqual(fixture.failures.read_bytes(), marker)


if __name__ == '__main__':
    unittest.main()
