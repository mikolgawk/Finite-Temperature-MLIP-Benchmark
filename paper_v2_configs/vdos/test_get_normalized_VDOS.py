"""Numerical, PBC and cache regressions for the corrected VDOS analysis."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from ase import Atoms
from ase.io import write


_SPEC = importlib.util.spec_from_file_location(
    "updated_vdos_core", Path(__file__).with_name("get_normalized_VDOS.py")
)
vdos = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vdos)


def oscillating_positions(n_frames=256):
    time = np.arange(n_frames)
    positions = np.zeros((n_frames, 2, 3))
    positions[:, :, 0] = 2.0 + 0.4 * np.sin(2 * np.pi * time[:, None] / 32)
    positions[:, :, 1] = 3.0 + 0.2 * np.cos(2 * np.pi * time[:, None] / 17)
    return positions


class CorrectedEstimatorTest(unittest.TestCase):
    def test_velocity_tones_preserve_amplitude_squared_weights(self):
        n_frames = 8192
        time = np.arange(n_frames)
        for separate_components in (False, True):
            with self.subTest(separate_components=separate_components):
                velocities = np.zeros((n_frames, 1, 3))
                velocities[:, 0, 0] = np.cos(2 * np.pi * 0.03 * time)
                velocities[:, 0, int(separate_components)] += 2 * np.cos(2 * np.pi * 0.12 * time)
                wavenumber, power = vdos.compute_vdos(
                    velocities, 1.0, samples_are_velocities=True,
                )
                cycles_per_frame = wavenumber * vdos.SPEED_OF_LIGHT_CM_S * 1e-15
                low = np.abs(cycles_per_frame - 0.03) < 0.002
                high = np.abs(cycles_per_frame - 0.12) < 0.002
                ratio = vdos.integrate(wavenumber[high], power[high]) / vdos.integrate(wavenumber[low], power[low])
                self.assertAlmostEqual(ratio, 4.0, places=5)

    def test_frequency_units_and_unit_area(self):
        time = np.arange(4096)
        velocities = np.zeros((len(time), 1, 3))
        velocities[:, 0, 0] = np.cos(2 * np.pi * time / 64)
        wavenumber, power = vdos.compute_vdos(velocities, 0.5, samples_are_velocities=True)
        expected = (1.0 / 64) / (0.5e-15 * vdos.SPEED_OF_LIGHT_CM_S)
        self.assertAlmostEqual(wavenumber[np.argmax(power)], expected, places=10)
        self.assertAlmostEqual(wavenumber[-1], 1 / (2 * 0.5e-15 * vdos.SPEED_OF_LIGHT_CM_S), places=9)
        energy, normalized = vdos.normalized_spectrum(wavenumber, power)
        self.assertAlmostEqual(vdos.integrate(energy, normalized), 1.0)
        self.assertTrue(np.all(power >= 0))

    def test_chunking_does_not_change_component_weights(self):
        positions = oscillating_positions()
        spectrum_one = vdos.compute_vdos(positions, 1, samples_are_velocities=False, chunk_size=1)
        spectrum_many = vdos.compute_vdos(positions, 1, samples_are_velocities=False, chunk_size=32)
        np.testing.assert_allclose(spectrum_one, spectrum_many, rtol=1e-13, atol=1e-28)

    def test_endpoint_policy_is_central_difference_with_endpoints_dropped(self):
        positions = oscillating_positions()
        numerical = (positions[2:] - positions[:-2]) / 2
        from_positions = vdos.compute_vdos(positions, 1, samples_are_velocities=False)
        from_velocities = vdos.compute_vdos(numerical, 1, samples_are_velocities=True)
        np.testing.assert_allclose(from_positions, from_velocities, rtol=1e-13)

    def test_invalid_or_zero_signals_fail_explicitly(self):
        for samples, timestep in ((np.zeros((32, 1, 3)), 1), (oscillating_positions(), 0),
                                  (oscillating_positions(), float("nan"))):
            with self.subTest(timestep=timestep), self.assertRaises(ValueError):
                vdos.compute_vdos(samples, timestep, samples_are_velocities=False)


class WelchEstimatorTest(unittest.TestCase):
    def test_welch_preserves_tone_power_and_mass_weights(self):
        time = np.arange(8192)
        velocities = np.zeros((len(time), 2, 3))
        velocities[:, 0, 0] = np.cos(2 * np.pi * 0.03125 * time)
        velocities[:, 1, 0] = 2 * np.cos(2 * np.pi * 0.125 * time)
        for weighting, expected in (("velocity", 4.0), ("mass", 16.0)):
            with self.subTest(weighting=weighting):
                wavenumber, power = vdos.compute_vdos(
                    velocities, 1, samples_are_velocities=True, estimator="welch",
                    weighting=weighting, masses=np.array([1., 4.]), window_fs=1024,
                    pad_factor=4,
                )
                frequency = wavenumber * vdos.SPEED_OF_LIGHT_CM_S * 1e-15
                low = np.abs(frequency - 0.03125) < 0.01
                high = np.abs(frequency - 0.125) < 0.01
                ratio = vdos.integrate(wavenumber[high], power[high]) / vdos.integrate(wavenumber[low], power[low])
                self.assertAlmostEqual(ratio, expected, places=5)
                energy, density = vdos.normalized_spectrum(wavenumber, power)
                self.assertAlmostEqual(vdos.integrate(energy, density), 1.0)

    def test_welch_reduces_white_noise_periodogram_variance(self):
        rng = np.random.default_rng(951)
        samples = rng.normal(size=(32768, 1, 3))
        full = vdos.compute_vdos(samples, 1, samples_are_velocities=True)
        averaged = vdos.compute_vdos(
            samples, 1, samples_are_velocities=True, estimator="welch", window_fs=1024,
        )
        # The flat PSD's coefficient of variation measures estimator noise,
        # independent of frequency-grid length or overall normalization.
        full_power, mean_power = full[1][10:-10], averaged[1][10:-10]
        ratio = (np.std(mean_power) / np.mean(mean_power)) / (np.std(full_power) / np.mean(full_power))
        self.assertLess(ratio, 0.25)

    def test_equal_kinetic_energy_modes_have_equal_mass_weighted_area(self):
        time = np.arange(8192)
        velocities = np.zeros((len(time), 2, 3))
        velocities[:, 0, 0] = np.cos(2 * np.pi * time / 32)
        velocities[:, 1, 0] = 0.5 * np.cos(2 * np.pi * time / 8)
        wavenumber, power = vdos.compute_vdos(
            velocities, 1, samples_are_velocities=True, estimator="welch",
            weighting="mass", masses=np.array([1., 4.]), window_fs=1024, pad_factor=4,
        )
        frequency = wavenumber * vdos.SPEED_OF_LIGHT_CM_S * 1e-15
        low, high = np.abs(frequency - 1 / 32) < 0.01, np.abs(frequency - 1 / 8) < 0.01
        self.assertAlmostEqual(vdos.integrate(wavenumber[low], power[low]) / vdos.integrate(wavenumber[high], power[high]), 1.0, places=6)

    def test_segment_windows_are_physical_durations_not_frame_counts(self):
        slow = vdos.spectral_settings(20000, 1, window_fs=2000)
        fast = vdos.spectral_settings(40000, 0.5, window_fs=2000)
        self.assertEqual(slow["segment_duration_fs"], 2000)
        self.assertEqual(fast["segment_duration_fs"], 2000)
        self.assertEqual(slow["resolution_cm-1"], fast["resolution_cm-1"])
        self.assertEqual(slow["n_segments"], fast["n_segments"])
        self.assertEqual(fast["nperseg"], 2 * slow["nperseg"])

    def test_short_window_adaptation_uses_largest_window_with_minimum_segments(self):
        settings = vdos.spectral_settings(1000, 1, min_segments=4, pad_factor=4)
        self.assertEqual(settings["nperseg"], 400)
        self.assertEqual(settings["n_segments"], 4)
        self.assertEqual(settings["segment_duration_fs"], 400)
        self.assertTrue(settings["window_shortened"])
        self.assertTrue(settings["minimum_segments_satisfied"])
        self.assertEqual(settings["n_fft"], 2048)
        self.assertGreater(settings["resolution_cm-1"], settings["bin_spacing_cm-1"])
        for n_frames in range(9, 100):
            settings = vdos.spectral_settings(n_frames, 1, min_segments=4)
            length = settings["nperseg"]
            self.assertGreaterEqual(settings["n_segments"], 4)
            self.assertGreater((length + 1) + 3 * ((length + 1) - (length + 1) // 2), n_frames)

    def test_mass_weighting_requires_valid_explicit_masses(self):
        for masses in (None, np.array([1.]), np.array([1., -1.]), np.array([1., np.nan])):
            with self.subTest(masses=masses), self.assertRaises(ValueError):
                vdos.compute_vdos(oscillating_positions(), 1, samples_are_velocities=False, weighting="mass", masses=masses)

    def test_welch_matches_central_difference_and_component_chunking(self):
        samples = oscillating_positions()
        options = dict(estimator="welch", weighting="mass", masses=np.array([1., 4.]), window_fs=64)
        positions = vdos.compute_vdos(samples, 1, samples_are_velocities=False, chunk_size=1, **options)
        velocities = vdos.compute_vdos((samples[2:] - samples[:-2]) / 2, 1, samples_are_velocities=True, chunk_size=32, **options)
        np.testing.assert_allclose(positions, velocities, rtol=1e-13)

    def test_periodogram_diagnostic_matches_original_corrected_estimator(self):
        from scipy import signal
        samples = oscillating_positions()
        velocity = ((samples[2:] - samples[:-2]) / 2).reshape(len(samples) - 2, -1)
        frequencies, powers = signal.periodogram(velocity, fs=1e15, window="hann", detrend="constant", nfft=256, axis=0, scaling="density")
        actual = vdos.compute_vdos(samples, 1, samples_are_velocities=False, estimator="periodogram", weighting="velocity", pad_factor=1)
        np.testing.assert_array_equal(actual[0], frequencies / vdos.SPEED_OF_LIGHT_CM_S)
        # Unit-weight multiplication can change the reduction's roundoff.
        np.testing.assert_allclose(actual[1], powers.sum(axis=1), rtol=1e-14, atol=0)


class PeriodicReconstructionTest(unittest.TestCase):
    def test_wrapped_and_unwrapped_positions_agree_in_orthogonal_and_skew_cells(self):
        time = np.arange(600)
        positions = np.zeros((len(time), 2, 3))
        positions[:, :, 0] = 0.04 * time[:, None] + 0.1 * np.sin(time[:, None] / 7)
        positions[:, :, 1] = 0.03 * time[:, None] + 0.2 * np.cos(time[:, None] / 11)
        positions[:, :, 2] = 2.0
        for cell in (np.diag([8., 9., 10.]), np.array([[8., 0., 0.], [7.5, 2., 0.], [0., 0., 10.]])):
            with self.subTest(cell=cell):
                wrapped = ((positions @ np.linalg.inv(cell)) % 1.0) @ cell
                reconstructed = vdos.unwrap_positions(wrapped, cell, np.ones(3, bool))
                # The initial image is arbitrary; subsequent displacements
                # (and therefore velocities/spectra) must match exactly.
                np.testing.assert_allclose(np.diff(reconstructed, axis=0), np.diff(positions, axis=0), atol=2e-14)
                ref = vdos.compute_vdos(positions, 1, samples_are_velocities=False)
                other = vdos.compute_vdos(reconstructed, 1, samples_are_velocities=False)
                self.assertLess(vdos.vdos_similarity(ref, other)[0], 1e-9)

    def test_nonperiodic_directions_are_not_unwrapped(self):
        positions = np.array([[[0., 0., 0.]], [[9., 9., 0.]], [[8., 18., 0.]]])
        result = vdos.unwrap_positions(positions, np.eye(3) * 10, [True, False, False])
        np.testing.assert_allclose(result[:, 0, 0], [0, -1, -2])
        np.testing.assert_array_equal(result[:, 0, 1], positions[:, 0, 1])

    def test_changing_or_missing_periodic_cells_fail_explicitly(self):
        positions = oscillating_positions()
        cells = np.repeat((np.eye(3) * 10)[None], len(positions), axis=0)
        cells[10, 0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "changing cells"):
            vdos.unwrap_positions(positions, cells, [True] * 3)
        with self.assertRaisesRegex(ValueError, "singular cell"):
            vdos.unwrap_positions(positions, np.zeros((3, 3)), [True] * 3)

    def test_reference_and_hdf5_loaders_give_zero_error_for_same_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref_path, mlip_path = root / "traj.extxyz", root / "nvt_model.h5"
            positions = oscillating_positions()
            cell = np.eye(3) * 4
            wrapped = positions % 4
            write(ref_path, [Atoms("H2", positions=frame, cell=cell, pbc=True) for frame in wrapped])
            # Load the rounded extxyz coordinates before making the H5 copy,
            # ensuring genuinely identical input rather than a format error.
            ref_positions = vdos.load_reference_positions(ref_path)
            with h5py.File(mlip_path, "w") as handle:
                handle["data/positions"] = ref_positions % 4
                handle["data/cell"] = cell[None]
                handle["data/pbc"] = [True] * 3
                # Intentionally incompatible stored velocities must never
                # affect the default symmetric position-based comparison.
                handle["data/velocities"] = np.ones_like(positions) * 1e20
                handle["data/atomic_numbers"] = [[1, 1]]
            mlip_positions, is_velocity = vdos.load_mlip_samples(mlip_path, len(positions))
            self.assertFalse(is_velocity)
            ref = vdos.compute_vdos(ref_positions, 1, samples_are_velocities=False)
            mlip = vdos.compute_vdos(mlip_positions, 1, samples_are_velocities=False)
            self.assertLess(vdos.vdos_similarity(ref, mlip)[0], 1e-10)
            self.assertIn("cell_sha256", vdos.trajectory_identity(mlip_path))
            np.testing.assert_array_equal(vdos.load_reference_atomic_numbers(ref_path), [1, 1])
            np.testing.assert_array_equal(vdos.load_mlip_atomic_numbers(mlip_path, n_atoms=2), [1, 1])
            np.testing.assert_allclose(vdos.standard_atomic_masses(np.array([1, 8])), [1.008, 15.999])

    def test_hdf5_identity_metadata_cannot_silently_default_to_unit_mass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nvt_model.h5"
            with h5py.File(path, "w") as handle:
                handle["data/positions"] = oscillating_positions()
            with self.assertRaisesRegex(ValueError, "missing data/atomic_numbers"):
                vdos.load_mlip_atomic_numbers(path)
            with h5py.File(path, "a") as handle:
                handle["data/atomic_numbers"] = [[1, 8], [8, 1]]
            with self.assertRaisesRegex(ValueError, "identities/order change"):
                vdos.load_mlip_atomic_numbers(path)


class ValidatedCacheTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "spectrum.csv"
        self.samples = oscillating_positions()

    def spectrum(self, **kwargs):
        options = dict(samples_are_velocities=False, pad_factor=1, chunk_size=32, overwrite=False)
        samples = kwargs.pop("samples", self.samples)
        timestep = kwargs.pop("timestep", 1.0)
        options.update(kwargs)
        return vdos.spectrum_for_samples(self.path, samples, timestep, **options)

    def test_unchanged_cache_is_reused(self):
        expected = self.spectrum()
        with patch.object(vdos, "compute_vdos", side_effect=AssertionError("unexpected recalculation")):
            actual = self.spectrum()
        np.testing.assert_allclose(expected, actual)

    def test_sample_frame_timestep_padding_chunk_and_identity_changes_regenerate(self):
        changed = self.samples.copy()
        changed[10, 0, 0] += 0.3
        for options in ({"samples": changed}, {"samples": self.samples[:128]}, {"timestep": 0.5},
                        {"pad_factor": 2}, {"chunk_size": 3}, {"input_identity": {"path": "new-input"}}):
            with self.subTest(options=list(options)):
                self.spectrum(overwrite=True)
                with patch.object(vdos, "compute_vdos", wraps=vdos.compute_vdos) as compute:
                    self.spectrum(**options)
                    self.assertEqual(compute.call_count, 1)

    def test_method_version_changes_regenerate(self):
        self.spectrum()
        with patch.object(vdos, "METHOD_VERSION", "new-estimator-v2"), patch.object(vdos, "compute_vdos", wraps=vdos.compute_vdos) as compute:
            self.spectrum()
            self.assertEqual(compute.call_count, 1)

    def test_estimator_weighting_masses_window_count_and_overlap_changes_regenerate(self):
        baseline = dict(estimator="welch", weighting="mass", masses=np.array([1., 4.]),
                        window_fs=64, min_segments=4, overlap=0.5)
        changes = ({"estimator": "periodogram"}, {"weighting": "velocity"},
                   {"masses": np.array([1., 5.])}, {"window_fs": 80},
                   {"min_segments": 5}, {"overlap": 0.25})
        for changed in changes:
            with self.subTest(changed=changed):
                self.spectrum(overwrite=True, **baseline)
                with patch.object(vdos, "compute_vdos", wraps=vdos.compute_vdos) as compute:
                    self.spectrum(**(baseline | changed))
                    self.assertEqual(compute.call_count, 1)
        signature = json.loads(self.path.with_suffix(".csv.json").read_text())["signature"]
        self.assertEqual(signature["weighting"], "mass")
        self.assertEqual(signature["atomic_masses_amu"], [1., 4.])
        self.assertIn("resolution_cm-1", signature["spectral_settings"])

    def test_corrupt_or_missing_sidecars_and_csv_content_regenerate(self):
        sidecar = self.path.with_suffix(".csv.json")
        for corruption in ("missing", "bad-json", "wrong-type", "csv"):
            with self.subTest(corruption=corruption):
                self.spectrum(overwrite=True)
                if corruption == "missing":
                    sidecar.unlink()
                elif corruption == "bad-json":
                    sidecar.write_text("{not valid}")
                elif corruption == "wrong-type":
                    sidecar.write_text("[]")
                else:
                    self.path.write_text(self.path.read_text().replace("e-", "e+", 1))
                with patch.object(vdos, "compute_vdos", wraps=vdos.compute_vdos) as compute:
                    self.spectrum()
                    self.assertEqual(compute.call_count, 1)
                self.assertEqual(json.loads(sidecar.read_text())["spectrum_sha256"], vdos._file_digest(self.path))
                self.assertEqual(list(self.path.parent.glob(".*")), [])

    def test_file_identity_detects_replaced_input(self):
        source = self.path.parent / "source.h5"
        source.write_bytes(b"first")
        self.spectrum(input_identity=vdos.trajectory_identity(source))
        source.write_bytes(b"second")
        with patch.object(vdos, "compute_vdos", wraps=vdos.compute_vdos) as compute:
            self.spectrum(input_identity=vdos.trajectory_identity(source))
            self.assertEqual(compute.call_count, 1)


if __name__ == "__main__":
    unittest.main()
