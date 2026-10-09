"""Failed MD must penalize percentage metrics and never enter dimensional errors."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from md_success import (
    discover_torchsim_md_trajectories,
    load_failed_md_runs,
    registered_md_failure_reason,
    torchsim_md_succeeded,
)
from test_system_filters import load_module


SOURCES = ("mlip-trajs-torchsim-eager", "mlip-trajs-torchsim-accelerated")
METAL = "bulkCu_1000K_Kapil"
HYDROGEN = "H_1050K_Rupp_QE"
INTERFACE = "Pt111w24H2O_380K_Heenen_VASP"
CRYSTAL = "naphthalene_295K_Sharma_S"
NO_STRESS = "bulkCuAu_500K-Artrith_VASP"
NEQUIP = "nequip-oam-l-force-only"
MACE = "mace-mp-0"
FAILED_MODEL = "never-completes"


class MdFailurePolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.reference = self.data / "ref-trajs"
        self.systems = (METAL, HYDROGEN, INTERFACE, NO_STRESS, CRYSTAL)
        self.metadata = {}
        for system in self.systems:
            folder = self.reference / system
            folder.mkdir(parents=True)
            (folder / "traj.extxyz").touch()
            self.metadata[system] = {"timestep": 1.0, "stress_print_stride": 1}
        self.metadata[NO_STRESS]["stress_print_stride"] = None
        (self.reference / "md_metadata.json").write_text(json.dumps(self.metadata))
        for source in SOURCES:
            for system in self.systems:
                for model in (NEQUIP, MACE):
                    folder = self.data / source / system
                    folder.mkdir(parents=True, exist_ok=True)
                    with h5py.File(folder / f"nvt_{model}.h5", "w") as handle:
                        handle.create_dataset("data/positions", data=np.zeros((4, 2, 3)))
                        handle.create_dataset("data/velocities", data=np.ones((4, 2, 3)))
                    timing = folder / f"md_timing_{model}.csv"
                    timing.write_text("" if model == NEQUIP and system == HYDROGEN else
                                      f"calculator,system,n_steps\n{model},{system},100\n")
            # Some failures leave no HDF5 trajectory at all.
            (self.data / source / HYDROGEN / f"md_timing_{FAILED_MODEL}.csv").touch()

    def tearDown(self):
        self.temp.cleanup()

    def pressure_inputs(self, source):
        folder = self.root / "pressure" / source
        (folder / "references").mkdir(parents=True)
        for model in ("nequip", MACE):
            rows = []
            for system in (METAL, HYDROGEN):
                for index in range(16):
                    value = 140 + index / 10 if system == HYDROGEN else index / 10
                    rows.append({"system": system, "frame_index": index,
                                 "trajectory_file": f"/old-machine/{system}/nvt_{model}.h5",
                                 "pressure_GPa": value})
            reference = pd.DataFrame(rows)
            prediction = reference.copy()
            if model == "nequip":
                prediction.loc[(prediction.system == HYDROGEN) &
                               (prediction.frame_index == 15), "pressure_GPa"] = 1e11
            suffix = "_same-simulation-length_pressure_per_frame.csv"
            prediction.to_csv(folder / f"{model}{suffix}", index=False)
            reference.to_csv(folder / "references" / f"{model}.csv", index=False)
            pd.DataFrame({
                "system": [METAL, HYDROGEN],
                "trajectory_file": [f"/old-machine/{system}/nvt_{model}.h5"
                                    for system in (METAL, HYDROGEN)],
                "absolute_mean_error_GPa": [2.0 if model == "nequip" else 1.0,
                                            1e10 if model == "nequip" else 1.0],
            }).to_csv(folder / f"{model}_same-simulation-length_pressure_trajectory_summary.csv",
                      index=False)
        return folder

    def compute_pressure(self, source, folder, *, models=None, cached=None):
        pressure = load_module("pressures/get_model_pressure_errors.py")
        output = self.root / "pressure-output" / source
        with contextlib.redirect_stdout(io.StringIO()):
            result = pressure.compute_pressure_metric(
                folder, None, pressure.DEFAULT_MODEL_FILE_SUFFIX, 80,
                output / "pairs.csv", output / "systems.csv", output / "models.csv",
                output / "types.csv", cached, models=models, source=source,
                md_data_dir=self.data,
            )
        return result, pd.read_csv(output / "pairs.csv"), output

    def test_completion_requires_matching_timing_and_keeps_failed_job_discovery(self):
        for source in SOURCES:
            jobs = discover_torchsim_md_trajectories(self.data / source)
            self.assertIn(FAILED_MODEL, jobs[HYDROGEN])
            self.assertFalse(torchsim_md_succeeded(jobs[HYDROGEN][FAILED_MODEL]))
            self.assertFalse(torchsim_md_succeeded(jobs[HYDROGEN][NEQUIP]))
            self.assertTrue(torchsim_md_succeeded(jobs[METAL][NEQUIP]))
            # A timing record belonging to another calculator cannot authorize this job.
            path = jobs[METAL][NEQUIP]
            path.with_name(f"md_timing_{NEQUIP}.csv").write_text(
                f"calculator,system,n_steps\nother,{METAL},100\n"
            )
            self.assertFalse(torchsim_md_succeeded(path))

    def test_audited_failures_override_completed_records_and_are_scoped(self):
        registry = load_failed_md_runs()
        expected = {
            (SOURCES[0], HYDROGEN, "mace-mh-omat"),
            (SOURCES[0], HYDROGEN, "mace-mpa-0"),
            (SOURCES[0], INTERFACE, "eSEN-30M-OAM-force-only"),
            (SOURCES[1], HYDROGEN, "grace-mp"),
            (SOURCES[1], HYDROGEN, "mace-mh-omat-compile"),
            (SOURCES[1], HYDROGEN, "mace-mpa-0-compile"),
        }
        self.assertEqual(set(registry), expected)
        for source, system, model in expected:
            other_source = SOURCES[1] if source == SOURCES[0] else SOURCES[0]
            for run_source, run_system, should_succeed in (
                (source, system, False),
                (other_source, system, True),
                ("mlip-trajs-ase", system, True),
                (source, METAL, True),
            ):
                with self.subTest(source=run_source, system=run_system, model=model):
                    folder = self.data / run_source / run_system
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / f"nvt_{model}.h5"
                    with h5py.File(path, "w"):
                        pass
                    path.with_name(f"md_timing_{model}.csv").write_text(
                        f"calculator,system,n_steps\n{model},{run_system},100\n"
                    )
                    self.assertEqual(torchsim_md_succeeded(path), should_succeed)
                    reason = registered_md_failure_reason(path)
                    self.assertEqual(reason, None if should_succeed else registry[source, system, model])
                    self.assertEqual(registered_md_failure_reason(
                        path.with_name(f"md_timing_{model}.csv")
                    ), reason)
                    self.assertIn(model, discover_torchsim_md_trajectories(self.data / run_source)[run_system])

    def test_timing_and_pareto_loaders_exclude_audited_completed_runs(self):
        loaders = [load_module(path).load_model_avg_timings for path in (
            "rdfs/figure_SI_14.py", "vdos/figure_SI_16.py",
            "pressures/figure_SI_15.py", "pareto_plots/figure_7.py",
        )]
        timings = load_module("pareto_plots/plot-model-timings.py")
        for source in SOURCES:
            folder = self.root / "timing-audit" / source
            registered = [(system, model) for entry_source, system, model in load_failed_md_runs()
                          if entry_source == source]
            for system, model in registered:
                for run_system, seconds in ((system, 10.0), (METAL, 0.1)):
                    target = folder / run_system
                    target.mkdir(parents=True, exist_ok=True)
                    (target / f"md_timing_{model}.csv").write_text(
                        "calculator,system,n_steps,elapsed_seconds,seconds_per_step\n"
                        f"{model},{run_system},100,{seconds * 100},{seconds}\n"
                    )
            for loader in loaders:
                with self.subTest(source=source, loader=loader.__module__):
                    frame = loader(folder)
                    self.assertEqual(len(frame), len(registered))
                    column = next(name for name in frame if name.startswith("mean_time"))
                    np.testing.assert_allclose(frame[column], 100.0)
            frame, skipped = timings.read_timing_files(folder)
            self.assertEqual(set(frame.system), {METAL})
            self.assertEqual(len(frame), len(registered))
            self.assertEqual(len(skipped), len(registered))
            self.assertTrue(all("audited failed MD" in reason for reason in skipped))

    def test_registered_completed_failures_exclude_cached_dimensional_errors(self):
        rmse = load_module("e_f_rmses/compute_mean_rmses_by_system_type.py")
        for source in SOURCES:
            with self.subTest(source=source):
                trajectory = self.data / source / HYDROGEN / f"nvt_{MACE}.h5"
                self.assertTrue(torchsim_md_succeeded(trajectory))
                folder = self.pressure_inputs(source)
                summary_path = folder / f"{MACE}_same-simulation-length_pressure_trajectory_summary.csv"
                summary = pd.read_csv(summary_path)
                summary.loc[summary.system == HYDROGEN, "absolute_mean_error_GPa"] = 1e10
                summary.to_csv(summary_path, index=False)
                predictions = self.root / "cached-rmse" / source
                predictions.mkdir(parents=True)
                pd.DataFrame({
                    "system": ["bulkCu", "H"],
                    "trajectory": [f"/old-machine/{system}/traj.extxyz" for system in (METAL, HYDROGEN)],
                    "energy_rmse": [1.0, 1e6], "force_rmse": [2.0, 1e6],
                }).to_csv(predictions / f"rmse-results-all_{MACE}.csv", index=False)
                reason = "Unphysical temperature blow-up during completed MD"
                failures = {**load_failed_md_runs(), (source, HYDROGEN, MACE): reason}
                with patch("md_success.load_failed_md_runs", return_value=failures):
                    self.assertFalse(torchsim_md_succeeded(trajectory))
                    models, pairs, _ = self.compute_pressure(source, folder)
                    h = pairs[(pairs.system == HYDROGEN) & (pairs.mlip_model == MACE)].iloc[0]
                    self.assertEqual(h.pressure_error_percent, 100.0)
                    self.assertEqual(h.failure_reason, reason)
                    model = models.set_index("model").loc[MACE]
                    self.assertEqual(model.pressure_error_percent, 50.0)
                    self.assertEqual(model.pressure_mae_GPa, 1.0)
                    frame = rmse.load_all_data(predictions, md_data_dir=self.data)
                    self.assertEqual(frame.system.tolist(), ["bulkCu"])
                    self.assertEqual(frame.energy_rmse.mean(), 1.0)
                    self.assertEqual(frame.force_rmse.mean(), 2.0)

    def test_pressure_overrides_partial_scores_and_rebuilds_cached_mae(self):
        for source in SOURCES:
            with self.subTest(source=source):
                folder = self.pressure_inputs(source)
                pressure = load_module("pressures/get_model_pressure_errors.py")
                old_pairs = pressure.build_pair_rows(folder, None, pressure.DEFAULT_MODEL_FILE_SUFFIX, 80)
                hydrogen = old_pairs[(old_pairs.system == HYDROGEN) & (old_pairs.mlip_model == "nequip")]
                self.assertAlmostEqual(hydrogen.pressure_error_percent.iloc[0], 6.25)
                cached = folder / "cached-comparison.csv"
                pd.DataFrame({"model": ["nequip", FAILED_MODEL], "error_GPa": [1e9, 0]}).to_csv(cached, index=False)
                original_cache = cached.read_bytes()
                models, pairs, output = self.compute_pressure(source, folder, cached=cached)
                self.assertEqual(set(pairs.system), {METAL, HYDROGEN})
                self.assertEqual(len(pairs), 6)
                h = pairs[(pairs.system == HYDROGEN) & (pairs.mlip_model == "nequip")].iloc[0]
                self.assertEqual(h.pressure_error_percent, 100.0)
                self.assertEqual(h.pressure_similarity, 0.0)
                self.assertEqual(h.n_mlip_frames, 0)
                self.assertIn("incomplete", h.failure_reason)
                model = models.set_index("model")
                self.assertEqual(model.loc["nequip", "pressure_mae_GPa"], 2.0)
                self.assertEqual(model.loc["nequip", "pressure_error_percent"], 50.0)
                self.assertEqual(model.loc[MACE, "pressure_error_percent"], 0.0)
                self.assertEqual(model.loc[FAILED_MODEL, "pressure_error_percent"], 100.0)
                self.assertTrue(pd.isna(model.loc[FAILED_MODEL, "pressure_mae_GPa"]))
                mae = pd.read_csv(output / "model_mean_pressure_comparison.csv")
                self.assertEqual(set(mae.model), {"nequip", MACE})
                self.assertEqual(cached.read_bytes(), original_cache)

    def test_completely_failed_model_gets_penalties_without_pressure_csv(self):
        for source in SOURCES:
            with self.subTest(source=source):
                folder = self.root / "empty-pressure" / source
                folder.mkdir(parents=True)
                models, pairs, output = self.compute_pressure(source, folder, models={FAILED_MODEL})
                self.assertEqual(set(pairs.system), {METAL, HYDROGEN})
                self.assertEqual(pairs.pressure_error_percent.tolist(), [100.0, 100.0])
                self.assertTrue(models.pressure_mae_GPa.isna().all())
                self.assertTrue(pd.read_csv(output / "model_mean_pressure_comparison.csv").empty)

    def test_model_selection_and_ase_scores_are_preserved(self):
        source = SOURCES[0]
        folder = self.pressure_inputs(source)
        models, pairs, _ = self.compute_pressure(source, folder, models={NEQUIP})
        self.assertEqual(models.model.tolist(), ["nequip"])
        self.assertEqual(len(pairs), 2)
        pressure = load_module("pressures/get_model_pressure_errors.py")
        old_pairs = pressure.build_pair_rows(folder, None, pressure.DEFAULT_MODEL_FILE_SUFFIX, 80)
        ase = pressure.add_failed_system_penalties(old_pairs, folder, "mlip-trajs-ase",
                                                   None, set(), 80, self.data)
        pd.testing.assert_frame_equal(ase, old_pairs)

    def test_rmse_excludes_saved_failed_md_rows_in_custom_layouts(self):
        rmse = load_module("e_f_rmses/compute_mean_rmses_by_system_type.py")
        for source in SOURCES:
            with self.subTest(source=source):
                folder = self.root / "predictions" / source / "custom-export"
                folder.mkdir(parents=True)
                pd.DataFrame({
                    "system": ["bulkCu", "H"],
                    "trajectory": [f"/old-machine/{system}/traj.extxyz" for system in (METAL, HYDROGEN)],
                    "energy_rmse": [1.0, 1e6], "force_rmse": [2.0, 1e6],
                }).to_csv(folder / f"rmse-results-all_{NEQUIP}.csv", index=False)
                frame = rmse.load_all_data(folder, md_data_dir=self.data)
                self.assertEqual(frame.system.tolist(), ["bulkCu"])
                self.assertEqual(frame.energy_rmse.mean(), 1.0)
                self.assertEqual(frame.force_rmse.mean(), 2.0)

    def test_pressure_plot_loaders_exclude_cached_failed_md(self):
        panel = load_module("pressures/figure_4.py")
        violin = load_module("pressures/figure_SI_4.py")
        percentage_panel = load_module("pressures/figure_SI_6.py")
        for source in SOURCES:
            with self.subTest(source=source), contextlib.redirect_stdout(io.StringIO()):
                folder = self.pressure_inputs(source)
                with patch("get_model_pressure_errors.DEFAULT_MD_DATA_DIR", self.data):
                    panels = panel.collect_histogram_panels(folder, None, 80)
                    hydrogen = next(item for item in panels if item[1] == HYDROGEN)
                    self.assertNotIn("nequip", hydrogen[3])
                    self.assertIn(MACE, hydrogen[3])
                    models, _ = percentage_panel.load_model_values(folder)
                    self.assertNotIn(HYDROGEN, models["nequip"])
                    frame = violin.build_pressure_dataframe(folder, None)
                    nequip = frame[frame.model == "nequip"]
                    self.assertEqual(len(nequip), 32)  # Metal reference and prediction only.
                    self.assertLess(nequip.pressure_GPa.max(), 2.0)

    def test_torchsim_rmse_cannot_bypass_completion_checks_without_trajectory(self):
        rmse = load_module("e_f_rmses/compute_mean_rmses_by_system_type.py")
        folder = self.root / "custom-export"
        folder.mkdir()
        pd.DataFrame({"system": ["H"], "energy_rmse": [1.0], "force_rmse": [2.0]}).to_csv(
            folder / f"rmse-results-all_{NEQUIP}.csv", index=False
        )
        with self.assertRaisesRegex(ValueError, "Missing trajectory column"):
            rmse.load_all_data(folder, md_data_dir=self.data)

    def test_rdf_and_vdos_penalize_failed_jobs_even_with_saved_curves(self):
        self.check_rdf_and_vdos_penalties()

    def test_rdf_and_vdos_penalize_registered_completed_jobs_with_saved_curves(self):
        failures = {
            **load_failed_md_runs(),
            **{(source, HYDROGEN, MACE): "Unphysical temperature blow-up during completed MD"
               for source in SOURCES},
        }
        with patch("md_success.load_failed_md_runs", return_value=failures):
            self.check_rdf_and_vdos_penalties(mace_hydrogen_failed=True)

    def check_rdf_and_vdos_penalties(self, mace_hydrogen_failed=False):
        rdf = load_module("rdfs/get-rdf-and-results-by-system-type-same-simulation-length.py")
        vdos = load_module("vdos/get_normalized_VDOS.py")
        curve = (np.array([0.0, 1.0, 2.0]), np.array([0.2, 0.7, 0.1]))
        for source in SOURCES:
            with self.subTest(source=source), contextlib.redirect_stdout(io.StringIO()):
                output = self.root / "rdf-output"
                saved = output / source / "rdf_same_simulation_length_saved" / "mlip" / NEQUIP / f"{HYDROGEN}.csv"
                rdf.save_rdf_csv(*curve, saved)
                if mace_hydrogen_failed:
                    rdf.save_rdf_csv(*curve, saved.parent.parent / MACE / saved.name)
                args = SimpleNamespace(results_dir=output, systems=None, models=None,
                                       excluded_system_types=["molecular crystals"], dry_run=False)
                with (
                    patch.object(rdf, "REF_TRAJ_BASE_DIR", self.reference),
                    patch.dict(rdf.MLIP_TRAJ_DIRS, {source: self.data / source}),
                    patch.object(rdf, "load_reference_trajectory", return_value=SimpleNamespace(n_frames=4)),
                    patch.object(rdf, "load_mlip_trajectory", return_value=object()),
                    patch.object(rdf, "compute_rdf", return_value=curve),
                ):
                    rdf.process_source(args, source)
                detailed = pd.read_csv(output / source / f"rdf_similarity_scores_same_simulation_length_{NEQUIP}.csv")
                self.assertEqual(detailed.loc[detailed.System == HYDROGEN, "RDF_Error"].tolist(), [100.0])
                scores = pd.read_csv(output / source / "rdf_similarity_scores_same_simulation_length.csv").set_index("Calculator")
                self.assertEqual(scores.loc[NEQUIP, "Mean RDF Error [%]"], 25.0)
                self.assertEqual(scores.loc[FAILED_MODEL, "Mean RDF Error [%]"], 100.0)
                self.assertEqual(scores.loc[MACE, "Mean RDF Error [%]"], 25.0 if mace_hydrogen_failed else 0.0)

                output = self.root / "vdos-output"
                saved = output / source / vdos.SPECTRA_DIR / "mlip" / NEQUIP / f"{HYDROGEN}.csv"
                vdos.save_spectrum(saved, curve)
                if mace_hydrogen_failed:
                    vdos.save_spectrum(saved.parent.parent / MACE / saved.name, curve)
                args = SimpleNamespace(results_dir=output, systems=None, models=None,
                                       excluded_system_types=["molecular crystals"], dry_run=False,
                                       numerical_mlip_velocities=False, pad_factor=1, chunk_size=4,
                                       overwrite=False, e_min=None, e_max=None)
                with (
                    patch.object(vdos, "load_reference_positions", return_value=np.zeros((4, 2, 3))),
                    patch.object(vdos, "spectrum_for_samples", return_value=curve),
                ):
                    vdos.process_source(args, source, data_dir=self.data,
                                        reference_dir=self.reference, metadata=self.metadata)
                pairs = pd.read_csv(output / source / vdos.PAIR_OUTPUT)
                self.assertEqual(pairs.loc[(pairs.model == NEQUIP) & (pairs.system == HYDROGEN),
                                          "vdos_error_percent"].tolist(), [100.0])
                if mace_hydrogen_failed:
                    hydrogen = pairs[(pairs.model == MACE) & (pairs.system == HYDROGEN)].iloc[0]
                    self.assertEqual(hydrogen.vdos_error_percent, 100.0)
                    self.assertIn("Unphysical temperature", hydrogen.failure_reason)
                scores = pd.read_csv(output / source / vdos.MODEL_OUTPUT).set_index("model")
                self.assertEqual(scores.loc[NEQUIP, "vdos_error_percent"], 25.0)
                self.assertEqual(scores.loc[FAILED_MODEL, "vdos_error_percent"], 100.0)
                self.assertEqual(scores.loc[MACE, "vdos_error_percent"], 25.0 if mace_hydrogen_failed else 0.0)


if __name__ == "__main__":
    unittest.main()
