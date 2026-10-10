"""The accelerated paper cohort uses physical eager eSEN/Equiformer runs."""

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from md_success import discover_torchsim_md_trajectories, load_failed_md_runs
from model_display_names import normalize_display_key
from metric_sources import (
    TORCHSIM_ACCELERATED as ACCELERATED, TORCHSIM_EAGER as EAGER,
    cohort_model_files, model_trajectory_source,
)
from test_system_filters import load_module

METAL = "bulkCu_1000K_Kapil"
HYDROGEN = "H_1050K_Rupp_QE"
ESEN = "esen-30m-oam-stress"
EQ = "eq-v2-M-omat-force-only"
OTHER = "mace-mp-0"
MODELS = (ESEN, EQ, OTHER)


class AcceleratedCohortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.reference = self.data / "ref-trajs"
        self.metadata = {}
        for system in (METAL, HYDROGEN):
            folder = self.reference / system
            folder.mkdir(parents=True)
            (folder / "traj.extxyz").touch()
            self.metadata[system] = {"timestep": 1.0, "stress_print_stride": 1}
            for source, multiplier in ((EAGER, 1), (ACCELERATED, 5)):
                folder = self.data / source / system
                folder.mkdir(parents=True)
                for number, model in enumerate(MODELS, 1):
                    with h5py.File(folder / f"nvt_{model}.h5", "w") as handle:
                        handle.create_dataset("data/positions", data=np.zeros((4, 2, 3)))
                        handle.create_dataset("data/velocities", data=np.ones((4, 2, 3)))
                    seconds = number * multiplier
                    (folder / f"md_timing_{model}.csv").write_text(
                        "" if source == EAGER and model == ESEN and system == HYDROGEN else
                        "calculator,system,n_steps,elapsed_seconds,seconds_per_step,engine\n"
                        f"{model},{system},100,{seconds * 100},{seconds},cueq-compile\n"
                    )
        (self.reference / "md_metadata.json").write_text(json.dumps(self.metadata))

    def pressure_inputs(self):
        pressure = load_module("pressures/get_model_pressure_errors.py")
        for source, multiplier in ((EAGER, 1), (ACCELERATED, 5)):
            folder = self.root / "pressure" / source
            (folder / "references").mkdir(parents=True)
            for number, model in enumerate(MODELS, 1):
                frame = pd.DataFrame({
                    "trajectory_file": [f"/old/{system}/traj.extxyz"
                                        for system in (METAL, HYDROGEN) for _ in range(4)],
                    "frame_index": [0, 1, 2, 3] * 2,
                    "pressure_GPa": np.tile(np.arange(4.) + number * multiplier, 2),
                })
                frame.to_csv(folder / (model + pressure.DEFAULT_MODEL_FILE_SUFFIX), index=False)
                frame.to_csv(folder / "references" / f"{model}.csv", index=False)
                pd.DataFrame({
                    "system": [METAL, HYDROGEN],
                    "absolute_mean_error_GPa": [number * multiplier, number * multiplier],
                }).to_csv(folder / (model + pressure.TRAJECTORY_SUMMARY_SUFFIX), index=False)
        return pressure, self.root / "pressure" / ACCELERATED

    def test_sources_aliases_and_missing_eager_files(self):
        for model in (ESEN, EQ, "eSEN-30m-oam", "eq-v2-m-omat-eager"):
            self.assertEqual(model_trajectory_source(ACCELERATED, model), EAGER)
        for source, model in ((EAGER, ESEN), (ACCELERATED, OTHER),
                              ("mlip-trajs-ase-accelerated", EQ)):
            self.assertEqual(model_trajectory_source(source, model), source)
        folder = self.root / "missing" / ACCELERATED
        folder.mkdir(parents=True)
        (folder / f"{ESEN}.csv").touch()
        self.assertEqual(cohort_model_files(folder, "*.csv", lambda path: path.stem), [])
        for layout in ("torchsim/md_accelerated", "mlip-trajs-torchsim-accelerated"):
            folder = self.root / "layouts" / layout
            eager = folder.parent / ("md_eager" if folder.name == "md_accelerated" else EAGER)
            for parent in (folder, eager):
                parent.mkdir(parents=True, exist_ok=True)
                for model in MODELS:
                    (parent / f"{model}.csv").touch()
            paths = cohort_model_files(folder, "*.csv", lambda path: path.stem)
            self.assertEqual(set(paths), {eager / f"{ESEN}.csv", eager / f"{EQ}.csv",
                                          folder / f"{OTHER}.csv"})

    def test_trajectory_discovery_uses_physical_paths_once(self):
        runs = discover_torchsim_md_trajectories(self.data / ACCELERATED)
        for system in (METAL, HYDROGEN):
            self.assertEqual(set(runs[system]), set(MODELS))
            for model, path in runs[system].items():
                self.assertEqual(path, self.data / model_trajectory_source(ACCELERATED, model)
                                 / system / f"nvt_{model}.h5")

    def test_rmse_accelerated_only_includes_eager_models_and_excludes_their_failed_md(self):
        rmse = load_module("e_f_rmses/compute_mean_rmses_by_system_type.py")
        for source, multiplier in ((EAGER, 1), (ACCELERATED, 5)):
            folder = self.root / "rmse" / source
            folder.mkdir(parents=True)
            for number, model in enumerate(MODELS, 1):
                pd.DataFrame({
                    "system": ["bulkCu", "H"],
                    "trajectory": [f"/ref/{system}/traj.extxyz" for system in (METAL, HYDROGEN)],
                    "energy_rmse": [number * multiplier] * 2,
                    "force_rmse": [number * multiplier * 10] * 2,
                }).to_csv(folder / f"rmse-results-all_{model}.csv", index=False)
        original = (self.root / "rmse" / EAGER / f"rmse-results-all_{ESEN}.csv").read_bytes()
        frame = rmse.load_all_data(self.root / "rmse", sources={ACCELERATED}, md_data_dir=self.data)
        self.assertEqual(set(frame.source), {ACCELERATED})
        self.assertEqual(frame.groupby("calculator").energy_rmse.mean().to_dict(),
                         {ESEN: 1., EQ: 2., OTHER: 15.})
        self.assertEqual(frame[frame.calculator == ESEN].system.tolist(), ["bulkCu"])
        self.assertEqual(frame.groupby("calculator").trajectory_source.first().to_dict(),
                         {ESEN: EAGER, EQ: EAGER, OTHER: ACCELERATED})
        both = rmse.load_all_data(self.root / "rmse", sources={EAGER, ACCELERATED}, md_data_dir=self.data)
        self.assertEqual(len(both[both.source == ACCELERATED]), len(frame))
        self.assertEqual(original, (self.root / "rmse" / EAGER / f"rmse-results-all_{ESEN}.csv").read_bytes())
        correlation = load_module("vdos/figure_SI_7_8_9.py")
        with patch.object(correlation, "CONFIG_DIR", self.root):
            means = correlation.force_rmse_excluding_systems(
                frame, ACCELERATED, exclude_hydrogen=True,
            ).set_index("calculator")
        self.assertEqual(means.loc["esen-30m-oam", "force_rmse"], 10.)
        self.assertEqual(means.loc["eq-v2-m-omat", "force_rmse"], 20.)

    def test_pressure_metrics_mae_and_plot_inputs_use_eager_references_and_failures(self):
        pressure, folder = self.pressure_inputs()
        output = self.root / "output"
        with contextlib.redirect_stdout(io.StringIO()):
            pressure.compute_pressure_metric(
                folder, None, pressure.DEFAULT_MODEL_FILE_SUFFIX, 8,
                output / "pairs.csv", output / "systems.csv", output / "models.csv",
                output / "types.csv", None, source=ACCELERATED, md_data_dir=self.data,
            )
        pairs = pd.read_csv(output / "pairs.csv")
        self.assertEqual(len(pairs), 6)
        self.assertEqual(set(pairs.source), {ACCELERATED})
        self.assertEqual(set(pairs["mode"]), {"md_accelerated"})
        models = pd.read_csv(output / "models.csv").set_index("model")
        self.assertEqual(models.loc["esen-30m-oam", "pressure_error_percent"], 50.)
        self.assertEqual(models.loc["esen-30m-oam", "pressure_mae_GPa"], 1.)
        self.assertEqual(models.loc["eq-v2-m-omat", "pressure_mae_GPa"], 2.)
        self.assertEqual(models.loc[OTHER, "pressure_mae_GPa"], 15.)
        eq_rows = pairs[pairs.mlip_model == "eq-v2-m-omat"]
        self.assertTrue(eq_rows.reference_file.str.contains(EAGER).all())
        self.assertTrue(eq_rows.model_file.str.contains(EAGER).all())
        self.assertEqual(set(eq_rows.trajectory_source), {EAGER})
        for filename in ("figure_4.py", "figure_SI_6.py", "figure_SI_4.py"):
            module = load_module("pressures/" + filename)
            with (patch("get_model_pressure_errors.DEFAULT_MD_DATA_DIR", self.data),
                  contextlib.redirect_stdout(io.StringIO())):
                if filename == "figure_SI_4.py":
                    frame = module.build_pressure_dataframe(folder, None)
                    self.assertEqual(len(frame[frame.model == "esen-30m-oam"]), 8)
                else:
                    panels = (module.collect_histogram_panels(folder, None, 8)
                              if filename == "figure_4.py" else
                              module.collect_histogram_panels(folder, None, 8,
                                                               [normalize_display_key(model) for model in MODELS]))
                    hydrogen = next(item for item in panels if item[1] == HYDROGEN)
                    self.assertNotIn("esen-30m-oam", hydrogen[3])
                    np.testing.assert_allclose(hydrogen[3]["eq-v2-m-omat"], np.arange(4.) + 2)
        reason = "Audited eager failure"
        failures = {**load_failed_md_runs(), (EAGER, METAL, EQ): reason}
        with (patch("md_success.load_failed_md_runs", return_value=failures),
              patch.object(pressure, "load_failed_md_runs", return_value=failures)):
            empty = pairs.iloc[:0].copy()
            penalties = pressure.add_failed_system_penalties(empty, folder, ACCELERATED,
                                                             {EQ}, set(), 8, self.data)
        failed = penalties[penalties.system == METAL].iloc[0]
        self.assertEqual(failed.failure_reason, reason)
        self.assertEqual(failed.pressure_error_percent, 100.)

    def test_all_pareto_and_timing_loaders_use_eager_step_times(self):
        for filename in ("rdfs/figure_SI_14.py", "vdos/figure_SI_16.py",
                         "pressures/figure_SI_15.py", "pareto_plots/figure_7.py",
                         "pareto_plots/figure_SI_13.py"):
            module = load_module(filename)
            frame = module.load_model_avg_timings(self.data / ACCELERATED).set_index("model")
            column = next(name for name in frame if name.startswith("mean_time"))
            self.assertEqual(frame[column].to_dict(),
                             {"esen-30m-oam": 1000., "eq-v2-m-omat": 2000., OTHER: 15000.})
        timing = load_module("pareto_plots/plot-model-timings.py")
        frame, _ = timing.read_timing_files(self.data / ACCELERATED, "Accelerated")
        esen = frame[frame.model.map(normalize_display_key) == "esen-30m-oam"]
        self.assertEqual(set(esen.trajectory_source), {EAGER})
        self.assertEqual(esen.seconds_per_step.tolist(), [1.])

    def test_rdf_and_vdos_compute_with_eager_paths_and_write_cohort_provenance(self):
        curve = (np.array([0., 1.]), np.array([1., 2.]))
        rdf = load_module("rdfs/get-rdf-and-results-by-system-type-same-simulation-length.py")
        output = self.root / "rdf-output"
        args = SimpleNamespace(results_dir=output, systems=None, models=None,
                               excluded_system_types=["molecular crystals"], dry_run=False)
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(rdf, "REF_TRAJ_BASE_DIR", self.reference),
            patch.dict(rdf.MLIP_TRAJ_DIRS, {ACCELERATED: self.data / ACCELERATED}),
            patch.object(rdf, "load_reference_trajectory", return_value=SimpleNamespace(n_frames=4)),
            patch.object(rdf, "load_mlip_trajectory", return_value=object()) as load,
            patch.object(rdf, "compute_rdf", return_value=curve),
        ):
            rdf.process_source(args, ACCELERATED)
        self.assertEqual({call.args[0] for call in load.call_args_list},
                         {self.data / model_trajectory_source(ACCELERATED, model) / system / f"nvt_{model}.h5"
                          for model in MODELS for system in (METAL, HYDROGEN)
                          if not (model == ESEN and system == HYDROGEN)})
        scores = pd.read_csv(output / ACCELERATED / "rdf_similarity_scores_same_simulation_length.csv").set_index("Calculator")
        self.assertEqual(scores.loc[ESEN, "Mean RDF Error [%]"], 50.)
        self.assertEqual(scores.loc[EQ, "Mean RDF Error [%]"], 0.)
        self.assertEqual(scores.loc[ESEN, "trajectory_source"], EAGER)
        vdos = load_module("vdos/get_normalized_VDOS.py")
        output = self.root / "vdos-output"
        args = SimpleNamespace(results_dir=output, systems=None, models=None,
                               excluded_system_types=["molecular crystals"], dry_run=False,
                               numerical_mlip_velocities=False, pad_factor=1, chunk_size=4,
                               overwrite=False, e_min=None, e_max=None)
        with (contextlib.redirect_stdout(io.StringIO()),
              patch.object(vdos, "load_reference_positions", return_value=np.zeros((8, 2, 3))),
              patch.object(vdos, "spectrum_for_samples", return_value=curve) as spectra):
            vdos.process_source(args, ACCELERATED, data_dir=self.data,
                                reference_dir=self.reference, metadata=self.metadata)
        scores = pd.read_csv(output / ACCELERATED / vdos.MODEL_OUTPUT).set_index("model")
        self.assertEqual(scores.loc[ESEN, "vdos_error_percent"], 50.)
        self.assertEqual(scores.loc[EQ, "vdos_error_percent"], 0.)
        self.assertEqual(scores.loc[ESEN, "source"], ACCELERATED)
        self.assertEqual(scores.loc[ESEN, "trajectory_source"], EAGER)
        for call in spectra.call_args_list:
            path = call.args[0]
            if path.parent.name in (ESEN, EQ):
                self.assertTrue(call.kwargs["overwrite"], f"Stale cohort spectrum reused: {path}")


if __name__ == "__main__":
    unittest.main()
