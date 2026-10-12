"""ASE eager CHGNet belongs to the accelerated paper cohort, with real provenance."""

import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from md_success import discover_torchsim_md_trajectories, torchsim_md_succeeded
from metric_sources import ASE_EAGER, cohort_model_files, cohort_trajectory_path, model_trajectory_source
from model_display_names import display_model_name, normalize_display_key
import test_accelerated_cohort as cohort
from test_accelerated_cohort import ACCELERATED, HYDROGEN, METAL, MODELS
from test_system_filters import load_module

CHGNET = "chgnet-force-only-ase"
ASE_DIRECTORY = "mlip-trajs-ase-eager"


class AseChgnetCohortTests(unittest.TestCase):
    def setUp(self):
        cohort.AcceleratedCohortTests.setUp(self)
        for system in (METAL, HYDROGEN):
            directory = self.data / ASE_DIRECTORY / system
            directory.mkdir(parents=True)
            with h5py.File(directory / f"nvt_{CHGNET}.h5", "w") as handle:
                handle.create_dataset("data/positions", data=np.zeros((4, 2, 3)))
                handle.create_dataset("data/velocities", data=np.ones((4, 2, 3)))
            (directory / f"md_timing_{CHGNET}.csv").write_text(
                "" if system == HYDROGEN else
                "calculator,system,n_steps,elapsed_seconds,seconds_per_step,engine\n"
                f"{CHGNET},{system},100,400,4,ase+chgnet\n"
            )
            native = self.data / ACCELERATED / system
            (native / "nvt_chgnet.h5").touch()
            (native / "md_timing_chgnet.csv").write_text(
                "calculator,system,n_steps,elapsed_seconds,seconds_per_step\n"
                f"chgnet,{system},100,9900,99\n"
            )

    def test_physical_paths_aliases_completion_and_no_accelerated_fallback(self):
        for name in ("chgnet", "chgnet-force-only", CHGNET, "CHGNet-ase"):
            self.assertEqual(model_trajectory_source(ACCELERATED, name), ASE_EAGER)
            self.assertEqual(display_model_name(name), "CHGNet")
            self.assertEqual(cohort_trajectory_path(self.data, ACCELERATED, METAL, name),
                             self.data / ASE_DIRECTORY / METAL / f"nvt_{CHGNET}.h5")
        runs = discover_torchsim_md_trajectories(self.data / ACCELERATED)
        self.assertEqual(set(runs[METAL]), {*MODELS, CHGNET})
        self.assertFalse(torchsim_md_succeeded(runs[HYDROGEN][CHGNET]))
        reason = "Audited ASE failure"
        for source in (ASE_EAGER, ASE_DIRECTORY):
            with patch("md_success.load_failed_md_runs", return_value={(source, METAL, CHGNET): reason}):
                self.assertFalse(torchsim_md_succeeded(runs[METAL][CHGNET]))
        for index, layout in enumerate(("torchsim/md_accelerated", "mlip-trajs-torchsim-accelerated")):
            root = self.root / "layouts" / str(index) / layout
            root.mkdir(parents=True)
            (root / "chgnet.csv").touch()
            self.assertEqual(cohort_model_files(root, "*.csv", lambda path: path.stem), [])
            ase = (root.parent.parent / "ase/md_eager" if root.parent.name == "torchsim"
                   else root.parent / ASE_DIRECTORY)
            ase.mkdir(parents=True)
            (ase / "chgnet.csv").touch()
            self.assertEqual(cohort_model_files(root, "*.csv", lambda path: path.stem), [ase / "chgnet.csv"])

    def test_rmse_correlations_use_ase_evaluations_and_exclude_failed_md(self):
        rmse = load_module("e_f_rmses/compute_mean_rmses_by_system_type.py")
        root = self.root / "rmse"
        for source, model, value in ((ACCELERATED, "chgnet", 999), (ASE_DIRECTORY, "chgnet-force-only", 4)):
            directory = root / source
            directory.mkdir(parents=True)
            pd.DataFrame({
                "system": ["bulkCu", "H"],
                "trajectory": [f"/ref/{system}/traj.extxyz" for system in (METAL, HYDROGEN)],
                "energy_rmse": [value, value * 100], "force_rmse": [value * 10, value * 1000],
            }).to_csv(directory / f"rmse-results-all_{model}.csv", index=False)
        frame = rmse.load_all_data(root, sources={ACCELERATED}, models={"chgnet"}, md_data_dir=self.data)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0].energy_rmse, 4)
        self.assertEqual(frame.iloc[0].source, ACCELERATED)
        self.assertEqual(frame.iloc[0].trajectory_source, ASE_EAGER)
        correlation = load_module("vdos/figure_SI_7_8_9.py")
        with patch.object(correlation, "CONFIG_DIR", self.root):
            means = correlation.force_rmse_excluding_systems(frame, ACCELERATED)
        self.assertEqual(means.iloc[0].calculator, "chgnet")
        self.assertEqual(means.iloc[0].force_rmse, 40)
        with patch("md_success.load_failed_md_runs", return_value={(ASE_EAGER, METAL, CHGNET): "bad MD"}):
            self.assertTrue(rmse.load_all_data(root, sources={ACCELERATED}, md_data_dir=self.data).empty)

    def test_pressure_references_mae_histogram_inputs_and_timing_models_match(self):
        pressure, folder = cohort.AcceleratedCohortTests.pressure_inputs(self)
        ase = self.root / "pressure" / "ase/md_eager"
        (ase / "references").mkdir(parents=True)
        frame = pd.DataFrame({
            "trajectory_file": [f"/old/{system}/traj.extxyz" for system in (METAL, HYDROGEN) for _ in range(4)],
            "frame_index": [0, 1, 2, 3] * 2,
            "pressure_GPa": np.tile(np.arange(4.), 2),
        })
        frame.to_csv(ase / ("chgnet" + pressure.DEFAULT_MODEL_FILE_SUFFIX), index=False)
        frame.to_csv(ase / "references/chgnet.csv", index=False)
        pd.DataFrame({"system": [METAL, HYDROGEN], "absolute_mean_error_GPa": [4., 999.]}).to_csv(
            ase / ("chgnet" + pressure.TRAJECTORY_SUMMARY_SUFFIX), index=False,
        )
        # Native accelerated values must be replaced, including stale dimensional errors.
        frame.assign(pressure_GPa=999.).to_csv(folder / ("chgnet" + pressure.DEFAULT_MODEL_FILE_SUFFIX), index=False)
        output = self.root / "output"
        with contextlib.redirect_stdout(io.StringIO()):
            pressure.compute_pressure_metric(
                folder, None, pressure.DEFAULT_MODEL_FILE_SUFFIX, 8,
                output / "pairs.csv", output / "systems.csv", output / "models.csv",
                output / "types.csv", None, source=ACCELERATED, md_data_dir=self.data,
            )
        model = pd.read_csv(output / "models.csv").set_index("model").loc["chgnet"]
        self.assertEqual(model.pressure_error_percent, 50.)
        self.assertEqual(model.pressure_mae_GPa, 4.)
        pairs = pd.read_csv(output / "pairs.csv")
        row = pairs[(pairs.mlip_model == "chgnet") & (pairs.system == METAL)].iloc[0]
        self.assertEqual(row.trajectory_source, ASE_EAGER)
        self.assertIn("ase/md_eager", row.reference_file)
        self.assertEqual(row.source, ACCELERATED)
        failures = {(ASE_DIRECTORY, METAL, CHGNET): "Audited ASE failure"}
        with (patch("md_success.load_failed_md_runs", return_value=failures),
              patch.object(pressure, "load_failed_md_runs", return_value=failures)):
            audited = pressure.add_failed_system_penalties(
                pairs, folder, ACCELERATED, {"chgnet"}, set(), 8, self.data,
            ).set_index("system")
        self.assertEqual(audited.loc[METAL, "failure_reason"], "Audited ASE failure")
        self.assertEqual(audited.loc[METAL, "pressure_error_percent"], 100.)
        for script in ("figure_4.py", "figure_SI_6.py", "figure_SI_4.py"):
            module = load_module("pressures/" + script)
            with patch("get_model_pressure_errors.DEFAULT_MD_DATA_DIR", self.data):
                if script == "figure_SI_4.py":
                    rows = module.build_pressure_dataframe(folder, None)
                    self.assertEqual(len(rows[(rows.model == "chgnet") & (rows.kind == "model")]), 4)
                else:
                    panels = (module.collect_histogram_panels(folder, None, 8) if script == "figure_4.py" else
                              module.collect_histogram_panels(folder, None, 8, ["chgnet"]))
                    hydrogen = next(panel for panel in panels if panel[1] == HYDROGEN)
                    self.assertNotIn("chgnet", hydrogen[3])
        for script in ("rdfs/figure_SI_14.py", "vdos/figure_SI_16.py", "pressures/figure_SI_15.py",
                       "pareto_plots/figure_7.py", "pareto_plots/figure_SI_13.py"):
            module = load_module(script)
            rows = module.load_model_avg_timings(self.data / ACCELERATED).set_index("model")
            column = next(name for name in rows if name.startswith("mean_time"))
            self.assertEqual(rows.loc["chgnet", column], 4000.)
        timing = load_module("pareto_plots/plot-model-timings.py")
        rows, _ = timing.read_timing_files(self.data / ACCELERATED, "Accelerated")
        chgnet = rows[rows.model.map(normalize_display_key) == "chgnet"]
        self.assertEqual(chgnet.seconds_per_step.tolist(), [4.])
        self.assertEqual(chgnet.model_display.tolist(), ["CHGNet"])

    def test_rdf_vdos_use_ase_positions_velocities_and_record_hydrogen_penalties(self):
        curve = (np.array([0., 1.]), np.array([1., 2.]))
        rdf = load_module("rdfs/get-rdf-and-results-by-system-type-same-simulation-length.py")
        output = self.root / "rdf-output"
        args = SimpleNamespace(results_dir=output, systems=None, models=["chgnet"],
                               excluded_system_types=["molecular crystals"], dry_run=False)
        with (contextlib.redirect_stdout(io.StringIO()),
              patch.object(rdf, "REF_TRAJ_BASE_DIR", self.reference),
              patch.dict(rdf.MLIP_TRAJ_DIRS, {ACCELERATED: self.data / ACCELERATED}),
              patch.object(rdf, "load_reference_trajectory", return_value=SimpleNamespace(n_frames=4)),
              patch.object(rdf, "load_mlip_trajectory", return_value=object()) as load,
              patch.object(rdf, "compute_rdf", return_value=curve)):
            rdf.process_source(args, ACCELERATED)
        self.assertIn(self.data / ASE_DIRECTORY / METAL / f"nvt_{CHGNET}.h5",
                      {call.args[0] for call in load.call_args_list})
        scores = pd.read_csv(output / ACCELERATED / "rdf_similarity_scores_same_simulation_length.csv")
        row = scores[scores.Calculator == CHGNET].iloc[0]
        self.assertEqual(row["Mean RDF Error [%]"], 50.)
        self.assertEqual(row.trajectory_source, ASE_EAGER)
        vdos = load_module("vdos/get_normalized_VDOS.py")
        output = self.root / "vdos-output"
        args = SimpleNamespace(results_dir=output, systems=None, models=["chgnet"],
                               excluded_system_types=["molecular crystals"], dry_run=False,
                               numerical_mlip_velocities=False, pad_factor=1, chunk_size=4,
                               overwrite=False, e_min=None, e_max=None)
        with (contextlib.redirect_stdout(io.StringIO()),
              patch.object(vdos, "load_reference_positions", return_value=np.zeros((8, 2, 3))),
              patch.object(vdos, "spectrum_for_samples", return_value=curve)):
            vdos.process_source(args, ACCELERATED, data_dir=self.data,
                                reference_dir=self.reference, metadata=self.metadata)
        rows = pd.read_csv(output / ACCELERATED / vdos.PAIR_OUTPUT)
        chgnet = rows[rows.model == CHGNET]
        self.assertEqual(set(chgnet.trajectory_source), {ASE_EAGER})
        self.assertEqual(chgnet.set_index("system").vdos_error_percent.to_dict(), {METAL: 0., HYDROGEN: 100.})
        for script in ("rdfs/fig_3.py", "vdos/figure_5.py", "vdos/figure_SI_7_8_9.py"):
            module = load_module(script)
            self.assertEqual(module.normalize_model_name(CHGNET), "chgnet")


if __name__ == "__main__":
    unittest.main()
