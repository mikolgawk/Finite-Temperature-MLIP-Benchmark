"""Data-only checks for stable MD selection shared by Figure 6 and SI correlations."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

import figure_6
import figure_SI_7_8_9 as correlations
import md_success
from _stable_md_inputs import stable_md_run
from matbench_scores import matbench_model_key


EAGER = "mlip-trajs-torchsim-eager"
ACCELERATED = "mlip-trajs-torchsim-accelerated"
ASE = "mlip-trajs-ase-eager"
CU = "bulkCu_1000K_Kapil"
AG = "bulkAg_600K_Kapil"
AU = "bulkAu_900K_Kapil"
H = "H_1050K_Rupp_QE"
PT_WATER = "Pt111w24H2O_380K_Heenen_VASP"
CRYSTAL = "picene_295K_Kapil"
TISE = "TiSe2_300K_Kapil"
MD_COLUMNS = ("force_rmse", "rdf_error_percent", "pressure_error_percent", "vdos_error_percent")


class StableFigure6Test(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.data = self.root / "data"
        self.data.mkdir()
        failures = patch.object(md_success, "load_failed_md_runs", return_value={})
        self.addCleanup(failures.stop)
        self.failure_registry = failures.start()

    def run_output(self, source, model, system, *, timing="complete", calculator=None):
        directory = self.data / source / system
        directory.mkdir(parents=True, exist_ok=True)
        trajectory = directory / f"nvt_{model}.h5"
        trajectory.touch()
        path = directory / f"md_timing_{model}.csv"
        if timing == "complete":
            path.write_text(
                f"calculator,system,n_steps\n{calculator or model},{system},100\n"
            )
        elif timing == "empty":
            path.write_text("")
        elif timing != "missing":
            path.write_text(timing)
        return trajectory

    def inputs(self, rows, *, source=EAGER):
        """Rows contain model, full system ID, force RMSE, RDF, pressure, VDOS."""
        frame = pd.DataFrame(rows, columns=("calculator", "system", *MD_COLUMNS))
        frame["source"] = source
        frame["trajectory"] = frame.system.map(lambda system: f"/old-machine/{system}/traj.extxyz")
        for filename, column in (("force.csv", "force_rmse"), ("rdf.csv", "rdf_error_percent"),
                                 ("pressure.csv", "pressure_error_percent"), ("vdos.csv", "vdos_error_percent")):
            frame[["calculator", "system", "source", "trajectory", column]].to_csv(
                self.root / filename, index=False,
            )
        models = sorted({matbench_model_key(model) for model in frame.calculator})
        for filename, column in (("f1.csv", "f1_score"), ("ksrme.csv", "ksrme_score")):
            pd.DataFrame({"calculator": models, column: [0.8] * len(models)}).to_csv(
                self.root / filename, index=False,
            )
        return SimpleNamespace(
            source=source, md_data_dir=self.data,
            force_rmse_input=self.root / "force.csv", rdf_file=self.root / "rdf.csv",
            pressure_file=self.root / "pressure.csv", pressure_backend=None, pressure_mode=None,
            vdos_model_means_file=self.root / "vdos.csv", f1_file=self.root / "f1.csv",
            ksrme_file=self.root / "ksrme.csv", exclude_hydrogen_force_rmse=False,
            exclude_pt_water=False,
        )

    def test_model_specific_failures_are_removed_before_equal_system_averages(self):
        args = self.inputs([
            ("mace-mp-0", CU, 1, 20, 10, 0),
            ("mace-mp-0", CU, 3, 60, 30, 20),
            ("mace-mp-0", AG, 4, 100, 100, 100),
            ("mace-mp-0", AU, 900, 100, 100, 100),
            ("orb-v2", CU, 900, 100, 100, 100),
            ("orb-v2", AG, 900, 100, 100, 100),
            ("orb-v2", AU, 11, 8, 9, 10),
        ])
        for system in (CU, AG, AU):
            self.run_output(EAGER, "mace-mp-0", system)
        self.run_output(EAGER, "orb-v2", CU, timing="empty")
        self.run_output(EAGER, "orb-v2", AG, timing="missing")
        self.run_output(EAGER, "orb-v2", AU)
        self.failure_registry.return_value = {(EAGER, AU, "mace-mp-0"): "unphysical temperature"}
        original = {path: path.read_bytes() for path in self.root.rglob("*.csv")}

        result = figure_6.load_joined_data(args).set_index("calculator")
        pd.testing.assert_frame_equal(result, correlations.load_joined_data(args).set_index("calculator"))

        self.assertEqual(result.loc["mace-mp-0", list(MD_COLUMNS)].tolist(), [3.0, 70.0, 60.0, 55.0])
        self.assertEqual(result.loc["orb-v2", list(MD_COLUMNS)].tolist(), [11.0, 8.0, 9.0, 10.0])
        # A completed stable system's genuine 100% disagreement contributes fully.
        self.assertEqual(original, {path: path.read_bytes() for path in self.root.rglob("*.csv")})

    def test_accelerated_cohort_uses_eager_and_ase_physical_run_status(self):
        esen = "eSEN-30M-OAM-force-only"
        eq = "eq-v2-m-omat"
        args = self.inputs([
            (esen, CU, 1, 10, 20, 30), (esen, AG, 900, 100, 100, 100),
            (eq, CU, 900, 100, 100, 100), (eq, AG, 2, 11, 21, 31),
            ("chgnet", CU, 3, 12, 22, 32), ("chgnet", AG, 900, 100, 100, 100),
        ], source=ACCELERATED)
        self.run_output(EAGER, esen, CU)
        self.run_output(EAGER, esen, AG)
        self.run_output(EAGER, eq, CU, timing="empty")
        self.run_output(EAGER, eq, AG)
        self.run_output(ASE, "chgnet-ase", CU, calculator="chgnet")
        self.run_output(ASE, "chgnet-ase", AG, timing="empty")
        for model in (esen, eq, "chgnet"):
            for system in (CU, AG):
                self.run_output(ACCELERATED, model, system)
        self.failure_registry.return_value = {(EAGER, AG, esen): "audited nonfinite positions"}

        result = figure_6.load_joined_data(args).set_index("calculator")
        pd.testing.assert_frame_equal(result, correlations.load_joined_data(args).set_index("calculator"))

        for model, expected in (("esen-30m-oam", [1, 10, 20, 30]),
                                (eq, [2, 11, 21, 31]), ("chgnet", [3, 12, 22, 32])):
            self.assertEqual(result.loc[model, list(MD_COLUMNS)].tolist(), expected)

    def test_completed_marker_validation_rejects_partial_or_mismatched_runs(self):
        for timing in ("missing", "empty", "calculator,system,n_steps\nwrong,bulkCu_1000K_Kapil,100\n",
                       "calculator,system,n_steps\nmace-mp-0,wrong-system,100\n",
                       "calculator,system,n_steps\nmace-mp-0,bulkCu_1000K_Kapil,nan\n",
                       "calculator,system,n_steps\nmace-mp-0,bulkCu_1000K_Kapil,0\n",
                       "calculator,system,n_steps\nmace-mp-0,bulkCu_1000K_Kapil,100\n"
                       "mace-mp-0,bulkCu_1000K_Kapil,200\n"):
            with self.subTest(timing=timing):
                directory = self.data / EAGER / CU
                if directory.exists():
                    for path in directory.iterdir():
                        path.unlink()
                trajectory = self.run_output(EAGER, "mace-mp-0", CU, timing=timing)
                self.assertFalse(stable_md_run(trajectory))
        trajectory = self.run_output(EAGER, "mace-mp-0", CU)
        trajectory.unlink()
        self.assertFalse(stable_md_run(trajectory))
        ase = self.run_output(ASE, "chgnet-ase", CU, calculator="chgnet")
        self.assertTrue(stable_md_run(ase))
        self.failure_registry.return_value = {(ASE, CU, "chgnet-ase"): "temperature instability"}
        self.assertFalse(stable_md_run(ase))

    def test_hydrogen_and_interface_exclusions_compose_with_stability_and_crystal_filter(self):
        args = self.inputs([
            ("mace-mp-0", CU, 1, 10, 10, 20), ("mace-mp-0", H, 100, 90, 90, 80),
            ("mace-mp-0", PT_WATER, 200, 80, 80, 60),
            ("mace-mp-0", TISE, 3, 30, 30, 40), ("mace-mp-0", CRYSTAL, 999, 100, 100, 100),
        ])
        for system in (CU, H, PT_WATER, TISE, CRYSTAL):
            self.run_output(EAGER, "mace-mp-0", system)
        original = figure_6.load_joined_data(args)
        pd.testing.assert_frame_equal(original, correlations.load_joined_data(args))
        self.assertEqual(original.loc[0, "force_rmse"], 76.0)
        self.assertAlmostEqual(original.loc[0, "pressure_error_percent"], 130 / 3)
        args.exclude_hydrogen_force_rmse = True
        args.exclude_pt_water = True
        result = figure_6.load_joined_data(args)
        pd.testing.assert_frame_equal(result, correlations.load_joined_data(args))
        self.assertEqual(result.loc[0, list(MD_COLUMNS)].tolist(), [2.0, 20.0, 20.0, 30.0])
        pd.testing.assert_frame_equal(original[["f1_score", "ksrme_score"]], result[["f1_score", "ksrme_score"]])

    def test_models_with_no_stable_md_keep_nan_metrics_and_independent_scores(self):
        args = self.inputs([
            ("mace-mp-0", CU, 1, 10, 20, 30), ("orb-v2", AG, 900, 100, 100, 100),
        ])
        self.run_output(EAGER, "mace-mp-0", CU)
        self.run_output(EAGER, "orb-v2", AG)
        self.failure_registry.return_value = {(EAGER, AG, "orb-v2"): "audited unstable completed run"}
        result = figure_6.load_joined_data(args).set_index("calculator")
        self.assertTrue(result.loc["orb-v2", list(MD_COLUMNS)].isna().all())
        self.assertEqual(result.loc["orb-v2", "f1_score"], 0.8)
        self.assertEqual(result.loc["orb-v2", "ksrme_score"], 0.8)
        self.assertEqual(result.loc["mace-mp-0", list(MD_COLUMNS)].tolist(), [1, 10, 20, 30])

    def test_stable_mode_rejects_summary_only_metric_overrides(self):
        args = self.inputs([("mace-mp-0", CU, 1, 10, 20, 30)])
        self.run_output(EAGER, "mace-mp-0", CU)
        for attr, column in (("force_rmse_input", "force_rmse"), ("rdf_file", "rdf_error_percent"),
                             ("pressure_file", "pressure_error_percent"),
                             ("vdos_model_means_file", "vdos_error_percent")):
            path = getattr(args, attr)
            original = path.read_bytes()
            pd.DataFrame({"calculator": ["mace-mp-0"], column: [77]}).to_csv(path, index=False)
            with self.subTest(metric=column), self.assertRaisesRegex((ValueError, FileNotFoundError), "per-system"):
                figure_6.load_joined_data(args)
            path.write_bytes(original)

    def test_explicit_legacy_summary_loader_keeps_benchmark_failure_penalties(self):
        args = self.inputs([
            ("mace-mp-0", CU, 1, 10, 20, 30), ("mace-mp-0", AG, 9, 100, 100, 100),
        ])
        self.run_output(EAGER, "mace-mp-0", CU)
        self.run_output(EAGER, "mace-mp-0", AG, timing="empty")
        for attr, column, mean in (("force_rmse_input", "force_rmse", 5), ("rdf_file", "rdf_error_percent", 55),
                                   ("pressure_file", "pressure_error_percent", 60),
                                   ("vdos_model_means_file", "vdos_error_percent", 65)):
            pd.DataFrame({"calculator": ["mace-mp-0"], column: [mean]}).to_csv(getattr(args, attr), index=False)
        with patch.object(correlations, "StableMDFilter", side_effect=AssertionError("legacy mode used stable filter")):
            legacy = correlations.load_joined_data(args, stable_md_only=False)
        self.assertEqual(legacy.loc[0, list(MD_COLUMNS)].tolist(), [5, 55, 60, 65])

    def test_both_plot_families_agree_for_each_source_and_manual_exclusion(self):
        for source in (EAGER, ACCELERATED):
            args = self.inputs([
                ("mace-mp-0", CU, 1, 10, 20, 30),
                ("mace-mp-0", H, 2, 40, 50, 60),
                ("mace-mp-0", PT_WATER, 3, 70, 80, 90),
                ("mace-mp-0", AG, 4, 100, 100, 100),
                ("mace-mp-0", AU, 900, 100, 100, 100),
            ], source=source)
            for system in (CU, H, PT_WATER, AG):
                self.run_output(source, "mace-mp-0", system)
            self.run_output(source, "mace-mp-0", AU, timing="empty")
            for exclude_hydrogen in (False, True):
                for exclude_pt_water in (False, True):
                    with self.subTest(source=source, exclude_hydrogen=exclude_hydrogen,
                                      exclude_pt_water=exclude_pt_water):
                        args.exclude_hydrogen_force_rmse = exclude_hydrogen
                        args.exclude_pt_water = exclude_pt_water
                        main = figure_6.load_joined_data(args)
                        si = correlations.load_joined_data(args)
                        pd.testing.assert_frame_equal(main, si)
                        # Genuine 100% metric disagreement on stable Ag remains.
                        if exclude_hydrogen and exclude_pt_water:
                            self.assertEqual(si.loc[0, list(MD_COLUMNS)].tolist(), [2.5, 55.0, 60.0, 65.0])

    def test_si_data_only_cli_uses_selected_md_data_and_does_not_render(self):
        args = self.inputs([
            ("mace-mp-0", CU, 1, 10, 20, 30),
            ("mace-mp-0", AG, 900, 100, 100, 100),
        ])
        self.run_output(EAGER, "mace-mp-0", CU)
        self.run_output(EAGER, "mace-mp-0", AG, timing="empty")
        options = ["--data-only", "--source", EAGER, "--md-data-dir", str(self.data)]
        for argument in ("force_rmse_input", "rdf_file", "pressure_file", "vdos_model_means_file",
                         "f1_file", "ksrme_file"):
            options.extend(["--" + argument.replace("_", "-"), str(getattr(args, argument))])
        output = io.StringIO()
        with patch("sys.argv", [correlations.__file__, *options]), \
                patch.object(correlations, "StableMDFilter", wraps=correlations.StableMDFilter) as stable_filter, \
                patch.object(correlations, "plot_single_error_figure") as render, \
                redirect_stdout(output):
            correlations.main()
        stable_filter.assert_called_once_with(self.data, EAGER)
        render.assert_not_called()
        self.assertIn("mace-mp-0", output.getvalue())
        self.assertNotIn("900", output.getvalue())
        self.assertFalse(list(self.root.rglob("*.pdf")))

    def test_default_summary_paths_use_detailed_caches_and_full_temperature_ids(self):
        cooler_cu = "bulkCu_700K_Kapil"
        args = self.inputs([
            ("mace-mp-0", CU, 1, 10, 20, 30), ("mace-mp-0", cooler_cu, 9, 100, 100, 100),
        ])
        self.run_output(EAGER, "mace-mp-0", CU)
        self.run_output(EAGER, "mace-mp-0", cooler_cu, timing="empty")
        args.force_rmse_input.rename(self.root / "rmse_per_system.csv")
        pd.DataFrame({"System": ["bulkCu", "bulkCu"], "temperature_K": [1000, 700],
                      "RDF_Error": [10, 100]}).to_csv(
            self.root / f"{correlations.RDF_SYSTEM_FILE_PREFIX}mace-mp-0.csv", index=False,
        )
        args.pressure_file.rename(self.root / correlations.SYSTEM_METRIC_FILES["pressure"])
        args.vdos_model_means_file.rename(self.root / correlations.SYSTEM_METRIC_FILES["VDOS"])
        for attr, column in (("force_rmse_input", "force_rmse"), ("rdf_file", "rdf_error_percent"),
                             ("pressure_file", "pressure_error_percent"),
                             ("vdos_model_means_file", "vdos_error_percent")):
            pd.DataFrame({"calculator": ["mace-mp-0"], column: [77]}).to_csv(getattr(args, attr), index=False)
        args.force_rmse_input = None
        with patch.object(correlations, "source_paths", return_value=(
            self.root / "force.csv", self.root / "rdf.csv", self.root / "vdos.csv",
        )):
            result = figure_6.load_joined_data(args)
        self.assertEqual(result.loc[0, list(MD_COLUMNS)].tolist(), [1, 10, 20, 30])


if __name__ == "__main__":
    unittest.main()
