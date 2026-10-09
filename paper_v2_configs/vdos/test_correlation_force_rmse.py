"""Check correlation inputs, hydrogen exclusion, and Matbench score matching."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

import figure_6
import figure_SI_7_8_9 as correlations
import plot_all
from matbench_scores import matbench_model_key


class MatbenchScoreMatchingTest(unittest.TestCase):
    def test_conservative_orb_aliases_remain_distinct_from_direct_orb(self):
        for alias in (
            "orb-v3-cons", "orb-v3-conservative-inf-omat", "orb-v3-omat",
            "orb-v3-omat-force-only-eager", "ORB-V3-CONS-stress",
        ):
            with self.subTest(alias=alias):
                self.assertEqual(matbench_model_key(alias), "orb-v3-omat")
        for alias in ("orb-v3-direct-omat-force-only", "orb-v3-direct-20-omat"):
            with self.subTest(alias=alias):
                self.assertEqual(matbench_model_key(alias), "orb-v3-direct-omat")

    def test_execution_variants_match_their_base_model_scores(self):
        for model, expected in (
            ("mace-mp-0-compile", "mace-mp-0"),
            ("mace-mpa-0-compile", "mace-mpa-0"),
            ("mace-mh-omat-compile", "mace-mh-omat"),
            ("MatterSim-v1-5M-compile-force-only", "mattersim-v1-5m"),
            ("grace-oam-force-only-compiled", "grace-oam"),
            ("pet-oam-xl-force-only-torchscript", "pet-oam-xl"),
            ("pet-omat-xl-stress-torchscript", "pet-omat-xl"),
            ("uma-s-omat-compile-force-only", "uma-s-omat"),
            ("uma-s-omat-turbo-force-only", "uma-s-omat"),
            ("uma-m-omat-compile-force-only", "uma-m-omat"),
            ("uma-m-omat-turbo-force-only", "uma-m-omat"),
            ("NequIP-OAM-L-force-only", "nequip"),
            ("eSEN-30M-OAM-force-only", "esen-30m-oam"),
        ):
            with self.subTest(model=model):
                self.assertEqual(matbench_model_key(model), expected)
        self.assertEqual(matbench_model_key("unknown-compile"), "unknown-compile")
        # Benchmark identity must not merge separate MD results or plot points.
        self.assertEqual(correlations.normalize_model_name("mace-mp-0-compile"), "mace-mp-0-compile")

    def test_aliases_work_for_each_score_column_and_preserve_missing_values(self):
        for column, value in (("f1_score", 0.905), ("ksrme_score", 0.2102),
                              ("cps_score", 0.86046)):
            with self.subTest(column=column):
                scores = correlations.standardize_score(
                    pd.DataFrame({"calculator": ["orb-v3-cons", "uma-s-omat"],
                                  column: [value, None]}),
                    value_candidates=[column], output_column=column, description=column,
                ).set_index("calculator")
                self.assertEqual(scores.loc["orb-v3-omat", column], value)
                self.assertTrue(pd.isna(scores.loc["uma-s-omat", column]))
                self.assertNotIn("orb-v3-direct-omat", scores.index)

    def test_conflicting_alias_scores_raise_instead_of_being_averaged(self):
        frame = pd.DataFrame({"calculator": ["orb-v3-cons", "orb-v3-omat"],
                              "f1_score": [0.905, 0.5]})
        with self.assertRaisesRegex(ValueError, "Conflicting.*orb-v3-omat"):
            correlations.standardize_score(
                frame, value_candidates=["f1_score"], output_column="f1_score", description="F1",
            )
        frame["f1_score"] = 0.905
        result = correlations.standardize_score(
            frame, value_candidates=["f1_score"], output_column="f1_score", description="F1",
        )
        self.assertEqual(result.to_dict("records"), [{"calculator": "orb-v3-omat", "f1_score": 0.905}])

    def test_join_preserves_models_and_md_metrics_with_shared_base_scores(self):
        models = ["orb-v3-omat", "orb-v3-direct-omat", "mace-mp-0",
                  "mace-mp-0-compile", "pet-oam-xl-torchscript", "uma-s-omat-turbo"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            force = pd.DataFrame({"calculator": models, "force_rmse": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
            force.to_csv(root / "force.csv", index=False)
            for filename, column in (("rdf.csv", "rdf_error_percent"),
                                      ("pressure.csv", "pressure_error_percent"),
                                      ("vdos.csv", "vdos_error_percent")):
                pd.DataFrame({"calculator": models, column: [10, 20, 30, 40, 50, 60]}).to_csv(
                    root / filename, index=False,
                )
            for filename, column, values in (
                ("f1.csv", "f1_score", [0.905, 0.669, 0.924, None]),
                ("ksrme.csv", "ksrme_score", [0.2102, 0.682, 0.119, None]),
            ):
                pd.DataFrame({"calculator": ["orb-v3-cons", "mace-mp-0", "pet-oam-xl", "uma-s-omat"],
                              column: values}).to_csv(root / filename, index=False)
            args = SimpleNamespace(
                source="mlip-trajs-ase", force_rmse_input=root / "force.csv",
                rdf_file=root / "rdf.csv", pressure_file=root / "pressure.csv",
                pressure_backend=None, pressure_mode=None, vdos_model_means_file=root / "vdos.csv",
                f1_file=root / "f1.csv", ksrme_file=root / "ksrme.csv",
                exclude_hydrogen_force_rmse=False,
            )
            result = correlations.load_joined_data(args).set_index("calculator")
            self.assertEqual(set(result.index), set(models))
            for model, original_force in zip(models, force.force_rmse):
                self.assertEqual(result.loc[model, "force_rmse"], original_force)
            self.assertEqual(result.loc["orb-v3-omat", "f1_score"], 0.905)
            self.assertEqual(result.loc["orb-v3-omat", "ksrme_score"], 0.2102)
            self.assertEqual(result.loc["mace-mp-0-compile", "f1_score"], 0.669)
            self.assertEqual(result.loc["pet-oam-xl-torchscript", "ksrme_score"], 0.119)
            self.assertTrue(pd.isna(result.loc["orb-v3-direct-omat", "f1_score"]))
            self.assertTrue(pd.isna(result.loc["uma-s-omat-turbo", "f1_score"]))
            self.assertEqual(result.loc["mace-mp-0-compile", "rdf_error_percent"], 40)
            self.assertEqual(result.loc["mace-mp-0-compile", "pressure_error_percent"], 40)
            self.assertEqual(result.loc["mace-mp-0-compile", "vdos_error_percent"], 40)


class HydrogenForceRmseTest(unittest.TestCase):
    def test_hydrogen_identifiers_do_not_match_other_elements(self):
        for value in (
            "H", "hydrogen", "H_1050K", "H_1050K_Rupp_QE",
            "/old-machine/H_1050K_Rupp_QE/traj.extxyz",
        ):
            with self.subTest(value=value):
                self.assertTrue(correlations.is_hydrogen_system(value))
        for value in ("Hf", "He", "Pt111w24H2O_380K", "bulkCu", None):
            with self.subTest(value=value):
                self.assertFalse(correlations.is_hydrogen_system(value))

    def test_average_is_per_system_and_keeps_hydrogen_only_models(self):
        frame = pd.DataFrame({
            "calculator": ["nequip-oam-l-force-only"] * 5 + ["only-hydrogen"],
            "system": ["H", "bulkCu", "bulkAg", "TiSe2", "picene", "H"],
            "system_type": ["hydrogen", "pure metals", "pure metals",
                            "metal dichalcogenides", "molecular crystals", "hydrogen"],
            "force_rmse": [1000.0, 1.0, 3.0, 8.0, 999.0, 500.0],
        })
        result = correlations.force_rmse_without_hydrogen(frame).set_index("calculator")
        self.assertEqual(result.loc["nequip", "force_rmse"], 4.0)
        self.assertTrue(pd.isna(result.loc["only-hydrogen", "force_rmse"]))

    def test_directory_input_keeps_system_identifiers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            pd.DataFrame({
                "reference_key": ["H_1050K", "bulkCu_1000K", "bulkAg_600K", "picene_295K"],
                "force_rmse": [1000.0, 1.0, 3.0, 2000.0],
            }).to_csv(path / "rmse-results-all_nequip-oam-l-force-only.csv", index=False)
            result = correlations.load_force_rmse(path, exclude_hydrogen=True)
            self.assertEqual(result.to_dict("records"), [{"calculator": "nequip", "force_rmse": 2.0}])

    def test_source_selection_happens_before_force_averaging(self):
        frame = pd.DataFrame({
            "calculator": ["mace-mp-0", "mace-mp-0", "other-source-only"],
            "system": ["bulkCu"] * 3,
            "source": ["mlip-trajs-ase", "mlip-trajs-ase-accelerated", "mlip-trajs-ase-accelerated"],
            "force_rmse": [1.0, 3.0, 5.0],
        })
        result = correlations.force_rmse_without_hydrogen(frame, "mlip-trajs-ase")
        self.assertEqual(result.to_dict("records"), [{"calculator": "mace-mp-0", "force_rmse": 1.0}])

    def test_hydrogen_only_torchsim_model_has_no_force_value(self):
        frame = pd.DataFrame({
            "calculator": ["mace-mp-0"], "system": ["H"], "force_rmse": [100.0],
            "trajectory": ["/old-machine/H_1050K_Rupp_QE/traj.extxyz"],
        })
        result = correlations.force_rmse_without_hydrogen(frame, "mlip-trajs-torchsim-eager")
        self.assertEqual(result.calculator.tolist(), ["mace-mp-0"])
        self.assertTrue(pd.isna(result.loc[0, "force_rmse"]))

    def test_summary_input_remains_supported_but_cannot_exclude_hydrogen(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "means.csv"
            pd.DataFrame({"calculator": ["mace-mp-0"], "force_rmse": [2.0]}).to_csv(path, index=False)
            self.assertEqual(correlations.load_force_rmse(path).iloc[0].force_rmse, 2.0)
            with self.assertRaisesRegex(ValueError, "per-system.*identifiers"):
                correlations.load_force_rmse(path, exclude_hydrogen=True)

    def test_failed_and_audited_md_are_excluded_before_averaging(self):
        source = "mlip-trajs-torchsim-eager"
        model = "eSEN-30M-OAM-force-only"
        systems = ["bulkCu_1000K_Kapil", "bulkAg_600K_Kapil",
                   "Pt111w24H2O_380K_Heenen_VASP", "H_1050K_Rupp_QE"]
        with tempfile.TemporaryDirectory() as folder:
            config_dir = Path(folder)
            for system in systems:
                directory = config_dir / "data" / source / system
                directory.mkdir(parents=True)
                (directory / f"nvt_{model}.h5").touch()
                (directory / f"md_timing_{model}.csv").write_text(
                    "" if system == systems[1] else
                    f"calculator,system,n_steps\n{model},{system},100\n"
                )
            frame = pd.DataFrame({
                "calculator": [model] * 4,
                "system": systems,
                "trajectory": [f"/old-machine/{system}/traj.extxyz" for system in systems],
                "force_rmse": [1.0, 100.0, 200.0, 300.0],
            })
            with patch.object(correlations, "CONFIG_DIR", config_dir):
                result = correlations.force_rmse_without_hydrogen(frame, source)
            self.assertEqual(result.to_dict("records"), [{"calculator": "esen-30m-oam", "force_rmse": 1.0}])

    def test_joined_metrics_and_other_score_columns_are_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pd.DataFrame({"calculator": ["mace-mp-0"], "force_rmse": [51.0]}).to_csv(
                root / "mean_metrics_by_model.csv", index=False,
            )
            pd.DataFrame({
                "calculator": ["mace-mp-0"] * 2,
                "system": ["H", "bulkCu"], "force_rmse": [100.0, 2.0],
            }).to_csv(root / "rmse_per_system.csv", index=False)
            for filename, column, value in (
                ("rdf.csv", "rdf_error_percent", 25.0),
                ("pressure.csv", "pressure_error_percent", 30.0),
                ("vdos.csv", "vdos_error_percent", 40.0),
                ("f1.csv", "f1_score", 0.9),
                ("ksrme.csv", "ksrme_score", 0.1),
            ):
                pd.DataFrame({"calculator": ["mace-mp-0"], column: [value]}).to_csv(
                    root / filename, index=False,
                )
            args = SimpleNamespace(
                source="mlip-trajs-ase", force_rmse_input=None, exclude_hydrogen_force_rmse=False,
                rdf_file=root / "rdf.csv", pressure_file=root / "pressure.csv",
                pressure_backend=None, pressure_mode=None,
                vdos_model_means_file=root / "vdos.csv", f1_file=root / "f1.csv",
                ksrme_file=root / "ksrme.csv",
            )
            with patch.object(correlations, "source_paths", return_value=(
                root / "mean_metrics_by_model.csv", root / "rdf.csv", root / "vdos.csv",
            )):
                original = correlations.load_joined_data(args)
                args.exclude_hydrogen_force_rmse = True
                variant = correlations.load_joined_data(args)
            self.assertEqual(original.loc[0, "force_rmse"], 51.0)
            self.assertEqual(variant.loc[0, "force_rmse"], 2.0)
            pd.testing.assert_frame_equal(
                original.drop(columns="force_rmse"), variant.drop(columns="force_rmse"),
            )

    def test_defaults_keep_original_and_extra_outputs_separate(self):
        for module, output_arguments in (
            (figure_6, ("output_file", "si_output_file")),
            (correlations, ("rdf_output_file", "pressure_output_file", "vdos_output_file")),
        ):
            with self.subTest(module=module.__name__):
                with patch("sys.argv", [module.__file__]):
                    original = module.parse_args()
                with patch("sys.argv", [module.__file__, "--exclude-hydrogen-force-rmse"]):
                    variant = module.parse_args()
                for argument in output_arguments:
                    original_path = Path(getattr(original, argument))
                    variant_path = Path(getattr(variant, argument))
                    self.assertEqual(variant_path.stem,
                                     original_path.stem + correlations.NO_HYDROGEN_FORCE_SUFFIX)
        with patch("sys.argv", [figure_6.__file__, "--exclude-hydrogen-force-rmse",
                                "--output-file", "custom.pdf", "--si-output-file", ""]):
            args = figure_6.parse_args()
        self.assertEqual(args.output_file, Path("custom.pdf"))
        self.assertEqual(args.si_output_file, "")

    def test_pipeline_variant_preserves_input_paths(self):
        command = ["python", "-u", "figure_6.py", "--source", "mlip-trajs-torchsim-eager",
                   "--rdf-file", "rdf.csv", "--output-file", "plots/figure_6.pdf",
                   "--si-output-file", "plots/figure_6_all_labels.pdf"]
        variant = plot_all.without_hydrogen_force_rmse(command)
        self.assertEqual(variant[:7], command[:7])
        self.assertEqual(command[8], "plots/figure_6.pdf")
        self.assertEqual(variant[8], "plots/figure_6_no_hydrogen_force_rmse.pdf")
        self.assertEqual(variant[10], "plots/figure_6_all_labels_no_hydrogen_force_rmse.pdf")
        self.assertEqual(variant[-1], "--exclude-hydrogen-force-rmse")


if __name__ == "__main__":
    unittest.main()
