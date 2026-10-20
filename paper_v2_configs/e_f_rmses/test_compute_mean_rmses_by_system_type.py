import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

from compute_mean_rmses_by_system_type import list_rmse_csv_files, load_all_data


class EnergyForceAggregationTests(unittest.TestCase):
    def test_load_all_data_filters_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "e-f-predictions-ase"
            data_dir.mkdir()
            for model in ("model-a", "model-b"):
                pd.DataFrame(
                    {
                        "system": ["bulkCu_300K_test"],
                        "trajectory": ["/ref/bulkCu_300K_test/traj.extxyz"],
                        "energy_rmse": [1.0],
                        "force_rmse": [2.0],
                    }
                ).to_csv(data_dir / f"rmse-results-all_{model}.csv", index=False)

            with patch("compute_mean_rmses_by_system_type.torchsim_md_succeeded", return_value=True):
                result = load_all_data(data_dir, {"model-b"})

            self.assertEqual(set(result["calculator"]), {"model-b"})

    def test_torchsim_rows_require_completed_matching_md(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            predictions = root / "predictions" / "md_eager"
            predictions.mkdir(parents=True)
            model = "grace-oam-force-only-eager"
            systems = ("bulkCu_300K_test", "H_1050K_test")
            pd.DataFrame({
                "system": ["bulkCu", "H"],
                "trajectory": [str(root / "ref-trajs" / name / "traj.extxyz") for name in systems],
                "energy_rmse": [1.0, 2.0],
                "force_rmse": [3.0, 4.0],
            }).to_csv(predictions / f"rmse-results-all_{model}.csv", index=False)
            md_root = root / "md-data" / "mlip-trajs-torchsim-eager"
            for system in systems:
                directory = md_root / system
                directory.mkdir(parents=True)
                (directory / f"nvt_{model}.h5").touch()
            (md_root / systems[0] / f"md_timing_{model}.csv").write_text(
                f"calculator,system,n_steps\n{model},{systems[0]},100\n"
            )
            (md_root / systems[1] / f"md_timing_{model}.csv").touch()

            result = load_all_data(root / "predictions", md_data_dir=root / "md-data")

            self.assertEqual(result["system"].tolist(), ["bulkCu"])

    def test_ase_rows_exclude_incomplete_missing_and_audited_md(self) -> None:
        cases = (
            ("mlip-trajs-ase", "mlip-trajs-ase", "md_eager"),
            ("mlip-trajs-ase", "mlip-trajs-ase-eager", "md_eager"),
            ("mlip-trajs-ase", "ase/md_eager", "md_eager"),
            ("mlip-trajs-ase-accelerated", "mlip-trajs-ase-accelerated", "md_accelerated"),
        )
        for source, physical_directory, mode in cases:
            with self.subTest(source=source, physical_directory=physical_directory), \
                    tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                predictions = root / "e-f-predictions-ase" / mode
                predictions.mkdir(parents=True)
                model = "nequip-oam-l-force-only"
                raw_model = model + "-ase"
                systems = ("bulkCu_300K_test", "H_1050K_test", "bulkAu_1500K_test", "TiSe2_400K_test")
                pd.DataFrame({
                    "system": ["bulkCu", "H", "bulkAu", "TiSe2"],
                    "trajectory": [f"/ref/{system}/traj.extxyz" for system in systems],
                    "energy_rmse": [1.0, 1e6, 1e6, 1e6],
                    "force_rmse": [2.0, 1e6, 1e6, 1e6],
                }).to_csv(predictions / f"rmse-results-all_{model}.csv", index=False)
                for system in systems:
                    directory = root / "md-data" / physical_directory / system
                    directory.mkdir(parents=True)
                    if system != systems[3]:
                        (directory / f"nvt_{raw_model}.h5").touch()
                    (directory / f"md_timing_{raw_model}.csv").write_text(
                        "" if system == systems[1] else
                        f"calculator,system,n_steps\n{model},{system},100\n"
                    )
                failures = {(source, systems[2], raw_model): "Audited unphysical dynamics"}
                with patch("md_success.load_failed_md_runs", return_value=failures):
                    result = load_all_data(predictions.parent, sources={source},
                                           md_data_dir=root / "md-data")
                self.assertEqual(result["system"].tolist(), ["bulkCu"])
                self.assertEqual(result["source"].tolist(), [source])
                self.assertEqual(result["energy_rmse"].mean(), 1.0)
                self.assertEqual(result["force_rmse"].mean(), 2.0)

    def test_ase_rows_require_reference_trajectory_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            predictions = Path(temporary_directory) / "e-f-predictions-ase" / "md_accelerated"
            predictions.mkdir(parents=True)
            pd.DataFrame({"system": ["bulkCu"], "energy_rmse": [1.0], "force_rmse": [2.0]}).to_csv(
                predictions / "rmse-results-all_model.csv", index=False,
            )
            with self.assertRaisesRegex(ValueError, "Missing trajectory column"):
                load_all_data(predictions)

    def test_legacy_discovery_prefers_current_results_per_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            predictions = root / "e-f-predictions"
            paths = [
                root / "md" / "rmse-results-all_model-a.csv",
                root / "md" / "rmse-results-all_model-b.csv",
                root / "md-accelerated" / "rmse-results-all_model-a.csv",
                predictions / "md_eager" / "rmse-results-all_model-a.csv",
                root / "e-f-predictions-ase" / "rmse-results-all_model-c.csv",
            ]
            for csv_file in paths:
                csv_file.parent.mkdir(parents=True, exist_ok=True)
                csv_file.touch()
            self.assertEqual(set(list_rmse_csv_files(predictions)), set(paths[1:]))

    def test_legacy_results_apply_md_completion_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy = root / "md"
            legacy.mkdir()
            pd.DataFrame({
                "system": ["bulkCu", "H"],
                "trajectory": ["/ref/bulkCu_300K_test/traj.extxyz", "/ref/H_1050K_test/traj.extxyz"],
                "energy_rmse": [1.0, 2.0],
                "force_rmse": [3.0, 4.0],
            }).to_csv(legacy / "rmse-results-all_model-a.csv", index=False)
            with patch("compute_mean_rmses_by_system_type.torchsim_md_succeeded",
                       side_effect=lambda trajectory: trajectory.parent.name == "bulkCu_300K_test") as completed:
                result = load_all_data(root / "e-f-predictions", md_data_dir=root / "md-data")
            self.assertEqual(result["system"].tolist(), ["bulkCu"])
            self.assertEqual(completed.call_args_list[0].args[0],
                             root / "md-data/mlip-trajs-torchsim-eager/bulkCu_300K_test/nvt_model-a.h5")

    def test_load_all_data_rejects_unknown_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "missing-model"):
                load_all_data(Path(temporary_directory), {"missing-model"})


if __name__ == "__main__":
    unittest.main()
