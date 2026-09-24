"""Regression tests for model-matched pressure figures and flexible timing files."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import figure_4
import figure_SI_4
import figure_SI_6
import figure_SI_15
from get_model_pressure_errors import resolve_model_reference_pressure_file


SYSTEM = "bulkAu_1500K_Kapil"


class PressureFiguresTest(unittest.TestCase):
    def test_each_model_uses_its_own_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "references").mkdir()
            for model, value in (("chgnet", 0.0), ("mace-mp-0", 10.0)):
                prediction = root / f"{model}_same-simulation-length_pressure_per_frame.csv"
                rows = pd.DataFrame({
                    "trajectory_file": [f"/sample/{SYSTEM}/traj.h5"] * 4,
                    "frame_index": range(4),
                    "pressure_GPa": [value, value + 0.1, value, value + 0.1],
                })
                rows.to_csv(prediction, index=False)
                rows.to_csv(root / "references" / f"{model}.csv", index=False)
                self.assertEqual(
                    resolve_model_reference_pressure_file(root, prediction),
                    root / "references" / f"{model}.csv",
                )

            mae_panel = figure_4.collect_histogram_panels(root, None, 5)[0]
            self.assertEqual(mae_panel[1], SYSTEM)
            self.assertAlmostEqual(mae_panel[4]["chgnet"], 0.0)
            self.assertAlmostEqual(mae_panel[4]["mace-mp-0"], 0.0)
            self.assertNotEqual(
                np.mean(mae_panel[2]["chgnet"]),
                np.mean(mae_panel[2]["mace-mp-0"]),
            )

            error_panel = figure_SI_6.collect_histogram_panels(
                root, None, 5, ["chgnet", "mace-mp-0"]
            )[0]
            self.assertAlmostEqual(error_panel[4]["chgnet"], 0.0)
            self.assertAlmostEqual(error_panel[4]["mace-mp-0"], 0.0)

            violin = figure_SI_4.build_pressure_dataframe(root, None)
            reference_means = violin[violin["kind"] == "reference"].groupby("model")[
                "pressure_GPa"
            ].mean()
            self.assertAlmostEqual(reference_means["chgnet"], 0.05)
            self.assertAlmostEqual(reference_means["mace-mp-0"], 10.05)

    def test_missing_matched_reference_does_not_use_legacy_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "references").mkdir()
            (root / "references" / "chgnet.csv").touch()
            legacy = root / "reference_pressure_per_frame.csv"
            legacy.touch()
            model = root / "mace-mp-0_same-simulation-length_pressure_per_frame.csv"
            with self.assertRaises(FileNotFoundError):
                resolve_model_reference_pressure_file(
                    root, model, legacy_candidates=(legacy,)
                )

    def test_timings_accept_extra_files_and_skip_empty_or_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            system_dir = Path(directory) / SYSTEM
            system_dir.mkdir()
            for index in range(16):
                pd.DataFrame({"seconds_per_step": [0.01], "n_steps": [100]}).to_csv(
                    system_dir / f"md_timing_model{index}.csv", index=False
                )
            (system_dir / "md_timing_empty.csv").touch()
            pd.DataFrame({"seconds_per_step": [0.01], "n_steps": [0]}).to_csv(
                system_dir / "md_timing_invalid.csv", index=False
            )
            timings = figure_SI_15.load_model_avg_timings(Path(directory))
            self.assertEqual(len(timings), 16)
            self.assertTrue((timings["mean_time_ms_per_step"] == 10.0).all())


if __name__ == "__main__":
    unittest.main()
