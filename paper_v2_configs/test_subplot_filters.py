"""Keep full-error models in statistics while omitting their subplot curves."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for directory in (HERE, HERE / "rdfs", HERE / "vdos", HERE / "pressures"):
    sys.path.insert(0, str(directory))

import fig_3 as rdf
import figure_5 as vdos
import figure_4 as pressure_mae
import figure_SI_6 as pressure
from get_model_pressure_errors import write_metric_outputs
from subplot_filters import (
    display_error_percent, display_full_error_legend, format_subplot_error_percent,
)


MODELS = ["mace-mp-0", "chgnet", "grace-mp"]
SYSTEM = "bulkAu_1500K_Kapil"


class SubplotFilterTests(unittest.TestCase):
    def test_full_error_tolerance_does_not_hide_nearby_valid_scores(self):
        self.assertTrue(display_error_percent(99.999))
        self.assertEqual(format_subplot_error_percent(99.999), "<100.0%")
        self.assertEqual(format_subplot_error_percent(100.0), "100.0%")
        for error in (100.0, 100.0 - 1e-10, np.nan, np.inf):
            self.assertFalse(display_error_percent(error))
        for error in (100.0, 100.0 - 1e-10):
            self.assertTrue(display_full_error_legend(error, has_data=True))
            self.assertFalse(display_full_error_legend(error, has_data=False))
        for error in (99.999, np.nan, np.inf, 101.0):
            self.assertFalse(display_full_error_legend(error, has_data=True))

    def test_rdf_full_error_legend_rejects_failed_md_with_stale_curves(self):
        source = "mlip-trajs-torchsim-eager"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rdf_dir = root / "results" / source / "saved"
            for model in MODELS:
                curve = rdf_dir / "mlip" / model / f"{SYSTEM}.csv"
                curve.parent.mkdir(parents=True)
                curve.write_text("r,g\n0,1\n1,1\n")
                trajectory = root / "data" / source / SYSTEM / f"nvt_{model}.h5"
                trajectory.parent.mkdir(parents=True, exist_ok=True)
                trajectory.touch()
                trajectory.with_name(f"md_timing_{model}.csv").write_text(
                    f"calculator,system,n_steps\n{model},{SYSTEM},10\n"
                    if model == MODELS[0] else ""
                )
            with patch.object(rdf, "DATA_DIR", root / "data"):
                self.assertEqual(rdf.full_error_models_for_legend(
                    SYSTEM, MODELS, dict.fromkeys(MODELS, 100.0), str(rdf_dir)
                ), [MODELS[0]])

    def test_vdos_full_error_legend_rejects_missing_data_and_failure_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            spectrum = Path(directory) / "spectrum.csv"
            spectrum.write_text("wavenumber,intensity\n0,1\n1,1\n")
            rows = pd.DataFrame({
                "mlip_model": MODELS,
                "vdos_error_percent": [100.0] * 3,
                "mlip_file": [str(spectrum), str(spectrum), ""],
                "ref_file": [str(spectrum)] * 3,
                "failure_reason": ["", "Failed MD", ""],
            })
            original = rows.copy(deep=True)
            self.assertEqual(vdos.full_error_models_for_legend(rows, MODELS), [MODELS[0]])
            pd.testing.assert_frame_equal(rows, original)

    def test_rdf_selects_valid_curves_and_keeps_full_error_in_means(self):
        scores = dict(zip(MODELS, (20.0, 80.0, 100.0)))
        with patch.object(rdf, "resolve_saved_rdf_path", return_value=Path("rdf.csv")):
            self.assertEqual(rdf.pick_best_worst_models(SYSTEM, MODELS, scores, "."),
                             (MODELS[0], MODELS[1]))
            self.assertEqual(rdf.pick_best_worst_models(
                SYSTEM, MODELS, dict.fromkeys(MODELS, 100.0), "."
            ), (None, None))
        self.assertAlmostEqual(rdf.tier_mean_rdf_error(MODELS, scores), 200.0 / 3)
        self.assertEqual(scores[MODELS[2]], 100.0)

    def test_vdos_selects_valid_curves_without_changing_worst_system_or_means(self):
        frame = pd.DataFrame({
            "system": [SYSTEM] * 3 + ["bulkCu_1000K_Kapil"] * 3,
            "mlip_model": MODELS * 2,
            "vdos_error_percent": [20.0, 80.0, 100.0, 55.0, 55.0, 55.0],
        })
        original = frame.copy(deep=True)
        self.assertEqual(vdos.select_system_with_worst_mean_vdos_error(frame, "Pure metals"),
                         SYSTEM)
        subset = frame[frame.system == SYSTEM]
        self.assertEqual(vdos.pick_best_worst_models_for_tier(subset, MODELS),
                         (MODELS[0], MODELS[1]))
        self.assertAlmostEqual(vdos.mean_system_vdos_error(subset), 200.0 / 3)
        model_means = vdos.model_mean_vdos_error_by_category(subset)
        self.assertAlmostEqual(vdos.tier_mean_vdos_error(MODELS, model_means), 200.0 / 3)
        pd.testing.assert_frame_equal(frame, original)

    def test_vdos_does_not_pick_a_full_error_row_within_a_valid_model(self):
        rows = pd.DataFrame({
            "system": [SYSTEM] * 2, "mlip_model": [MODELS[0]] * 2,
            "vdos_error_percent": [80.0, 100.0],
        })
        self.assertEqual(vdos.pick_model_row(rows, SYSTEM, MODELS[0], False)
                         .vdos_error_percent, 80.0)
        full_error = rows[rows.vdos_error_percent == 100.0]
        self.assertIsNone(vdos.pick_model_row(full_error, SYSTEM, MODELS[0], False))
        self.assertEqual(vdos.pick_best_worst_models_for_tier(full_error, MODELS),
                         (None, None))

    def test_pressure_keeps_penalties_in_tier_means_and_selects_valid_curves(self):
        scores = dict(zip(MODELS, (20.0, 80.0, 100.0)))
        best, worst, tier_scores = pressure.choose_best_worst(MODELS, scores)
        self.assertEqual((best, worst), (MODELS[0], MODELS[1]))
        self.assertEqual(tier_scores, scores)
        self.assertAlmostEqual(pressure.mean_finite(tier_scores.values()), 200.0 / 3)
        best, worst, tier_scores = pressure.choose_best_worst(
            MODELS, dict.fromkeys(MODELS, 100.0)
        )
        self.assertEqual((best, worst), (None, None))
        self.assertEqual(pressure.mean_finite(tier_scores.values()), 100.0)

    def test_pressure_mae_uses_percentages_instead_of_gpa_for_curve_filter(self):
        reference = np.array([0.0, 100.0, 200.0, 300.0])
        references = dict.fromkeys(MODELS, reference)
        values = {MODELS[0]: reference, MODELS[1]: reference + 100.0,
                  MODELS[2]: reference + 1000.0, "empty": np.array([])}
        references["empty"] = reference
        visible = pressure_mae.visible_pressure_models(references, values, 10)
        self.assertEqual(visible, set(MODELS[:2]))
        self.assertEqual(pressure_mae.full_error_pressure_models(references, values, 10),
                         {MODELS[2]})
        scores = dict(zip(MODELS, (0.0, 100.0, 1000.0)))
        best, worst, tier_scores = pressure_mae.choose_best_worst_from_scores(
            MODELS, scores, visible_models=visible
        )
        self.assertEqual((best, worst), (MODELS[0], MODELS[1]))
        self.assertEqual(tier_scores, scores)
        self.assertAlmostEqual(pressure_mae.mean_finite(tier_scores.values()), 1100.0 / 3)

    def test_pressure_metric_exports_still_include_full_error_rows(self):
        pairs = pd.DataFrame({
            "system": [SYSTEM, "bulkCu_1000K_Kapil"],
            "system_type": ["pure metals"] * 2, "mlip_model": [MODELS[0]] * 2,
            "pressure_similarity": [1.0, 0.0], "bins": [10] * 2,
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, means, _ = write_metric_outputs(
                pairs, root / "pairs.csv", root / "systems.csv",
                root / "models.csv", root / "types.csv", None,
            )
            self.assertEqual(len(pd.read_csv(root / "pairs.csv")), 2)
            self.assertEqual(means.final_mean_pressure_error_percent.tolist(), [50.0])


if __name__ == "__main__":
    unittest.main()
