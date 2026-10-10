"""Regression tests for model-matched pressure figures and flexible timing files."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import figure_4
import figure_SI_4
import figure_SI_6
import figure_SI_15
from get_model_pressure_errors import resolve_model_reference_pressure_file


SYSTEM = "bulkAu_1500K_Kapil"
INTERFACE = "Pt111w24H2O_380K_Heenen_VASP"


class PressureFiguresTest(unittest.TestCase):
    def test_full_error_models_are_hidden_from_histograms_and_legends(self) -> None:
        values = np.array([0.1, 0.2, 0.3, 0.4])
        references = {"mace-mp-0": values, "orb-v2": values}
        predictions = {"mace-mp-0": values, "orb-v2": values + 1000}
        ranking = pd.DataFrame({
            "model": ["mace-mp-0", "orb-v2", "mace-mpa-0"],
            "error_GPa": [0.0, 1000.0, np.nan],
            "final_mean_pressure_error_percent": [0.0, 100.0, 100.0],
        })
        original = ranking.copy(deep=True)
        full_error_models = ["orb-v3-omat", "orb-v3-direct-omat", "pet-oam-xl", "pet-omat-xl"]
        valid_models = ["mattersim-v1-5m", "esen-30m-oam"]
        for model in full_error_models:
            references[model] = values
            predictions[model] = values + 1000
        for model, offset in zip(valid_models, (0.01, 0.02)):
            references[model] = values
            predictions[model] = values + offset
        for module, overview, scores in (
            (figure_4, "draw_overall_pressure_mae_plot",
             {"mace-mp-0": 0.0, "orb-v2": 1000.0}),
            (figure_SI_6, "draw_overall_pressure_error_plot",
             {"mace-mp-0": 0.0, "orb-v2": 100.0, "mace-mpa-0": 100.0}),
        ):
            scores.update(dict.fromkeys(full_error_models, 1000.0 if module is figure_4 else 100.0))
            scores.update(dict.fromkeys(valid_models, 0.02))
            panels = [("Hydrogen", "H_1050K_Rupp_QE", references, predictions,
                       scores, np.linspace(0.1, 1000.4, 6))]
            with self.subTest(figure=module.__name__), tempfile.TemporaryDirectory() as directory:
                with (
                    patch.object(module, "collect_histogram_panels", return_value=panels),
                    patch.object(module, overview),
                    patch.object(module.plt, "savefig"),
                    patch.object(module.plt, "close"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    module.plot_combined(Path(directory), None, ranking, 5,
                                         Path(directory) / "unused.pdf")
                    fig = module.plt.gcf()
                legend_text = [text.get_text() for ax in fig.axes if ax.get_legend()
                               for text in ax.get_legend().get_texts()]
                drawn_labels = [label for ax in fig.axes
                                for label in ax.get_legend_handles_labels()[1]]
                self.assertFalse(any("orb-v2" in label for label in legend_text))
                self.assertFalse(any("MACE-MPA" in label for label in legend_text))
                self.assertFalse(any("orb-v2" in label for label in drawn_labels))
                self.assertFalse(any("MACE-MPA" in label for label in drawn_labels))
                full_error_labels = [label for label in legend_text
                                     if "100.0%" in label and "mean error:" not in " ".join(label.split())]
                self.assertEqual(full_error_labels, [])
                if module is figure_SI_6:
                    self.assertTrue(any("Tier 2 mean error: 100.0%" in " ".join(label.split())
                                        for label in legend_text))
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                for ax in fig.axes:
                    if ax.get_legend():
                        legend_bounds = ax.get_legend().get_window_extent(renderer)
                        panel = ax.get_subplotspec().get_topmost_subplotspec()
                        self.assertLess(legend_bounds.height, ax.get_window_extent(renderer).height)
                        self.assertLess(legend_bounds.width,
                                        panel.get_position(fig).width * fig.bbox.width)
                module.plt.close(fig)
        pd.testing.assert_frame_equal(ranking, original)

    def test_hydrogen_is_kept_as_the_fifth_pressure_panel(self) -> None:
        categories = [
            ("Pure metals", SYSTEM),
            ("Perovskites", "CsSnI3_500K_Ivor_VASP"),
            ("Metal dichalcogenides", "TiSe2_400K_Ivor_VASP"),
            ("Metal alloys", "bulkPt3Co_300K_J.Kioseoglou_VASP"),
            ("Hydrogen", "H_1050K_Rupp_QE"),
        ]
        values = np.array([0.1, 0.2, 0.3, 0.4])
        panels = [
            (kind, system, {"mace-mp-0": values}, {"mace-mp-0": values},
             {"mace-mp-0": 0.0}, np.linspace(0.1, 0.4, 6))
            for kind, system in categories
        ]
        ranking = pd.DataFrame({"model": ["mace-mp-0"], "error_GPa": [0.0],
                                "final_mean_pressure_error_percent": [0.0]})
        for module, overview in (
            (figure_4, "draw_overall_pressure_mae_plot"),
            (figure_SI_6, "draw_overall_pressure_error_plot"),
        ):
            with self.subTest(figure=module.__name__), tempfile.TemporaryDirectory() as directory:
                figures = []
                create_figure = module.plt.figure

                def capture_figure(*args, **kwargs):
                    fig = create_figure(*args, **kwargs)
                    figures.append(fig)
                    return fig

                output = Path(directory) / "unused.pdf"
                with (
                    patch.object(module, "collect_histogram_panels", return_value=panels),
                    patch.object(module, overview),
                    patch.object(module.plt, "figure", side_effect=capture_figure),
                    patch.object(module.plt, "savefig"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    module.plot_combined(Path(directory), None, ranking, 5, output)
                fig = figures[0]
                titles = [ax.get_title().split("\n")[0]
                          for ax in fig.axes if ax.get_title()]
                self.assertEqual(titles, [kind for kind, _ in categories])
                panels_with_titles = [ax for ax in fig.axes if ax.get_title()]
                specs = [ax.get_subplotspec().get_topmost_subplotspec()
                         for ax in panels_with_titles]
                self.assertEqual([spec.rowspan.start for spec in specs], [2, 2, 2, 3, 3])
                bounds = [ax.get_position() for ax in panels_with_titles]
                np.testing.assert_allclose([box.width for box in bounds], bounds[0].width)
                top_center = (bounds[1].x0 + bounds[1].x1) / 2
                bottom_center = (bounds[3].x0 + bounds[4].x1) / 2
                self.assertAlmostEqual(bottom_center, top_center)
                self.assertFalse(output.exists())

    def test_each_model_uses_its_own_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "references").mkdir()
            for model, value in (("chgnet", 0.0), ("mace-mp-0", 10.0)):
                prediction = root / f"{model}_same-simulation-length_pressure_per_frame.csv"
                rows = pd.DataFrame({
                    "trajectory_file": [f"/sample/{system}/traj.h5"
                                        for system in (SYSTEM, INTERFACE) for _ in range(4)],
                    "frame_index": list(range(4)) * 2,
                    "pressure_GPa": [value, value + 0.1, value, value + 0.1] * 2,
                })
                rows.to_csv(prediction, index=False)
                rows.to_csv(root / "references" / f"{model}.csv", index=False)
                self.assertEqual(
                    resolve_model_reference_pressure_file(root, prediction),
                    root / "references" / f"{model}.csv",
                )

            mae_panels = figure_4.collect_histogram_panels(root, None, 5)
            self.assertEqual([panel[1] for panel in mae_panels], [SYSTEM])
            mae_panel = mae_panels[0]
            self.assertEqual(mae_panel[1], SYSTEM)
            self.assertAlmostEqual(mae_panel[4]["chgnet"], 0.0)
            self.assertAlmostEqual(mae_panel[4]["mace-mp-0"], 0.0)
            self.assertNotEqual(
                np.mean(mae_panel[2]["chgnet"]),
                np.mean(mae_panel[2]["mace-mp-0"]),
            )

            error_panels = figure_SI_6.collect_histogram_panels(
                root, None, 5, ["chgnet", "mace-mp-0"]
            )
            self.assertEqual([panel[1] for panel in error_panels], [SYSTEM])
            error_panel = error_panels[0]
            self.assertAlmostEqual(error_panel[4]["chgnet"], 0.0)
            self.assertAlmostEqual(error_panel[4]["mace-mp-0"], 0.0)

            violin = figure_SI_4.build_pressure_dataframe(root, None)
            self.assertEqual(len(violin), 16)
            self.assertEqual(
                set(figure_SI_4.load_pressure_per_frame_csv(
                    root / "references/chgnet.csv", include_molecular_crystals=True
                ).structure), {SYSTEM},
            )
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
            interface_dir = Path(directory) / INTERFACE
            interface_dir.mkdir()
            pd.DataFrame({"seconds_per_step": [99.0], "n_steps": [100]}).to_csv(
                interface_dir / "md_timing_model0.csv", index=False
            )
            timings = figure_SI_15.load_model_avg_timings(Path(directory))
            self.assertEqual(len(timings), 16)
            self.assertTrue((timings["mean_time_ms_per_step"] == 10.0).all())


if __name__ == "__main__":
    unittest.main()
