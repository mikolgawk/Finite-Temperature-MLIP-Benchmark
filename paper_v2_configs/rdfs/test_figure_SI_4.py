"""Validate the pinned Pt3Co RDF panels without generating figures."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_rdf_pipeline as pipeline


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rdf_figure_si_4", HERE / "figure_SI_4.py")
panel = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = panel
spec.loader.exec_module(panel)


class FigureSI4Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = panel.DEFAULT_SOURCE
        self.scores = self.root / "results" / self.source
        self.saved = self.scores / "rdf_same_simulation_length_saved"
        self.scores.mkdir(parents=True)
        self.curve(None)

    def curve(self, model):
        folder = self.saved / ("reference" if model is None else "mlip/" + model)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{panel.SYSTEM}.csv"
        path.write_text("r_A,g_r\n0,0\n1,1\n2,1\n")
        return path

    def model(self, name, error, with_curve=True):
        pd.DataFrame({
            "System": [panel.SYSTEM, "bulkCuAu_500K-Artrith_VASP"],
            "RDF_Error": [error, 1.],
        }).to_csv(self.scores / f"rdf_similarity_scores_same_simulation_length_{name}.csv", index=False)
        if with_curve:
            self.curve(name)

    def test_means_and_selections_use_pt3co_and_preserve_full_error_penalties(self):
        self.model("mace-mp-0-compile", 20.)
        self.model("chgnet-force-only-ase", 80.)
        self.model("grace-mp", 100., with_curve=False)
        self.model("eSEN-30M-OAM-force-only", 10.)
        self.model("eq-v2-M-omat-force-only", 12.)
        self.model("uma-m-omat-compile-force-only", 21.)
        self.model("uma-m-omat-turbo-force-only", 22.)
        reference, tiers = panel.load_panel_data(self.scores, self.saved)
        self.assertEqual(len(tiers), 4)
        np.testing.assert_array_equal(reference[0], [0, 1, 2])
        self.assertAlmostEqual(tiers[0].mean_error, 200. / 3)
        self.assertEqual([curve[0] for curve in tiers[0].curves],
                         ["MACE-MP-0 (20.0%)", "CHGNet (80.0%)"])
        self.assertEqual([curve[2] for curve in tiers[0].curves], ["-", "--"])
        self.assertEqual(tiers[0].legend_only, [])
        self.assertTrue(np.isnan(tiers[1].mean_error))
        self.assertEqual(tiers[1].curves, [])
        self.assertEqual(tiers[2].mean_error, 11.)
        self.assertEqual([curve[0] for curve in tiers[2].curves],
                         ["eSEN-30M-OAM (10.0%)", "EquiformerV2 (12.0%)"])
        self.assertEqual(tiers[3].mean_error, 21.5)
        self.assertEqual([curve[0] for curve in tiers[3].curves],
                         ["UMA-M-P1 (compile) (21.0%)", "UMA-M-P1 (turbo) (22.0%)"])

    def test_single_model_is_drawn_once_and_valid_full_error_is_legend_only(self):
        self.model("mace-mp-0-compile", 20.)
        self.model("grace-mp", 100.)
        trajectory = self.root / "data" / self.source / panel.SYSTEM / "nvt_grace-mp.h5"
        trajectory.parent.mkdir(parents=True)
        trajectory.touch()
        trajectory.with_name("md_timing_grace-mp.csv").write_text(
            f"calculator,system,n_steps\ngrace-mp,{panel.SYSTEM},10\n"
        )
        with patch.object(panel.rdf, "DATA_DIR", self.root / "data"):
            _, tiers = panel.load_panel_data(self.scores, self.saved)
            self.assertEqual(tiers[0].mean_error, 60.)
            self.assertEqual(len(tiers[0].curves), 1)
            self.assertEqual(tiers[0].legend_only, ["GRACE-2L-MPtrj (100.0%)"])
            trajectory.with_name("md_timing_grace-mp.csv").write_text("")
            _, tiers = panel.load_panel_data(self.scores, self.saved)
            self.assertEqual(tiers[0].mean_error, 60.)
            self.assertEqual(tiers[0].legend_only, [])

    def test_data_only_cli_never_calls_plotting(self):
        self.model("mace-mp-0-compile", 20.)
        argv = ["figure_SI_4.py", "--source", self.source,
                "--results-dir", str(self.root / "results"), "--data-only"]
        with (patch.object(sys, "argv", argv),
              patch.object(panel, "plot_panel", side_effect=AssertionError("Must not create figures")) as plot,
              contextlib.redirect_stdout(io.StringIO()) as output):
            panel.main()
        plot.assert_not_called()
        self.assertIn(panel.SYSTEM, output.getvalue())
        self.assertFalse(list(self.root.rglob("*.pdf")))

    def test_pipeline_selects_source_paths_and_supports_skipping_si4(self):
        argv = ["run_rdf_pipeline.py", "--plots-only", "--dry-run", "--source", self.source,
                "--results-dir", str(self.root / "results"), "--plots-dir", str(self.root / "plots")]
        with (patch.object(sys, "argv", argv), patch.object(pipeline, "run_stage") as stage,
              contextlib.redirect_stdout(io.StringIO())):
            pipeline.main()
        commands = [call.args[1] for call in stage.call_args_list
                    if Path(call.args[1][2]).name == "figure_SI_4.py"]
        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertEqual(command[command.index("--rdf-csv-dir") + 1], str(self.scores))
        self.assertEqual(command[command.index("--output-file") + 1],
                         str(self.root / "plots" / self.source / "figure_SI_4.pdf"))
        with (patch.object(sys, "argv", argv + ["--skip-figure-si-4"]),
              patch.object(pipeline, "run_stage") as stage,
              contextlib.redirect_stdout(io.StringIO())):
            pipeline.main()
        self.assertFalse(any(Path(call.args[1][2]).name == "figure_SI_4.py"
                             for call in stage.call_args_list))


if __name__ == "__main__":
    unittest.main()
