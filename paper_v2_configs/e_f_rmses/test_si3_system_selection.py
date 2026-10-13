"""Check SI3 exclusions and actual model means without creating figures."""

import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import _plot_inputs


class SI3SystemSelectionTests(unittest.TestCase):
    def test_short_names_benchmark_paths_and_categories_are_excluded(self):
        cases = {
            "system": (["H", "H_1050K_Rupp_QE", "hydrogen", "naphthalene_295K_Sharma_S"],
                       ["bulkPt3Co", "Pt111w24H2O", "Pt111w24H2O_380K_Heenen_VASP"]),
            "system_type": (["hydrogen", "molecular crystals"],
                            ["metal alloys", "Metal-water interfaces"]),
            "trajectory": (["/ref/H_1050K_Rupp_QE/traj.extxyz",
                            "/ref/naphthalene_295K_Sharma_S/traj.extxyz"],
                           ["/ref/bulkPt3Co_300K/traj.extxyz",
                            "/ref/Pt111w24H2O_380K_Heenen_VASP/traj.extxyz"]),
        }
        for column, (excluded, allowed) in cases.items():
            with self.subTest(column=column):
                values = excluded + allowed
                frame = pd.DataFrame({column: values}, index=range(20, 20 + len(values)))
                original = frame.copy(deep=True)
                filtered = _plot_inputs.filter_si3_systems(frame)
                self.assertEqual(filtered[column].tolist(), allowed)
                pd.testing.assert_frame_equal(frame, original)

    def test_plot_script_averages_filtered_systems_for_each_source(self):
        script = Path(__file__).with_name("figure_SI_4.py")
        tree = ast.parse(script.read_text())
        # Execute only the real data preparation, stopping before any figure is created.
        cutoff = next(index for index, node in enumerate(tree.body)
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Tuple)
                              and [name.id for name in target.elts] == ["fig", "axes"]
                              for target in node.targets))
        prefix = ast.Module(body=tree.body[:cutoff], type_ignores=[])
        frame = pd.DataFrame({
            "calculator": ["orb-v2-force-only"] * 5 + ["eq-v2-m-omat"] * 2,
            "system": ["bulkCu", "bulkPt3Co", "H", "Pt111w24H2O", "naphthalene",
                       "bulkCu", "H"],
            "energy_rmse": [1., 3., 100., 5., 10000., 6., 999.],
            "force_rmse": [2., 8., 200., 14., 20000., 12., 999.],
        })
        for source in ("mlip-trajs-torchsim-eager", "mlip-trajs-torchsim-accelerated"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary) / source
                folder.mkdir()
                with (patch.object(_plot_inputs, "plot_args", return_value=(folder, folder)),
                      patch.object(_plot_inputs, "read_metric_csv", return_value=frame.copy())):
                    namespace = {"__file__": str(script)}
                    exec(compile(prefix, str(script), "exec"), namespace)
                summary = pd.read_csv(folder / "mean_metrics_by_model_no_molecular_crystals.csv")
                orb = summary[summary.calculator == "orb-v2"].iloc[0]
                self.assertEqual(orb.energy_rmse, 3.)
                self.assertEqual(orb.force_rmse, 8.)
                eq = summary[summary.calculator == "eq-v2-m-omat"].iloc[0]
                self.assertEqual(eq.energy_rmse, 6.)
                self.assertEqual(eq.force_rmse, 12.)
                self.assertEqual(namespace["excluded_count"], 3)
                self.assertFalse(list(folder.glob("*.pdf")))

    def test_regular_plot_inputs_keep_hydrogen_and_interfaces(self):
        frame = pd.DataFrame({"system": ["bulkCu", "H", "Pt111w24H2O", "naphthalene"]})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.csv"
            frame.to_csv(path, index=False)
            with patch.object(_plot_inputs, "INCLUDE_MOLECULAR_CRYSTALS", False):
                ordinary = _plot_inputs.read_metric_csv(path)
            self.assertEqual(ordinary.system.tolist(), ["bulkCu", "H", "Pt111w24H2O"])
            with patch.object(_plot_inputs, "INCLUDE_MOLECULAR_CRYSTALS", True):
                si3 = _plot_inputs.filter_si3_systems(_plot_inputs.read_metric_csv(path))
            self.assertEqual(si3.system.tolist(), ["bulkCu", "Pt111w24H2O"])


if __name__ == "__main__":
    unittest.main()
