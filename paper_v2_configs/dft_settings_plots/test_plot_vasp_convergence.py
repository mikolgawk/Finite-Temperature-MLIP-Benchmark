"""Check convergence values against the supplied VASP data without plotting."""

from pathlib import Path
import unittest

import plot_vasp_convergence as convergence


class VaspConvergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path(__file__).with_name("convergence_tests")
        cls.data, cls.summary = convergence.load_convergence_data(cls.data_dir)

    def rows_from(self, filename):
        return self.data.loc[
            self.data["source_file"].map(lambda value: Path(value).name == filename)
        ]

    def test_pt_uses_all_seven_raw_frames_and_the_900_ev_reference(self):
        rows = self.rows_from("encut_convergence.extxyz")
        self.assertEqual(len(rows), 7)
        self.assertEqual(set(rows["cutoff_ev"]), {300, 400, 500, 600, 700, 800, 900})
        self.assertEqual(set(rows["reference_cutoff_ev"]), {900})
        self.assertEqual(set(rows["n_atoms"]), {108})
        at_700 = rows.loc[rows.cutoff_ev.eq(700)].iloc[0]
        self.assertAlmostEqual(at_700.energy_error_mev_per_atom, 0.62703009259, places=9)
        self.assertAlmostEqual(at_700.max_force_component_error_mev_per_angstrom, 23.99, places=8)
        reference = rows.loc[rows.cutoff_ev.eq(900)].iloc[0]
        self.assertEqual(reference.energy_error_mev_per_atom, 0)
        self.assertEqual(reference.max_force_component_error_mev_per_angstrom, 0)

    def test_anisotropic_mesh_headers_keep_their_components(self):
        rows = self.rows_from("kpoints_convergence.extxyz")
        self.assertEqual(len(rows), 7)
        self.assertEqual(set(rows["k_grid"]),
                         {"4_3_1", "7_6_1", "10_8_1", "13_11_1", "16_14_1", "19_16_1", "22_19_1"})
        self.assertEqual(set(rows["reference_k_grid"]), {"22_19_1"})
        coarsest = rows.loc[rows.k_grid.eq("4_3_1")].iloc[0]
        self.assertAlmostEqual(coarsest.k_spacing_inv_angstrom, 0.217328, places=5)
        self.assertAlmostEqual(coarsest.inverse_k_spacing_angstrom,
                               1 / coarsest.k_spacing_inv_angstrom, places=10)

    def test_mapbbr3_energy_differences_are_normalized_per_96_atoms(self):
        rows = self.data.loc[self.data.system.eq("MAPbBr3") & self.data.sweep.eq("encut")]
        self.assertEqual(set(rows["source_file"].map(lambda value: Path(value).name)),
                         {"gemini-code-1779112066944.txt"})
        self.assertEqual(set(rows["n_atoms"]), {96})
        self.assertEqual(set(rows["reference_cutoff_ev"]), {1000})
        at_700 = rows.loc[rows.cutoff_ev.eq(700)].iloc[0]
        self.assertAlmostEqual(at_700.energy_error_mev_per_atom, 0.65625, places=10)
        self.assertAlmostEqual(at_700.abs_energy_error_mev_per_atom, 0.65625, places=10)

    def test_cu3au_reports_its_actual_740_ev_reference_and_atom_count(self):
        rows = self.rows_from("Cu3Au_Ecut_total_energy.csv")
        self.assertFalse(rows.empty)
        self.assertEqual(set(rows["n_atoms"]), {128})
        self.assertEqual(set(rows["reference_cutoff_ev"]), {740})
        at_700 = rows.loc[rows.cutoff_ev.eq(700)].iloc[0]
        self.assertAlmostEqual(at_700.energy_error_mev_per_atom, -0.4910653125, places=8)
        self.assertAlmostEqual(at_700.abs_energy_error_mev_per_atom, 0.4910653125, places=8)

    def test_fhi_aims_sweeps_are_excluded_from_vasp_data_and_summary(self):
        sources = self.data["source_file"].astype(str)
        for forbidden in ("Cu_bulk", "naphthalene_bulk", "aims.out", "control.in"):
            self.assertFalse(sources.str.contains(forbidden, regex=False).any())
        self.assertFalse(self.summary.duplicated(["system", "sweep"]).any())
        self.assertEqual(
            set(self.summary[["system", "sweep"]].itertuples(index=False, name=None)),
            set(self.data[["system", "sweep"]].itertuples(index=False, name=None)),
        )


if __name__ == "__main__":
    unittest.main()
