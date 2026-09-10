"""CPU regression tests; no MLIP checkpoints or GPU required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace
import h5py
import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
import pressure_pipeline as pipeline
from get_model_pressure_errors import build_pair_rows

HERE = Path(__file__).resolve().parent


class PressurePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.structure = "bulkCu_300K_test"
        self.meta = {"timestep": 2., "position_print_stride": 2}

    def tearDown(self):
        self.temp.cleanup()

    def hdf(self, name="test", n=4, steps=True):
        p = self.root / "mlip" / self.structure / f"nvt_{name}.h5"
        p.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(p, "w") as f:
            f.create_dataset("data/positions", data=np.zeros((n, 1, 3)))
            f.create_dataset("data/stress", data=np.repeat(np.diag([1., 2., 3.])[None], n, axis=0))
            if steps:
                f.create_dataset("steps/positions", data=np.arange(n)*2)
                f.create_dataset("steps/stress", data=np.arange(n)*2)
        return p

    def test_units_sign_shear_and_2d(self):
        stress = [1, 2, 6, 4, 5, 3]
        row = pipeline.stress_row(stress, "bulkCu_300K_test")
        self.assertAlmostEqual(row["pressure_GPa"], -3*pipeline.EV_A3_TO_GPA)
        self.assertEqual(row["stress_xy_eV_A3"], 3)
        self.assertAlmostEqual(pipeline.stress_row(stress, "TiSe2_300K")["pressure_GPa"], -1.5*pipeline.EV_A3_TO_GPA)
        with self.assertRaises(ValueError): pipeline.stress_matrix([np.nan]*6)

    def test_hdf_step_alignment_and_stride(self):
        p = self.hdf()
        df = pipeline.read_stress_frames(p, self.meta)
        np.testing.assert_equal(df.time_fs.to_numpy(), [0, 4, 8, 12])
        with h5py.File(p, "a") as f: f["steps/stress"][2] = 3
        with self.assertRaisesRegex(ValueError, "step numbers"):
            pipeline.read_stress_frames(p, self.meta)

    def test_incomplete_stress_rejected(self):
        p = self.hdf()
        with h5py.File(p, "a") as f:
            del f["data/stress"]
            f.create_dataset("data/stress", data=np.zeros((2, 3, 3)))
        with self.assertRaisesRegex(ValueError, "different frame counts"):
            pipeline.read_stress_frames(p, self.meta)

    def test_shared_time_window_uses_times(self):
        ref = pd.DataFrame({"time_fs": [0, 2, 4, 6, 8], "pressure_GPa": [0]*5})
        model = pd.DataFrame({"time_fs": [0, 4, 8, 12], "pressure_GPa": [0]*4})
        r, m, start, end = pipeline.match_time_window(ref, model)
        self.assertEqual((len(r), len(m), start, end), (5, 3, 0, 8))

    def test_ase_writer_stress_roundtrip(self):
        import ast
        from ase import units
        # Execute the actual writer class independently of the MD integrator imports.
        source = ast.parse((HERE / "md/ase-scripts/_ase_md.py").read_text())
        writer_class = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "HDF5TrajectoryWriter")
        namespace = dict(h5py=h5py, np=np, Path=Path, units=units)
        exec(compile(ast.Module(body=[writer_class], type_ignores=[]), "_ase_md.py", "exec"), namespace)
        atoms = Atoms("Cu", positions=[[0,0,0]], cell=[3,3,3], pbc=True)
        atoms.set_velocities([[0,0,0]])
        atoms.calc = SinglePointCalculator(atoms, energy=0, forces=np.zeros((1,3)), stress=[1,2,3,4,5,6])
        p = self.root / self.structure / "nvt_ase.h5"
        p.parent.mkdir()
        writer = namespace["HDF5TrajectoryWriter"](p, atoms, "test")
        writer.file.attrs["timestep_fs"] = 2.
        writer.file.attrs["record_interval"] = 3
        writer.write()
        writer.write()
        writer.close()
        df = pipeline.read_stress_frames(p, self.meta)
        np.testing.assert_equal(df.time_fs.to_numpy(), [0,6])
        np.testing.assert_equal(df.stress_yz_eV_A3.to_numpy(), [4,4])

    def test_complete_pipeline_and_matched_reference_rescoring(self):
        self.hdf(n=4)
        metadata = self.root / "metadata.json"
        metadata.write_text(json.dumps({self.structure: self.meta}))
        ref_path = self.root / "ref" / self.structure / "traj.extxyz"
        ref_path.parent.mkdir(parents=True)
        frames = []
        for i in range(3):
            atoms = Atoms("Cu", positions=[[0,0,0]], cell=[3,3,3], pbc=True)
            atoms.calc = SinglePointCalculator(atoms, energy=0, forces=np.zeros((1,3)), stress=[1,2,3,0,0,0])
            frames.append(atoms)
        write(ref_path, frames)
        output = self.root / "out"
        args = Namespace(metadata=metadata, output_dir=output, reference_file=None,
                         traj_dir=[self.root/"mlip"], prefix=None, model=None, recompute=False,
                         include_interfaces=False, ref_dir=self.root/"ref", reference_first_step=0,
                         bins=8, no_plots=False)
        pairs = pipeline.run_pipeline(args)
        self.assertEqual(pairs.n_mlip_frames.iloc[0], 3)
        self.assertEqual(len(pd.read_csv(output/"test_stress_per_frame.csv")), 4)
        self.assertAlmostEqual(pairs.absolute_mean_error_GPa.iloc[0], 0)
        self.assertAlmostEqual(pairs.pressure_similarity.iloc[0], 1)
        self.assertTrue((output/"plots/pressure_model_comparison.png").is_file())
        self.assertTrue((output/"plots/test"/f"{self.structure}.png").is_file())
        # A full reference containing an unmatched tail must not change the score.
        full_ref = output/pipeline.REFERENCE_NAME
        ref = pd.read_csv(full_ref)
        tail = ref.iloc[[-1]].copy()
        tail["frame_index"], tail["pressure_GPa"] = 99, 1e6
        pd.concat([ref,tail]).to_csv(full_ref,index=False)
        score = build_pair_rows(output, full_ref, pipeline.SUFFIX, 8)
        self.assertAlmostEqual(score.pressure_similarity.iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
