"""CPU tests for post-processing saved frames without integrating MD."""
from argparse import Namespace
from pathlib import Path
import json
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

import pressure_evaluator as evaluator


class ConstantStress(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(self.atoms), 3)),
            "stress": np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]),
        }


class PressureEvaluatorTests(unittest.TestCase):
    def test_ase_evaluator_reads_existing_hdf5_without_running_md(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            system = "bulkCu_300K_test"
            trajectory = root / "trajectories" / system / "nvt_dummy.h5"
            trajectory.parent.mkdir(parents=True)
            with h5py.File(trajectory, "w") as handle:
                handle.create_dataset("data/atomic_numbers", data=[[29]])
                handle.create_dataset("data/pbc", data=[True, True, True])
                handle.create_dataset("data/positions", data=np.zeros((4, 1, 3)))
                handle.create_dataset(
                    "data/cell", data=np.repeat(np.diag([3.0, 4.0, 5.0])[None], 4, axis=0)
                )
                handle.create_dataset("steps/positions", data=[0, 2, 4, 6])

            reference_path = root / "references" / system / "traj.extxyz"
            reference_path.parent.mkdir(parents=True)
            reference_frames = []
            for _ in range(3):
                atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[3, 4, 5], pbc=True)
                atoms.calc = SinglePointCalculator(
                    atoms, energy=0, forces=np.zeros((1, 3)),
                    stress=[1, 2, 3, 0, 0, 0],
                )
                reference_frames.append(atoms)
            write(reference_path, reference_frames)

            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({
                system: {"timestep": 1.0, "position_print_stride": 2}
            }))
            output = root / "output"
            evaluator._ARGS = Namespace(
                traj_dir=root / "trajectories",
                ref_dir=root / "references",
                metadata=metadata,
                output_dir=output,
                trajectory_model=None,
                max_frames=None,
                force=False,
                no_progress=True,
                debug=False,
            )
            evaluator.run_ase_pressure("dummy", ConstantStress, "test-ase")

            result = pd.read_csv(
                output / "dummy_same-simulation-length_pressure_per_frame.csv"
            )
            self.assertEqual(len(result), 3)
            self.assertTrue(np.allclose(result.pressure_GPa, -2 * 160.21766208))
            full = pd.read_csv(output / "dummy_stress_per_frame.csv")
            self.assertEqual(len(full), 4)
            self.assertTrue((output / "references" / "dummy.csv").is_file())


if __name__ == "__main__":
    unittest.main()
