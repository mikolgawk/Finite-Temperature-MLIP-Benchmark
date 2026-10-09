"""CPU tests for post-processing saved frames without integrating MD."""
from argparse import Namespace
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

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
    def test_trajectory_directory_default_can_be_overridden(self):
        script = Path(__file__).resolve().parent / "md_eager/ase-scripts/pressure_orb_v3_omat.py"
        default = script.parents[3] / "data/mlip-trajs-ase"
        custom = Path("/tmp/custom-omat-trajectories")
        with patch.object(evaluator, "_ARGS"), patch.object(evaluator, "_SCRIPT"), patch.object(evaluator, "_BACKEND"):
            with patch.object(sys, "argv", [str(script)]):
                evaluator.early_cli(script, "ase", default_traj_dir=default)
                self.assertEqual(evaluator._ARGS.traj_dir, default)
            with patch.object(sys, "argv", [str(script), "--traj-dir", str(custom)]):
                evaluator.early_cli(script, "ase", default_traj_dir=default)
                self.assertEqual(evaluator._ARGS.traj_dir, custom)
            with patch.object(sys, "argv", [str(script)]):
                evaluator.early_cli(script, "torchsim")
                self.assertEqual(evaluator._ARGS.traj_dir, script.parents[3] / "data/mlip-trajs-torchsim-eager")

    def test_stress_matrix_symmetrizes_full_tensor(self):
        stress = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])

        result = evaluator._stress_matrix(stress)

        np.testing.assert_allclose(result, 0.5 * (stress + stress.T))
        np.testing.assert_allclose(np.diag(result), np.diag(stress))

    def test_csv_retry_replaces_selected_system_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pressure.csv"
            pd.DataFrame([
                {"system": "hydrogen", "frame_index": 0, "pressure_GPa": 1.0},
                {"system": "copper", "frame_index": 0, "pressure_GPa": 2.0},
                {"system": "Pt111w24H2O_380K_Heenen_VASP", "frame_index": 0,
                 "pressure_GPa": 100.0},
            ]).to_csv(output, index=False)

            evaluator._write_csv_records(
                output,
                [
                    {"system": "hydrogen", "frame_index": 0, "pressure_GPa": 3.0},
                    {"system": "hydrogen", "frame_index": 1, "pressure_GPa": 4.0},
                ],
                {"hydrogen"},
            )

            result = pd.read_csv(output)
            self.assertEqual(result.system.tolist(), ["copper", "hydrogen", "hydrogen"])
            self.assertEqual(result.pressure_GPa.tolist(), [2.0, 3.0, 4.0])

    def test_failure_system_uses_trajectory_parent(self):
        self.assertEqual(
            evaluator._failure_system({"file": "/trajectories/hydrogen/nvt_model.h5"}),
            "hydrogen",
        )

    def test_frame_count_respects_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            trajectory = Path(temporary) / "trajectory.h5"
            with h5py.File(trajectory, "w") as handle:
                handle.create_dataset("data/positions", data=np.zeros((7, 1, 3)))

            self.assertEqual(evaluator._frame_count(trajectory, None), 7)
            self.assertEqual(evaluator._frame_count(trajectory, 3), 3)
            self.assertEqual(evaluator._frame_count(trajectory, 20), 7)

    def test_trajectory_model_tags_are_canonicalized_for_lookup(self):
        self.assertEqual(evaluator._trajectory_model("orb-v3-omat-force-only-eager"), "orb-v3-omat")
        self.assertEqual(evaluator._trajectory_model("orb-v3-omat-stress-eager"), "orb-v3-omat")
        self.assertEqual(evaluator._trajectory_model("pet-oam-xl-force-only-torchscript"),
                         "pet-oam-xl-torchscript")
        self.assertEqual(evaluator._trajectory_model("pet-oam-xl-stress-torchscript"),
                         "pet-oam-xl-torchscript")
        self.assertEqual(evaluator._trajectory_model("nequip-oam-l-force-only"), "nequip")
        self.assertEqual(evaluator._trajectory_model("chgnet-force-only-ase"), "chgnet-ase")

        with tempfile.TemporaryDirectory() as temporary:
            trajectory = (
                Path(temporary) / "system" / "nvt_orb-v3-omat-force-only-eager.h5"
            )
            trajectory.parent.mkdir()
            trajectory.touch()
            self.assertEqual(
                evaluator._trajectory_paths(Path(temporary), "orb-v3-omat-stress-eager"),
                [trajectory],
            )

    def test_ase_trajectory_names_match_pressure_models(self):
        cases = [
            ("chgnet", "chgnet-force-only-ase"),
            ("chgnet-force-only-ase", "chgnet-force-only-ase"),
            ("nequip-oam-l", "nequip-oam-l-ase"),
            ("nequip-oam-l-stress", "nequip-oam-l-force-only-ase"),
            ("mace-mpa-0", "mace-mpa-0-ase"),
            ("mace-mpa-0-compile", "mace-mpa-0-compile-ase"),
            ("mattersim-v1-5M-compile-stress", "mattersim-v1-5M-compile-force-only-ase"),
            ("orb-v3-direct-omat-stress", "orb-v3-direct-omat-force-only"),
        ]
        for model, name in cases:
            with self.subTest(model=model, name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                trajectory = root / "system" / f"nvt_{name}.h5"
                trajectory.parent.mkdir()
                trajectory.touch()
                (trajectory.parent / f"nvt_{name}.h5.inprogress").touch()
                (trajectory.parent / "nvt_unrelated-ase.h5").touch()
                self.assertEqual(evaluator._trajectory_paths(root, model), [trajectory])

    def test_ase_trajectory_is_preferred_over_legacy_copy_per_system(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            modern = root / "system" / "nvt_orb-v3-direct-omat-ase.h5"
            legacy = modern.parent / "nvt_orb-v3-direct-omat.h5"
            fallback = root / "other-system" / legacy.name
            modern.parent.mkdir()
            fallback.parent.mkdir()
            for path in (modern, legacy, fallback):
                path.touch()
            for model in ("orb-v3-direct-omat", "orb-v3-direct-omat-ase"):
                with self.subTest(model=model):
                    self.assertEqual(
                        evaluator._trajectory_paths(root, model), sorted([modern, fallback])
                    )

    def test_ase_evaluator_reads_existing_hdf5_without_running_md(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            system = "bulkCu_300K_test"
            trajectory = root / "trajectories" / system / "nvt_dummy-force-only-ase.h5"
            trajectory.parent.mkdir(parents=True)
            with h5py.File(trajectory, "w") as handle:
                handle.create_dataset("data/atomic_numbers", data=[[29]])
                handle.create_dataset("data/pbc", data=[True, True, True])
                handle.create_dataset("data/positions", data=np.zeros((4, 1, 3)))
                handle.create_dataset(
                    "data/cell", data=np.repeat(np.diag([3.0, 4.0, 5.0])[None], 4, axis=0)
                )
                handle.create_dataset("steps/positions", data=[0, 2, 4, 6])

            no_stress_system = "bulkCuAu_500K_test"
            no_stress_trajectory = (
                root / "trajectories" / no_stress_system / "nvt_dummy-force-only-ase.h5"
            )
            no_stress_trajectory.parent.mkdir(parents=True)
            with h5py.File(no_stress_trajectory, "w") as handle:
                handle.create_dataset("data/atomic_numbers", data=[[29]])
                handle.create_dataset("data/pbc", data=[True, True, True])
                handle.create_dataset("data/positions", data=np.zeros((4, 1, 3)))
                handle.create_dataset(
                    "data/cell",
                    data=np.repeat(np.diag([3.0, 4.0, 5.0])[None], 4, axis=0),
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
                system: {"timestep": 1.0, "position_print_stride": 2},
                no_stress_system: {
                    "timestep": 1.0,
                    "position_print_stride": 2,
                    "stress_print_stride": None,
                },
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
            self.assertFalse((output / "dummy_pressure_failures.json").exists())


if __name__ == "__main__":
    unittest.main()
