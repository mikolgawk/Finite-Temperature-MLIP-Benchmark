"""Keep constrained-atom force RMSEs comparable between ASE and TorchSim."""

import importlib.util
import io
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import read, write


HERE = Path(__file__).resolve().parent
RAW_FORCES = np.array([[0.12, -0.18, 0.24], [-0.03, 0.09, -0.15]])
ENERGY = -12.5


def load_backend(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ASE = load_backend('constraints_test_ase_rmse', 'rmse_ase_scripts/ase_rmse.py')
TORCHSIM = load_backend('constraints_test_torchsim_rmse',
                        'rmse_torchsim_scripts/torchsim_rmse.py')


def reference_frame(*, constrained=True):
    """Round-trip real DFT-like forces and frozen atoms through extended XYZ."""
    atoms = Atoms('PtH', positions=[[0, 0, 0], [1, 1, 1]],
                  cell=[5, 5, 5], pbc=True)
    if constrained:
        atoms.set_constraint(FixAtoms(indices=[0]))
    atoms.calc = SinglePointCalculator(atoms, energy=ENERGY, forces=RAW_FORCES)
    stream = io.StringIO()
    columns = ['symbols', 'positions', 'forces']
    if constrained:
        columns.append('move_mask')
    write(stream, atoms, format='extxyz', columns=columns)
    stream.seek(0)
    return read(stream, format='extxyz')


class RmseForceConstraintTests(unittest.TestCase):
    def assert_fixed_atom_preserved(self, atoms):
        self.assertEqual(len(atoms.constraints), 1)
        np.testing.assert_array_equal(atoms.constraints[0].get_indices(), [0])
        np.testing.assert_array_equal(atoms.get_forces()[0], [0, 0, 0])
        np.testing.assert_allclose(atoms.get_forces(apply_constraint=False), RAW_FORCES)

    def test_both_references_recover_raw_forces_without_removing_constraints(self):
        for backend in (ASE, TORCHSIM):
            with self.subTest(backend=backend.__name__):
                atoms = reference_frame()
                energy, forces = backend.reference(atoms)
                self.assertEqual(energy, ENERGY)
                np.testing.assert_allclose(forces, RAW_FORCES)
                self.assert_fixed_atom_preserved(atoms)

    def test_perfect_ase_prediction_has_zero_all_atom_force_rmse(self):
        atoms = reference_frame()
        original_calculator = atoms.calc
        calculator = SinglePointCalculator(atoms, energy=ENERGY, forces=RAW_FORCES)

        reference_energy, reference_forces = ASE.reference(atoms)
        energy, forces = ASE.predict(calculator, atoms)

        np.testing.assert_allclose(forces, RAW_FORCES)
        self.assertEqual(energy, reference_energy)
        self.assertEqual(float(np.sqrt(np.mean((forces - reference_forces) ** 2))), 0)
        self.assertIs(atoms.calc, original_calculator)
        self.assert_fixed_atom_preserved(atoms)

    def test_perfect_torchsim_prediction_has_zero_all_atom_force_rmse(self):
        try:
            import torch
        except ImportError:
            self.skipTest('CPU Torch tensors are required for the TorchSim prediction test')
        atoms = reference_frame()
        torch_sim = ModuleType('torch_sim')

        def initialize_state(frame, device, dtype):
            self.assertIs(frame, atoms)
            self.assertEqual(device, torch.device('cpu'))
            self.assertEqual(dtype, torch.float64)
            return frame

        torch_sim.initialize_state = initialize_state

        def perfect_model(state):
            self.assertIs(state, atoms)
            return {'energy': torch.tensor([ENERGY], dtype=torch.float64),
                    'forces': torch.tensor(RAW_FORCES, dtype=torch.float64)}

        reference_energy, reference_forces = TORCHSIM.reference(atoms)
        with patch.dict(sys.modules, {'torch_sim': torch_sim}):
            energy, forces = TORCHSIM.predict(
                perfect_model, atoms, torch.device('cpu'), torch.float64,
            )

        self.assertEqual(energy, reference_energy)
        self.assertEqual(float(np.sqrt(np.mean((forces - reference_forces) ** 2))), 0)
        self.assert_fixed_atom_preserved(atoms)

    def test_explicit_reference_tags_keep_priority_and_raw_fixed_atom_forces(self):
        for backend in (ASE, TORCHSIM):
            with self.subTest(backend=backend.__name__):
                atoms = reference_frame()
                atoms.info['REF_energy'] = ENERGY - 1
                explicit_forces = RAW_FORCES * 2
                atoms.new_array('REF_forces', explicit_forces)

                energy, forces = backend.reference(atoms)

                self.assertEqual(energy, ENERGY - 1)
                np.testing.assert_allclose(forces, explicit_forces)
                self.assert_fixed_atom_preserved(atoms)

    def test_unconstrained_reference_and_ase_prediction_are_unchanged(self):
        for backend in (ASE, TORCHSIM):
            with self.subTest(backend=backend.__name__):
                atoms = reference_frame(constrained=False)
                reference_energy, reference_forces = backend.reference(atoms)
                energy, forces = ASE.predict(atoms.calc, atoms)

                self.assertEqual(reference_energy, ENERGY)
                self.assertEqual(energy, ENERGY)
                np.testing.assert_allclose(reference_forces, RAW_FORCES)
                np.testing.assert_allclose(forces, RAW_FORCES)
                self.assertEqual(atoms.constraints, [])


if __name__ == '__main__':
    unittest.main()
