# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "ase>=3.26", "h5py>=3.11", "numpy>=1.26,<2",
#   "tensorpotential==0.5.7.2", "nvidia-cuda-nvcc-cu12==12.8.*",
#   "tf-keras==2.19.*", "torch",
# ]
# [tool.uv]
# override-dependencies = ["tensorflow>=2.17,<2.20"]
# ///
"""Stress-enabled md_grace_mp.py: record potential stress for every saved trajectory frame."""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
if os.environ["TF_ENABLE_ONEDNN_OPTS"] != "0":
    raise RuntimeError("GRACE requires TF_ENABLE_ONEDNN_OPTS=0")

from pathlib import Path
import torch
from _ase_md import REPO, run_ase_md



from ase.calculators.calculator import Calculator, all_changes
from tensorpotential.calculator import TPCalculator

class WithStressTPCalculator(TPCalculator):
    """Native neighbor construction with explicit E/F/virial conversion."""
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy", "forces", "stress"), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.get_data(self.atoms)
        output = self.stress_compute(self.data)
        energy = float(output["total_energy"].numpy().reshape(-1)[0])
        forces = output["total_f"].numpy()[:len(self.atoms)]
        # TensorPotential virial order: xx yy zz xy xz yz; ASE: xx yy zz yz xz xy.
        virial = output["virial"].numpy().reshape(-1)
        stress = -virial[[0, 1, 2, 5, 4, 3]] / self.atoms.get_volume()
        self.results = {"energy": energy, "free_energy": energy, "forces": forces, "stress": stress}





def make_calculator():
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from tensorpotential.calculator.foundation_models import grace_fm
    original = grace_fm("GRACE-2L-MP-r6", float_dtype="float64")
    # Retain the native loader's data builders and replace its calculation method.
    original.__class__ = WithStressTPCalculator
    calculator = original
    if len(calculator.models) != 1:
        raise RuntimeError("Expected one GRACE-MP model")
    calculator.stress_compute = calculator.models[0].compute
    calculator.compute_properties = ["energy", "forces", "stress"]
    return calculator

if __name__ == "__main__":
    run_ase_md("grace-mp", make_calculator, "ase+tensorpotential-xla-stress")
