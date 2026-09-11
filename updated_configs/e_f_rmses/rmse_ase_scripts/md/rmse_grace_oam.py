# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "tqdm>=4.66",
#   "ase>=3.26", "h5py>=3.11", "numpy>=1.26,<2",
#   "tensorpotential==0.5.7.2", "nvidia-cuda-nvcc-cu12==12.8.*",
#   "tf-keras==2.19.*", "torch",
# ]
# [tool.uv]
# override-dependencies = ["tensorflow>=2.17,<2.20"]
# ///

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ase_rmse import REPO, early_cli, run_rmse
if __name__ == "__main__" and "--compile-force-only" not in sys.argv[1:]:
    early_cli(__file__)

"""Native ASE energy/force-only counterpart to torchsim-scripts/md_grace_oam.py.

Run: uv run ase-scripts/md_grace_oam.py
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
if os.environ["TF_ENABLE_ONEDNN_OPTS"] != "0":
    raise RuntimeError("GRACE requires TF_ENABLE_ONEDNN_OPTS=0")

"""ASE RMSE evaluator derived from the matching MD calculator."""

import csv
import json
import time
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import torch
from ase import units
from ase.io import read


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-matched"

CHAIN_LENGTH = 1
CHAIN_STEPS = 1
SEED = 42


torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False










from ase.calculators.calculator import Calculator, all_changes
from tensorpotential.calculator import TPCalculator

class ForceOnlyTPCalculator(TPCalculator):
    """Keep native TensorPotential neighbor construction, evaluate E/F only."""
    implemented_properties = ["energy", "free_energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.get_data(self.atoms)
        output = self.force_compute(self.data)
        energy = float(output["total_energy"].numpy().reshape(-1)[0])
        forces = output["total_f"].numpy()[:len(self.atoms)]
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}

"""Reconstruct GRACE Python instructions for genuine TensorFlow eager inference.

This does not invoke a SavedModel graph, tf.function, or XLA during inference.
GRACE-MP deliberately remains on its stress-pruned XLA SavedModel path because
its public artifact lacks the Python model definition required here.
"""

import os
from pathlib import Path


def load_eager_grace(family):
    import tensorflow as tf
    from tensorpotential.instructions.base import load_instructions
    from tensorpotential.tpmodel import TPModel

    if family != "grace_oam":
        raise ValueError("Only GRACE-OAM has the artifacts required for eager inference")
    artifact = "GRACE-2L-OMAT-large-ft-AM"
    cache = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache/grace"))
    root = cache / "checkpoints" / artifact
    potential = Path(os.environ.get("GRACE_OAM_POTENTIAL", root / "model-single.yaml"))
    checkpoint = Path(os.environ.get(
        "GRACE_OAM_CHECKPOINT", root / "checkpoint-single-fp32",
    ))
    if not potential.is_file() or not Path(str(checkpoint) + ".index").is_file():
        raise FileNotFoundError(
            f"True eager {artifact} needs its Python instruction YAML and matching "
            f"fp32 checkpoint. Set GRACE_OAM_POTENTIAL and GRACE_OAM_CHECKPOINT "
            f"(checkpoint prefix without .index). Looked for "
            f"{potential} and {checkpoint}.index. A SavedModel is not an eager fallback."
        )
    model = TPModel(load_instructions(str(potential)))
    model.build(tf.float32)
    status = tf.train.Checkpoint(model=model).read(str(checkpoint))
    status.assert_existing_objects_matched()
    status.expect_partial()  # Training step/optimizer are intentionally unused.

    def compute(input_data):
        from tensorpotential import constants as C
        from tensorpotential.tpmodel import execute_instructions

        if not tf.executing_eagerly():
            raise RuntimeError("GRACE eager inference was called inside a traced graph")
        data = dict(input_data)
        # Match TensorPotential's single-structure force convention exactly,
        # but never construct its virial reduction.
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(data[C.BOND_VECTOR])
            execute_instructions(data, model.instructions, training=False)
            atomic_energy = tf.reshape(data[C.PREDICT_ATOMIC_ENERGY], [-1, 1])
        pair_forces = -tape.gradient(atomic_energy, data[C.BOND_VECTOR])
        n_atoms = tf.reshape(data[C.N_ATOMS_BATCH_TOTAL], [])
        forces = tf.math.unsorted_segment_sum(
            pair_forces, data[C.BOND_IND_J], n_atoms
        ) - tf.math.unsorted_segment_sum(pair_forces, data[C.BOND_IND_I], n_atoms)
        return {
            "total_energy": tf.reduce_sum(atomic_energy, axis=0, keepdims=True),
            "total_f": forces,
        }

    return compute


def make_calculator():
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    model_path = REPO / "updated_configs/data/models/GRACE-2L-OMAT-large-ft-AM-fp32"
    if not model_path.is_dir():
        raise FileNotFoundError(f"Converted GRACE metadata artifact missing: {model_path}")
    calculator = ForceOnlyTPCalculator(str(model_path), float_dtype="float32")
    calculator.force_compute = load_eager_grace("grace_oam")
    calculator.compute_properties = ["energy", "forces"]
    return calculator

if __name__ == "__main__":
    run_rmse("grace-oam", make_calculator, "ase+tensorpotential-force-only")
