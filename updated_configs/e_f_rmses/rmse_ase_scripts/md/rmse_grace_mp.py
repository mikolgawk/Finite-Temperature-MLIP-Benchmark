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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ase_rmse import REPO, early_cli, run_rmse
if __name__ == "__main__" and "--compile-force-only" not in sys.argv[1:]:
    early_cli(__file__)

"""Native ASE energy/force-only counterpart to torchsim-scripts/md_grace_mp.py.

Run: uv run ase-scripts/md_grace_mp.py
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

"""Remove GRACE's virial branch from a loaded TensorFlow SavedModel graph."""


def prune_grace_stress(restored_compute):
    """Return a callable that computes only total energy and atomic forces.

    GRACE SavedModels expose one XLA-compiled ``compute`` function whose normal
    output dictionary also contains virial, atomic-energy, and pair-force
    tensors. Selecting two entries after calling that function is too late to
    avoid those output branches. TensorFlow's graph-pruning machinery instead
    traces backwards from just the two outputs required by NVT MD.

    This uses TensorFlow internals, so fail loudly if the pinned TensorFlow API
    or the GRACE artifact layout changes rather than silently benchmarking the
    unpruned graph.
    """
    import tensorflow as tf
    from tensorflow.python.eager.wrap_function import VariableHolder, WrappedFunction

    concrete_functions = restored_compute.concrete_functions
    if len(concrete_functions) != 1:
        raise RuntimeError(
            f"Expected one GRACE compute graph, found {len(concrete_functions)}"
        )
    concrete = concrete_functions[0]
    outputs = tf.nest.pack_sequence_as(concrete.structured_outputs, concrete.outputs)
    required = {"total_energy", "total_f"}
    if not required.issubset(outputs) or "virial" not in outputs:
        raise RuntimeError(
            f"Unexpected GRACE outputs: {sorted(outputs)}"
        )

    # WrappedFunction.prune works for this SavedModel wrapper but expects the
    # holder attribute normally installed by tf.compat.v1.wrap_function. All
    # model variables remain existing external captures; no variables are made.
    concrete._variable_holder = VariableHolder()  # noqa: SLF001
    pruned = WrappedFunction.prune(
        concrete,
        feeds=concrete.inputs,
        fetches={name: outputs[name] for name in sorted(required)},
        name="grace_energy_forces",
        input_signature=concrete.structured_input_signature,
    )
    # prune() does not propagate function attributes. Re-wrap the reduced graph
    # with the original XLA attributes so this remains the same compiled
    # inference workload rather than falling back to uncompiled TensorFlow.
    reduced = pruned
    pruned = WrappedFunction(
        reduced.graph,
        variable_holder=VariableHolder(),
        attrs=dict(concrete.function_def.attr),
    )
    pruned._num_positional_args = reduced._num_positional_args  # noqa: SLF001
    pruned._arg_keywords = reduced._arg_keywords  # noqa: SLF001
    if set(pruned.structured_outputs) != required:
        raise RuntimeError("GRACE force-only graph pruning failed")
    if (concrete.function_def.attr.get("_XlaMustCompile") is not None
            and not pruned.function_def.attr["_XlaMustCompile"].b):
        raise RuntimeError("GRACE force-only graph lost XLA compilation")
    return pruned


def make_calculator():
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from tensorpotential.calculator.foundation_models import grace_fm
    original = grace_fm("GRACE-2L-MP-r6", float_dtype="float64")
    # Retain the native loader's data builders and replace its calculation method.
    original.__class__ = ForceOnlyTPCalculator
    calculator = original
    if len(calculator.models) != 1:
        raise RuntimeError("Expected one GRACE-MP model")
    calculator.force_compute = prune_grace_stress(calculator.models[0].compute)
    calculator.compute_properties = ["energy", "forces"]
    return calculator

if __name__ == "__main__":
    run_rmse("grace-mp", make_calculator, "ase+tensorpotential-force-only")
