# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["ase>=3.26", "h5py>=3.11", "mace-torch==0.3.16", "numpy>=1.26", "torch"]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ase_rmse import REPO, early_cli, run_rmse
if __name__ == "__main__" and "--compile-force-only" not in sys.argv[1:]:
    early_cli(__file__)

"""Native ASE energy/force-only counterpart to torchsim-scripts/md_mace_mh_omat.py.

Run: uv run ase-scripts/md_mace_mh_omat.py
"""

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











def disable_mace_stress(calculator):
    """Override MACE's unconditional stress request before model evaluation."""
    def force_only(module, args, kwargs):
        kwargs.update(compute_stress=False, compute_virials=False,
                      compute_displacement=False, compute_atomic_stresses=False,
                      compute_edge_forces=False)
        return args, kwargs
    if calculator.use_compile:
        raise RuntimeError("Expected eager MACE")
    for model in calculator.models:
        # Match the TorchSim MACE loader: e3nn may contain scripted kernels.
        if isinstance(model, torch.jit.ScriptModule):
            raise RuntimeError("Expected a Python MACE model")
        model.register_forward_pre_hook(force_only, with_kwargs=True)
    calculator.implemented_properties = ["energy", "free_energy", "forces"]
    return calculator


def make_calculator():
    from mace.calculators import mace_mp

    return disable_mace_stress(mace_mp(
        model="mh-1", default_dtype="float32", device="cuda", head="omat_pbe"
    ))


if __name__ == "__main__":
    run_rmse("mace-mh-omat", make_calculator, "ase+mace-torch-0.3.16")
