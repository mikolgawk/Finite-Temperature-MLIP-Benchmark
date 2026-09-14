# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tqdm>=4.66",
#   "h5py>=3.11", "numpy>=1.26",
#   "mattersim",
#   "ase>=3.26",
#   "torch==2.11.0+cu128",
#   "torchvision==0.26.0+cu128",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
# mattersim = { git = "https://github.com/microsoft/mattersim.git", rev = "df68847af1d66c6725bfadc58c4e3d3ec705bc5f" }
# ///

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ase_rmse import REPO, early_cli, run_rmse
if __name__ == "__main__" and "--compile-force-only" not in sys.argv[1:]:
    early_cli(__file__)

"""Native ASE energy/force-only counterpart to torchsim-scripts/md_mattersim_v1_5M.py.

Run: uv run ase-scripts/md_mattersim_v1_5M.py
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










"""Uncompiled PyTorch loaders for the separate eager MD runners."""


def assert_eager_module(model):
    """Reject TorchScript and torch.compile modules, including nested modules."""
    import torch
    from torch._dynamo.eval_frame import OptimizedModule

    for name, module in model.named_modules():
        if isinstance(module, (torch.jit.ScriptModule, OptimizedModule)):
            raise RuntimeError(f"Compiled module in eager model: {name or '<root>'}")
        if getattr(module, "_compiled_call_impl", None) is not None:
            raise RuntimeError(f"Compiled call in eager model: {name or '<root>'}")



def make_calculator():
    from mattersim.forcefield import MatterSimCalculator
    calculator = MatterSimCalculator(
        load_path="MatterSim-v1.0.0-5M.pth", device="cuda", compute_stress=False,
    )
    assert_eager_module(calculator.potential.model)
    calculator.implemented_properties = ["energy", "free_energy", "forces"]
    return calculator

if __name__ == "__main__":
    run_rmse("mattersim-v1-5M", make_calculator, "ase+mattersim-eager-force-only")
