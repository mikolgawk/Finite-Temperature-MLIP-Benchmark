# /// script
# requires-python = ">=3.12"
# dependencies = ["tqdm>=4.66", "ase>=3.26", "h5py>=3.11", "numpy>=1.26", "torch", "upet==0.2.6"]
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

"""Native ASE energy/force-only counterpart to torchsim-scripts/md_pet_oam_xl.py.

Run: uv run ase-scripts/md_pet_oam_xl.py
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


def load_eager_pet(*, model, size, version, checkpoint_path=None):
    """Follow UPET 0.2.6's loader but omit its final torch.jit.script call."""
    from upet._models import _get_upet_exported_atomistic_model
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    if checkpoint_path is None:
        filename = f"models/{model}-{size}-v{version}.ckpt"
        cached = try_to_load_from_cache("lab-cosmo/upet", filename)
        checkpoint_path = cached if isinstance(cached, str) else hf_hub_download(
            "lab-cosmo/upet", filename=filename
        )
    model = _get_upet_exported_atomistic_model(
        model=model, size=size, version=version, checkpoint_path=checkpoint_path
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert_eager_module(model)
    return model


def make_calculator():
    from metatomic_ase import MetatomicCalculator
    model = load_eager_pet(model="pet-oam", size="xl", version="1.0.0")
    model = model.to(device="cuda", dtype=torch.float32)
    model._capabilities.dtype = "float32"
    calculator = MetatomicCalculator(model, device="cuda")
    # Metatomic differentiates the cell only when stress is requested.
    calculator.implemented_properties = ["energy", "free_energy", "forces"]
    return calculator

if __name__ == "__main__":
    run_rmse("pet-oam-xl", make_calculator, "ase+upet-0.2.6-eager-force-only")
