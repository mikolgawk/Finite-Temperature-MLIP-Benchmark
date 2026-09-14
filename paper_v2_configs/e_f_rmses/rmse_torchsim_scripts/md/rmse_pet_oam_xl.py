# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tqdm>=4.66",
#   "torch-sim-atomistic[metatomic]==0.6.1",
#   "upet==0.2.6",
#   "metatomic-torchsim==0.1.3",  # 0.1.4 calls vesin's NeighborList(skin=...),
#                                 # a kwarg only added in vesin>=0.6, but 0.1.4
#                                 # itself pins vesin<0.6 - always TypeErrors.
#   "ase>=3.26",
#   "torch",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchsim_rmse import run_rmse, early_cli
early_cli(__file__)

"""Evaluate reference energy/force RMSE with the matching TorchSim MD model."""

import csv
import json
import os
import time
from pathlib import Path

# Vesin reads this at import time.  PET's large cutoff produces 2,098
# neighbors/atom for dense periodic H at 1050 K, above Vesin's 1,000 default.
os.environ["VESIN_CUDA_MAX_PAIRS_PER_POINT"] = "4096"

import torch

# no TF32, no autotuned kernels
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

import torch_sim as ts
from ase.io import read
"""Evaluate reference energy/force RMSE with the matching TorchSim MD model."""


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
"""Evaluate reference energy/force RMSE with the matching TorchSim MD model."""

import csv
import json
import time
from pathlib import Path

import torch
import torch_sim as ts
from ase.io import read


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-matched"

CHAIN_LENGTH = 1
CHAIN_STEPS = 1
SY_STEPS = 3
SEED = 42
STATE_DTYPE = torch.float32



from torch_sim.models.metatomic import MetatomicModel
import metatomic_torchsim._neighbors as metatomic_neighbors

def main():
    # settings
    MODEL_NAME = "pet-oam-xl-force-only-eager"
    HERE = Path(__file__).resolve().parent
    REPO = HERE.parents[3]
    METADATA_FILE = REPO / "updated_configs" /  "data" / "ref-trajs" / "md_metadata.json"
    OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-matched"

    CHAIN_LENGTH = 1              # Nose-Hoover chain settings, as in the ASE benchmark
    CHAIN_STEPS = 1
    SY_STEPS = 3
    SEED = 42                     # Maxwell-Boltzmann velocity seed
    STATE_DTYPE = torch.float32

    device = torch.device("cuda")

    # nvalchemiops' CUDA full-list implementation has a fixed per-atom neighbor
    # capacity (848 in the installed version).  Dense periodic H at 1050 K needs
    # 2,098 neighbors, so use metatomic-torchsim's Vesin fallback instead.
    # This must be set before MetatomicModel constructs its neighbor calculators.
    metatomic_neighbors.HAS_NVALCHEMIOPS = False

    # model
    # PET, xl size, OAM checkpoint (MP-consistent PBE), version pinned as in the ASE benchmark
    model = MetatomicModel(
        model=load_eager_pet(model="pet-oam", size="xl", version="1.0.0"),
        device=device, compute_stress=False,
    )

    run_rmse(MODEL_NAME, model, "torch-sim-0.6.1+pet-eager-force-only")


if __name__ == "__main__":
    main()
