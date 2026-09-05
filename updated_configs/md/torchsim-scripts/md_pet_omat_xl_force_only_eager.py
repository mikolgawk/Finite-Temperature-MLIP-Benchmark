# /// script
# requires-python = ">=3.12"
# dependencies = [
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
"""Eager force-only TorchSim NVT production MD — pet-omat-xl.

Per-system MD parameters come from data/ref-trajs/md_metadata.json, which
records how each reference AIMD was run. The trajectory is saved as HDF5
with positions and velocities:

    <OUT_ROOT>/<system>/nvt_pet-omat-xl-force-only-eager.h5

Run:  uv run md_pet_omat_xl_force_only_eager.py
"""

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
from eager_models import load_eager_pet
from torchsim_md_runner import run_torchsim_md

from torch_sim.models.metatomic import MetatomicModel
import metatomic_torchsim._neighbors as metatomic_neighbors

def main():
    # settings
    MODEL_NAME = "pet-omat-xl-force-only-eager"
    HERE = Path(__file__).resolve().parent
    REPO = HERE.parent.parent
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
    # PET, xl size, OMat-only checkpoint (PBE, not MP-consistent), latest version as in
    # the ASE benchmark
    model = MetatomicModel(
        model=load_eager_pet(model="pet-omat", size="xl", version="1.0.0"),
        device=device, compute_stress=False,
    )

    run_torchsim_md(MODEL_NAME, model, "torch-sim-0.6.1+pet-eager-force-only")


if __name__ == "__main__":
    main()
