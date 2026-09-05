# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch-sim-atomistic[mace,vesin]==0.6.1",
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
"""TorchSim NVT production MD — mace-mp-0.

Per-system MD parameters come from data/ref-trajs/md_metadata.json, which
records how each reference AIMD was run. The trajectory is saved as HDF5
with positions and velocities:

    <OUT_ROOT>/<system>/nvt_mace-mp-0.h5

Run:  uv run md_mace_mp_0.py
"""

import csv
import json
import time
from pathlib import Path

import torch

# no TF32, no autotuned kernels
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

import torch_sim as ts
from ase.io import read

from mace.calculators import mace_mp
from torch_sim.models.mace import MaceModel
from torch_sim.neighbors import vesin_nl_ts
from torchsim_md_runner import run_torchsim_md

# settings
MODEL_NAME = "mace-mp-0"
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

# model
model = MaceModel(
    model=mace_mp(model="medium", return_raw_model=True, default_dtype="float32"),
    device=device, dtype=torch.float32, compute_forces=True, compute_stress=False,
    neighbor_list_fn=vesin_nl_ts,  # default NVIDIA NL overflows on dense fluids like H at 1050 K
)

run_torchsim_md(MODEL_NAME, model, "torch-sim-0.6.1")
