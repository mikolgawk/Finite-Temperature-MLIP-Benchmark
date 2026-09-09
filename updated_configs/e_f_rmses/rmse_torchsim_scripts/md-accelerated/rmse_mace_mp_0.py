# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch-sim-atomistic[mace,vesin]==0.6.1",
#   "ase>=3.26",
#   "cuequivariance-torch>=0.4",
#   "cuequivariance-ops-torch-cu12",
# ]
# ///
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchsim_rmse import run_rmse, early_cli
early_cli(__file__)

"""Evaluate reference energy/force RMSE with the matching TorchSim MD model."""

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

# settings
MODEL_NAME = "mace-mp-0-compile"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-accelerated"

CHAIN_LENGTH = 1              # Nose-Hoover chain settings, as in the ASE benchmark
CHAIN_STEPS = 1
SY_STEPS = 3
ACCELERATION = True           # cuEquivariance fused CUDA kernels for MACE
COMPILE_MODE = "reduce-overhead"  # torch.compile + CUDA graphs around the CuEq model
SEED = 42                     # Maxwell-Boltzmann velocity seed
STATE_DTYPE = torch.float32

device = torch.device("cuda")

# model
# wrapper dtype defaults to fp64, so set fp32 explicitly
model = MaceModel(
    model=mace_mp(model="medium", return_raw_model=True, default_dtype="float32"),
    device=device, dtype=torch.float32, compute_forces=True, compute_stress=False,
    enable_cueq=ACCELERATION,
    compile_mode=COMPILE_MODE,
    neighbor_list_fn=vesin_nl_ts,  # robust for dense systems such as H at 1050 K
)


# NVT MD loop, one entry per system in the metadata file
run_rmse(MODEL_NAME, model, state_dtype=STATE_DTYPE)
