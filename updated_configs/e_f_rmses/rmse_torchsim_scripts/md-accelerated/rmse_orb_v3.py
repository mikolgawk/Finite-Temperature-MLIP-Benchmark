# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tqdm>=4.66",
#   "torch-sim-atomistic==0.6.1",
#   "orb-models",
#   "ase>=3.26",
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

from orb_models.forcefield import pretrained
from torch_sim.models.orb import OrbModel

# settings
MODEL_NAME = "orb-v3-force-only"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-accelerated"

CHAIN_LENGTH = 1              # Nose-Hoover chain settings, as in the ASE benchmark
CHAIN_STEPS = 1
SY_STEPS = 3
ACCELERATION = True           # torch.compile kernel fusion; same fp32 numerics
SEED = 42                     # Maxwell-Boltzmann velocity seed
STATE_DTYPE = torch.float32

device = torch.device("cuda")

# model
# conservative variant
orb_ff, adapter = pretrained.orb_v3_conservative_inf_mpa(
    device=device,
    precision="float32-highest",   # always true fp32 matmuls, never TF32
    compile=ACCELERATION,
)
orb_ff.disable_stress()
model = OrbModel(orb_ff, adapter, device=device, dtype=torch.float32)


# NVT MD loop, one entry per system in the metadata file
run_rmse(MODEL_NAME, model, state_dtype=STATE_DTYPE)
