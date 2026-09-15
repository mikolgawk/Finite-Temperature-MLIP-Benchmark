# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tqdm>=4.66",
#   "pandas>=2.2",
#   "h5py>=3.11",
#   "torch-sim-atomistic==0.6.1",
#   "orb-models",
#   "ase>=3.26",
# ]
# ///

import sys
from pathlib import Path
_PRESSURE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "pressures")
sys.path.insert(0, str(_PRESSURE_ROOT))
from pressure_evaluator import early_cli, run_torchsim_pressure
early_cli(__file__, "torchsim")

"""Stress-enabled md_orb_v3_direct.py: record potential stress for every saved trajectory frame."""

FAILED_SYSTEMS = []

import csv
import json
import time
from pathlib import Path

import torch


def frame_stress(state, model):
    """Record one finite potential stress tensor per reported system."""
    stress = model(state)["stress"].detach()
    if stress.numel() != state.n_systems * 9 or not bool(torch.isfinite(stress).all()):
        raise ValueError("Model returned an invalid per-frame stress tensor")
    return stress.reshape(state.n_systems, 3, 3)


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
MODEL_NAME = "orb-v3-direct-stress"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
METADATA_FILE = REPO / "paper_v2_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "paper_v2_configs" / "data" / "mlip-trajs-torchsim-accelerated-stress"

CHAIN_LENGTH = 1              # Nose-Hoover chain settings, as in the ASE benchmark
CHAIN_STEPS = 1
SY_STEPS = 3
ACCELERATION = True           # torch.compile kernel fusion; same fp32 numerics
SEED = 42                     # Maxwell-Boltzmann velocity seed
STATE_DTYPE = torch.float32

device = torch.device("cuda")

# model
# direct-force variant
orb_ff, adapter = pretrained.orb_v3_direct_20_mpa(
    device=device,
    precision="float32-highest",   # always true fp32 matmuls, never TF32
    compile=ACCELERATION,
)

model = OrbModel(orb_ff, adapter, device=device, dtype=torch.float32)


# NVT MD loop, one entry per system in the metadata file
METADATA = json.loads(METADATA_FILE.read_text())


run_torchsim_pressure(MODEL_NAME, model, "torchsim-stress-postprocessing")
