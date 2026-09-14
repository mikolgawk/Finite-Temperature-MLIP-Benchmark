# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas>=2.2",
#   "h5py>=3.11",
#   "torch-sim-atomistic==0.6.1",
#   "fairchem-core",
#   "ase>=3.26",
# ]
# ///

import sys
from pathlib import Path
_PRESSURE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "pressures")
sys.path.insert(0, str(_PRESSURE_ROOT))
from pressure_evaluator import early_cli, run_torchsim_pressure
early_cli(__file__, "torchsim")

"""Stress-enabled md_uma_s_omat_turbo.py: record potential stress for every saved trajectory frame."""

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


# FairChem temporarily configures TF32 inside the UMA predictor for each mode.
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

import torch_sim as ts
from ase.io import read

from torch_sim.models.fairchem import FairChemModel

def retain_uma_stress(calculator):
    """Preserve the native energy/force/stress model before compilation."""
    return calculator


# settings
INFERENCE_MODE = "turbo"
if INFERENCE_MODE not in {"compile", "turbo"}:
    raise ValueError(f"Unsupported UMA inference mode: {INFERENCE_MODE!r}")

MODEL_NAME = f"uma-s-omat-{INFERENCE_MODE}-stress"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
METADATA_FILE = REPO / "paper_v2_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "paper_v2_configs" / "data" / "mlip-trajs-torchsim-accelerated-stress"

CHAIN_LENGTH = 1              # Nose-Hoover chain settings, as in the ASE benchmark
CHAIN_STEPS = 1
SY_STEPS = 3
SEED = 42                     # Maxwell-Boltzmann velocity seed
STATE_DTYPE = torch.float32

device = torch.device("cuda")

# model
# HF-gated checkpoint: run `huggingface-cli login` first
# fairchem's torch-sim wrapper has no inference_settings passthrough; it calls
# pretrained_mlip.get_predict_unit(...) internally, so inject the named mode at
# the factory.
from fairchem.core import pretrained_mlip
from fairchem.core.units.mlip_unit import InferenceSettings

FAIRCHEM_INFERENCE_SETTINGS = (
    InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=False,
        compile=True,
    )
    if INFERENCE_MODE == "compile"
    else "turbo"
)

_get_predict_unit = pretrained_mlip.get_predict_unit


def _get_predict_unit_with_mode(model_name, **kwargs):
    kwargs.setdefault("inference_settings", FAIRCHEM_INFERENCE_SETTINGS)
    return _get_predict_unit(model_name, **kwargs)


pretrained_mlip.get_predict_unit = _get_predict_unit_with_mode

model = FairChemModel(
    model="uma-s-1p1", task_name="omat", device=device, dtype=torch.float32,
    compute_stress=True,
)

# Remove stress tasks and strain derivatives before lazy compilation/merging.
retain_uma_stress(model)

# FairChem 2.22 prepares merged MOLE experts before its normal device move.
# TorchSim's first state is already on CUDA, so turbo must move the predictor
# early to keep embedding weights and indices on the same device during merging.
if INFERENCE_MODE == "turbo":
    model.predictor.move_to_device()


# NVT MD loop, one entry per system in the metadata file
METADATA = json.loads(METADATA_FILE.read_text())


run_torchsim_pressure(MODEL_NAME, model, "torchsim-stress-postprocessing")
