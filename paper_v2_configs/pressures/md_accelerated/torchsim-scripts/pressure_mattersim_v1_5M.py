# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas>=2.2",
#   "h5py>=3.11",
#   "torch-sim-atomistic[mattersim]==0.6.1",
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
_PRESSURE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "pressures")
sys.path.insert(0, str(_PRESSURE_ROOT))
from pressure_evaluator import early_cli, run_torchsim_pressure
early_cli(__file__, "torchsim")

"""Stress-enabled md_mattersim_v1_5M.py: record potential stress for every saved trajectory frame."""

FAILED_SYSTEMS = []

import csv
import json
import os
import time
from pathlib import Path

import torch


def frame_stress(state, model):
    """Record one finite potential stress tensor per reported system."""
    stress = model(state)["stress"].detach()
    if stress.numel() != state.n_systems * 9 or not bool(torch.isfinite(stress).all()):
        raise ValueError("Model returned an invalid per-frame stress tensor")
    return stress.reshape(state.n_systems, 3, 3)


# TorchInductor's generated C++ precompiled header includes CUDA headers without
# always adding the wheel's CUDA runtime include directory. Make that directory
# visible before MatterSim invokes the in-process compiler.
CUDA_HOME = Path(os.environ.get("MATTERSIM_CUDA_HOME", "/usr/local/cuda")).resolve()
_site_packages = Path(torch.__file__).resolve().parent.parent
_wheel_cuda_include = _site_packages / "nvidia" / "cuda_runtime" / "include"
_cuda_include = (
    _wheel_cuda_include
    if (_wheel_cuda_include / "cuda_fp8.h").is_file()
    else CUDA_HOME / "include"
)
if not (_cuda_include / "cuda_fp8.h").is_file():
    raise FileNotFoundError(
        f"CUDA headers not found under {_cuda_include}; set MATTERSIM_CUDA_HOME "
        "to a complete CUDA toolkit"
    )
os.environ["CUDA_HOME"] = str(CUDA_HOME)
os.environ["CUDA_PATH"] = str(CUDA_HOME)
os.environ["CUDAToolkit_ROOT"] = str(CUDA_HOME)
os.environ["CPATH"] = os.pathsep.join(
    value
    for value in (
        str(_cuda_include),
        str(CUDA_HOME / "include"),
        os.environ.get("CPATH", ""),
    )
    if value
)

# no TF32, no autotuned kernels
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

import torch_sim as ts
from ase.io import read

from mattersim.forcefield import Potential
from mattersim.torchsim.graph_construction import build_graph_from_simstate
from mattersim.torchsim.torchsim_wrapper import TorchSimWrapper


class WithStressTorchSimWrapper(TorchSimWrapper):
    """Use native MatterSim stress conversion from GPa to eV/A^3."""
    pass


# settings
MODEL_NAME = "mattersim-v1-5M-compile-stress"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
METADATA_FILE = REPO / "paper_v2_configs" /  "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "paper_v2_configs" / "data" / "mlip-trajs-torchsim-accelerated-stress"

CHAIN_LENGTH = 1              # Nose-Hoover chain settings, as in the ASE benchmark
CHAIN_STEPS = 1
SY_STEPS = 3
SEED = 42                     # Maxwell-Boltzmann velocity seed
STATE_DTYPE = torch.float32

device = torch.device("cuda")

# model
# Checkpoint name resolved/downloaded by MatterSim itself, same as the ASE
# benchmark. Compile the M3GNet forward pass before wrapping it for TorchSim.
# system pays the one-time TorchInductor compilation cost. This revision includes
# the self-image fix from merged PR #164.
potential = Potential.from_checkpoint(
    load_path="MatterSim-v1.0.0-5M.pth", model_name="m3gnet",
    device=device, load_training_state=False,
)
potential.model.forward = torch.compile(potential.model.forward)
model = WithStressTorchSimWrapper(
    model=potential,
    device=device,
)

# NVT MD loop, one entry per system in the metadata file
METADATA = json.loads(METADATA_FILE.read_text())


run_torchsim_pressure(MODEL_NAME, model, "torchsim-stress-postprocessing")
