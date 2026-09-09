# /// script
# requires-python = ">=3.12"
# dependencies = [
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchsim_rmse import run_rmse, early_cli
early_cli(__file__)

"""Evaluate reference energy/force RMSE with the matching TorchSim MD model."""

import csv
import json
import os
import time
from pathlib import Path

import torch

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


class ForceOnlyTorchSimWrapper(TorchSimWrapper):
    """Run MatterSim for TorchSim NVT without computing unused stresses."""

    def __init__(self, model: Potential, *, device: torch.device | str) -> None:
        super().__init__(model=model, device=device)
        self._compute_stress = False
        self.implemented_properties = ["energy", "forces"]

    def forward(self, state: ts.SimState) -> dict[str, torch.Tensor]:
        if state.device != self._device:
            state = state.to(self._device)

        output_dtype = state.dtype
        graph_input = build_graph_from_simstate(
            state,
            twobody_cutoff=self.two_body_cutoff,
            threebody_cutoff=self.three_body_cutoff,
            max_num_neighbors_threshold=self._max_neighbors,
        )
        if output_dtype != self._model_dtype:
            graph_input = {
                key: value.to(dtype=self._model_dtype)
                if value.is_floating_point()
                else value
                for key, value in graph_input.items()
            }

        result = self.model.forward(
            graph_input,
            include_forces=True,
            include_stresses=False,
        )
        return {
            "energy": result["total_energy"].to(dtype=output_dtype).detach(),
            "forces": result["forces"].to(dtype=output_dtype).detach().reshape(-1, 3),
        }


# settings
MODEL_NAME = "mattersim-v1-5M-compile-force-only"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
METADATA_FILE = REPO / "updated_configs" /  "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-accelerated"

CHAIN_LENGTH = 1              # Nose-Hoover chain settings, as in the ASE benchmark
CHAIN_STEPS = 1
SY_STEPS = 3
SEED = 42                     # Maxwell-Boltzmann velocity seed
STATE_DTYPE = torch.float32

device = torch.device("cuda")

# model
# Checkpoint name resolved/downloaded by MatterSim itself, same as the ASE
# benchmark. Compile the M3GNet forward pass before wrapping it for TorchSim.
# Stress is intentionally omitted because NVT does not consume it. The first
# system pays the one-time TorchInductor compilation cost. This revision includes
# the self-image fix from merged PR #164.
potential = Potential.from_checkpoint(
    load_path="MatterSim-v1.0.0-5M.pth", model_name="m3gnet",
    device=device, load_training_state=False,
)
potential.model.forward = torch.compile(potential.model.forward)
model = ForceOnlyTorchSimWrapper(
    model=potential,
    device=device,
)

# NVT MD loop, one entry per system in the metadata file
run_rmse(MODEL_NAME, model, state_dtype=STATE_DTYPE)
