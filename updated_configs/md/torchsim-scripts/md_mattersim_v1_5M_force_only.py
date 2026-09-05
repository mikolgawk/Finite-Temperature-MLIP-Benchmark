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
"""Force-only eager TorchSim NVT production MD — mattersim-v1-5M.

Per-system MD parameters come from data/ref-trajs/md_metadata.json, which
records how each reference AIMD was run. The trajectory is saved as HDF5
with positions and velocities:

    <OUT_ROOT>/<system>/nvt_mattersim-v1-5M-force-only.h5

Run:  uv run md_mattersim_v1_5M_force_only.py
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

from mattersim.forcefield import Potential
from mattersim.torchsim.graph_construction import build_graph_from_simstate
from mattersim.torchsim.torchsim_wrapper import TorchSimWrapper
from torchsim_md_runner import run_torchsim_md


class ForceOnlyTorchSimWrapper(TorchSimWrapper):
    """Run eager MatterSim with the same energy/force workload as force-only AOTI."""

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
MODEL_NAME = "mattersim-v1-5M-force-only"
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
# Eager checkpoint with stress disabled. This uses the same MatterSim revision
# and energy/force workload as the force-only AOTI runner.
potential = Potential.from_checkpoint(
    load_path="MatterSim-v1.0.0-5M.pth", model_name="m3gnet",
    device=device, load_training_state=False,
)
model = ForceOnlyTorchSimWrapper(model=potential, device=device)

run_torchsim_md(MODEL_NAME, model, "torch-sim-0.6.1+mattersim-eager-force-only")
