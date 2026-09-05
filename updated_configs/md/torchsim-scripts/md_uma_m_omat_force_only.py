# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch-sim-atomistic[fairchem]==0.6.1",
#   "fairchem-core==2.21.0",
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
"""Force-only TorchSim NVT MD for UMA-M-1p1, OMat task."""

import torch

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

from uma_force_only import load_force_only_uma
from torchsim_md_runner import run_torchsim_md


def main():
    model = load_force_only_uma("uma-m-1p1", torch.device("cuda"))
    run_torchsim_md(
        "uma-m-omat-force-only", model,
        "torch-sim-0.6.1+fairchem-2.21.0-force-only",
    )


if __name__ == "__main__":
    main()
