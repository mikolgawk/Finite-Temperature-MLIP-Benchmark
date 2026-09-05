# /// script
# requires-python = ">=3.12,<3.13"  # torch 2.4.1+cu124 supplies wheels only through CPython 3.12
# dependencies = [
#   "torch-sim-atomistic==0.5.2",  # 0.6.1 needs torch>=2.8, fairchem 1.x pins <2.5;
#                                  # 0.5.2 has the same integrate API incl. nvt_vrescale
#   "fairchem-core==1.10.0",       # v1: only API that loads this legacy checkpoint
#   "torch==2.4.1",
#   "torch-scatter",               # runtime imports of fairchem v1, undeclared there
#   "torch-sparse",
#   "scipy<1.17",                # fairchem-core 1.10 imports the removed scipy.special.sph_harm
#   "ase>=3.26",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# explicit = true
#
# [[tool.uv.index]]
# name = "pyg-cu124"
# url = "https://data.pyg.org/whl/torch-2.4.0+cu124.html"
# format = "flat"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu124" }
# torch-scatter = { index = "pyg-cu124" }
# torch-sparse = { index = "pyg-cu124" }
# ///
"""Force-only eager TorchSim MD for eSEN-30M-OAM; original checkpoint weights."""

from pathlib import Path

import torch

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

from fairchem_force_only import load_force_only_legacy
from torchsim_md_runner import run_torchsim_md


def main():
    checkpoint = Path(__file__).resolve().parent.parent / "data" / "models" / "esen_30m_oam.pt"
    model = load_force_only_legacy(checkpoint, "esen", torch.device("cuda"))
    run_torchsim_md(
        "eSEN-30M-OAM-force-only", model,
        "torch-sim-0.5.2+fairchem-1.10.0-force-only",
    )


if __name__ == "__main__":
    main()
