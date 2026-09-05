# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch-sim-atomistic[orb]==0.6.1",
#   "orb-models==0.6.2",
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
"""Force-only eager TorchSim NVT production MD using ORB v2."""

import torch

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

from orb_models.forcefield import pretrained
from torch_sim.models.orb import OrbModel

from torchsim_md_runner import run_torchsim_md
from eager_models import assert_eager_module


def main():
    device = torch.device("cuda")
    orb_ff, atoms_adapter = pretrained.orb_v2(
        device=device, precision="float32-highest", compile=False
    )
    # This disables the stress head/gradient inside ORB, before inference.
    orb_ff.disable_stress()
    assert_eager_module(orb_ff)
    model = OrbModel(orb_ff, atoms_adapter, device=device)

    run_torchsim_md(
        "orb-v2-force-only-eager", model, "torch-sim-0.6.1+orb-models-0.6.2-force-only-eager"
    )


if __name__ == "__main__":
    main()
