# /// script
# requires-python = ">=3.12"
# dependencies = ["ase>=3.26", "h5py>=3.11", "numpy>=1.26", "orb-models==0.6.2", "torch==2.11.0+cu128", "tqdm>=4.66"]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""ASE counterpart to torchsim-scripts/md_orb_v3_omat.py."""

from _ase_md import run_ase_md


import torch

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False


def make_calculator():
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    model, adapter = pretrained.orb_v3_conservative_inf_omat(
        device="cuda", precision="float32-highest", compile=True,
    )
    model.disable_stress()
    # Preserve grad_forces: ORBCalculator maps it to ASE's forces.
    calculator = ORBCalculator(model, atoms_adapter=adapter, device="cuda")
    return calculator


if __name__ == "__main__":
    run_ase_md("orb-v3-omat-force-only", make_calculator,
               "ase+orb+compile+force-only", filename_suffix="")
