# /// script
# requires-python = ">=3.12"
# dependencies = ["ase>=3.26", "h5py>=3.11", "numpy>=1.26", "torch", "upet==0.2.6"]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ase_rmse import REPO, early_cli, run_rmse
if __name__ == "__main__" and "--compile-force-only" not in sys.argv[1:]:
    early_cli(__file__)

"""Native ASE energy/force-only counterpart to torchsim-scripts/md_pet_omat_xl_force_only_compile.py.

Run: uv run ase-scripts/md_pet_omat_xl_force_only_compile.py
"""

from pathlib import Path
import torch



def make_calculator():
    from metatomic_ase import MetatomicCalculator
    from upet._models import _get_upet_exported_atomistic_model
    from huggingface_hub import hf_hub_download
    checkpoint = hf_hub_download("lab-cosmo/upet", "models/pet-omat-xl-v1.0.0.ckpt")
    model = _get_upet_exported_atomistic_model(
        model="pet-omat", size="xl", version="1.0.0", checkpoint_path=checkpoint
    )
    model = model.to(device="cuda", dtype=torch.float32)
    model._capabilities.dtype = "float32"
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = torch.jit.script(model)
    calculator = MetatomicCalculator(model, device="cuda")
    # Metatomic differentiates the cell only when stress is requested.
    calculator.implemented_properties = ["energy", "free_energy", "forces"]
    return calculator

if __name__ == "__main__":
    run_rmse("pet-omat-xl", make_calculator, "ase+upet-0.2.6-torchscript-force-only")
