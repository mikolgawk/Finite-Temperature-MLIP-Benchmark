# /// script
# requires-python = ">=3.12"
# dependencies = ["ase>=3.26", "h5py>=3.11", "numpy>=1.26", "orb-models==0.6.2", "torch==2.11.0+cu128", "pandas>=2.2", "tqdm>=4.66"]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///

import sys
from pathlib import Path
_PRESSURE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "pressures")
sys.path.insert(0, str(_PRESSURE_ROOT))
from pressure_evaluator import early_cli, run_ase_pressure
early_cli(__file__, "ase", default_traj_dir=_PRESSURE_ROOT.parent / "data" / "mlip-trajs-ase-accelerated")

"""Stress-enabled md_orb_v3_direct_omat.py: record potential stress for every saved trajectory frame."""




import torch

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False


def make_calculator():
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    model, adapter = pretrained.orb_v3_direct_20_omat(
        device="cuda", precision="float32-high", compile=True,
    )

    calculator = ORBCalculator(model, atoms_adapter=adapter, device="cuda")
    calculator.implemented_properties = ["energy", "free_energy", "forces", "stress"]
    return calculator


if __name__ == "__main__":
    run_ase_pressure("orb-v3-direct-omat-stress", make_calculator,
               "ase+orb+compile+stress")
