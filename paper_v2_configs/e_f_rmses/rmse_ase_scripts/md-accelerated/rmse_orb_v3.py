# /// script
# requires-python = ">=3.12"
# dependencies = ["tqdm>=4.66", "ase>=3.26", "h5py>=3.11", "numpy>=1.26", "orb-models==0.6.2", "torch==2.11.0+cu128"]
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

"""ASE counterpart to torchsim-scripts/md_orb_v3.py."""



def make_calculator():
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    model, adapter = pretrained.orb_v3_conservative_inf_mpa(
        device="cuda", precision="float32-highest", compile=True,
    )
    model.disable_stress()
    calculator = ORBCalculator(model, atoms_adapter=adapter, device="cuda")
    calculator.implemented_properties = ["energy", "free_energy", "forces"]
    return calculator


if __name__ == "__main__":
    run_rmse("orb-v3-force-only", make_calculator,
               "ase+orb+compile+force-only")
