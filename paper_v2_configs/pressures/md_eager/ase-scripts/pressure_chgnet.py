# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "pandas>=2.2",
#   "ase>=3.26", "h5py>=3.11", "numpy>=1.26",
#   "chgnet==0.4.2",
#   "torch-sim-atomistic==0.6.1",
#   "torch",
#   "tqdm>=4.66",
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

import sys
from pathlib import Path
_PRESSURE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "pressures")
sys.path.insert(0, str(_PRESSURE_ROOT))
from pressure_evaluator import early_cli, run_ase_pressure
early_cli(__file__, "ase")



def make_calculator():
    from chgnet.model.dynamics import CHGNetCalculator
    return CHGNetCalculator(use_device="cuda")

if __name__ == "__main__":
    run_ase_pressure("chgnet", make_calculator, "ase+chgnet-stress")
