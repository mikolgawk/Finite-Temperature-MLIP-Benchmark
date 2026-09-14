# /// script
# requires-python = ">=3.12"
# dependencies = ["tqdm>=4.66", "ase>=3.26", "h5py>=3.11", "numpy>=1.26", "torch", "chgnet==0.4.2"]
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

"""Native ASE CHGNet MD, requesting energy and forces only."""

import torch


def make_calculator():
    from chgnet.model.dynamics import CHGNetCalculator
    from chgnet.model.model import CHGNet

    class ForceOnlyCHGNetCalculator(CHGNetCalculator):
        implemented_properties = ["energy", "free_energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=None):
            super().calculate(atoms=atoms, properties=properties,
                              system_changes=system_changes, task="ef")

    model = CHGNet.load(use_device="cuda")
    return ForceOnlyCHGNetCalculator(model=model, use_device="cuda", on_isolated_atoms="error")


if __name__ == "__main__":
    run_rmse("chgnet-force-only", make_calculator, "ase+chgnet-0.4.2-force-only")
