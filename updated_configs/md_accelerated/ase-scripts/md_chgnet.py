# /// script
# requires-python = ">=3.12"
# dependencies = ["ase>=3.26", "h5py>=3.11", "numpy>=1.26", "torch", "chgnet==0.4.2"]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""Native ASE CHGNet MD, requesting energy and forces only."""

import torch
from _ase_md import run_ase_md


def make_calculator():
    from chgnet.model.dynamics import CHGNetCalculator
    from chgnet.model.model import CHGNet

    class ForceOnlyCHGNetCalculator(CHGNetCalculator):
        implemented_properties = ["energy", "free_energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=None):
            super().calculate(atoms=atoms, properties=properties,
                              system_changes=system_changes, task="ef")

    model = CHGNet.load(use_device="cuda")
    model.forward = torch.compile(model.forward, dynamic=True, fullgraph=False)
    return ForceOnlyCHGNetCalculator(model=model, use_device="cuda", on_isolated_atoms="error")


if __name__ == "__main__":
    run_ase_md("chgnet-force-only", make_calculator, "ase+chgnet-0.4.2-compile-force-only")
