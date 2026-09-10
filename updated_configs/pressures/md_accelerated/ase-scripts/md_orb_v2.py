# /// script
# requires-python = ">=3.12"
# dependencies = ["ase>=3.26", "h5py>=3.11", "numpy>=1.26", "orb-models==0.6.2", "torch==2.11.0+cu128"]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""Stress-enabled md_orb_v2.py: record potential stress for every saved trajectory frame."""

from _ase_md import run_ase_md


def make_calculator():
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    model, adapter = pretrained.orb_v2(
        device="cuda", precision="float32-highest", compile=True,
    )

    calculator = ORBCalculator(model, atoms_adapter=adapter, device="cuda")
    calculator.implemented_properties = ["energy", "free_energy", "forces", "stress"]
    return calculator


if __name__ == "__main__":
    run_ase_md("orb-v2-stress", make_calculator,
               "ase+orb+compile+stress")
