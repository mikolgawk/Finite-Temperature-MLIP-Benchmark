# /// script
# requires-python = ">=3.12"
# dependencies = ["ase>=3.26", "h5py>=3.11", "numpy>=1.26", "orb-models==0.6.2", "torch==2.11.0+cu128", "pandas>=2.2"]
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
early_cli(__file__, "ase")

"""Stress-enabled md_orb_v2.py: record potential stress for every saved trajectory frame."""




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
    run_ase_pressure("orb-v2-stress", make_calculator,
               "ase+orb+compile+stress")
