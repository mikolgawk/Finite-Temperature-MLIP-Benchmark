# /// script
# requires-python = ">=3.12"
# dependencies = ["ase>=3.26", "h5py>=3.11", "numpy>=1.26", "fairchem-core==2.22.0", "torch==2.11.0+cu128", "pandas>=2.2"]
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

"""Stress-enabled md_uma_s_omat_turbo.py: record potential stress for every saved trajectory frame."""


from _uma import retain_uma_stress


def make_calculator():
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from fairchem.core.units.mlip_unit import InferenceSettings

    settings = "turbo"
    predictor = pretrained_mlip.get_predict_unit(
        "uma-s-1p1", device="cuda", inference_settings=settings,
    )
    calculator = retain_uma_stress(FAIRChemCalculator(predictor, task_name="omat"))
    predictor.move_to_device()
    return calculator


if __name__ == "__main__":
    run_ase_pressure("uma-s-omat-turbo-stress", make_calculator,
               "ase+fairchem-2.22.0+turbo+stress", per_system_calculator=True)
