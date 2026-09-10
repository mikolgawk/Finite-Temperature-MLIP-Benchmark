# /// script
# requires-python = ">=3.12"
# dependencies = ["ase>=3.26", "h5py>=3.11", "numpy>=1.26", "fairchem-core==2.22.0", "torch==2.11.0+cu128"]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""Stress-enabled md_uma_s_omat_compile.py: record potential stress for every saved trajectory frame."""

from _ase_md import run_ase_md
from _uma import retain_uma_stress


def make_calculator():
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from fairchem.core.units.mlip_unit import InferenceSettings

    settings = InferenceSettings(
        tf32=False, activation_checkpointing=False, merge_mole=False, compile=True,
    )
    predictor = pretrained_mlip.get_predict_unit(
        "uma-s-1p1", device="cuda", inference_settings=settings,
    )
    calculator = retain_uma_stress(FAIRChemCalculator(predictor, task_name="omat"))
    predictor.move_to_device()
    return calculator


if __name__ == "__main__":
    run_ase_md("uma-s-omat-compile-stress", make_calculator,
               "ase+fairchem-2.22.0+compile+stress", per_system_calculator=True)
