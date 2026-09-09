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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ase_rmse import REPO, early_cli, run_rmse
if __name__ == "__main__" and "--compile-force-only" not in sys.argv[1:]:
    early_cli(__file__)

"""ASE counterpart to torchsim-scripts/md_uma_m_omat_compile.py."""

from _uma import disable_uma_stress


def make_calculator():
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from fairchem.core.units.mlip_unit import InferenceSettings

    settings = InferenceSettings(
        tf32=False, activation_checkpointing=False, merge_mole=False, compile=True,
    )
    predictor = pretrained_mlip.get_predict_unit(
        "uma-m-1p1", device="cuda", inference_settings=settings,
    )
    calculator = disable_uma_stress(FAIRChemCalculator(predictor, task_name="omat"))
    predictor.move_to_device()
    return calculator


if __name__ == "__main__":
    run_rmse("uma-m-omat-compile-force-only", make_calculator,
               "ase+fairchem-2.22.0+compile+force-only", per_system_calculator=True)
