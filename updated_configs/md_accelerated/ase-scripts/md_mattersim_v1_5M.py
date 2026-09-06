# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "h5py>=3.11", "numpy>=1.26",
#   "mattersim",
#   "ase>=3.26",
#   "torch==2.11.0+cu128",
#   "torchvision==0.26.0+cu128",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
# mattersim = { git = "https://github.com/microsoft/mattersim.git", rev = "df68847af1d66c6725bfadc58c4e3d3ec705bc5f" }
# ///
"""ASE counterpart using the same compiled MatterSim forward, without stress."""

import os
from pathlib import Path
import torch
from _ase_md import run_ase_md

# TorchInductor's generated C++ precompiled header includes CUDA headers without
# always adding the wheel's CUDA runtime include directory. Make that directory
# visible before MatterSim invokes the in-process compiler.
CUDA_HOME = Path(os.environ.get("MATTERSIM_CUDA_HOME", "/usr/local/cuda")).resolve()
_site_packages = Path(torch.__file__).resolve().parent.parent
_wheel_cuda_include = _site_packages / "nvidia" / "cuda_runtime" / "include"
_cuda_include = (
    _wheel_cuda_include
    if (_wheel_cuda_include / "cuda_fp8.h").is_file()
    else CUDA_HOME / "include"
)
if not (_cuda_include / "cuda_fp8.h").is_file():
    raise FileNotFoundError(
        f"CUDA headers not found under {_cuda_include}; set MATTERSIM_CUDA_HOME "
        "to a complete CUDA toolkit"
    )
os.environ["CUDA_HOME"] = str(CUDA_HOME)
os.environ["CUDA_PATH"] = str(CUDA_HOME)
os.environ["CUDAToolkit_ROOT"] = str(CUDA_HOME)
os.environ["CPATH"] = os.pathsep.join(
    value
    for value in (
        str(_cuda_include),
        str(CUDA_HOME / "include"),
        os.environ.get("CPATH", ""),
    )
    if value
)


def make_calculator():
    from mattersim.forcefield import MatterSimCalculator

    calculator = MatterSimCalculator(
        load_path="MatterSim-v1.0.0-5M.pth", device="cuda", compute_stress=False,
    )
    calculator.potential.model.forward = torch.compile(calculator.potential.model.forward)
    calculator.implemented_properties = ["energy", "free_energy", "forces"]
    return calculator


if __name__ == "__main__":
    run_ase_md("mattersim-v1-5M-compile-force-only", make_calculator,
               "ase+mattersim+torch-compile+force-only")
