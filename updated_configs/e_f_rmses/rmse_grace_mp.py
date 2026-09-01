# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "ase==3.29.0",
#   "numpy==2.1.3",
#   "pandas==2.3.3",
#   "matplotlib==3.11.1",
#   "tqdm==4.70.0",
#   "tensorpotential==0.5.7.2",
#   "torch-sim-atomistic[vesin]==0.6.1",
#   "torch==2.11.0+cu128",
#   "nvidia-cuda-nvcc-cu12==12.8.93",
#   "tf-keras==2.19.0",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
#
# [tool.uv]
# override-dependencies = ["tensorflow==2.19.1"]
# ///
"""Evaluate GRACE-2L-MP on every reference trajectory."""

import os
import runpy
from pathlib import Path

os.environ["MODEL_NAME"] = "grace-mp"
runpy.run_path(str(Path(__file__).with_name("rmse_script-generic.py")), run_name="__main__")
