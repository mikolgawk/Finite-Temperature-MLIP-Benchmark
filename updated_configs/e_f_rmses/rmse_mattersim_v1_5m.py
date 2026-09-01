# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "ase>=3.26", "matplotlib>=3.8", "numpy>=1.26", "pandas>=2.2", "torch==2.11.0+cu128",
#   "torch-sim-atomistic[mattersim]==0.6.1", "torchvision==0.26.0+cu128",
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
# ///
"""Run static energy/force RMSE evaluation for MatterSim-v1.0.0-5M."""

import os
import runpy
from pathlib import Path

os.environ.setdefault("MODEL_NAME", "mattersim-v1-5M")
runpy.run_path(Path(__file__).with_name("rmse_script-generic.py"), run_name="__main__")
