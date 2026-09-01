# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "ase==3.29.0",
#   "numpy==2.5.2",
#   "pandas==3.0.5",
#   "matplotlib==3.11.1",
#   "torch-sim-atomistic[mattersim]==0.6.1",
#   "mattersim==1.2.5",
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
# ///
"""Evaluate MatterSim-v1 5M on every reference trajectory."""

import os
import runpy
from pathlib import Path

os.environ["MODEL_NAME"] = "mattersim-v1-5M"
runpy.run_path(str(Path(__file__).with_name("rmse_script-generic.py")), run_name="__main__")
