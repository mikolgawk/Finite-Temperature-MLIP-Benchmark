# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "ase==3.29.0",
#   "numpy==2.4.6",
#   "pandas==3.0.5",
#   "matplotlib==3.11.1",
#   "torch-sim-atomistic[fairchem]==0.6.1",
#   "fairchem-core==2.21.0",
#   "torch==2.8.0+cu128",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""Evaluate UMA-S-1p1 with the OMat task on every reference trajectory."""

import os
import runpy
from pathlib import Path

os.environ["MODEL_NAME"] = "uma-s-omat"
runpy.run_path(str(Path(__file__).with_name("rmse_script-generic.py")), run_name="__main__")
