# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "ase==3.29.0",
#   "numpy==2.5.2",
#   "pandas==3.0.5",
#   "matplotlib==3.11.1",
#   "tqdm==4.70.0",
#   "torch-sim-atomistic[mace,vesin]==0.6.1",
#   "mace-torch==0.3.16",
#   "torch==2.11.0+cu128",
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
"""Evaluate the local MACE-MPA-0 medium checkpoint on every reference trajectory."""

import os
import runpy
from pathlib import Path

os.environ["MODEL_NAME"] = "mace-mpa-0"
runpy.run_path(str(Path(__file__).with_name("rmse_script-generic.py")), run_name="__main__")
