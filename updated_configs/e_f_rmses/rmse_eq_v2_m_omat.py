# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "ase>=3.26", "fairchem-core==1.10.0", "matplotlib>=3.8", "numpy>=1.26", "pandas>=2.2",
#   "scipy<1.17", "torch==2.4.1", "torch-scatter", "torch-sim-atomistic==0.5.2", "torch-sparse",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# explicit = true
#
# [[tool.uv.index]]
# name = "pyg-cu124"
# url = "https://data.pyg.org/whl/torch-2.4.0+cu124.html"
# format = "flat"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu124" }
# torch-scatter = { index = "pyg-cu124" }
# torch-sparse = { index = "pyg-cu124" }
# ///
"""Run static energy/force RMSE evaluation for EquiformerV2."""

import os
import runpy
from pathlib import Path

os.environ.setdefault("MODEL_NAME", "eq-v2-M-omat")
runpy.run_path(Path(__file__).with_name("rmse_script-generic.py"), run_name="__main__")
