# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "ase==3.29.0",
#   "numpy==1.26.4",
#   "pandas==3.0.5",
#   "matplotlib==3.11.1",
#   "tqdm==4.70.0",
#   "torch-sim-atomistic==0.5.2",
#   "fairchem-core==1.10.0",
#   "torch==2.4.1",
#   "torch-scatter==2.1.2",
#   "torch-sparse==0.6.18",
#   "scipy==1.16.3",
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
"""Evaluate eSEN-30M-OAM on every reference trajectory."""

import os
import runpy
from pathlib import Path

os.environ["MODEL_NAME"] = "eSEN-30M-OAM"
runpy.run_path(str(Path(__file__).with_name("rmse_script-generic.py")), run_name="__main__")
