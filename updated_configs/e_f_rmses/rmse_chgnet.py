# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "ase==3.29.0",
#   "numpy==2.5.2",
#   "pandas==3.0.5",
#   "matplotlib==3.11.1",
#   "chgnet==0.4.2",
#   "torch-sim-atomistic==0.6.1",
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
"""Evaluate CHGNet on every reference trajectory. Run: uv run rmse_chgnet.py"""

import os
import runpy
from pathlib import Path

os.environ["MODEL_NAME"] = "chgnet"
runpy.run_path(str(Path(__file__).with_name("rmse_script-generic.py")), run_name="__main__")
