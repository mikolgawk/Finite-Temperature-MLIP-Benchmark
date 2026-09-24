# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "ase>=3.26",
#   "chgnet==0.4.2",
#   "torch-sim-atomistic==0.6.1",
#   "torch",
#   "tqdm>=4.66",
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
"""Force-only CHGNet NVT production MD using the energy/force task."""

import sys
from pathlib import Path


# The reusable implementation lives one directory above this entry point.  Add
# that directory before importing so this file does not resolve itself as
# ``md_chgnet`` and create a circular import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from md_chgnet_force_only import main


if __name__ == "__main__":
    main()
