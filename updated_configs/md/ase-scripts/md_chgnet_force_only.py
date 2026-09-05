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

from md_chgnet import main


if __name__ == "__main__":
    main(force_only=True)
