# /// script
# requires-python = ">=3.12"
# dependencies = ["ase>=3.26", "h5py>=3.11", "numpy>=1.26", "mace-torch==0.3.16", "torch==2.11.0+cu128", "cuequivariance-torch>=0.4", "cuequivariance-ops-torch-cu12"]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""Stress-enabled md_mace_mp_0.py: record potential stress for every saved trajectory frame."""

from _ase_md import REPO, run_ase_md
from _mace import retain_mace_stress


def make_calculator():
    from mace.calculators import MACECalculator, mace_mp

    settings = dict(default_dtype="float32", device="cuda",
                    enable_cueq=True, compile_mode="reduce-overhead")
    return retain_mace_stress(mace_mp(model="medium", **settings))


if __name__ == "__main__":
    run_ase_md("mace-mp-0-compile", make_calculator,
               "ase+mace+cueq+compile-reduce-overhead")
