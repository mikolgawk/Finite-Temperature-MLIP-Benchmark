#!/usr/bin/env python3
"""Accelerated TorchSim entry point for custom-model V2 production MD.

The supplied factory must construct the accelerated model. Compilation,
artifact loading, and other setup happen before the MD timing boundary.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "md_eager"))

from _torchsim_runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(accelerated=True))
