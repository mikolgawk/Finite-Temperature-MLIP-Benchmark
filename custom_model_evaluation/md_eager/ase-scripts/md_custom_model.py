#!/usr/bin/env python3
"""Baseline ASE entry point for custom-model V2 production MD."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _ase_runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(accelerated=False))
