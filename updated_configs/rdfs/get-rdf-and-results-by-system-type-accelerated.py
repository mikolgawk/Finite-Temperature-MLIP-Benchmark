#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "ase>=3.26",
#   "h5py>=3.11",
#   "mdtraj>=1.10",
#   "numpy>=1.26",
#   "pandas>=2.2",
# ]
# ///
"""Compute RDFs for the accelerated TorchSim MLIP trajectories.

This is a separate entry point for ``data/mlip-trajs-torchsim-accelerated``.
It reuses the matched-trajectory RDF implementation and writes to
``rdfs/results_accelerated`` by default, keeping both result sets independent.

Run with::

    uv run get-rdf-and-results-by-system-type-accelerated.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RDF_IMPLEMENTATION = (
    BASE_DIR / "get-rdf-and-results-by-system-type-same-simulation-length.py"
)


def load_rdf_implementation():
    spec = importlib.util.spec_from_file_location(
        "matched_rdf_implementation",
        RDF_IMPLEMENTATION,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load RDF implementation: {RDF_IMPLEMENTATION}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    rdf = load_rdf_implementation()
    rdf.MLIP_TRAJ_BASE_DIR = rdf.DATA_DIR / "mlip-trajs-torchsim-accelerated"
    rdf.RESULTS_DIR = rdf.BASE_DIR / "results_accelerated"
    rdf.main()


if __name__ == "__main__":
    main()
