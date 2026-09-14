# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "tqdm>=4.66",
#   "ase>=3.26", "fairchem-core==1.10.0", "h5py>=3.11",
#   "numpy==1.26.4", "scipy<1.17", "torch==2.4.1",
#   "torch-scatter", "torch-sparse",
# ]
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# explicit = true
# [[tool.uv.index]]
# name = "pyg-cu124"
# url = "https://data.pyg.org/whl/torch-2.4.0+cu124.html"
# format = "flat"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cu124" }
# torch-scatter = { index = "pyg-cu124" }
# torch-sparse = { index = "pyg-cu124" }
# ///

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ase_rmse import REPO, early_cli, run_rmse
if __name__ == "__main__" and "--compile-force-only" not in sys.argv[1:]:
    early_cli(__file__)

"""Native ASE energy/force-only counterpart to torchsim-scripts/md_eq_v2_M_omat.py.

Run: uv run ase-scripts/md_eq_v2_M_omat.py
"""

"""ASE RMSE evaluator derived from the matching MD calculator."""

import csv
import json
import time
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import torch
from ase import units
from ase.io import read


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-matched"

CHAIN_LENGTH = 1
CHAIN_STEPS = 1
SEED = 42


torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False










"""Disable stress inside the two supported legacy FairChem checkpoints.

Apply after checkpoint loading, before inference. This changes only the live
model and inference targets; checkpoint files and training metadata stay intact.
"""


def disable_legacy_stress(calculator, family: str) -> None:
    """Configure an eqV2 or eSEN FairChemV1Model for energy/forces only."""
    if family not in {"eqv2", "esen"}:
        raise ValueError(f"Unsupported legacy model family: {family}")
    trainer = calculator.trainer
    models = [m for m in trainer.model.modules() if hasattr(m, "output_heads")]
    if len(models) != 1:
        raise RuntimeError("Expected exactly one legacy Hydra model")
    model = models[0]
    heads = model.output_heads
    required = {"energy", "forces"}
    for outputs in (trainer.output_targets, trainer.config["outputs"], calculator.config["outputs"]):
        if not required.issubset(outputs):
            raise RuntimeError("Checkpoint does not expose energy and forces")

    if family == "eqv2":
        if set(heads) != {"energy", "forces", "stress"}:
            raise RuntimeError(f"Unexpected eqV2 heads: {list(heads)}")
        # Remove the actual network head, including its decomposed stress outputs.
        del heads["stress"]
    else:
        if set(heads) != {"mptrj"}:
            raise RuntimeError(f"Unexpected eSEN heads: {list(heads)}")
        backbone, head = model.backbone, heads["mptrj"]
        if head.__class__.__name__ != "MLP_EFS_Head":
            raise RuntimeError("Expected eSEN MLP_EFS_Head")
        for component in (backbone, head):
            if not component.regress_forces or not component.regress_stress:
                raise RuntimeError("Expected joint eSEN energy/force/stress model")
        if backbone.direct_forces:
            raise RuntimeError("Expected conservative eSEN backbone")
        # Both flags matter: the backbone constructs the strain graph, while
        # the head chooses positions-only versus joint position/strain gradients.
        backbone.regress_stress = False
        head.regress_stress = False

    # predict() and _forward() each consult a different target dictionary.
    # Whitelisting also removes eqV2's stress_isotropic/stress_anisotropic targets.
    trainer.output_targets = {k: v for k, v in trainer.output_targets.items() if k in required}
    trainer.config["outputs"] = {k: v for k, v in trainer.config["outputs"].items() if k in required}
    calculator.config["outputs"] = {k: v for k, v in calculator.config["outputs"].items() if k in required}
    calculator.implemented_properties = ["energy", "forces"]
    # The 0.5.2 legacy adapter exposes read-only public properties.
    calculator._compute_stress = False
    calculator._compute_forces = True


CHECKPOINT = REPO / "updated_configs/data/models/eqV2_86M_omat_mp_salex.pt"


def make_calculator():
    from fairchem.core import OCPCalculator

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"eqV2 checkpoint not found: {CHECKPOINT}")
    calculator = OCPCalculator(
        checkpoint_path=str(CHECKPOINT), cpu=False, seed=42, disable_amp=True
    )
    disable_legacy_stress(calculator, "eqv2")
    return calculator


if __name__ == "__main__":
    run_rmse("eq-v2-M-omat", make_calculator, "ase+fairchem-core-1.10.0")
