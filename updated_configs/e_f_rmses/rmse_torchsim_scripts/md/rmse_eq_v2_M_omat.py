# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tqdm>=4.66",
#   "torch-sim-atomistic==0.5.2",  # 0.6.1 needs torch>=2.8, fairchem 1.x pins <2.5;
#                                  # 0.5.2 has the same integrate API incl. nvt_vrescale
#   "fairchem-core==1.10.0",       # v1: only API that loads this legacy checkpoint
#   "torch==2.4.1",
#   "torch-scatter",               # runtime imports of fairchem v1, undeclared there
#   "torch-sparse",
#   "scipy<1.17",                # fairchem-core 1.10 imports the removed scipy.special.sph_harm
#   "ase>=3.26",
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
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchsim_rmse import run_rmse, early_cli
early_cli(__file__)

"""Evaluate reference energy/force RMSE with the matching TorchSim MD model."""

from pathlib import Path

import torch

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

"""Evaluate reference energy/force RMSE with the matching TorchSim MD model."""


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


def load_force_only_legacy(checkpoint, family: str, device, seed: int = 42):
    """Load original weights with the existing legacy adapter, then disable stress."""
    from torch_sim.models.fairchem_legacy import FairChemV1Model

    calculator = FairChemV1Model(
        model=str(checkpoint), device=device, compute_stress=False,
        seed=seed, pbc=True, disable_amp=True,
    )
    disable_legacy_stress(calculator, family)
    return calculator
"""Evaluate reference energy/force RMSE with the matching TorchSim MD model."""

import csv
import json
import time
from pathlib import Path

import torch
import torch_sim as ts
from ase.io import read


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-matched"

CHAIN_LENGTH = 1
CHAIN_STEPS = 1
SY_STEPS = 3
SEED = 42
STATE_DTYPE = torch.float32




def main():
    checkpoint = REPO / "updated_configs" / "data" / "models" / "eqV2_86M_omat_mp_salex.pt"
    model = load_force_only_legacy(checkpoint, "eqv2", torch.device("cuda"))
    run_rmse(
        "eq-v2-M-omat-force-only", model,
        "torch-sim-0.5.2+fairchem-1.10.0-force-only",
    )


if __name__ == "__main__":
    main()
