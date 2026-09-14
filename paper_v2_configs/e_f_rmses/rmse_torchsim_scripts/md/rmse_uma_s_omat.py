# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tqdm>=4.66",
#   "torch-sim-atomistic[fairchem]==0.6.1",
#   "fairchem-core==2.21.0",
#   "ase>=3.26",
#   "torch",
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
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchsim_rmse import run_rmse, early_cli
early_cli(__file__)

"""Evaluate reference energy/force RMSE with the matching TorchSim MD model."""

import torch

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

"""Evaluate reference energy/force RMSE with the matching TorchSim MD model."""


def disable_uma_stress(calculator):
    """Disable derivative stress and remove its predictor tasks before first use."""
    predictor = calculator.predictor
    if predictor.lazy_model_intialized:
        raise RuntimeError("Disable UMA stress before the first prediction/compilation")
    model = predictor.model.module
    config = model.backbone.regress_config
    if config.direct_forces or config.direct_stress or not config.forces or config.hessian:
        raise RuntimeError("Expected conservative UMA energy/forces without Hessians")
    if not config.stress:
        raise RuntimeError("Expected the original stress-enabled UMA model")
    tasks = {name: task for name, task in model.tasks.items() if task.property != "stress"}
    omat_properties = {task.property for task in tasks.values() if "omat" in task.datasets}
    if not {"energy", "forces"}.issubset(omat_properties):
        raise RuntimeError("UMA checkpoint is missing OMat energy/force tasks")

    # Task routing/normalization must agree with the outputs the heads produce.
    # Preserve the existing Task objects, including normalizers and atom references.
    from fairchem.core.models.base import _get_dataset_to_tasks_map

    model._tasks = tasks
    model._dataset_to_tasks = _get_dataset_to_tasks_map(tasks.values())
    model.backbone.validate_tasks(model._dataset_to_tasks)
    predictor.inference_settings.auto_add_default_untrained_tasks = False
    predictor.inference_settings.predict_untrained_stress = set()
    # EFS heads share this config with the backbone. Also check every head so a
    # package/API change cannot silently leave joint differentiation enabled.
    config.stress = False
    for module in model.modules():
        regression = getattr(module, "regress_config", None)
        if regression is not None and regression.stress:
            raise RuntimeError("UMA contains a head with an unshared stress configuration")
    calculator._compute_stress = False
    calculator.implemented_properties = ["energy", "forces"]
    return calculator


def load_force_only_uma(model_name, device):
    """Keep the standard loader, precision and inference settings; disable stress."""
    import torch
    from torch_sim.models.fairchem import FairChemModel

    calculator = FairChemModel(
        model=model_name, task_name="omat", device=device,
        dtype=torch.float32, compute_stress=False,
    )
    return disable_uma_stress(calculator)
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
    model = load_force_only_uma("uma-s-1p1", torch.device("cuda"))
    run_rmse(
        "uma-s-omat-force-only", model,
        "torch-sim-0.6.1+fairchem-2.21.0-force-only",
    )


if __name__ == "__main__":
    main()
