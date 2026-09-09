# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch-sim-atomistic==0.6.1",
#   "fairchem-core",
#   "ase>=3.26",
# ]
# ///
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchsim_rmse import run_rmse, early_cli
early_cli(__file__)

"""Evaluate reference energy/force RMSE with the matching TorchSim MD model."""

import csv
import json
import time
from pathlib import Path

import torch

# FairChem temporarily configures TF32 inside the UMA predictor for each mode.
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

import torch_sim as ts
from ase.io import read

from torch_sim.models.fairchem import FairChemModel

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


# settings
INFERENCE_MODE = "turbo"
if INFERENCE_MODE not in {"compile", "turbo"}:
    raise ValueError(f"Unsupported UMA inference mode: {INFERENCE_MODE!r}")

MODEL_NAME = f"uma-m-omat-{INFERENCE_MODE}-force-only"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-accelerated"

CHAIN_LENGTH = 1              # Nose-Hoover chain settings, as in the ASE benchmark
CHAIN_STEPS = 1
SY_STEPS = 3
SEED = 42                     # Maxwell-Boltzmann velocity seed
STATE_DTYPE = torch.float32

device = torch.device("cuda")

# model
# fairchem's torch-sim wrapper has no inference_settings passthrough; it calls
# pretrained_mlip.get_predict_unit(...) internally, so inject the named mode at
# the factory.
from fairchem.core import pretrained_mlip
from fairchem.core.units.mlip_unit import InferenceSettings

FAIRCHEM_INFERENCE_SETTINGS = (
    InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=False,
        compile=True,
    )
    if INFERENCE_MODE == "compile"
    else "turbo"
)

_get_predict_unit = pretrained_mlip.get_predict_unit


def _get_predict_unit_with_mode(model_name, **kwargs):
    kwargs.setdefault("inference_settings", FAIRCHEM_INFERENCE_SETTINGS)
    return _get_predict_unit(model_name, **kwargs)


pretrained_mlip.get_predict_unit = _get_predict_unit_with_mode

model = FairChemModel(
    model="uma-m-1p1", task_name="omat", device=device, dtype=torch.float32,
    compute_stress=False,
)

# Remove stress tasks and strain derivatives before lazy compilation/merging.
disable_uma_stress(model)

# FairChem 2.22 prepares merged MOLE experts before its normal device move.
# TorchSim's first state is already on CUDA, so turbo must move the predictor
# early to keep embedding weights and indices on the same device during merging.
if INFERENCE_MODE == "turbo":
    model.predictor.move_to_device()


# NVT MD loop, one entry per system in the metadata file
run_rmse(MODEL_NAME, model, state_dtype=STATE_DTYPE)
