# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tqdm>=4.66",
#   "torch-sim-atomistic[nequip]==0.6.1",
#   "nequip>=0.19.1",
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

import csv
import json
import time
from pathlib import Path

import torch_sim as ts
from ase.io import read

from nequip.data import AtomicDataDict
from torch_sim.models.nequip_framework import NequIPFrameworkModel


class PositionGradientOnly(torch.nn.Module):
    """Add forces to an energy-only NequIP graph without cell differentiation."""

    def __init__(self, energy_model: torch.nn.Module) -> None:
        super().__init__()
        self.energy_model = energy_model

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Detaching makes positions an independent differentiation variable and
        # deliberately leaves the cell outside the autograd graph.
        positions = data[AtomicDataDict.POSITIONS_KEY].detach().requires_grad_(True)
        data = dict(data)
        data[AtomicDataDict.POSITIONS_KEY] = positions

        output = self.energy_model(data)
        energy = output[AtomicDataDict.TOTAL_ENERGY_KEY]
        forces = -torch.autograd.grad(
            energy.sum(), positions, create_graph=self.training
        )[0]
        output[AtomicDataDict.FORCE_KEY] = forces
        return output


def load_force_only_nequip(
    model_source: str, device: torch.device | str
) -> NequIPFrameworkModel:
    """Load an OAM package and replace joint force/stress differentiation."""
    calculator = NequIPFrameworkModel._from_saved_model(
        model_path=model_source,
        device=device,
        chemical_species_to_atom_type_map=True,
        allow_tf32=False,
        compile_mode="eager",
    )

    # Packaged models carry their own NequIP code. Older ForceStressOutput
    # implementations have no do_derivatives branch, so assigning that flag
    # silently leaves their joint backward pass active. Remove the wrapper
    # itself, retaining its energy function and every other model layer.
    removed = 0

    def remove_joint_output(module):
        nonlocal removed
        if module.__class__.__name__ == "ForceStressOutput":
            if not isinstance(getattr(module, "func", None), torch.nn.Module):
                raise RuntimeError("Unsupported NequIP ForceStressOutput structure")
            removed += 1
            return remove_joint_output(module.func)
        for name, child in list(module.named_children()):
            replacement = remove_joint_output(child)
            if replacement is not child:
                setattr(module, name, replacement)
        return module

    energy_model = remove_joint_output(calculator.model)
    if removed == 0:
        raise RuntimeError("NequIP package has no ForceStressOutput layer to remove")
    if any(layer.__class__.__name__ == "ForceStressOutput"
           for layer in energy_model.modules()):
        raise RuntimeError("failed to remove NequIP's joint force/stress layer")
    calculator.model = PositionGradientOnly(energy_model).to(device).eval()
    calculator.compute_stress = False
    calculator.compute_forces = True
    return calculator


def assert_eager_module(model):
    """Reject TorchScript and torch.compile modules, including nested modules."""
    import torch
    from torch._dynamo.eval_frame import OptimizedModule

    for name, module in model.named_modules():
        if isinstance(module, (torch.jit.ScriptModule, OptimizedModule)):
            raise RuntimeError(f"Compiled module in eager model: {name or '<root>'}")
        if getattr(module, "_compiled_call_impl", None) is not None:
            raise RuntimeError(f"Compiled call in eager model: {name or '<root>'}")


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
    device = torch.device("cuda")
    model = load_force_only_nequip(
        "nequip.net:mir-group/NequIP-OAM-L:0.1", device=device
    )
    assert_eager_module(model.model)
    run_rmse(
        "nequip-oam-l-force-only", model, "torch-sim-0.6.1+nequip-eager-force-only"
    )


if __name__ == "__main__":
    main()
