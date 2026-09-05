# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch-sim-atomistic[vesin]==0.6.1",
#   "tensorpotential==0.5.7.2",
#   "ase>=3.26",
#   "torch",
#   "nvidia-cuda-nvcc-cu12==12.8.*",
#   "tf-keras==2.19.*",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
#
# [tool.uv]
# override-dependencies = ["tensorflow>=2.17,<2.20"]
# ///
"""TorchSim NVT production MD — grace-mp.

GRACE is a TensorFlow XLA SavedModel; GraceModel below bridges it into torch-sim:
neighbor list and bond vectors are built in torch and tensors cross via DLPack.
For NVT, the loaded graph is pruned to its energy/force dependency subgraph so
its virial branch is not executed. Each system is cross-checked against the
stock TPCalculator before its run.

GRACE-2L-MP-r6 ships as an fp64 SavedModel. Its public download does not include
the original potential YAML required by TensorPotential's offline casting utility,
so this script keeps the model and TorchSim state in native fp64 precision.

Per-system MD parameters come from updated_configs/data/ref-trajs/md_metadata.json, which
records how each reference AIMD was run. The trajectory is saved as HDF5
with positions and velocities:

    <OUT_ROOT>/<system>/nvt_grace-mp.h5

Run:  uv run md_grace_mp.py
"""

import csv
import json
import math
import os
import time
from pathlib import Path

import torch

# no TF32, no autotuned kernels
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)  # else TF takes all VRAM from torch
tf.config.experimental.enable_tensor_float_32_execution(False)

import torch_sim as ts
from ase.data import atomic_numbers as ASE_Z
from ase.io import read
from tensorflow.experimental import dlpack as tf_dlpack
from tensorpotential.calculator import TPCalculator
from torch.utils import dlpack as torch_dlpack
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.transforms import compute_cell_shifts

from grace_force_only import prune_grace_stress
from torchsim_md_runner import run_torchsim_md

FAR_AWAY = 52.0  # fake bond vector component, beyond any cutoff


class GraceModel(ModelInterface):
    """torch-sim interface to a GRACE TPCalculator's XLA SavedModel via DLPack."""

    def __init__(self, model, device=None, dtype=torch.float32,
                 compute_stress=False, pad_fraction=0.05):
        super().__init__()
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype
        self._compute_forces = True
        self._compute_stress = compute_stress
        self._memory_scales_with = "n_atoms_x_density"
        self._pad_fraction = pad_fraction
        self._bond_buckets = []

        self._calc = model
        if len(model.models) != 1:
            raise ValueError("GRACE ensembles are not supported, pass a single model")
        self._tp = model.models[0]
        self._compute = (self._tp.compute if compute_stress
                         else prune_grace_stress(self._tp.compute))

        specs = self._tp.signatures["serving_default"].structured_input_signature[1]
        self._data_keys = set(model.data_keys)
        self._tf_float = specs["bond_vector"].dtype
        self._torch_float = (torch.float64 if self._tf_float == tf.float64
                             else torch.float32)
        self._cutoff = float(model.cutoff)

        # artifact element map as a Z-indexed lookup table
        z_to_mu = torch.full((119,), -1, dtype=torch.int32, device=self._device)
        for sym, mu in model.element_map.items():
            z_to_mu[ASE_Z[sym]] = mu
        self._z_to_mu = z_to_mu

        print(f"[grace-torchsim] cutoff {self._cutoff} A, graph dtype "
              f"{self._tf_float.name}, {len(model.element_map)} elements")

    def _bucket(self, n_bonds):
        # XLA compiles one executable per input shape: pad bonds to grow-only buckets
        for bound in self._bond_buckets:
            if bound >= n_bonds:
                return bound
        bound = max(256, math.ceil(n_bonds * (1.0 + self._pad_fraction)))
        self._bond_buckets.append(bound)
        self._bond_buckets.sort()
        return bound

    def forward(self, state, **_kwargs):
        if int(state.system_idx[-1]) != 0:
            raise ValueError("GraceModel handles one system per state")
        pos = state.positions.detach()
        cell = state.row_vector_cell.detach()
        n_real = pos.shape[0]
        dev = pos.device

        mapping, sys_map, unit_shifts = vesin_nl_ts(
            pos, cell, state.pbc, self._cutoff, state.system_idx)
        shifts = compute_cell_shifts(cell, unit_shifts, sys_map)
        ind_i, ind_j = mapping[0], mapping[1]
        bond_vector = pos[ind_j] - pos[ind_i] + shifts

        atomic_mu = self._z_to_mu[state.atomic_numbers]
        if bool((atomic_mu < 0).any()):
            raise ValueError("structure contains elements outside the model's element map")

        # fake atom appended last; fake bonds attach to it with 52 A vectors beyond the cutoff
        fake = n_real
        n_bonds = ind_i.shape[0] + n_real
        n_fill = self._bucket(n_bonds) - n_bonds
        ind_i_p = torch.cat([ind_i, torch.arange(n_real, device=dev),
                             torch.full((n_fill,), fake, device=dev)])
        ind_j_p = torch.cat([ind_j, torch.full((n_real + n_fill,), fake, device=dev)])
        atomic_mu_p = torch.cat([atomic_mu, atomic_mu[:1]])
        pad_vec = torch.full((n_real + n_fill, 3), FAR_AWAY,
                             dtype=self._torch_float, device=dev)
        bond_vector_p = torch.cat([bond_vector.to(self._torch_float), pad_vec])

        feed = {
            "atomic_mu_i": atomic_mu_p,
            "bond_vector": bond_vector_p,
            "ind_i": ind_i_p.to(torch.int32),
            "ind_j": ind_j_p.to(torch.int32),
            "mu_i": atomic_mu_p[ind_i_p],
            "mu_j": atomic_mu_p[ind_j_p],
        }

        if dev.type == "cuda":
            torch.cuda.synchronize()  # TF reads on its own stream, no implicit ordering
        tf_feed = {k: tf_dlpack.from_dlpack(torch_dlpack.to_dlpack(v.contiguous()))
                   for k, v in feed.items() if k in self._data_keys}
        tf_feed["batch_tot_nat"] = tf.constant(n_real + 1, dtype=tf.int32)
        tf_feed["batch_tot_nat_real"] = tf.constant(n_real, dtype=tf.int32)
        missing = self._data_keys - set(tf_feed)
        if missing:
            raise ValueError(f"artifact wants unsupported input keys: {sorted(missing)}")

        out = self._compute({k: tf_feed[k] for k in self._data_keys})

        # TF's to_dlpack drains its compute stream: outputs are materialized
        energy = torch.from_dlpack(tf_dlpack.to_dlpack(out["total_energy"]))
        forces = torch.from_dlpack(tf_dlpack.to_dlpack(out["total_f"]))
        results = {
            "energy": energy.reshape(-1)[:1].to(device=dev, dtype=self._dtype),
            "forces": forces[:n_real].to(device=dev, dtype=self._dtype).clone(),
        }
        if self._compute_stress:
            v = torch.from_dlpack(tf_dlpack.to_dlpack(out["virial"])).reshape(-1)
            xx, yy, zz, xy, xz, yz = (v[i] for i in range(6))
            virial = torch.stack([torch.stack([xx, xy, xz]),
                                  torch.stack([xy, yy, yz]),
                                  torch.stack([xz, yz, zz])])
            volume = torch.abs(torch.det(cell[0].to(self._torch_float)))
            results["stress"] = (-virial / volume).to(
                device=dev, dtype=self._dtype).unsqueeze(0)
        return results

    def validate(self, atoms, energy_atol_per_atom=1e-4, force_atol=1e-3):
        """Cross-check this bridge against the stock TPCalculator host path."""
        state = ts.initialize_state(atoms.copy(), self.device, self.dtype)
        out = self(state)
        ref = atoms.copy()
        ref.calc = self._calc
        de = abs(out["energy"].item() - ref.get_potential_energy()) / len(atoms)
        df = float((torch.from_numpy(ref.get_forces()).to(out["forces"])
                    - out["forces"]).abs().max())
        if de > energy_atol_per_atom or df > force_atol:
            raise RuntimeError(
                f"bridge disagrees with TPCalculator: |dE|/atom={de:.2e} eV, "
                f"max|dF|={df:.2e} eV/A")
        return de, df


# settings
MODEL_NAME = "grace-mp"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-matched"

CHAIN_LENGTH = 1              # Nose-Hoover chain settings, as in the ASE benchmark
CHAIN_STEPS = 1
SY_STEPS = 3
SEED = 42                     # Maxwell-Boltzmann velocity seed
STATE_DTYPE = torch.float64

device = torch.device("cuda")

# model
# float_dtype declares the graph input dtype and must match the fixed fp64 signature.
from tensorpotential.calculator.foundation_models import grace_fm

model = GraceModel(
    model=grace_fm("GRACE-2L-MP-r6", float_dtype="float64"),
    device=device, dtype=STATE_DTYPE, compute_stress=False,
)

run_torchsim_md(MODEL_NAME, model, "torch-sim-0.6.1+xla-dlpack-force-only", state_dtype=STATE_DTYPE, validate=model.validate)
