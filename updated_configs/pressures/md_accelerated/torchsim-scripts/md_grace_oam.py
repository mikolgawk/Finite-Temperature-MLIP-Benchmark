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
"""Stress-enabled md_grace_oam.py: record potential stress for every saved trajectory frame."""

FAILED_SYSTEMS = []

import csv
import json
import math
import os
import time
from pathlib import Path

import torch


def frame_stress(state, model):
    """Record one finite potential stress tensor per reported system."""
    stress = model(state)["stress"].detach()
    if stress.numel() != state.n_systems * 9 or not bool(torch.isfinite(stress).all()):
        raise ValueError("Model returned an invalid per-frame stress tensor")
    return stress.reshape(state.n_systems, 3, 3)


# no TF32, no autotuned kernels
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
# TensorPotential 0.5.7.2's compiled fp32 path is not numerically consistent with
# its exported XLA model when oneDNN graph rewrites are enabled.
if os.environ.get("TF_ENABLE_ONEDNN_OPTS", "0") != "0":
    raise RuntimeError("GRACE compiled inference requires TF_ENABLE_ONEDNN_OPTS=0")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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

# Embedded implementation: this runner has no local helper imports.

# Embedded implementation: this runner has no local helper imports.
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
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-accelerated-stress"

CHAIN_LENGTH = 1
CHAIN_STEPS = 1
SY_STEPS = 3
SEED = 42
STATE_DTYPE = torch.float32


def run_torchsim_md(
    model_name: str,
    model,
    engine: str,
    *,
    require_stress_enabled: bool = True,
    warmup: bool = True,
    state_dtype: torch.dtype = STATE_DTYPE,
    validate=None,
) -> None:
    """Run the catalog's matched NVT workload for one TorchSim model."""
    metadata = json.loads(METADATA_FILE.read_text())
    device = torch.device("cuda")

    if require_stress_enabled and not model.compute_stress:
        raise ValueError(f"{model_name} does not support stress")

    for name, meta in metadata.items():
        out_dir = OUT_ROOT / name
        out_h5 = out_dir / f"nvt_{model_name}.h5"
        out_csv = out_dir / f"md_timing_{model_name}.csv"
        if out_h5.is_file() and out_csv.is_file() and out_csv.stat().st_size > 0:
            print(f"[{model_name}] {name}: output exists, skipping")
            continue

        try:
            init_file = next((p for p in (REPO / meta["initfile_path"], REPO / "updated_configs" / meta["initfile_path"]) if p.is_file()), REPO / meta["initfile_path"])
            if not init_file.is_file():
                print(f"[{model_name}] {name}: init file missing, skipping ({init_file})")
                continue

            temp_k = float(meta["temperature"])
            dt_fs = float(meta["timestep"])
            tau_fs = float(meta["thermostat_coupling_constant"])
            thermostat = meta["thermostat_type"]
            stride = int(meta["position_print_stride"] or 1)
            trajectory_length_ps = float(meta["trajectory_length_ps"])
            n_steps = round(trajectory_length_ps * 1000.0 / dt_fs)

            if thermostat == "Langevin":
                integrator = ts.Integrator.nvt_langevin
                init_kwargs = {}
                step_kwargs = {"gamma": 1000.0 / tau_fs}
            elif thermostat == "Nose-Hoover":
                integrator = ts.Integrator.nvt_nose_hoover
                init_kwargs = {
                    "tau": tau_fs / 1000.0,
                    "chain_length": CHAIN_LENGTH,
                    "chain_steps": CHAIN_STEPS,
                    "sy_steps": SY_STEPS,
                }
                step_kwargs = {}
            else:
                integrator = ts.Integrator.nvt_vrescale
                init_kwargs = {}
                step_kwargs = {"tau": tau_fs / 1000.0}

            atoms0 = read(init_file, index=0)
            out_dir.mkdir(parents=True, exist_ok=True)

            state = ts.initialize_state(atoms0, device, state_dtype)
            state.rng = SEED

            # Uniform policy: exactly one force/energy evaluation on the initial
            # state, outside the timed region, without changing the MD state.
            if warmup:
                # A validation call already evaluates the production model once.
                # Use it as that system's warmup rather than evaluating twice.
                if validate is not None:
                    validate(atoms0)
                else:
                    model(state)
                torch.cuda.synchronize()

            print(
                f"[{model_name}] {name}: T={temp_k:.0f} K, dt={dt_fs} fs, "
                f"{thermostat} tau={tau_fs} fs, {n_steps} steps, "
                f"{len(atoms0)} atoms, stride {stride}"
            )
            torch.cuda.synchronize()
            start = time.perf_counter()

            ts.integrate(
                system=state,
                model=model,
                integrator=integrator,
                n_steps=n_steps,
                temperature=temp_k,
                timestep=dt_fs / 1000.0,
                init_kwargs=init_kwargs,
                trajectory_reporter={
                    "filenames": [str(out_h5)],
                    "state_frequency": stride,
                    "prop_calculators": {stride: {"stress": frame_stress}},
                    "metadata": {"stress_units": "eV/Angstrom^3",
                                 "stress_kind": "potential; tensile-positive; no kinetic term"},
                    "state_kwargs": {"save_velocities": True, "save_forces": False},
                },
                pbar=True,
                **step_kwargs,
            )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            with out_csv.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "calculator", "system", "temperature_K", "n_steps",
                        "time_step_fs", "thermostat", "tau_fs", "record_interval",
                        "elapsed_seconds", "seconds_per_step", "engine", "seed",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "calculator": model_name,
                        "system": name,
                        "temperature_K": temp_k,
                        "n_steps": n_steps,
                        "time_step_fs": dt_fs,
                        "thermostat": thermostat,
                        "tau_fs": tau_fs,
                        "record_interval": stride,
                        "elapsed_seconds": f"{elapsed:.2f}",
                        "seconds_per_step": f"{elapsed / n_steps:.6f}",
                        "engine": engine,
                        "seed": SEED,
                    }
                )
            print(
                f"[{model_name}] {name}: saved {out_h5} "
                f"({elapsed:.1f} s, {elapsed / n_steps * 1e3:.2f} ms/step)"
            )
        except Exception as exc:
            print(f"[{model_name}] {name}: FAILED ({type(exc).__name__}: {exc})")
            FAILED_SYSTEMS.append(name)
            # A zero-byte timing CSV marks this model/system as failed.
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                out_csv.write_bytes(b"")
            except OSError as marker_error:
                print(f"[{model_name}] {name}: could not write failure CSV: {marker_error}")

    print(f"[{model_name}] done.")

    if FAILED_SYSTEMS:

        raise RuntimeError(f"MD failed for {FAILED_SYSTEMS}")


FAR_AWAY = 52.0  # fake bond vector component, beyond any cutoff


class GraceModel(ModelInterface):
    """TorchSim interface to GRACE compiled inference via DLPack."""

    def __init__(self, model, device=None, dtype=torch.float32,
                 compute_stress=True, pad_fraction=0.05):
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
        self._compute = self._tp.compute

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
        # Keep the original bridge padding policy for comparable inputs
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

        # TPCalculator turns isolated ASE structures into a sufficiently large
        # periodic cell before building its neighbour list.  Do the same here;
        # Vesin otherwise sees a non-periodic zero-cell molecule.
        nl_cell, nl_pbc = cell, state.pbc
        if not bool(state.pbc.any()):
            extent = torch.linalg.vector_norm(pos - pos[0], dim=1).max()
            side = 2.0 * (extent + self._cutoff)
            nl_cell = torch.eye(3, dtype=pos.dtype, device=dev).unsqueeze(0) * side
            nl_pbc = torch.ones_like(state.pbc)
        mapping, sys_map, unit_shifts = vesin_nl_ts(
            pos, nl_cell, nl_pbc, self._cutoff, state.system_idx)
        shifts = compute_cell_shifts(nl_cell, unit_shifts, sys_map)
        ind_i, ind_j = mapping[0], mapping[1]
        bond_vector = pos[ind_j] - pos[ind_i] + shifts

        atomic_mu = self._z_to_mu[state.atomic_numbers]
        if bool((atomic_mu < 0).any()):
            raise ValueError("structure contains elements outside the model's element map")

        # Match TensorPotential's padding: only atoms with no real neighbours
        # get a far-away bond; all remaining padded bonds are fake-to-fake.
        fake = n_real
        has_neighbour = torch.zeros(n_real, dtype=torch.bool, device=dev)
        has_neighbour[ind_i] = True
        isolated = torch.arange(n_real, device=dev)[~has_neighbour]
        n_bonds = ind_i.shape[0] + isolated.shape[0]
        n_fill = self._bucket(n_bonds) - n_bonds
        ind_i_p = torch.cat([ind_i, isolated,
                             torch.full((n_fill,), fake, device=dev)])
        ind_j_p = torch.cat([ind_j, torch.zeros_like(isolated),
                             torch.full((n_fill,), fake, device=dev)])
        atomic_mu_p = torch.cat([atomic_mu, atomic_mu[:1]])
        pad_vec = torch.full((isolated.shape[0] + n_fill, 3), FAR_AWAY,
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
        ds = float((torch.from_numpy(ref.get_stress(voigt=False)).to(out["stress"])
                    - out["stress"][0]).abs().max())
        if ds > 1e-4:
            raise RuntimeError(f"GRACE stress disagrees with TPCalculator: max|dS|={ds:.2e} eV/A^3")
        if de > energy_atol_per_atom or df > force_atol:
            raise RuntimeError(
                f"bridge disagrees with TPCalculator: |dE|/atom={de:.2e} eV, "
                f"max|dF|={df:.2e} eV/A")
        return de, df


def main():
    # settings
    MODEL_NAME = "grace-oam-stress-compiled"
    HERE = Path(__file__).resolve().parent
    REPO = HERE.parents[3]
    METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
    OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-accelerated-stress"
    MODEL_PATH = REPO / "updated_configs" / "data" / "models" / "GRACE-2L-OMAT-large-ft-AM-fp32"

    CHAIN_LENGTH = 1              # Nose-Hoover chain settings, as in the ASE benchmark
    CHAIN_STEPS = 1
    SY_STEPS = 3
    SEED = 42                     # Maxwell-Boltzmann velocity seed
    STATE_DTYPE = torch.float32
    VALIDATE_ONLY = os.environ.get("VALIDATE_ONLY") == "1"

    device = torch.device("cuda")

    # model
    # the fp32 artifact used by the ASE benchmark; the official checkpoint is already fp32
    if not MODEL_PATH.is_dir():
        import glob
        from types import SimpleNamespace
        from tensorpotential.calculator.foundation_models import get_or_download_checkpoint
        from tensorpotential.scripts.grace_utils import cast_model
        ckpt_dir = get_or_download_checkpoint("GRACE-2L-OMAT-large-ft-AM")
        ckpt = glob.glob(os.path.join(ckpt_dir, "*.index"))[0].removesuffix(".index")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        os.chdir(MODEL_PATH.parent)
        cast_model(SimpleNamespace(potential=os.path.join(ckpt_dir, "model.yaml"),
                                   checkpoint_path=ckpt, output_suffix="-fp32",
                                   curr="fp32", to="fp32"))
        os.replace("casted_model", MODEL_PATH.name)
        os.chdir(HERE)
    model = GraceModel(
        model=TPCalculator(str(MODEL_PATH), float_dtype="float32"),
        device=device, dtype=STATE_DTYPE, compute_stress=True,
    )

    run_torchsim_md(MODEL_NAME, model, "torch-sim-0.6.1+tf-compiled-dlpack-stress", validate=None)


if __name__ == "__main__":
    main()
