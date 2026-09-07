# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch-sim-atomistic[mattersim]==0.6.1",
#   "mattersim",
#   "ase>=3.26",
#   "torch==2.11.0+cu128",
#   "torchvision==0.26.0+cu128",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
# mattersim = { git = "https://github.com/microsoft/mattersim.git", rev = "df68847af1d66c6725bfadc58c4e3d3ec705bc5f" }
# ///
"""Force-only torch.compile TorchSim NVT production MD — mattersim-v1-5M.

Per-system MD parameters come from data/ref-trajs/md_metadata.json, which
records how each reference AIMD was run. The trajectory is saved as HDF5
with positions and velocities:

    <OUT_ROOT>/<system>/nvt_mattersim-v1-5M-compile-force-only.h5

Run:  uv run md_mattersim_v1_5M.py
"""

import csv
import json
import os
import time
from pathlib import Path

import torch

# TorchInductor's generated C++ precompiled header includes CUDA headers without
# always adding the wheel's CUDA runtime include directory. Make that directory
# visible before MatterSim invokes the in-process compiler.
CUDA_HOME = Path(os.environ.get("MATTERSIM_CUDA_HOME", "/usr/local/cuda")).resolve()
_site_packages = Path(torch.__file__).resolve().parent.parent
_wheel_cuda_include = _site_packages / "nvidia" / "cuda_runtime" / "include"
_cuda_include = (
    _wheel_cuda_include
    if (_wheel_cuda_include / "cuda_fp8.h").is_file()
    else CUDA_HOME / "include"
)
if not (_cuda_include / "cuda_fp8.h").is_file():
    raise FileNotFoundError(
        f"CUDA headers not found under {_cuda_include}; set MATTERSIM_CUDA_HOME "
        "to a complete CUDA toolkit"
    )
os.environ["CUDA_HOME"] = str(CUDA_HOME)
os.environ["CUDA_PATH"] = str(CUDA_HOME)
os.environ["CUDAToolkit_ROOT"] = str(CUDA_HOME)
os.environ["CPATH"] = os.pathsep.join(
    value
    for value in (
        str(_cuda_include),
        str(CUDA_HOME / "include"),
        os.environ.get("CPATH", ""),
    )
    if value
)

# no TF32, no autotuned kernels
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

import torch_sim as ts
from ase.io import read

from mattersim.forcefield import Potential
from mattersim.torchsim.graph_construction import build_graph_from_simstate
from mattersim.torchsim.torchsim_wrapper import TorchSimWrapper


class ForceOnlyTorchSimWrapper(TorchSimWrapper):
    """Run MatterSim for TorchSim NVT without computing unused stresses."""

    def __init__(self, model: Potential, *, device: torch.device | str) -> None:
        super().__init__(model=model, device=device)
        self._compute_stress = False
        self.implemented_properties = ["energy", "forces"]

    def forward(self, state: ts.SimState) -> dict[str, torch.Tensor]:
        if state.device != self._device:
            state = state.to(self._device)

        output_dtype = state.dtype
        graph_input = build_graph_from_simstate(
            state,
            twobody_cutoff=self.two_body_cutoff,
            threebody_cutoff=self.three_body_cutoff,
            max_num_neighbors_threshold=self._max_neighbors,
        )
        if output_dtype != self._model_dtype:
            graph_input = {
                key: value.to(dtype=self._model_dtype)
                if value.is_floating_point()
                else value
                for key, value in graph_input.items()
            }

        result = self.model.forward(
            graph_input,
            include_forces=True,
            include_stresses=False,
        )
        return {
            "energy": result["total_energy"].to(dtype=output_dtype).detach(),
            "forces": result["forces"].to(dtype=output_dtype).detach().reshape(-1, 3),
        }


# settings
MODEL_NAME = "mattersim-v1-5M-compile-force-only"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
METADATA_FILE = REPO / "updated_configs" /  "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-accelerated"

CHAIN_LENGTH = 1              # Nose-Hoover chain settings, as in the ASE benchmark
CHAIN_STEPS = 1
SY_STEPS = 3
SEED = 42                     # Maxwell-Boltzmann velocity seed
STATE_DTYPE = torch.float32

device = torch.device("cuda")

# model
# Checkpoint name resolved/downloaded by MatterSim itself, same as the ASE
# benchmark. Compile the M3GNet forward pass before wrapping it for TorchSim.
# Stress is intentionally omitted because NVT does not consume it. The first
# system pays the one-time TorchInductor compilation cost. This revision includes
# the self-image fix from merged PR #164.
potential = Potential.from_checkpoint(
    load_path="MatterSim-v1.0.0-5M.pth", model_name="m3gnet",
    device=device, load_training_state=False,
)
potential.model.forward = torch.compile(potential.model.forward)
model = ForceOnlyTorchSimWrapper(
    model=potential,
    device=device,
)

# NVT MD loop, one entry per system in the metadata file
METADATA = json.loads(METADATA_FILE.read_text())

for name, meta in METADATA.items():
    init_file = REPO / meta["initfile_path"]
    if not init_file.is_file():
        print(f"[{MODEL_NAME}] {name}: init file missing, skipping ({init_file})")
        continue

    out_dir = OUT_ROOT / name
    out_h5 = out_dir / f"nvt_{MODEL_NAME}.h5"
    out_csv = out_dir / f"md_timing_{MODEL_NAME}.csv"
    if out_csv.exists():
        print(f"[{MODEL_NAME}] {name}: output exists, skipping")
        continue

    temp_k = float(meta["temperature"])
    dt_fs = float(meta["timestep"])                        # timestep_units: fs
    tau_fs = float(meta["thermostat_coupling_constant"])   # coupling units: fs
    thermostat = meta["thermostat_type"]
    stride = int(meta["position_print_stride"] or 1)
    trajectory_length_ps = float(meta["trajectory_length_ps"])
    n_steps = round(trajectory_length_ps * 1000.0 / dt_fs)

    if thermostat == "Langevin":
        integrator = ts.Integrator.nvt_langevin
        init_kwargs = {}
        step_kwargs = {"gamma": 1000.0 / tau_fs}           # 1/tau, in ps^-1
    elif thermostat == "Nose-Hoover":  # Nose-Hoover
        integrator = ts.Integrator.nvt_nose_hoover
        init_kwargs = {"tau": tau_fs / 1000.0,             # ps
                       "chain_length": CHAIN_LENGTH,
                       "chain_steps": CHAIN_STEPS, "sy_steps": SY_STEPS}
        step_kwargs = {}
    else:  # velocity rescaling (CSVR)
        integrator = ts.Integrator.nvt_vrescale
        init_kwargs = {}
        step_kwargs = {"tau": tau_fs / 1000.0}              # ps

    atoms0 = read(init_file, index=0)                      # reference frame 0
    out_dir.mkdir(parents=True, exist_ok=True)

    state = ts.initialize_state(atoms0, device, STATE_DTYPE)
    state.rng = SEED

    print(f"[{MODEL_NAME}] {name}: T={temp_k:.0f} K, dt={dt_fs} fs, {thermostat} "
          f"tau={tau_fs} fs, {n_steps} steps, {len(atoms0)} atoms, stride {stride}")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    ts.integrate(
        system=state,
        model=model,
        integrator=integrator,
        n_steps=n_steps,
        temperature=temp_k,                  # Kelvin
        timestep=dt_fs / 1000.0,             # picoseconds, torch-sim metal units
        init_kwargs=init_kwargs,
        trajectory_reporter={
            "filenames": [str(out_h5)],
            "state_frequency": stride,
            "state_kwargs": {"save_velocities": True, "save_forces": False},
        },
        pbar=True,
        **step_kwargs,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["calculator", "system", "temperature_K",
                                               "n_steps", "time_step_fs", "thermostat",
                                               "tau_fs", "record_interval",
                                               "elapsed_seconds", "seconds_per_step",
                                               "engine", "seed"])
        writer.writeheader()
        writer.writerow({"calculator": MODEL_NAME, "system": name,
                         "temperature_K": temp_k, "n_steps": n_steps,
                         "time_step_fs": dt_fs, "thermostat": thermostat,
                         "tau_fs": tau_fs, "record_interval": stride,
                         "elapsed_seconds": f"{elapsed:.2f}",
                         "seconds_per_step": f"{elapsed / n_steps:.6f}",
                         "engine": "torch-sim-0.6.1+mattersim-torch-compile-force-only",
                         "seed": SEED})
    print(f"[{MODEL_NAME}] {name}: saved {out_h5} "
          f"({elapsed:.1f} s, {elapsed / n_steps * 1e3:.2f} ms/step)")

print(f"[{MODEL_NAME}] done.")
