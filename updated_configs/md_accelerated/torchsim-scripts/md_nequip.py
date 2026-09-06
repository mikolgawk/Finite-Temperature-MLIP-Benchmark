# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "torch-sim-atomistic[nequip]==0.6.1",
#   "ase>=3.26",
#   "torch",
#   "nequip==0.19.0",
#   "openequivariance==0.6.8",
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
# [tool.uv.config-settings-package.openequivariance]
# "cmake.define.CUDAToolkit_ROOT" = "/usr/local/cuda"
# ///
"""TorchSim NVT production MD -- NequIP-OAM-L with OpenEquivariance.

The NequIP model is AOTInductor-compiled for NequIP's ``batch`` target with
``enable_OpenEquivariance`` and position-only energy derivatives (no stress or
virial). The compiled artifact is cached outside the timed
region and evaluated by TorchSim; LAMMPS and JAX are not used.

Per-system parameters come from ``md_metadata.json``. Trajectories are saved
as:

    <OUT_ROOT>/<system>/nvt_nequip-oam-l-force-only.h5

Run:  uv run md_nequip.py
"""

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

# Match the numerical settings used by the other accelerated TorchSim runs.
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

# Importing OpenEquivariance registers the custom operators before the
# AOTInductor model is loaded. New NequIP artifacts also record this dependency,
# but the explicit import keeps older/reused artifacts working.
import openequivariance  # noqa: F401, E402
import torch_sim as ts  # noqa: E402
from ase.io import read  # noqa: E402
from nequip.integrations.torchsim import NequIPTorchSimCalc  # noqa: E402


MODEL_NAME = "nequip-oam-l-force-only"
MODEL_ID = os.environ.get(
    "NEQUIP_MODEL_ID", "nequip.net:mir-group/NequIP-OAM-L:0.1"
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = (
    REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-accelerated"
)
COMPILED_MODEL = Path(
    os.environ.get(
        "NEQUIP_OEQ_MODEL",
        REPO
        / "updated_configs"
        / "data"
        / "models"
        / "nequip-oam-l-oeq-torchsim-force-only.nequip.pt2",
    )
).resolve()

CHAIN_LENGTH = 1
CHAIN_STEPS = 1
SY_STEPS = 3
SEED = 42
STATE_DTYPE = torch.float32
DEVICE = torch.device("cuda")
CUDA_HOME = Path(os.environ.get("NEQUIP_CUDA_HOME", "/usr/local/cuda")).resolve()


def position_only_forward(self, data):
    """Differentiate energy with respect to positions, without strain or virial."""
    from nequip.data import AtomicDataDict

    data = dict(data)
    positions = data[AtomicDataDict.POSITIONS_KEY]
    positions.requires_grad_(True)
    data = self.func(data)
    data[AtomicDataDict.FORCE_KEY] = -torch.autograd.grad(
        data[AtomicDataDict.TOTAL_ENERGY_KEY].sum(), positions,
        create_graph=self.training,
    )[0]
    return data


def compile_force_only(args):
    """Apply OEq normally, then replace joint derivatives before AOT export."""
    from types import MethodType
    from nequip.data import AtomicDataDict
    from nequip.scripts import compile as compiler

    original_modify = compiler.modify

    def modify_force_only(model, modifiers):
        model = original_modify(model, modifiers)
        layers = [m for m in model.modules()
                  if m.__class__.__name__ == "ForceStressOutput"]
        if len(layers) != 1:
            raise RuntimeError("Expected one NequIP ForceStressOutput layer")
        layer = layers[0]
        layer.forward = MethodType(position_only_forward, layer)
        for key in (AtomicDataDict.STRESS_KEY, AtomicDataDict.VIRIAL_KEY,
                    AtomicDataDict.EDGE_FORCE_KEY):
            layer.irreps_out.pop(key, None)
        return model

    compiler.modify = modify_force_only
    # Keep the batch target's inputs and dynamic-shape policy; export E/F only.
    target = dict(compiler.COMPILE_TARGET_DICT["batch"])
    target["output"] = [AtomicDataDict.TOTAL_ENERGY_KEY, AtomicDataDict.FORCE_KEY]
    compiler.COMPILE_TARGET_DICT["batch"] = target
    compiler.main(args)


def compile_configuration() -> dict[str, object]:
    """Describe the cached model so a stale artifact is not silently reused."""
    return {
        "model": MODEL_ID,
        "mode": "aotinductor",
        "target": "batch",
        "modifier": "enable_OpenEquivariance",
        "derivatives": "positions-only-v1",
        "outputs": ["total_energy", "forces"],
        "device": "cuda",
        "allow_tf32": False,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_home": str(CUDA_HOME),
    }


def ensure_compiled_model() -> Path:
    """Compile and cache the OpenEquivariance TorchSim model when needed."""
    config_file = Path(str(COMPILED_MODEL) + ".benchmark-config.json")
    configuration = compile_configuration()

    if COMPILED_MODEL.is_file() and config_file.is_file():
        try:
            if json.loads(config_file.read_text()) == configuration:
                return COMPILED_MODEL
        except json.JSONDecodeError:
            pass

    # Rebuild artifacts without a matching force-only configuration sidecar.

    if not torch.cuda.is_available():
        raise RuntimeError(
            "a CUDA GPU is required to compile and run the OpenEquivariance model"
        )

    COMPILED_MODEL.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = COMPILED_MODEL.with_name(
        f".{COMPILED_MODEL.name}.inprogress.nequip.pt2"
    )
    temporary_model.unlink(missing_ok=True)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--compile-force-only",
        MODEL_ID,
        str(temporary_model),
        "--device",
        "cuda",
        "--mode",
        "aotinductor",
        "--target",
        "batch",
        "--modifiers",
        "enable_OpenEquivariance",
        "--no-tf32",
    ]
    environment = os.environ.copy()
    environment["CUDA_HOME"] = str(CUDA_HOME)
    environment["CUDA_PATH"] = str(CUDA_HOME)
    environment["CUDAToolkit_ROOT"] = str(CUDA_HOME)
    environment["PATH"] = os.pathsep.join(
        (str(CUDA_HOME / "bin"), environment.get("PATH", ""))
    )

    # PyTorch 2.11's AOTInductor precompiled-header command includes its own
    # CUDA wrapper but omits the CUDA include directory. Prefer the CUDA 12.8
    # headers installed beside the cu128 PyTorch wheel, which avoids mixing the
    # wrapper with the obsolete system CUDA 11.5 headers found in /usr/include.
    site_packages = Path(torch.__file__).resolve().parent.parent
    torch_cuda_include = site_packages / "nvidia" / "cuda_runtime" / "include"
    cuda_include = (
        torch_cuda_include
        if (torch_cuda_include / "cuda_fp8.h").is_file()
        else CUDA_HOME / "include"
    )
    if not (cuda_include / "cuda_fp8.h").is_file():
        raise FileNotFoundError(
            f"CUDA headers not found under {cuda_include}; set NEQUIP_CUDA_HOME "
            "to a complete CUDA toolkit"
        )
    environment["CPATH"] = os.pathsep.join(
        value
        for value in (
            str(cuda_include),
            str(CUDA_HOME / "include"),
            environment.get("CPATH", ""),
        )
        if value
    )

    print(
        f"[{MODEL_NAME}] compiling {MODEL_ID} for TorchSim with "
        f"OpenEquivariance -> {COMPILED_MODEL}"
    )
    try:
        subprocess.run(command, check=True, env=environment)
        if not temporary_model.is_file():
            raise RuntimeError("nequip-compile did not produce the expected artifact")
        temporary_model.replace(COMPILED_MODEL)
        config_file.write_text(json.dumps(configuration, indent=2) + "\n")
    except BaseException:
        temporary_model.unlink(missing_ok=True)
        raise

    return COMPILED_MODEL


def load_model() -> NequIPTorchSimCalc:
    compile_path = ensure_compiled_model()
    model = NequIPTorchSimCalc.from_compiled_model(
        compile_path=compile_path,
        device=DEVICE,
        chemical_species_to_atom_type_map=True,
        # NequIP's GPU-native, batched neighbor-list implementation.
        neighborlist_backend="alchemiops",
    )
    model.compute_forces = True
    model.compute_stress = False
    return model


def thermostat_settings(
    name: str, tau_fs: float
) -> tuple[ts.Integrator, dict[str, object], dict[str, float]]:
    if name == "Langevin":
        return ts.Integrator.nvt_langevin, {}, {"gamma": 1000.0 / tau_fs}
    if name == "Nose-Hoover":
        return (
            ts.Integrator.nvt_nose_hoover,
            {
                "tau": tau_fs / 1000.0,
                "chain_length": CHAIN_LENGTH,
                "chain_steps": CHAIN_STEPS,
                "sy_steps": SY_STEPS,
            },
            {},
        )
    if name.lower().startswith("velocity"):
        return ts.Integrator.nvt_vrescale, {}, {"tau": tau_fs / 1000.0}
    raise ValueError(f"unknown thermostat type {name!r}")


def run_system(name: str, meta: dict[str, object], model: NequIPTorchSimCalc) -> None:
    init_file = REPO / str(meta["initfile_path"])
    if not init_file.is_file():
        print(f"[{MODEL_NAME}] {name}: init file missing, skipping ({init_file})")
        return

    out_dir = OUT_ROOT / name
    out_h5 = out_dir / f"nvt_{MODEL_NAME}.h5"
    out_csv = out_dir / f"md_timing_{MODEL_NAME}.csv"
    if out_h5.exists() and out_csv.exists():
        print(f"[{MODEL_NAME}] {name}: output exists, skipping")
        return

    temperature_k = float(meta["temperature"])
    timestep_fs = float(meta["timestep"])
    tau_fs = float(meta["thermostat_coupling_constant"])
    thermostat = str(meta["thermostat_type"])
    stride = int(meta["position_print_stride"] or 1)
    n_steps = round(float(meta["trajectory_length_ps"]) * 1000.0 / timestep_fs)
    integrator, init_kwargs, step_kwargs = thermostat_settings(thermostat, tau_fs)

    atoms = read(init_file, index=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = ts.initialize_state(atoms, DEVICE, STATE_DTYPE)
    state.rng = SEED

    print(
        f"[{MODEL_NAME}] {name}: T={temperature_k:.0f} K, dt={timestep_fs:g} fs, "
        f"{thermostat} tau={tau_fs:g} fs, {n_steps} steps, {len(atoms)} atoms, "
        f"stride {stride}"
    )
    torch.cuda.synchronize()
    start = time.perf_counter()

    ts.integrate(
        system=state,
        model=model,
        integrator=integrator,
        n_steps=n_steps,
        temperature=temperature_k,
        timestep=timestep_fs / 1000.0,
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
    elapsed = time.perf_counter() - start

    with out_csv.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "calculator",
                "system",
                "temperature_K",
                "n_steps",
                "time_step_fs",
                "thermostat",
                "tau_fs",
                "record_interval",
                "elapsed_seconds",
                "seconds_per_step",
                "engine",
                "kernel",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "calculator": MODEL_NAME,
                "system": name,
                "temperature_K": temperature_k,
                "n_steps": n_steps,
                "time_step_fs": timestep_fs,
                "thermostat": thermostat,
                "tau_fs": tau_fs,
                "record_interval": stride,
                "elapsed_seconds": f"{elapsed:.2f}",
                "seconds_per_step": f"{elapsed / n_steps:.6f}",
                "engine": "torch-sim-0.6.1+aotinductor-force-only",
                "kernel": "OpenEquivariance",
                "seed": SEED,
            }
        )

    print(
        f"[{MODEL_NAME}] {name}: saved {out_h5} "
        f"({elapsed:.1f} s, {elapsed / n_steps * 1e3:.2f} ms/step)"
    )


def main() -> None:
    metadata = json.loads(METADATA_FILE.read_text())
    model = load_model()
    for name, meta in metadata.items():
        run_system(name, meta, model)
    print(f"[{MODEL_NAME}] done.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compile-force-only":
        compile_force_only(sys.argv[2:])
    else:
        main()
