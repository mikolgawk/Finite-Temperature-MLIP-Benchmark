# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "tqdm>=4.66",
#   "pandas>=2.2",
#   "h5py>=3.11",
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

import sys
from pathlib import Path
_PRESSURE_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "pressures")
sys.path.insert(0, str(_PRESSURE_ROOT))
from pressure_evaluator import early_cli, run_torchsim_pressure

# The model compiler is run in a clean subprocess by invoking this script with
# an internal flag.  Do not let the pressure evaluator consume compiler args.
_COMPILE_STRESS_MODE = sys.argv[1:2] == ["--compile-stress"]
if not _COMPILE_STRESS_MODE:
    early_cli(__file__, "torchsim")

"""Stress-enabled md_nequip.py: record potential stress for every saved trajectory frame."""

FAILED_SYSTEMS = []

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


def frame_stress(state, model):
    """Record one finite potential stress tensor per reported system."""
    stress = model(state)["stress"].detach()
    if stress.numel() != state.n_systems * 9 or not bool(torch.isfinite(stress).all()):
        raise ValueError("Model returned an invalid per-frame stress tensor")
    return stress.reshape(state.n_systems, 3, 3)


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


MODEL_NAME = "nequip-oam-l-stress"
MODEL_ID = os.environ.get(
    "NEQUIP_MODEL_ID", "nequip.net:mir-group/NequIP-OAM-L:0.1"
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
METADATA_FILE = REPO / "updated_configs" / "data" / "ref-trajs" / "md_metadata.json"
OUT_ROOT = (
    REPO / "updated_configs" / "data" / "mlip-trajs-torchsim-accelerated-stress"
)
COMPILED_MODEL = Path(
    os.environ.get(
        "NEQUIP_OEQ_STRESS_MODEL",
        REPO
        / "updated_configs"
        / "data"
        / "models"
        / "nequip-oam-l-oeq-torchsim-stress.nequip.pt2",
    )
).resolve()

CHAIN_LENGTH = 1
CHAIN_STEPS = 1
SY_STEPS = 3
SEED = 42
STATE_DTYPE = torch.float32
DEVICE = torch.device("cuda")
CUDA_HOME = Path(os.environ.get("NEQUIP_CUDA_HOME", "/usr/local/cuda")).resolve()





def compile_with_stress(args):
    """Compile the standard joint force/stress graph with OpenEquivariance."""
    from nequip.scripts import compile as compiler
    compiler.main(args)


def compile_configuration() -> dict[str, object]:
    """Describe the cached model so a stale artifact is not silently reused."""
    return {
        "model": MODEL_ID,
        "mode": "aotinductor",
        # Pressure post-processing evaluates one trajectory frame at a time.
        # NequIP's batch target requires at least two frames, while its ASE
        # target is the standard static single-frame AOT signature.
        "target": "ase",
        "modifier": "enable_OpenEquivariance",
        "derivatives": "positions-and-strain-v1",
        "outputs": ["total_energy", "forces", "stress"],
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

    # Rebuild artifacts without a matching stress configuration sidecar.

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
        "--compile-stress",
        MODEL_ID,
        str(temporary_model),
        "--device",
        "cuda",
        "--mode",
        "aotinductor",
        "--target",
        "ase",
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


class WithStressNequIPTorchSimCalc(NequIPTorchSimCalc):
    """Load the static single-frame export through the TorchSim integration."""

    @classmethod
    def _get_aoti_compile_target(cls):
        # NequIPTorchSimCalc normally requires the multi-frame ``batch``
        # signature. This evaluator always supplies one system, so use the
        # compatible static signature while retaining TorchSim transforms and
        # neighbor-list construction.
        from nequip.scripts._compile_utils import (
            AOTI_ASE_TARGET,
            COMPILE_TARGET_DICT,
        )

        return COMPILE_TARGET_DICT[AOTI_ASE_TARGET]


def load_model() -> NequIPTorchSimCalc:
    compile_path = ensure_compiled_model()
    model = WithStressNequIPTorchSimCalc.from_compiled_model(
        compile_path=compile_path,
        device=DEVICE,
        chemical_species_to_atom_type_map=True,
        # NequIP's GPU-native, batched neighbor-list implementation.
        neighborlist_backend="alchemiops",
    )
    model.compute_forces = True
    model.compute_stress = True
    return model



if __name__ == "__main__":
    if _COMPILE_STRESS_MODE:
        compile_with_stress(sys.argv[2:])
    else:
        run_torchsim_pressure(
            MODEL_NAME, load_model(), "torchsim-nequip-stress-postprocessing"
        )
