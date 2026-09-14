"""Canonical data locations used by the energy/force-RMSE scripts."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
UPDATED_CONFIGS_DIR = BASE_DIR.parent

# TorchSim-generated trajectories, reference AIMD trajectories, and model
# assets live with the updated configuration data.
DATA_DIR = UPDATED_CONFIGS_DIR / "data"
MLIP_TRAJ_DIR = DATA_DIR / "mlip-trajs-torchsim-eager"
REF_TRAJ_DIR = DATA_DIR / "ref-trajs"

# Per-model RMSE summaries are kept with the energy/force analysis code.
RMSE_RESULTS_DIR = BASE_DIR / "data" / "e-f-predictions"
