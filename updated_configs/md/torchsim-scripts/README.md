# TorchSim MD scripts

Python scripts for running the benchmark's NVT molecular dynamics workloads
with MLIPs. Each `md_*.py` file declares its own
Python requirements and dependencies in an inline `uv` script metadata block.
The production loops use CUDA and evaluate energies and forces with stress disabled.

## Requirements

- `uv` available on `PATH`, and Bash for the batch runner.
- A CUDA-capable NVIDIA GPU with a driver compatible with the packages declared
  in the selected script. These production loops require CUDA.
- Python satisfying each script's metadata (all require at least 3.12; some
  explicitly require Python 3.12).
- The repository's input data and any required model checkpoints. Dependency and
  model downloads need network access unless already cached; model loaders may
  also require credentials or access to the relevant model repository.

Keep this directory in its repository location: scripts derive the repository
root from their own file paths.

## Run all models

From this directory:

```bash
./run_all_md.sh
```

The runner discovers every `md_*.py` beside it and runs them sequentially in shell
glob order using `uv run --script`. It changes into this directory automatically,
so it can also be invoked by its path from another working directory.

```bash
# Preview the commands without installing dependencies or running MD.
./run_all_md.sh --dry-run

# Select a GPU for the entire batch.
CUDA_VISIBLE_DEVICES=0 ./run_all_md.sh

# Show runner usage.
./run_all_md.sh --help
```

Output is streamed to the terminal and saved separately for each script:

```text
logs/md/<YYYYMMDD_HHMMSS>_<pid>/<script-name>.log
```

A failed script or logging operation is recorded and the batch continues. The
runner prints a failure summary and exits with status 1 if any run failed, or 0
if all scripts succeeded. Interrupting the batch stops it. Success reflects
process exit status; missing input structures can still be skipped by the MD scripts.

## Run one model

```bash
uv run --script md_orb_v3_direct.py

# Select a GPU for a single model.
CUDA_VISIBLE_DEVICES=0 uv run --script md_orb_v3_direct.py
```

The scripts do not expose a shared command-line configuration interface. Simulation
settings come from the metadata file and constants in each script.

## Available scripts

| Script | Model |
| --- | --- |
| `md_eq_v2_M_omat.py` | EquiformerV2 M OMat mp salex |
| `md_esen_30M_OAM.py` | eSEN 30M OAM |
| `md_grace_mp.py` | GRACE MP |
| `md_grace_oam.py` | GRACE OAM |
| `md_mace_mh_omat.py` | MACE MH OMat |
| `md_mace_mp_0.py` | MACE MP 0 |
| `md_mace_mpa_0.py` | MACE MPA 0 |
| `md_mattersim_v1_5M.py` | MatterSim v1 5M |
| `md_nequip.py` | NequIP OAM L |
| `md_orb_v2.py` | ORB v2 |
| `md_orb_v3.py` | ORB v3 |
| `md_orb_v3_direct.py` | ORB v3 direct forces |
| `md_pet_oam_xl.py` | PET OAM XL |
| `md_pet_omat_xl.py` | PET OMat XL |
| `md_uma_m_omat.py` | UMA M, OMat task |
| `md_uma_s_omat.py` | UMA S, OMat task |

## Inputs and model files

Paths below are relative to the repository root. All scripts read:

```text
updated_configs/data/ref-trajs/md_metadata.json
```

This JSON maps system names to settings including `initfile_path`,
`temperature`, `timestep`, `thermostat_type`, `thermostat_coupling_constant`,
`position_print_stride`, and `trajectory_length_ps`. Initial structure paths are
resolved relative to the repository root. Missing structures are reported and skipped.

The legacy FairChem scripts expect these local checkpoints:

| Script | Checkpoint |
| --- | --- |
| `md_eq_v2_M_omat.py` | `updated_configs/data/models/eqV2_86M_omat_mp_salex.pt` |
| `md_esen_30M_OAM.py` | `updated_configs/data/models/esen_30m_oam.pt` |

Other scripts use their model libraries' loaders and caches. GRACE OAM also
prepares a local artifact at
`updated_configs/data/models/GRACE-2L-OMAT-large-ft-AM-fp32` if absent. Its eager
loader accepts `GRACE_CACHE`, `GRACE_OAM_POTENTIAL`, and `GRACE_OAM_CHECKPOINT`
overrides; the checkpoint value is a prefix without `.index`.

## Outputs and reruns

Simulation results are written under:

```text
updated_configs/data/mlip-trajs-torchsim-matched/<system>/
    nvt_<model-name>.h5
    md_timing_<model-name>.csv
```

The model name is defined inside each script and may differ from its filename.
HDF5 trajectories save velocities, with force saving disabled. Timing CSVs record
the simulation settings, elapsed time, time per step, engine, and seed. Each
system has a warmup evaluation before the timed integration.

An existing timing CSV causes that model/system pair to be skipped. Rerunning
the batch therefore skips completed pairs, but does not resume an interrupted
trajectory from its last frame. To intentionally repeat a completed pair, move
its existing trajectory and timing CSV aside before rerunning the model.
