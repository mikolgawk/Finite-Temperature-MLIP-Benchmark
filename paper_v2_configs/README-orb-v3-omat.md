# ORB v3 OMAT analysis

Run these commands from `paper_v2_configs/`. The conservative and direct OMAT
checkpoints have their own model identifiers and outputs.

The OMAT RMSE and pressure scripts use the shared metric implementations for both TorchSim and ASE, in eager and accelerated modes.
They preserve the same shared evaluators, CLI options, dependencies, and output
conventions. The model names identify the OMAT checkpoints, and direct-model precision
matches the OMAT MD scripts.

| Mode | Conservative trajectory identifier | Direct trajectory identifier |
| --- | --- | --- |
| Eager | `orb-v3-omat-force-only-eager` | `orb-v3-direct-omat-force-only-eager` |
| Accelerated | `orb-v3-omat-force-only` | `orb-v3-direct-omat-force-only` |

The corresponding MD runners must finish first. RMSE and pressure evaluation
require CUDA, `uv`, and access to the model packages/checkpoints. The analysis
uses only model/system pairs with a completed timing CSV and an HDF5 trajectory.

## Eager

Evaluate energies and forces on the reference frames and stress on the saved
production frames:

```bash
uv run --script e_f_rmses/rmse_torchsim_scripts/md_eager/rmse_orb_v3_omat.py
uv run --script e_f_rmses/rmse_torchsim_scripts/md_eager/rmse_orb_v3_direct_omat.py
uv run --script pressures/md_eager/torchsim-scripts/pressure_orb_v3_omat.py
uv run --script pressures/md_eager/torchsim-scripts/pressure_orb_v3_direct_omat.py
```

Then aggregate RMSE and pressure results and compute matched-length RDF/VDOS:

```bash
uv run --script generate_metrics.py \
  --source mlip-trajs-torchsim-eager \
  --model orb-v3-omat-force-only-eager \
  --model orb-v3-direct-omat-force-only-eager
```

## Accelerated

```bash
uv run --script e_f_rmses/rmse_torchsim_scripts/md-accelerated/rmse_orb_v3_omat.py
uv run --script e_f_rmses/rmse_torchsim_scripts/md-accelerated/rmse_orb_v3_direct_omat.py
uv run --script pressures/md_accelerated/torchsim-scripts/pressure_orb_v3_omat.py
uv run --script pressures/md_accelerated/torchsim-scripts/pressure_orb_v3_direct_omat.py

uv run --script generate_metrics.py \
  --source mlip-trajs-torchsim-accelerated \
  --model orb-v3-omat-force-only \
  --model orb-v3-direct-omat-force-only
```

`--model` filters use the full trajectory identifiers in the table. Add
`--dry-run` to `generate_metrics.py` to inspect its commands. Omitting the model
filters processes all discovered models, including the new OMAT variants.
The RMSE and pressure batch launchers discover the new runners automatically.

## ASE metric counterparts

The matching ASE runners are also available in both modes:

```bash
uv run --script e_f_rmses/rmse_ase_scripts/md_eager/rmse_orb_v3_omat.py
uv run --script e_f_rmses/rmse_ase_scripts/md_eager/rmse_orb_v3_direct_omat.py
uv run --script pressures/md_eager/ase-scripts/pressure_orb_v3_omat.py
uv run --script pressures/md_eager/ase-scripts/pressure_orb_v3_direct_omat.py

uv run --script e_f_rmses/rmse_ase_scripts/md-accelerated/rmse_orb_v3_omat.py
uv run --script e_f_rmses/rmse_ase_scripts/md-accelerated/rmse_orb_v3_direct_omat.py
uv run --script pressures/md_accelerated/ase-scripts/pressure_orb_v3_omat.py
uv run --script pressures/md_accelerated/ase-scripts/pressure_orb_v3_direct_omat.py
```

Eager ASE RMSE identifiers are `orb-v3-omat`
and `orb-v3-direct-omat`; accelerated RMSE identifiers add `-force-only`.
ASE prediction summaries go to
`e_f_rmses/data/e-f-predictions-ase/{md_eager,md-accelerated}/`. ASE pressure
evaluators read the saved TorchSim trajectories and write to
`pressures/results/ase/{md_eager,md_accelerated}/`. Their results remain separate
from the TorchSim metric outputs. RDF and VDOS use the existing shared pipelines.

## Results and plots

RMSE predictions go to `e_f_rmses/data/e-f-predictions/md_eager/` or
`e_f_rmses/data/e-f-predictions/md-accelerated/`. Pressure frame results go to
`pressures/results/torchsim/md_eager/` or
`pressures/results/torchsim/md_accelerated/`, using the canonical
`orb-v3-omat` and `orb-v3-direct-omat` names.

The four metric summaries go to their respective `results/<source>/` directories,
keeping eager and accelerated results separate. As with the rest of the current
metric pipeline, `generate_metrics.py` excludes molecular crystals. Model-filtered
runs write summaries for the selected models; run again without model filters
when producing figures for the complete benchmark.

The energy/force, pressure, RDF, VDOS, and Pareto figures recognize both OMAT
variants alongside the existing Tier 3 ORB models, with labels
`orb-v3-conservative-inf-omat` and `orb-v3-direct-20-omat`. The legacy comparison
scripts omit these new checkpoints because they have no v1 counterpart.

Evaluator settings match the MD runners: conservative OMAT uses strict FP32
(`float32-highest`, TF32 disabled), while direct OMAT uses `float32-high`
(TF32 enabled). Eager evaluation disables compilation; accelerated evaluation
enables it. Pressure evaluators leave stress enabled and evaluate existing MD
frames without running a second simulation.
