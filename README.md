# Finite-Temperature MLIP Benchmark

Run MD, evaluate energies and forces, compute RDF/pressure/VDOS metrics, and
plot the results using `paper_v2_configs/`. Run the commands below from the
repository root. Complete the [requirements and data setup](INFO.md#requirements)
first; GPU evaluations require the relevant model dependencies and checkpoints.

Reference inputs must follow this layout:

```text
paper_v2_configs/data/ref-trajs/
├── md_metadata.json
└── <system>/traj.extxyz
```

## Run MD

Run the eager and accelerated TorchSim workloads:

```bash
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/md_eager/torchsim-scripts/run_all_md.sh
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/md_accelerated/torchsim-scripts/run_all_md.sh
```

For ASE workloads:

```bash
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/md_eager/ase-scripts/run_all_md.sh
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/md_accelerated/ase-scripts/run_all_md.sh
```

Change `CUDA_VISIBLE_DEVICES` to select a GPU. Add `--dry-run` to a batch
launcher to list its commands. Launchers run models sequentially and save logs
under their script directories. To run one model, invoke its script directly:

```bash
uv run --script paper_v2_configs/md_accelerated/torchsim-scripts/md_mace_mp_0.py
```

Check each runner's checkpoint and input/output paths before launching it.
For ASE-eager CHGNet, configure the shared
[CHGNet runner](paper_v2_configs/md_eager/md_chgnet_force_only.py) for the V2
reference and output directories; its current defaults point to `updated_configs`.
See the [MD guide](INFO.md#running-v2-torchsim-md) for trajectory and timing layouts.

## Evaluate energies, forces and pressures

Before aggregating metrics, evaluate the models on the reference configurations
for energy/force RMSEs and on their saved MD trajectories for pressures.

For TorchSim:

```bash
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/e_f_rmses/rmse_torchsim_scripts/md_eager/run_all_rmses.sh
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/e_f_rmses/rmse_torchsim_scripts/md-accelerated/run_all_rmses.sh
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/pressures/md_eager/torchsim-scripts/run_all_pressures.sh
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/pressures/md_accelerated/torchsim-scripts/run_all_pressures.sh
```

For ASE:

```bash
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/e_f_rmses/rmse_ase_scripts/md_eager/run_all_rmses.sh
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/e_f_rmses/rmse_ase_scripts/md-accelerated/run_all_rmses.sh
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/pressures/md_eager/ase-scripts/run_all_pressures.sh
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/pressures/md_accelerated/ase-scripts/run_all_pressures.sh
```

Individual evaluators accept `--help` for input/output options. If an ASE pressure
evaluator's default trajectory directory differs from your MD output directory,
set its `--traj-dir` explicitly.

The accelerated TorchSim analysis also uses **TorchSim-eager eSEN and eqV2** and
**ASE-eager CHGNet**. Their MD outputs and evaluation CSVs must therefore be
available. To evaluate CHGNet separately:

```bash
uv run --script paper_v2_configs/e_f_rmses/rmse_ase_scripts/md_eager/rmse_chgnet.py
uv run --script paper_v2_configs/pressures/md_eager/ase-scripts/pressure_chgnet.py
```

RMSE evaluators resume saved results; normal resume does not recompute completed
rows. When recomputing selected systems, use a separate `--output-dir` and merge
the corrected rows afterwards. Combining `--force`, a restricted `--ref-dir`
and the original output directory discards the other saved rows. Details:
[TorchSim RMSE](paper_v2_configs/e_f_rmses/rmse_torchsim_scripts/README-torchsim.md)
and [ASE RMSE](paper_v2_configs/e_f_rmses/rmse_ase_scripts/README.md).

## Compute metrics

Generate all four metric sets from the MD outputs and evaluation CSVs:

```bash
uv run --script paper_v2_configs/generate_metrics.py
```

To process only the accelerated TorchSim analysis cohort:

```bash
uv run --script paper_v2_configs/generate_metrics.py \
  --source mlip-trajs-torchsim-accelerated
```

Select individual metrics and sources by repeating `--metric` and `--source`:

```bash
uv run --script paper_v2_configs/generate_metrics.py \
  --metric energy-force --metric pressure \
  --source mlip-trajs-torchsim-eager \
  --source mlip-trajs-torchsim-accelerated

uv run --script paper_v2_configs/generate_metrics.py \
  --metric rdf --metric vdos \
  --source mlip-trajs-ase-accelerated
```

Supported source names are:

| Runner | Eager | Accelerated |
| --- | --- | --- |
| TorchSim | `mlip-trajs-torchsim-eager` | `mlip-trajs-torchsim-accelerated` |
| ASE | `mlip-trajs-ase` | `mlip-trajs-ase-accelerated` |

ASE-eager trajectories may be stored in `data/mlip-trajs-ase-eager/` or the
older `data/mlip-trajs-ase/`; use `--source mlip-trajs-ase` for either layout.

Use repeatable `--model MODEL_NAME` to select models and `--dry-run` to inspect
the commands. Model-filtered runs replace the selected source's aggregate
summaries with that subset; rerun without `--model` before plotting the full
benchmark. Molecular crystals are excluded by default; add
`--include-molecular-crystals` to include them. Pressure analysis excludes
Pt/H2O, and CuAu has no reference stress data.

Metric tables are saved under
`paper_v2_configs/<metric-directory>/results/<source>/`, where the metric
directories are `e_f_rmses`, `pressures`, `rdfs` and `vdos`.
The metric command aggregates existing energy/force and pressure evaluations
and computes RDF/VDOS comparisons; it does not launch MD or create plots.

## Generate plots

Once the metric tables are available, run the plotting pipelines:

```bash
uv run --script paper_v2_configs/e_f_rmses/plot_all.py
uv run --script paper_v2_configs/rdfs/plot_all.py
uv run --script paper_v2_configs/pressures/plot_all.py
uv run --script paper_v2_configs/vdos/plot_all.py
uv run --script paper_v2_configs/pareto_plots/plot_all.py
```

For only the accelerated TorchSim plots:

```bash
uv run --script paper_v2_configs/e_f_rmses/plot_all.py --source mlip-trajs-torchsim-accelerated
uv run --script paper_v2_configs/rdfs/plot_all.py --source mlip-trajs-torchsim-accelerated
uv run --script paper_v2_configs/pressures/plot_all.py --dataset torchsim:md_accelerated
uv run --script paper_v2_configs/vdos/plot_all.py --source mlip-trajs-torchsim-accelerated
uv run --script paper_v2_configs/pareto_plots/plot_all.py --source mlip-trajs-torchsim-accelerated
```

Plots are saved under `paper_v2_configs/<metric-directory>/plots/`.
Correlation plots require per-system metrics, MD trajectories and timing records,
plus the F1 and k-SRME CSVs in `paper_v2_configs/data/matbench-scores/`.
Pareto plots also require MD timings and all three observable metrics.
For style or label changes, rerun plotting without recomputing metrics.

Individual plot scripts can also be run directly:

```bash
# Main accuracy-cost Pareto plot.
uv run --script paper_v2_configs/pareto_plots/figure_7.py --source mlip-trajs-torchsim-accelerated

# TorchSim versus ASE timings and Pareto comparisons.
uv run --script paper_v2_configs/pareto_plots/figure_SI_17.py
uv run --script paper_v2_configs/pareto_plots/figure_SI_18.py

# VASP convergence figure from the supplied convergence data.
uv run --script paper_v2_configs/dft_settings_plots/plot_vasp_convergence.py
```

SI17 needs completed benchmark workloads for models available in both runners.
SI18 additionally needs RDF, pressure and VDOS metrics for both accelerated
sources. See the [analysis guide](INFO.md#analysis-pipeline) and
[VDOS instructions](paper_v2_configs/vdos/README.md) for additional options.

## Evaluate a new potential

Create a Python module with a zero-argument `make_calculator()` function returning
an ASE calculator, or a `make_model()` function returning a TorchSim-compatible
model. Run the entry point in an environment containing your model's dependencies.

First check one MD step on one system, writing to a separate output directory:

```bash
python custom_model_evaluation/md_eager/ase-scripts/md_custom_model.py \
  --model-name my-model \
  --model-loader ./my_ase_model.py:make_calculator \
  --system bulkAg_600K_Kapil --max-steps 1 \
  --output-root /tmp/my-model-ase-check

python custom_model_evaluation/md_eager/torchsim-scripts/md_custom_model.py \
  --model-name my-model \
  --model-loader ./my_torchsim_model.py:make_model \
  --system bulkAg_600K_Kapil --max-steps 1 \
  --output-root /tmp/my-model-torchsim-check
```

Then run the full workload, placing the outputs where the standard analysis
commands can discover them:

```bash
python custom_model_evaluation/md_eager/ase-scripts/md_custom_model.py \
  --model-name my-model \
  --model-loader ./my_ase_model.py:make_calculator \
  --output-root paper_v2_configs/data/mlip-trajs-ase-eager

python custom_model_evaluation/md_eager/torchsim-scripts/md_custom_model.py \
  --model-name my-model \
  --model-loader ./my_torchsim_model.py:make_model \
  --output-root paper_v2_configs/data/mlip-trajs-torchsim-eager
```

For accelerated MD, use the corresponding entry points under
`custom_model_evaluation/md_accelerated/` and the matching accelerated output
roots. Enable compilation or custom kernels in your loader function.

To complete the evaluation, add model-specific energy/force and pressure
wrappers using the same checkpoint and precision as MD. Reuse the
[ASE RMSE helper](paper_v2_configs/e_f_rmses/rmse_ase_scripts/ase_rmse.py),
[TorchSim RMSE helper](paper_v2_configs/e_f_rmses/rmse_torchsim_scripts/torchsim_rmse.py)
and [pressure pipeline](paper_v2_configs/pressures/pressure_pipeline.py).
Once those evaluations exist, run the metric commands with `--model my-model`
and the appropriate `--source`, then generate plots.
See the [custom-model guide](INFO.md#evaluating-a-custom-model) for loader
interfaces and evaluator setup.
