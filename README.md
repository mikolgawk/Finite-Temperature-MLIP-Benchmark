# Finite-Temperature MLIP Benchmark

A benchmarking suite for evaluating machine-learned interatomic potentials
(MLIPs) with finite-temperature molecular dynamics. Detailed installation,
model, protocol, and output-format documentation is available in
[INFO.md](INFO.md).

## Reproducing our plots

The current benchmark workflow is under `paper_v2_configs/`. First complete the
[requirements and data setup](INFO.md#requirements), then check and run the
baseline and accelerated TorchSim workloads from the repository root:

```bash
# Inspect the commands without starting MD.
bash paper_v2_configs/md_eager/torchsim-scripts/run_all_md.sh --dry-run
bash paper_v2_configs/md_accelerated/torchsim-scripts/run_all_md.sh --dry-run

# Run the reference-matched workloads on GPU 0.
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/md_eager/torchsim-scripts/run_all_md.sh
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/md_accelerated/torchsim-scripts/run_all_md.sh
```

Obtain the model-specific energy/force and pressure evaluations first:

```bash
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/e_f_rmses/rmse_torchsim_scripts/md_eager/run_all_rmses.sh
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/e_f_rmses/rmse_torchsim_scripts/md-accelerated/run_all_rmses.sh
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/pressures/md_eager/torchsim-scripts/run_all_pressures.sh
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/pressures/md_accelerated/torchsim-scripts/run_all_pressures.sh
```

Then generate all four metric sets with the `uv` script:

```bash
uv run --script paper_v2_configs/generate_metrics.py
```

Next, generate the timing and Pareto plots:

```bash
python paper_v2_configs/pareto_plots/plot-model-timings.py
python paper_v2_configs/pareto_plots/plot-pareto-combined-vdos-rdf-pressure-average-similarity-same-simulation-length.py
```

These commands assume that the reference AIMD trajectories are available under
[`paper_v2_configs/data/ref-trajs/`](paper_v2_configs/data/ref-trajs/), with a
`traj.extxyz` file for each system and the shared `md_metadata.json` file. The MD
runners generate the model trajectories and timing data, while the subsequent
energy/force, pressure, RDF, and VDOS stages generate the CSV files consumed by
the plotting scripts. See the [V2 MD instructions](INFO.md#running-v2-torchsim-md)
and [analysis-pipeline guide](INFO.md#analysis-pipeline) for the expected file
layout and the available command-line options.

## Evaluating a new potential

Use the two baseline entry points in `custom_model_evaluation/` with functions
that construct your ASE calculator and TorchSim model. Start with a one-step
check on one system:

```bash
python custom_model_evaluation/md_eager/ase-scripts/md_custom_model.py \
  --model-name my-model \
  --model-loader ./my_ase_model.py:make_calculator \
  --system bulkAg_600K_Kapil \
  --max-steps 1

python custom_model_evaluation/md_eager/torchsim-scripts/md_custom_model.py \
  --model-name my-model \
  --model-loader ./my_torchsim_model.py:make_model \
  --system bulkAg_600K_Kapil \
  --max-steps 1
```

Remove `--system` and `--max-steps` to evaluate every V2 system. Optional
accelerated entry points are available under `custom_model_evaluation/md_accelerated/`.
The runners produce V2-compatible trajectories and timings; follow the
[custom-model guide](INFO.md#evaluating-a-custom-model) for model setup,
accelerated adapters, output validation, and the remaining analysis stages.
