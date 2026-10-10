# Finite-Temperature MLIP Benchmark

A benchmarking suite for evaluating machine-learned interatomic potentials
(MLIPs) with finite-temperature molecular dynamics. Detailed installation,
model, protocol, and output-format documentation is available in
[INFO.md](INFO.md).

## Reproducing our plots

The current benchmark workflow is under `paper_v2_configs/`. Before running it,
complete the [requirements and data setup](INFO.md#requirements) to access the input data. The workflow
assumes that the reference AIMD trajectories are available under
[`paper_v2_configs/data/ref-trajs/`](paper_v2_configs/data/ref-trajs/), with a
`traj.extxyz` file for each system and the shared `md_metadata.json` file. The MD
runners generate the model trajectories and timing data, while the subsequent
energy/force, pressure, RDF, and VDOS stages generate the CSV files used by
the plotting scripts. See the [V2 MD instructions](INFO.md#running-v2-torchsim-md)
and [analysis-pipeline guide](INFO.md#analysis-pipeline) for the expected file
layout and the available command-line options.

Then check and run the baseline and accelerated TorchSim workloads from the
repository root:

```bash

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

V2 metric computations, raw energy/force and pressure evaluations, and timing
analysis exclude the five molecular crystals by default: anthracene,
naphthalene, pentacene, picene, and tetracene. Pass
`--include-molecular-crystals` to the relevant analysis command to include them.
Standalone RDF and VDOS commands use the same default. The remaining workload
contains 14 systems before any metric-specific exclusions or failed MD runs.
Pressure evaluation, aggregation, and plots always exclude the Pt(111) + 24 H2O
metal-water interface, including rows in previously saved pressure results.
The small RDF, pressure, and VDOS panels omit model curves with 100% error.
Successful RDF and VDOS runs with valid data and 100% error retain text-only
legend entries; failed runs and missing data stay out of the legends.
Pressure subplot legends also omit model names with 100% histogram error.
Pressure-MAE values and tier means retain their GPa units.
Those models still contribute to metric computations, averages, and worst-system
selection.
The pressure panels include every available system type, including hydrogen;
their layout adds rows as needed.

To generate all four metric sets for only one model, pass its exact model name
(the part after `nvt_` in its trajectory filenames):

```bash
uv run --script paper_v2_configs/generate_metrics.py --model mace-mp-0
```

`--model` is repeatable and can be combined with the repeatable `--metric`
option, for example:

```bash
uv run --script paper_v2_configs/generate_metrics.py \
  --model mace-mp-0 \
  --metric rdf \
  --metric vdos
```

All four metrics are stored separately by trajectory source under
`paper_v2_configs/<metric-directory>/results/<source>/`, including
`mlip-trajs-torchsim-eager` and `mlip-trajs-torchsim-accelerated`.
The TorchSim accelerated paper cohort also includes the **eager eSEN-30M-OAM and
EquiformerV2** runs by default. This applies to RDF, VDOS, pressure histogram
errors, pressure MAE, energy/force RMSE, correlations, and timing/Pareto plots.
For these two models, analysis reads the eager trajectories, evaluations,
model-matched references, and actual eager step times; every other model uses
its own source. Eager and ASE analyses retain their existing source selection.
The output directory and `source`/pressure `mode` identify the paper cohort;
`trajectory_source` identifies the physical execution source. Raw files and
the source-specific failed-run registry are preserved. Eager failures receive
100% RDF/VDOS/pressure histogram error and are excluded from RMSE/pressure MAE,
using the same rules as other models. Each model appears once in the cohort.

After changing to this cohort, regenerate the accelerated metric tables and
then rerun the plotting commands. No new MD or model evaluation is needed when
the eager evaluation CSVs already exist:

```bash
uv run --script paper_v2_configs/generate_metrics.py \
  --source mlip-trajs-torchsim-accelerated
```

Energy/force means and pressure MAEs are otherwise computed independently for
each source, even when model names are identical. Existing raw RMSE and pressure input layouts
are still supported. Older aggregate CSVs at the results root are no longer used
by the plotting entry points.

RMSE aggregation also discovers evaluations under
`paper_v2_configs/e_f_rmses/data/e-f-predictions-torchsim/`, including its
`md_eager/` and `md_accelerated/` subdirectories.

To generate only eager energy/force and pressure metrics:

```bash
uv run --script paper_v2_configs/generate_metrics.py \
  --metric energy-force --metric pressure \
  --source mlip-trajs-torchsim-eager
```

`--source` is repeatable and defaults to available sources. Filtered runs use
the standard output paths, so their aggregate summary CSVs contain only the
selected model(s). RMSE plots also accept `--source`; pressure plots retain the
`--dataset backend:mode` option.

Once the metric CSV files are available, each analysis directory provides a
`plot_all.py` entry point that creates every plot supported by that directory.

The v2 RMSE two-panel overviews use the v1 dimensions of 7.06 × 3.53 inches
and 8-point text. SI1/SI2 use larger 12-point text and a 10.59 × 8.47-inch canvas
for six system-type panels. Median connectors stay vertical when labels are spaced.
RDF Figure 3 uses an 11.12 × 14.30-inch canvas for two rows of systems, with
12-point text and 8-point legends. Pressure panel Figures 4 and SI6 use the
v1 sizes: a 10.59 × 12.53-inch canvas, 10-point text, and 6-point legends.
They retain the centered 3×2 system layout, with v1's tighter column and tier
spacing and the same relative heights for the overview and system panels.
Overview tier labels, median values, and panel letters sit inside the plot box;
vertical median connectors are split at axis breaks. Pressure tier legends omit
all model names with 100% histogram error. Those model histograms remain hidden,
and every model still contributes to metric computations and tier means under
the existing failure policy.
VDOS comparison panels use the v1 RDF canvas of 10.59 × 12.53 inches, with
10-point text and 6-point legends.
Additional system rows and crowded RDF or pressure legends can add height.
To apply style changes to existing results, rerun plotting only.

VDOS correlation plots use the v1 proportions: Figure 6 has a 10.59 × 10.59-inch
canvas, and the 1×3 RDF/pressure/VDOS correlations use 10.59 × 7.06 inches.
Both use 12-point text and tier legends, with equal-width panels and no extra
horizontal stretching. Figure 6 uses 10-point outlier model labels and 12-point
labels in its larger all-model companion, whose default canvas is scaled by 1.45.
This style also applies to the versions without hydrogen and without Pt+H₂O.

The E/F RMSE supplementary outputs are numbered SI1–SI3. The existing scripts
`figure_SI_2.py`, `figure_SI_3.py` and `figure_SI_4.py` save
`figure_SI_1.pdf`, `figure_SI_2.pdf` and `figure_SI_3.pdf`, respectively.

Run the plotting pipelines with:

```bash
uv run --script paper_v2_configs/e_f_rmses/plot_all.py
uv run --script paper_v2_configs/rdfs/plot_all.py
uv run --script paper_v2_configs/vdos/plot_all.py
uv run --script paper_v2_configs/pressures/plot_all.py
uv run --script paper_v2_configs/pareto_plots/plot_all.py
```

The VDOS plotting entry point includes Figure 6 and its SI companion, saving the
two 3×3 correlation figures under `vdos/plots/<source>/`.
It also creates no-hydrogen versions of Figure 6, its companion, and the 1×3
correlations. These exclude hydrogen from force RMSE, RDF, pressure histogram
error, and VDOS averages, using the existing per-system metric CSVs. F1 and
k-SRME scores are unchanged. Use `--exclude-hydrogen` when calling a correlation
script directly; `--exclude-hydrogen-force-rmse` remains an alias. The existing
`_no_hydrogen_force_rmse` output suffix is retained for compatibility. Rerun
plotting only to update these figures when the per-system CSVs are available.

The pipeline also creates a third correlation set excluding both hydrogen and
Pt+H₂O from the force RMSE, RDF, and VDOS averages; pressure already excludes
Pt+H₂O. These PDFs use the `_no_hydrogen_no_pt_water` suffix, keeping the original
and no-hydrogen sets. F1 and k-SRME scores remain unchanged, and failed systems
outside these exclusions still contribute their 100% RDF/pressure/VDOS errors.
For direct calls, use `--exclude-hydrogen --exclude-pt-water` on `figure_6.py` or
`figure_SI_7_8_9.py`. Using only `--exclude-pt-water` keeps hydrogen and saves
PDFs with the `_no_pt_water` suffix. Existing per-system CSVs are sufficient;
only plotting needs to be rerun.

To plot only the accelerated TorchSim results, use:

```bash
uv run --script paper_v2_configs/rdfs/plot_all.py \
  --source mlip-trajs-torchsim-accelerated
uv run --script paper_v2_configs/vdos/plot_all.py \
  --source mlip-trajs-torchsim-accelerated
uv run --script paper_v2_configs/pressures/plot_all.py \
  --dataset torchsim:md_accelerated
uv run --script paper_v2_configs/pareto_plots/plot_all.py \
  --source mlip-trajs-torchsim-accelerated
```

The correlation figures require the F1 and k-SRME score CSVs described
in the [analysis-pipeline guide](INFO.md#analysis-pipeline). A wrapper attempts
all of its plot stages and reports any unavailable prerequisites at the end.

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
