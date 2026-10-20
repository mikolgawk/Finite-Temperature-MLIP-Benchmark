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

The default VDOS analysis is the paper's refined mass-weighted Welch method in
[`vdos/`](paper_v2_configs/vdos/README.md): a 2 ps target segment,
50% overlap, at least four segments and fourfold FFT padding. Its metrics and
plots are saved under `vdos/results/` and `vdos/plots/`.
The standard Pareto scripts read these refined metrics too. The original
`vdos_legacy/` analysis remains available for explicit historical comparisons.

V2 metric computations, raw energy/force and pressure evaluations, and timing
analysis exclude the five molecular crystals by default: anthracene,
naphthalene, pentacene, picene, and tetracene. Pass
`--include-molecular-crystals` to the relevant analysis command to include them.
Standalone RDF and VDOS commands use the same default. The remaining workload
contains 14 systems before any metric-specific exclusions or failed MD runs.
Pressure evaluation, aggregation, and plots always exclude the Pt(111) + 24 H2O
metal-water interface, including rows in previously saved pressure results.
The small RDF, VDOS, and pressure-histogram-error panels omit model curves with
100% error.
Successful RDF and VDOS runs with valid data and 100% error retain text-only
legend entries; failed runs and missing data stay out of the legends.
The pressure MAE panel in Figure 4 draws each tier's best model as a solid
histogram and its worst model as a dashed histogram, including successful runs
with 100% histogram error. The legends report their MAEs in GPa. If the same
model is both best and worst, it is drawn once. Failed MD runs remain excluded
from the MAE analysis. Figure SI6's legends continue to omit model names with
100% histogram error.
Figure 4 draws one reference histogram per tier, labeled `Reference`, using the
best plotted model's matched reference when available.
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
`mlip-trajs-torchsim-eager` and `mlip-trajs-torchsim-accelerated`. The metric
directories are `e_f_rmses`, `pressures`, `rdfs`, and `vdos`.
The TorchSim accelerated paper cohort also includes **TorchSim eager eSEN-30M-OAM
and EquiformerV2, and ASE eager CHGNet** runs by default. This applies to RDF, VDOS, pressure histogram
errors, pressure MAE, energy/force RMSE, correlations, and timing/Pareto plots.
For these three models, analysis reads the eager trajectories, evaluations,
model-matched references, and actual eager step times; every other model uses
its own source. Eager and ASE analyses retain their existing source selection.
The output directory and `source`/pressure `mode` identify the paper cohort;
`trajectory_source` identifies the physical execution source. Raw files and
the source-specific failed-run registry are preserved. Eager failures receive
100% RDF/VDOS/pressure histogram error and are excluded from RMSE/pressure MAE,
using the same rules as other models. Each model appears once in the cohort.
CHGNet trajectories are read from `data/mlip-trajs-ase-eager/` (with the older
`data/mlip-trajs-ase/` layout also supported). Its `chgnet-force-only-ase` MD
identifier and `chgnet-force-only` evaluator identifier both join as CHGNet.
The legacy source key `mlip-trajs-ase` in `trajectory_source` denotes ASE eager.

After changing to this cohort, regenerate the accelerated metric tables and
then rerun the plotting commands. No new MD or model evaluation is needed when
the eager evaluation CSVs already exist:

```bash
uv run --script paper_v2_configs/generate_metrics.py \
  --source mlip-trajs-torchsim-accelerated
```

If CHGNet's pressure evaluation CSVs are missing, evaluate stresses on its saved
ASE eager MD frames before aggregating pressure metrics (this does not rerun MD):

```bash
uv run --script paper_v2_configs/pressures/md_eager/ase-scripts/pressure_chgnet.py
uv run --script paper_v2_configs/generate_metrics.py \
  --metric pressure --source mlip-trajs-torchsim-accelerated
```

The CHGNet pressure evaluator defaults to `data/mlip-trajs-ase-eager/` and writes
its raw per-frame pressures, matched references, and MAE summaries under
`pressures/results/ase/md_eager/`. The accelerated cohort reads these files
directly, without copying them into a TorchSim raw-data directory.

Energy/force means and pressure MAEs are otherwise computed independently for
each source, even when model names are identical. Existing raw RMSE and pressure input layouts
are still supported. Older aggregate CSVs at the results root are no longer used
by the plotting entry points.

RMSE aggregation excludes incomplete, missing, and audited physically failed MD
runs for both ASE and TorchSim, in eager and accelerated modes. Successful runs
require a matching trajectory and completed timing record. The ASE eager data
directory may be named `mlip-trajs-ase-eager` or `mlip-trajs-ase`.

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
Overview tier labels and median values sit inside the plot box; Figure 4's
overview panel letter sits outside it. Vertical median connectors are split at
axis breaks. Figure 4 draws the best and worst pressure histograms as solid and
dashed lines, respectively. Figure SI6 omits all model names with 100% histogram
error. Metric computations and tier means retain the existing failure policy.
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
Figure 6 labels every model meeting its outlier threshold in each panel by
default. Labels identify departures from the fitted linear trend using absolute
externally studentized residuals of at least 2.5. Each residual is scaled by the
noise estimated with that model omitted and adjusted for its leverage. Extreme
x/y metric values alone do not trigger a label. This cutoff is descriptive,
not a significance test; all eligible points remain in the fit and correlation.
Use `--outlier-threshold` to change the cutoff (`--outlier-z-threshold` remains
an alias). `--max-labels-per-panel N` applies an optional cap; `0` hides labels.
Use `--data-only` on `vdos/figure_6.py` to print the selected names without
generating figures. Nonfinite metric pairs are excluded from dots, fits,
correlations, and outlier selection.
Figure 6, its all-label companion, and the SI7–SI9 1×3 correlation plots average
force RMSE, RDF, VDOS, and pressure errors only over each model's stable MD
systems, using completed trajectories
and matching timing records and excluding registered physical failures. This
also applies to its no-hydrogen and no-hydrogen/no-Pt+H₂O versions. A successful
run with 100% metric error is still included. Matbench scores are unchanged.
Benchmark metric CSVs retain their failure penalties. Both correlation scripts
require per-system metric CSVs and the MD outputs; use `--md-data-dir` to
override the MD data root. Use `--data-only` on `figure_SI_7_8_9.py` to inspect
the model averages without generating figures. Rerun plotting only to apply
this change.

The E/F RMSE supplementary outputs are numbered SI1–SI3. The existing scripts
`figure_SI_2.py`, `figure_SI_3.py` and `figure_SI_4.py` save
`figure_SI_1.pdf`, `figure_SI_2.pdf` and `figure_SI_3.pdf`, respectively.
SI3 excludes molecular crystals and hydrogen before averaging energy and force
RMSEs by model and calculating tier medians. Metal-water interfaces are included. Existing
`rmse_per_system.csv` files are sufficient; rerun SI3 plotting to apply this filter.

RDF Figure SI4 shows the bulk Pt₃Co system at 300 K in four stacked tier panels,
using Figure 3's best/worst curve selection, fonts, colors, and legend rules.
Its 3.71 × 5.82-inch canvas matches one Figure 3 system column, with 12-point
text, 8-point legends, and the same tier spacing. Crowded legends can add height.
Its model errors and tier means are specific to Pt₃Co. It is included in the RDF
plotting pipeline and saves `rdfs/plots/<source>/figure_SI_4.pdf`. Run it separately
from existing results with:

```bash
uv run --script paper_v2_configs/rdfs/figure_SI_4.py \
  --source mlip-trajs-torchsim-accelerated
```

Use `--data-only` to check the selections without creating a figure, or
`--skip-figure-si-4` on the RDF pipeline to skip this figure. No RDF recomputation
is needed when the per-model scores and saved curves are already available.

Run the plotting pipelines with:

```bash
uv run --script paper_v2_configs/e_f_rmses/plot_all.py
uv run --script paper_v2_configs/rdfs/plot_all.py
uv run --script paper_v2_configs/vdos/plot_all.py
uv run --script paper_v2_configs/pressures/plot_all.py
uv run --script paper_v2_configs/pareto_plots/plot_all.py
```

The Pareto plotting pipeline also includes Figure SI17, a timings-only comparison
of native accelerated TorchSim and ASE runs. Only models with a terminal success
or failure for every selected reference system in both runners are included.
Empty timing CSVs are explicit failure markers; missing outputs or an unfinished
trajectory without a terminal marker do not establish completion. Failed systems
are allowed in a finished model's coverage but do not contribute timings.
For each shared model it averages
step times over the same systems with completed, matching MD workloads in both
runners, excluding molecular crystals and registered failed runs. The panels
show mean step times and their ASE/TorchSim ratio; ratios above one mean TorchSim
is faster. Each model label includes its number of matched systems. Borrowed
eager runs in the paper's accelerated cohort are not used in this comparison.
The pipeline skips SI17 when no paired timing data is available.
PET-OAM-XL and PET-OMAT-XL are matched across runners using their timing engine
metadata: ASE's unsuffixed model names match TorchSim's `-torchscript` names
when the ASE records explicitly identify TorchScript inference.

Run SI17 independently from the existing MD timing CSVs and matching trajectories:

```bash
uv run --script paper_v2_configs/pareto_plots/figure_SI_17.py
```

This saves `pareto_plots/plots/torchsim_vs_ase_accelerated/figure_SI_17.pdf` and
summary and per-system CSVs under `pareto_plots/results/torchsim_vs_ase_accelerated/`.
Add `--data-only` to export just the timing tables. No RDF, VDOS, pressure, or
energy/force metric computations need to be rerun for SI17.

Figure 7 uses only the TorchSim eager or accelerated paper cohort. For the
accelerated main-paper figure, run:

```bash
uv run --script paper_v2_configs/pareto_plots/figure_7.py \
  --source mlip-trajs-torchsim-accelerated
```

This saves `pareto_plots/plots/mlip-trajs-torchsim-accelerated/figure_7.pdf`,
with `figure_7_metrics.csv` and `figure_7_frontier.csv` under the corresponding
results directory. The TorchSim paper cohort still includes the previously
selected eager EQ/eSEN and ASE eager CHGNet results.

Figure SI18 compares accelerated TorchSim and ASE on the same axes:

```bash
uv run --script paper_v2_configs/pareto_plots/figure_SI_18.py
```

The runners keep separate Pareto frontiers: TorchSim uses circles and a solid
line; ASE uses triangles and a dashed line. Colours still denote model tiers,
and model names remain beside their points. Each runner retains its own metrics
and available model configurations, including native ASE PET-OAM-XL when its
metrics are available. For each configuration shown in both runners, step times
are averaged over the same successful systems with matching step counts, time
steps and recording intervals, using SI17's native timing validation. Accuracy
metrics retain all existing failure penalties. Configurations shown for only
one runner retain their usual cohort timing. The SI18 metric CSV records
`n_timing_systems`, `timing_systems` and `timing_basis`; a shared configuration
without any matched timing workloads is omitted.
`ASE` UMA-S-P1 (turbo) and UMA-M-P1 (turbo) are excluded from SI18 only;
their TorchSim counterparts and the underlying benchmark metrics are retained.
The comparison is saved as
`pareto_plots/plots/torchsim_vs_ase_accelerated/figure_SI_18.pdf`, with
`figure_SI_18_metrics.csv` and `figure_SI_18_frontier.csv` under the matching
results directory. Existing metrics are sufficient; add `--data-only` to either
script to export tables without creating a figure. Add `--layout panels` to
SI18 for separate panels. The standard Pareto pipeline creates SI18 whenever
both accelerated sources are selected or discovered, and creates Figure 7 only
for selected TorchSim sources.

Native ASE timings and all three native ASE observable metrics are required;
TorchSim errors are never substituted. Models without pressure evaluations are
omitted until those evaluations are available. If the ASE summaries have not
yet been generated, compute them from the saved trajectories and pressure
evaluations:

```bash
uv run --script paper_v2_configs/generate_metrics.py \
  --metric rdf --metric vdos --metric pressure \
  --source mlip-trajs-ase-accelerated
```

Timing means use completed trajectory/timing pairs, excluding registered
failures. Molecular crystals remain excluded by default.

The refined VDOS plotting entry point includes Figure 6 and its SI companion,
saving the two 3×3 correlation figures under `vdos/plots/<source>/`.
It creates the standard Figure 6 and 1×3 correlations only; automatic
no-hydrogen and combined no-hydrogen/no-Pt+H₂O sets have been removed.
To create an exclusion variant manually, use `--exclude-hydrogen`,
`--exclude-pt-water`, or both on `figure_6.py` or `figure_SI_7_8_9.py`.
These filter the selected systems from force RMSE, RDF, pressure, and VDOS
averages; pressure already excludes Pt+H₂O. Matbench scores stay unchanged.
`--exclude-hydrogen-force-rmse` remains an alias for `--exclude-hydrogen`, and
the `_no_hydrogen_force_rmse`, `_no_pt_water`, and `_no_hydrogen_no_pt_water`
output suffixes remain supported for manual runs. Figure 6 and the 1×3 SI
figures both use stable runs only. Existing per-system CSVs and MD outputs are
sufficient; rerun plotting only.

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

For an explicit comparison with the historical VDOS scores, select their result
root rather than replacing the refined defaults:

```bash
uv run --script paper_v2_configs/pareto_plots/plot_all.py \
  --source mlip-trajs-torchsim-accelerated \
  --vdos-results-dir paper_v2_configs/vdos_legacy/results \
  --plots-dir /tmp/mlip-legacy-pareto/plots \
  --results-dir /tmp/mlip-legacy-pareto/results
```

Native ASE VDOS can be computed and plotted with the refined pipeline. Its
ordinary VDOS figures are supported; TorchSim-only dependent plotting stages
are skipped. Figure SI18 uses both accelerated runners' refined summaries:

```bash
uv run --script paper_v2_configs/vdos/run_vdos_pipeline.py \
  --source mlip-trajs-ase-accelerated --compute-only
uv run --script paper_v2_configs/vdos/figure_SI_18.py
```

To collect existing figures and metrics for sharing, use
`python paper_v2_configs/vdos/package_plots.py`. The ZIP includes the
TorchSim figures and summaries, native ASE summaries, and the SI18 comparison
figure and tables when available. See the
[refined VDOS guide](paper_v2_configs/vdos/README.md) for packaging options.

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
