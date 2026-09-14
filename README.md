# Finite-Temperature MLIP Benchmark

Benchmarking suite for evaluating machine-learned interatomic potentials
(MLIPs) under finite-temperature molecular dynamics (MD). The benchmark
compares MLIP trajectories with ab initio MD (AIMD) references using:

- energy and force errors;
- radial distribution functions (RDFs);
- pressure errors;
- vibrational density of states (VDOS); and
- MD throughput and accuracy/speed Pareto plots.

## Repository layout

The configurations are organized by paper version:

| Directory | Purpose |
| --- | --- |
| `paper_v1_configs/` | Original benchmark configurations and scripts used for the first paper version. |
| `paper_v2_configs/` | Revised benchmark with reference-matched MD, eager and accelerated runners, expanded analysis pipelines, refreshed checkpoints, and an audited precision policy. |
| `custom_model_evaluation/` | Test your own model to the V2 ASE and TorchSim interfaces and evaluate it with the same physical and analysis pipeline. |

The main V2 directories are:

| Directory | Contents |
| --- | --- |
| `data/` | Reference metadata. |
| `md/` | Baseline, non-accelerated NVT MD implementations for ASE and TorchSim. |
| `md_accelerated/` | NVT MD with model-specific inference acceleration for ASE and TorchSim. |
| `e_f_rmses/` | Energy/force evaluation on reference, baseline-MD, and accelerated-MD structures. |
| `pressures/` | Per-frame stress/pressure evaluation and pressure-error aggregation. It also contains pressure-specific copies of the MD runners. |
| `rdfs/` | RDF calculation and comparison with reference trajectories. |
| `vdos/` | VDOS calculation from the velocity autocorrelation function. |
| `pareto_plots/` | Timing, system-size scaling, and combined accuracy/speed plots. |

The V2 calculator catalog contains 17 models: CHGNet, EquiformerV2,
eSEN-30M-OAM, GRACE MP, GRACE OAM, MACE-MH-OMAT, MACE-MP-0, MACE-MPA-0,
MatterSim v1 5M, NequIP OAM L, ORB v2, ORB v3, ORB v3 direct, PET OAM XL,
PET OMat XL, UMA S OMat, and UMA M OMat. Dedicated runner coverage differs
slightly between workflows; the available `md_*.py` files are the source of
truth for a particular runner directory.

<!-- ## What changed in V2

The principal changes from V1 are:

- V2 has separate baseline (`md/`) and inference-accelerated
  (`md_accelerated/`) workflows, each with native ASE and TorchSim variants.
- The physical MD workload is matched to each reference trajectory through
  [`md_metadata.json`](paper_v2_configs/data/ref-trajs/md_metadata.json), rather
  than using one fixed production length for every system.
- The model panel grew from 15 to 17 entries. ORB v3 direct and PET OMat XL were
  added, and MatterSim moved from the 1M to the 5M checkpoint.
- PyTorch workloads use an audited strict-fp32 policy where supported:
  `torch.set_float32_matmul_precision("highest")`, TF32 disabled, and cuDNN
  autotuning disabled. GRACE MP remains fp64 because its distributed SavedModel
  cannot be recast without unavailable source metadata. UMA turbo is an explicit
  performance variant that enables TF32.
- Production calculations request energy and forces only; stress/virial branches
  are disabled. Stress is evaluated separately by the pressure pipeline.
- V2 adds TorchSim/ASE RMSE runners, a complete pressure pipeline, VDOS analysis,
  and timing/Pareto plotting.

For reproducibility, model constructors, checkpoints, versions, and precision
notes are recorded in:

```text
paper_v2_configs/md/model_calculators.json
paper_v2_configs/md/md_calculator_versions.json
paper_v2_configs/md_accelerated/model_calculators.json
paper_v2_configs/e_f_rmses/model_calculators.json
paper_v2_configs/pressures/model_calculators.json -->

## V2 MD settings

### Shared physical protocol

Both `paper_v2_configs/md/` and `paper_v2_configs/md_accelerated/` run the same
physical workload. **Accelerated** settings refer to faster model inference.

### Per-system settings

These values are taken directly from the current V2 metadata:

| System | T (K) | dt (fs) | Thermostat | tau (fs) | Length (ps) |
| --- | ---: | ---: | --- | ---: | ---: |
| Anthracene | 293 | 0.5 | Nose-Hoover | 20 | 8.0000 |
| Ag | 600 | 1.0 | Nose-Hoover | 40 | 24.8810 |
| Au | 1500 | 1.0 | Nose-Hoover | 40 | 20.1620 |
| Cu | 1000 | 1.0 | Nose-Hoover | 40 | 36.0070 |
| CuAu | 500 | 1.0 | Nose-Hoover | 40 | 1.1050 |
| CuZrAl | 1500 | 1.0 | Nose-Hoover | 40 | 2.7180 |
| LiMgAlZnSn | 600 | 1.0 | Nose-Hoover | 40 | 3.0150 |
| LiMgAlZnSn | 900 | 1.0 | Nose-Hoover | 40 | 2.1810 |
| MoS2 | 300 | 1.0 | velocity rescaling | 1 | 20.0000 |
| Pt3Co | 300 | 1.0 | Nose-Hoover | 40 | 20.9270 |
| CsSnI3 | 500 | 1.0 | Nose-Hoover | 40 | 14.7580 |
| H | 1050 | 0.2 | velocity rescaling | 10 | 0.3568 |
| MAPbBr3 | 300 | 0.5 | Nose-Hoover | 20 | 30.0000 |
| Naphthalene | 295 | 0.5 | Nose-Hoover | 20 | 3.7785 |
| Pentacene | 295 | 0.5 | Nose-Hoover | 20 | 7.7895 |
| Picene | 295 | 0.5 | Nose-Hoover | 20 | 7.0755 |
| Pt(111) + 24 H2O | 380 | 1.0 | Nose-Hoover | 40 | 3.8430 |
| Tetracene | 295 | 0.5 | Nose-Hoover | 20 | 8.0000 |
| TiSe2 | 400 | 1.0 | Nose-Hoover | 40 | 14.7160 |

All systems currently use a position, energy, and force print stride of one.
Stress also has stride one where it exists in the reference calculation; the
CuAu reference contains no stress.

### Baseline MD

`paper_v2_configs/md/` provides the comparison workload without optional
inference acceleration:

- PyTorch models run eagerly; ORB explicitly uses `compile=False`, PET bypasses
  its default TorchScript export, and NequIP uses eager mode.
- Strict fp32 and disabled TF32 are used wherever supported.
- TorchSim output is written under
  `paper_v2_configs/data/mlip-trajs-torchsim-matched/`.
- ASE uses the same directory with `-ase` in run names, allowing engine-matched
  trajectories and timings to coexist.

The baseline TorchSim directory contains 15 model scripts. CHGNet is available
through ASE only, and GRACE MP does not have a baseline dedicated runner.

### Accelerated MD

`paper_v2_configs/md_accelerated/` keeps the physical settings fixed and changes
only the model execution path:

| Model family | Acceleration used |
| --- | --- |
| MACE | cuEquivariance fused CUDA kernels plus `torch.compile` in `reduce-overhead` mode. |
| ORB | ORB's compiled inference path (`compile=True`). |
| MatterSim | `torch.compile` applied to the model forward pass. |
| PET | Force-only TorchScript model. |
| NequIP | Force-only AOTInductor artifact with OpenEquivariance. |
| GRACE | Stress-pruned, force-only TensorFlow XLA graph bridged to TorchSim through DLPack. |
| UMA | Separate `compile` and `turbo` scripts. Compile keeps TF32 off; turbo enables TF32 and `merge_mole`, with a separate predictor for each fixed-composition system. |

Accelerated TorchSim output is written under
`paper_v2_configs/data/mlip-trajs-torchsim-accelerated/`; accelerated ASE output
uses `paper_v2_configs/data/mlip-trajs-ase-accelerated/`.

The accelerated directories contain runners for GRACE, MACE, MatterSim,
NequIP, ORB, PET, and UMA. CHGNet, EquiformerV2, and eSEN currently have no
dedicated accelerated runner.

## Running V2 TorchSim MD

### Requirements

- Linux or WSL;
- `uv` on `PATH`;
- a CUDA-capable NVIDIA GPU compatible with the dependencies declared in each
  script's inline `uv` metadata;
- reference trajectories under
  `paper_v2_configs/data/ref-trajs/<system>/traj.extxyz`; and
- any required local checkpoints under `paper_v2_configs/data/models/`.

Each model script declares its own Python dependencies and can be run directly
with `uv`. From the repository root:

```bash
# Run every baseline TorchSim model sequentially.
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/md/torchsim-scripts/run_all_md.sh

# Run every accelerated TorchSim variant sequentially.
CUDA_VISIBLE_DEVICES=0 bash paper_v2_configs/md_accelerated/torchsim-scripts/run_all_md.sh
```

Use `--dry-run` to list the commands without launching MD:

```bash
bash paper_v2_configs/md/torchsim-scripts/run_all_md.sh --dry-run
bash paper_v2_configs/md_accelerated/torchsim-scripts/run_all_md.sh --dry-run
```

Run one model with:

```bash
uv run --script paper_v2_configs/md/torchsim-scripts/md_orb_v3.py
uv run --script paper_v2_configs/md_accelerated/torchsim-scripts/md_orb_v3.py
```

Batch logs are saved below the selected script directory at
`logs/md/<timestamp>_<pid>/`. The launchers continue after a model process
fails and return a non-zero status if any model failed.

TorchSim scripts skip a model/system pair when its timing CSV already exists.
This includes a zero-byte CSV, which is used as a failure marker. Remove the
empty timing CSV and any partial HDF5 trajectory before retrying a failed pair.
Missing initial structures are reported and skipped.

> **Migration note:** the configuration trees have been renamed, but some V2
> runner constants and script docstrings still contain the former
> `updated_configs` path (and some metadata paths still start at `data/`). These
> references must be changed to the new `paper_v2_configs` layout before the
> commands above can run solely against the renamed tree.

## Evaluating a custom model

`custom_model_evaluation/` is intended to let a model developer run the same
benchmark protocol as `paper_v2_configs/` without adding their model to the
paper's fixed 17-model catalog. The developer supplies one native ASE runner
and one TorchSim runner; the model-independent analysis stages then consume
their trajectories and timings.

At present, `custom_model_evaluation/` is an empty integration workspace. It
does not yet contain templates, copied analysis entry points, or a one-command
launcher. The description below defines the intended layout and compatibility
contract; those files must be added before the custom workflow is runnable.

### Intended layout

```text
custom_model_evaluation/
├── data/
│   ├── ref-trajs/                   AIMD inputs and md_metadata.json
│   ├── models/                      local custom checkpoints
│   ├── mlip-trajs-ase/              baseline ASE results
│   ├── mlip-trajs-torchsim/         baseline TorchSim results
│   ├── mlip-trajs-ase-accelerated/  accelerated ASE results, if available
│   └── mlip-trajs-torchsim-accelerated/
├── md/
│   ├── ase-scripts/md_<model>.py
│   └── torchsim-scripts/md_<model>.py
├── md_accelerated/                  optional optimized counterparts
├── e_f_rmses/                       custom ASE/TorchSim RMSE adapters
├── pressures/                       stress-enabled custom calculator
├── rdfs/                            V2 RDF pipeline or a configured wrapper
├── vdos/                            V2 VDOS pipeline or a configured wrapper
└── pareto_plots/                    combined custom-model report
```

The reference trajectories and metadata should be copied or linked from
`paper_v2_configs/data/ref-trajs/`. Do not change their temperatures,
timesteps, thermostat settings, trajectory lengths, or print strides: using the
same metadata is what makes the custom result directly comparable with V2.

### Model identifier

Choose one stable base identifier such as `my-model` and use it everywhere:

- the `model_name` passed to the MD and RMSE drivers;
- the stem after `nvt_` and `md_timing_` in output filenames;
- the calculator name in result CSVs; and
- `--model my-model` when an analysis script supports model filtering.

Preserve deliberate execution suffixes in that stem. For example, the shared
ASE driver appends `-ase`, while eager and accelerated implementations may use
`my-model-eager` and `my-model-compile`. These names prevent results from
overwriting each other and ensure that downstream discovery treats them as
separate executions.

### ASE adapter

The ASE script must construct an `ase.calculators.calculator.Calculator`
compatible object. For production MD it must provide:

- total energy in eV through `atoms.get_potential_energy()`;
- forces in eV/Angstrom through `atoms.get_forces()`;
- support for the elements and periodic cells in all selected systems; and
- stress disabled so the timed MD workload remains energy/force only.

Copying the structure of an existing runner, such as
[`md_orb_v3.py`](paper_v2_configs/md/ase-scripts/md_orb_v3.py), is the simplest
starting point. Replace its inline `uv` dependencies, model imports,
`make_calculator()` implementation, model identifier, and checkpoint path. Keep
the shared metadata-driven NVT loop and HDF5 writer unchanged.

A second, stress-enabled calculator is required by the pressure stage. Its ASE
stress must use the ASE sign convention and eV/Angstrom^3 units. Pressure is
deliberately evaluated separately so stress computation is not included in the
production MD timing.

### TorchSim adapter

The TorchSim script must wrap the model in a TorchSim-compatible model object
that accepts a TorchSim simulation state and returns total energy and forces.
It must use the same checkpoint, dtype policy, cutoff, and physical model as the
ASE adapter. Production stress must be disabled.

Use an existing model family with a similar API as the template; for example,
[`md_orb_v3.py`](paper_v2_configs/md/torchsim-scripts/md_orb_v3.py) shows the
general metadata-driven loop and an existing TorchSim adapter. Only the inline
dependencies, model loader/adapter, model identifier, checkpoint, and any
model-specific validation should need to change.

If no native TorchSim adapter exists, the custom wrapper is responsible for:

- converting atomic numbers, positions, cells, periodicity, and batches from
  the TorchSim state into the model's input representation;
- preserving gradients when forces are obtained from energy derivatives;
- returning outputs in TorchSim's expected units and shapes; and
- rebuilding neighbour information when required by the model.

The ASE and TorchSim runners should be tested first on one system and a few
steps. Compare their initial energy and forces before launching the full set;
matching trajectories alone is not a sufficient adapter validation.

### Output contract

Downstream V2 analysis discovers results by directory and filename, so custom
runners must preserve the V2 schema:

```text
data/<trajectory-source>/<system>/
├── nvt_<model>.h5
└── md_timing_<model>.csv
```

The HDF5 trajectory must contain, under `data/`, atomic numbers, masses,
positions, cell vectors, periodic-boundary flags, and velocities. Positions and
cells use Angstrom; velocities use Angstrom/ps. A stress-enabled trajectory
additionally stores `data/stress` in eV/Angstrom^3 and identifies the stress
units in an attribute. Frame counts must agree across all time-dependent
datasets.

The timing CSV uses the V2 columns:

```text
calculator,system,temperature_K,n_steps,time_step_fs,thermostat,tau_fs,
record_interval,elapsed_seconds,seconds_per_step,engine,seed
```

Follow the timing boundary of the V2 runner used as the template and synchronize
the accelerator before and after it. Model loading and explicit artifact export
belong outside the reported interval. Compilation may occur eagerly during
setup or lazily on the first model call, so record whether compilation/warmup is
inside or outside the timing boundary. Include enough detail in `engine` to
distinguish ASE/TorchSim and eager/accelerated execution.

### Running the V2-equivalent pipeline

Each stage has a different custom-model integration point:

| Stage | Custom-model requirement |
| --- | --- |
| MD and timing | Run both custom MD scripts over every entry in `md_metadata.json`. Optional accelerated scripts must retain the same physical settings. |
| Energy/force RMSE | Add matching ASE and TorchSim RMSE wrappers that load exactly the same model/checkpoint as MD and call the shared `run_rmse(...)` implementation on the AIMD reference frames. |
| Pressure | Supply a stress-enabled ASE calculator or a stress-enabled HDF5 trajectory, then run `pressure_pipeline.py` with the custom trajectory root. |
| RDF | Place HDF5 trajectories in one of the four standard trajectory-source directories. The RDF pipeline discovers `nvt_<model>.h5` automatically and accepts `--model <model>`. |
| VDOS | Supply trajectories with velocities and matching timestep/stride settings, then point the VDOS batch stage at the custom trajectory root. |
| Pareto report | Combine the custom timing CSVs with its RMSE, RDF, VDOS, and pressure metric rows, then pass those files to the Pareto plotting script. Unknown models are displayed in the `Other` tier. |

A typical implementation sequence is:

1. Copy the V2 reference metadata and make all referenced AIMD trajectories
   available under `custom_model_evaluation/data/ref-trajs/`.
2. Implement `md_<model>.py` for ASE and TorchSim by adapting the closest V2
   examples. Run a short one-system validation and compare energy/force output.
3. Run the complete reference-matched NVT workload with both engines. Confirm
   that every requested system has a non-empty HDF5 trajectory and timing CSV.
4. Add ASE and TorchSim RMSE wrappers using the same loader functions, then
   evaluate the AIMD frames and aggregate energy/force errors.
5. Add a stress-enabled ASE calculator for pressure evaluation. Do not reuse a
   production calculator whose stress branch was intentionally removed.
6. Run RDF and VDOS comparison on the produced trajectories.
7. Merge all metric rows with MD timings and generate the final Pareto report.

Useful V2 entry points to reuse or wrap are:

```text
paper_v2_configs/e_f_rmses/rmse_ase_scripts/ase_rmse.py
paper_v2_configs/e_f_rmses/rmse_torchsim_scripts/torchsim_rmse.py
paper_v2_configs/pressures/pressure_pipeline.py
paper_v2_configs/rdfs/run_rdf_pipeline.py
paper_v2_configs/vdos/get_normalized_VDOS.py
paper_v2_configs/pareto_plots/plot-pareto-combined-vdos-rdf-pressure-average-similarity-same-simulation-length.py
```

Current repository limitations matter when setting this up:

- the V2 analysis scripts still assume their own `paper_v2_configs/data/`
  location in several places, so a custom wrapper, configurable data-root
  option, or link into one of the standard source directories is required; and
- `get_normalized_VDOS.py` imports VDOS batch/normalization helper modules and a
  `vdos_settings_mlip.csv` file that are not currently present in the V2 tree.
  Those components must be restored before the complete VDOS stage can run; and
- the combined Pareto script currently considers a timing-system directory
  complete only when it contains exactly 15 timing CSV files. That completeness
  rule must be made configurable before it can consume a custom-only result set
  or a V2 result set with an additional custom model.

Until templates and a top-level launcher are added under
`custom_model_evaluation/`, this is a manual integration workflow rather than a
single-command custom-model benchmark.

## Analysis pipeline

The V2 analysis stages consume reference, baseline, or accelerated trajectories:

```text
e_f_rmses/   Evaluate per-frame energy and force errors, including isolated-atom
             energy corrections and aggregation by system type.
pressures/   Evaluate stress and hydrostatic pressure separately from the
             force-only production loop, then aggregate pressure errors.
rdfs/        Compare MLIP and reference RDFs over matched simulated time.
vdos/        Fourier-transform the Hann-windowed velocity autocorrelation
             function and normalize the resulting VDOS.
pareto_plots/ Combine accuracy/similarity metrics with MD timings and scaling.
```

The pressure stage excludes Pt(111) + 24 H2O, and CuAu cannot contribute a
reference pressure because its AIMD trajectory has no stress values.

## V1 historical MD

The original `paper_v1_configs/md_production/` ASE workflow ran 80,000 steps at
0.25 fs (20 ps), saved every tenth frame, and used a thermostat damping time of
`100 * timestep`. The five molecular crystals were run separately through the
i-PI harness documented in
[`paper_v1_configs/md_production/molecular_crystals_ipi/generic/README.md`](paper_v1_configs/md_production/molecular_crystals_ipi/generic/README.md).

`paper_v1_configs/md_timings/` contains the original short timing workload. It
runs 0.2 ps over a reduced system set with a 25 fs damping time and includes
startup costs in the measured interval.
