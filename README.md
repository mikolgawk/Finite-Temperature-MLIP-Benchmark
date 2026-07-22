# Finite-Temperature MLIP Benchmark

Benchmarking suite for evaluating MLIPs under finite-temperature molecular dynamics. For a panel of
foundation MLIPs, the pipeline runs NVT MD on a set of reference systems and
compares the resulting trajectories to AIMD reference trajectories along four
axes: energy/force accuracy, pressure, radial distribution functions (RDFs),
and vibrational density of states (VDOS).

## Two configuration trees

In the repo, the same pipeline exists
twice, under two top-level directories:

| Tree | What it is |
| --- | --- |
| `paper_configs/` | The pipeline that produced the results on arxiv. |
| `updated_configs/` | The revised pipeline — audited numerical precision, refreshed checkpoints. Where new work goes. |

The main differences between the two:

- **Precision.** `paper_configs` ran each model at whatever its constructor
  defaulted to or at fp64 (`mace_mp(default_dtype='float64')`,
  `pretrained.orb_v2(precision='float64')`). `updated_configs` normalizes to
  fp32 wherever precision is settable, and each model config entry records
  `weight_dtype`, `matmul_precision`, `precision_settable`, `verified`, and a
  `dtype_note` explaining how that was established. The one exception is
  `paper_configs/md_timings`, whose MACE and ORB entries run at fp32 /
  `float32-high` like the updated tree — a speed comparison is only meaningful
  if both sides are clocked at the same precision.
- **Checkpoints.** `mattersim-v1-1M` → `mattersim-v1-5M`; `grace-oam` moves
  from the shipped fp64 `GRACE-2L-OMAT-large-ft-AM` to an offline-recast fp32
  artifact. `grace-mp` remains the one fp64 model in the updated model config and
  is flagged as not precision-matched.
- **MD settings.** `paper_configs` runs 80 000 steps × 0.25 fs (20 ps),
  recording every 10th frame, with `tdamp = 100 × timestep`, and excludes the
  molecular crystals (those run under i-PI, see below). `updated_configs`
  runs 22 ps with a per-system timestep (1.0 fs default, 0.5 fs for
  H-containing systems, 2.0 fs for CuAu), records every step, and uses a
  fixed 20 fs `tdamp` for all systems.

### Model config files

The four stages that build a calculator — `md_production`, `md_timings`,
`e_f_rmses` and `pressures` — each carry a `model_calculators.json` declaring,
per model, the imports needed and a self-contained Python expression that
constructs the ASE calculator. `rdfs` and `vdos` read trajectories off disk and
have no catalog.


The two trees also differ in panel size: `paper_configs` covers 15 models,
`updated_configs` 17 — `orb-v3-direct` and `pet-omat-xl` were added after the
paper and appear only in the updated tree.


#### `md_production`

The paths to the model config files are:

```
paper_configs/md_production/model_calculators.json
updated_configs/md_production/model_calculators.json
```

Calculator expressions that differ between the trees:

| Model | `paper_configs` | `updated_configs` |
| --- | --- | --- |
| `chgnet` | `stress_weight=0.01` | argument dropped |
| `grace-oam` | `grace_fm('GRACE-2L-OMAT-large-ft-AM')` — the shipped fp64 checkpoint | `TPCalculator('../data/models/GRACE-2L-OMAT-large-ft-AM-fp32', float_dtype='float32')` — an offline recast, by explicit path |
| `mace-mp-0`, `mace-mpa-0`, `mace-mh-omat` | `default_dtype='float64'` | `default_dtype='float32'` |
| MatterSim | `mattersim-v1-1M`: `MatterSimCalculator(device='cuda')`, i.e. the 1M default | `mattersim-v1-5M`: `load_path='MatterSim-v1.0.0-5M.pth'` |
| `orb-v2`, `orb-v3` | `precision='float64'` | `precision='float32-high'` — fp32 weights with TF32 matmuls, and a process-global setting |
| `pet-oam-xl` | no `dtype` — defers to the checkpoint | `dtype=torch.float32`, pinned explicitly |
| `orb-v3-direct`, `pet-omat-xl` | not in the panel | added, at `'float32-high'` and `dtype=torch.float32` respectively |

`eq-v2-M-omat`, `eSEN-30M-OAM`, `grace-mp`, `nequip`, `uma-s-omat` and
`uma-m-omat` carry identical calculator expressions in both trees: none of them
expose a settable precision, so there was nothing to change. (The updated tree's
entries still differ in carrying the precision metadata fields above.)

#### `md_timings`

The paths to the model config files are:

```
paper_configs/md_timings/model_calculators.json
updated_configs/md_timings/model_calculators.json
```

Differences between paper and updated model config files:

- `chgnet` — `stress_weight=0.01` dropped.
- `grace-oam` — fp64 `grace_fm(...)` → the recast fp32 `TPCalculator(...)`.
- MatterSim — `mattersim-v1-1M` (`MatterSim-v1.0.0-1M.pth`) →
  `mattersim-v1-5M` (`MatterSim-v1.0.0-5M.pth`). Both trees pin the checkpoint
  by path here, unlike `md_production`, where the paper tree takes the 1M
  default implicitly.
- `pet-oam-xl` — `dtype=torch.float32` pinned explicitly.
- `orb-v3-direct`, `pet-omat-xl` — present only in the updated tree.

MACE and ORB do **not** differ in this stage: the paper catalog runs them at
`default_dtype='float32'` and `precision='float32-high'`, matching the updated
tree, so the two timing sets are taken at the same precision.

##### How the timing is taken

Both trees run the same short benchmark — 0.2 ps per system, per-system
timestep, `tdamp = 25 fs`, every step recorded, over a reduced system set
(molecular crystals, `H_1050K_Rupp_QE` and `Pt111w24H2O_380K_Heenen_VASP` are
skipped) — and wrap `dyn.run(n_steps)` in a CUDA-synchronized
`time.perf_counter()`, writing `md_timing_<model>.csv` alongside the
trajectory. They differ in what falls inside the clock:

- `paper_configs` times the whole 0.2 ps from the first step, so one-time
  costs (CUDA kernel autotune, first-call compilation, lazily built neighbour
  lists) are included in `seconds_per_step`.
- `updated_configs` first runs `NVT_WARMUP_FRACTION = 0.1` of the steps
  untimed, synchronizes, and only then starts the clock — a steady-state
  per-step cost, with the startup transient excluded rather than averaged in.
  The CSV carries an extra `warmup_steps` column recording this.

Output goes to `../data/output-trajs-timings-paper/` and
`../data/output-trajs-timings-updated/` respectively.

#### `e_f_rmses`

```
paper_configs/e_f_rmses/model_calculators.json
updated_configs/e_f_rmses/model_calculators.json
```

Within each tree this catalog matches that tree's `md_production` catalog
entry for entry — the RMSEs are evaluated with exactly the calculators that
produced the trajectories. So the paper/updated differences are the ones in the
`md_production` table above.

#### `pressures`

The paths to the model config files are:

```
paper_configs/pressures/model_calculators.json
updated_configs/pressures/model_calculators.json
```


What is specific to this stage:

- `chgnet` — `compute_stress=True` in **both** trees. These are the only two
  entries anywhere that turn stress on; every other catalog sets
  `compute_stress=False`. The stage needs the stress tensor and CHGNet only
  returns it when asked. The trees differ in the neighbouring flags: the paper
  copy also sets `compute_hessian=True`, the updated one leaves it `False`, and
  the paper copy drops `stress_weight` here (unlike its other stages).
- Every checkpoint path in both catalogs resolves under `../data/models/`,
  including `nequip`'s `compile_path`.

Only `updated_configs` ships a pressure script; the paper tree carries the
catalog with no script to feed it. The script skips any system whose directory
starts with `Pt111w24H2O_` — a slab in a padded cell, where a cell-averaged
pressure is not a meaningful quantity to compare.



## Benchmark systems

Reference trajectories span several system types:

- **Pure metals** — bulk Ag (600 K), Au (1500 K), Cu (1000 K)
- **Metal alloys** — CuAu (500 K), CuZrAl (1500 K), LiMgAlZnSn (600/900 K),
  Pt3Co (300 K)
- **Metal dichalcogenides** — MoS2 (300 K), TiSe2 (400 K)
- **Perovskites** — CsSnI3 (500 K), MAPbBr3 (300 K)
- **Molecular crystals** — anthracene (293 K), naphthalene, pentacene,
  picene, tetracene (295 K)
- **Metal–water interfaces** — Pt(111) with 24 H2O (380 K) — `updated_configs`
  only
- **Hydrogen** — H at 1050 K — `updated_configs` only

`paper_configs` covers the first five categories.

Per-system settings (temperature, stride, timestep) are recorded in the
`*_settings_ref.csv` carried by the analysis stages that need them —
`updated_configs/{pressures,rdfs,vdos}/` and the i-PI harness's
`ipi_settings_ref.csv`. The MD stages take temperature from the system
directory name (the `\d+K` in it) and the timestep from the per-system rule
above, so they carry no settings file.


## Pipeline

Both trees use the same stage names, but the paper tree is not complete:

```
md_production/   NVT MD production runs for every model/system pair.
md_timings/      Standalone timing harness sharing the same MD driver: a
                 short 0.2 ps run over a reduced system set.
e_f_rmses/       Energy/force RMSE of each MLIP against reference AIMD
                 trajectories, with per-system isolated-atom energy
                 corrections and per-system-type aggregation.
pressures/       Per-frame stress and trajectory-averaged pressure, matched
                 to reference trajectories by simulated time, plus error
                 aggregation.  Script in `updated_configs` only.
rdfs/            Radial distribution functions from MLIP vs. reference
                 trajectories (via MDTraj), matched by simulation length.
vdos/            Vibrational density of states via the Fourier transform of
                 the velocity autocorrelation function (Hann-windowed),
                 matched by simulation length, with normalization/plotting.
                 `updated_configs` only.
```

New analysis work goes in `updated_configs`.

Alongside the analysis scripts, each stage carries the plotting scripts for
the figures it produces — `figure_2.py`, `fig_3.py`, `figure_4.py` and the
`figure_SI_*.py` set — plus the aggregation steps they read from
(`compute_mean_rmses_by_system_type.py`, `get_model_pressure_errors.py`,
`get_normalized_VDOS.py`).

Additionally, `paper_configs/md_production/molecular_crystals_ipi/generic/`
holds the unified i-PI harness used for the five molecular crystals, which
run under i-PI rather than the ASE driver. It has its own
[README](paper_configs/md_production/molecular_crystals_ipi/generic/README.md)
covering the `SYSTEM` × `MODEL_NAME` submission grid. Note that the molecular
crystals are excluded from `md_timings` in both trees.

## Data layout

No `data/` directory is committed — trajectories and checkpoints are too large
to track. Each tree's scripts resolve paths relative to their own stage
directory, so create `paper_configs/data/` and `updated_configs/data/` with
this layout before running anything:

```
data/ref-trajs/<system>/traj.extxyz     Reference AIMD trajectories
data/mlip-trajs-20fs-tau/<system>/      MD output, written by `updated` md_production
data/mlip-trajs/<system>/               MD output, written by `paper` md_production
data/output-trajs-timings-updated/      Timing output, written by `updated` md_timings
data/output-trajs-timings-paper/        Timing output, written by `paper` md_timings
data/models/                            Local model checkpoints
```

The updated tree keeps to this convention throughout: its scripts reference
only their own directory or `../data/`. The paper tree is less consistent and
needs editing before it will run elsewhere —
`paper_configs/rdfs/get-rdf-and-results-by-system-type-same-simulation-length.py`
and `fig_3.py` still carry absolute `/home/mjgawkowski/…` paths, and
`paper_configs/e_f_rmses/compute_mean_rmses_by_system_type.py` reads a
stage-local `data/` where its updated counterpart reads `../data/`.

## Usage

Each stage is invoked with the model selected via the `MODEL_NAME`
environment variable, from inside the stage directory:

```bash
# Run NVT MD production for one model over all reference systems
cd updated_configs/md_production
MODEL_NAME=mace-mpa-0 python md_script-generic.py

# Compute energy/force RMSEs against AIMD reference trajectories
cd ../e_f_rmses
MODEL_NAME=mace-mpa-0 python rmse_script-generic.py

# Compute pressures, matched to the reference simulation length
cd ../pressures
MODEL_NAME=mace-mpa-0 python pressure_script-generic.py
```

An unset or unrecognized `MODEL_NAME` fails immediately with the list of
valid names. MD production skips any system whose output trajectory already
exists, so reruns are resumable.

