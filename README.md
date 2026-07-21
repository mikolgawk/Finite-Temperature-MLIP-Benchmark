# Finite-Temperature MLIP Benchmark

Benchmarking suite for evaluating (MLIPs) under finite-temperature molecular dynamics. For a panel of
foundation MLIPs, the pipeline runs NVT MD on a set of reference systems and
compares the resulting trajectories to AIMD reference trajectories along four
axes: energy/force accuracy, pressure, radial distribution functions (RDFs),
and vibrational density of states (VDOS).

## Two configuration trees

The repository tracks configurations, not data. The same pipeline exists
twice, under two top-level directories:

| Tree | What it is |
| --- | --- |
| `paper_configs/` | The configurations that produced the results on arxiv. Frozen; kept for reproducibility. |
| `updated_configs/` | The revised panel — audited numerical precision, refreshed checkpoints. Where new work goes. |

The substantive differences:

- **Precision.** `paper_configs` ran each model at whatever its constructor
  defaulted to or at fp64 (`mace_mp(default_dtype='float64')`,
  `pretrained.orb_v2(precision='float64')`). `updated_configs` normalizes to
  fp32 wherever precision is settable, and each catalog entry records
  `weight_dtype`, `matmul_precision`, `precision_settable`, `verified`, and a
  `dtype_note` explaining how that was established (source reading,
  runtime probe, or unverified). Models where precision genuinely cannot be
  set — CHGNet, eqV2, eSEN, UMA — are documented as such rather than silently
  assumed.
- **Checkpoints.** `mattersim-v1-1M` → `mattersim-v1-5M`; `grace-oam` moves
  from the shipped fp64 `GRACE-2L-OMAT-large-ft-AM` to an offline-recast fp32
  artifact. `grace-mp` remains the one fp64 model in the updated catalog and
  is flagged as not precision-matched.
- **MD settings.** `paper_configs` runs 80 000 steps × 0.25 fs (20 ps),
  recording every 10th frame, with `tdamp = 100 × timestep`, and excludes the
  molecular crystals (those run under i-PI, see below). `updated_configs`
  runs 22 ps with a per-system timestep (1.0 fs default, 0.5 fs for
  H-containing systems, 2.0 fs for CuAu), records every step, and uses a
  fixed 20 fs `tdamp` for all systems.

### Model catalogs

Each stage carries a `model_calculators.json` declaring, per model, the
imports needed and a self-contained Python expression that constructs the ASE
calculator. The catalog is what `MODEL_NAME` selects from at runtime.

Calculator expressions that differ between the trees:

| Model | `paper_configs` | `updated_configs` |
| --- | --- | --- |
| `chgnet` | `stress_weight=0.01` | argument dropped |
| `grace-oam` | `grace_fm('GRACE-2L-OMAT-large-ft-AM')` — the shipped fp64 checkpoint | `TPCalculator('../data/models/GRACE-2L-OMAT-large-ft-AM-fp32', float_dtype='float32')` — an offline recast, by explicit path |
| `mace-mp-0`, `mace-mpa-0`, `mace-mh-omat` | `default_dtype='float64'` | `default_dtype='float32'` |
| MatterSim | `mattersim-v1-1M`: `MatterSimCalculator(device='cuda')`, i.e. the 1M default | `mattersim-v1-5M`: `load_path='MatterSim-v1.0.0-5M.pth'` |
| `nequip` | bare `compile_path='mir-group__NequIP-OAM-XL__0.1.nequip.pth'` | `compile_path='../data/models/…'` |
| `orb-v2`, `orb-v3`, `orb-v3-direct` | `precision='float64'` | `precision='float32-high'` — fp32 weights with TF32 matmuls, and a process-global setting |
| `pet-oam-xl`, `pet-omat-xl` | no `dtype` — defers to the checkpoint | `dtype=torch.float32`, pinned explicitly |

`eq-v2-M-omat`, `eSEN-30M-OAM`, `grace-mp`, `uma-s-omat` and `uma-m-omat` are
byte-identical between the trees: none of them expose a settable precision, so
there was nothing to change.

The schema differs as well. `paper_configs` entries carry only `name`,
`package`, `imports`, `calculator_expr` and a `<package>_version`.
`updated_configs` adds the precision audit — `weight_dtype`,
`matmul_precision`, `precision_settable`, `weight_dtype_verified`, `verified`
(`source` / `runtime-probe` / `unverified`) and a prose `dtype_note` recording
what was measured and what was retracted — plus `conversion` where a
checkpoint was recast, `nequip_artifact`, `probe_checkpoint`, and a top-level
`shared_notes` block that entries reference via `shared_note_ref` (the UMA
inference-settings note is shared by both UMA entries). That is why the
updated catalogs are ~20 KB against ~6 KB.

Catalogs also vary *within* a tree, deliberately:

- `paper_configs/e_f_rmses` omits `orb-v3-direct` and `pet-omat-xl` (15
  models, not 17).
- `paper_configs/md_timings` was already fp32: MACE at
  `default_dtype='float32'`, ORB at `'float32-high'`, and MatterSim pinned to
  `MatterSim-v1.0.0-1M.pth` rather than left to the default.
- `updated_configs/pressures` sets `compute_stress=True` for `chgnet` — the
  stage needs the stress tensor — and its `nequip` entry still uses the bare
  `compile_path` filename rather than `../data/models/`.
- `rdfs/` and `vdos/` have no catalog: they read trajectories off disk and
  never construct a calculator.

## Benchmark systems

Reference trajectories span several system types (see
`e_f_rmses/compute_mean_rmses_by_system_type.py` for the exact
classification):

- **Pure metals** — bulk Ag (600 K), Au (1500 K), Cu (1000 K)
- **Metal alloys** — CuAu (500 K), CuZrAl (1500 K), LiMgAlZnSn (600/900 K),
  Pt3Co (300 K)
- **Metal dichalcogenides** — MoS2 (300 K), TiSe2 (400 K)
- **Perovskites** — CsSnI3 (500 K), MAPbBr3 (300 K)
- **Molecular crystals** — anthracene (293 K), naphthalene, pentacene,
  picene, tetracene (295 K)
- **Metal–water interfaces** — Pt(111) with 24 H2O (380 K)
- **Hydrogen** — H at 1050 K

Per-system settings (temperature, stride, timestep) are recorded in each
analysis directory's `*_settings_ref.csv`.


## Pipeline stages

Both trees use the same stage names:

```
md_production/   NVT MD production runs for every model/system pair; also
                 emits per-step timing alongside each trajectory.
md_timings/      Standalone timing harness sharing the same MD driver: a
                 short 0.2 ps run with a warm-up fraction excluded from the
                 timed region, over a reduced system set.
e_f_rmses/       Energy/force RMSE of each MLIP against reference AIMD
                 trajectories, with per-system isolated-atom energy
                 corrections, per-system-type aggregation, and figures
                 (figure_2, figure_SI_2/3/4).
pressures/       Per-frame stress and trajectory-averaged pressure, matched
                 to reference trajectories by simulated time, plus error
                 aggregation and figures (figure_4, figure_SI_4/6/15).
rdfs/            Radial distribution functions from MLIP vs. reference
                 trajectories (via MDTraj), matched by simulation length,
                 plus figures (fig_3, figure_SI_14).
vdos/            Vibrational density of states via the Fourier transform of
                 the velocity autocorrelation function (Hann-windowed),
                 matched by simulation length, with normalization/plotting.
data/            Placeholder for local inputs; see below.
```

Additionally, `paper_configs/md_production/molecular_crystals_ipi/generic/`
holds the unified i-PI harness used for the five molecular crystals, which
run under i-PI rather than the ASE driver. It has its own
[README](paper_configs/md_production/molecular_crystals_ipi/generic/README.md)
covering the `SYSTEM` × `MODEL_NAME` submission grid.

## Data layout

`data/` is tracked as an empty placeholder — trajectories and checkpoints are
large and live outside the repository. Populate it, in the tree you are
running:

```
data/ref-trajs/<system>/traj.extxyz     Reference AIMD trajectories
data/mlip-trajs-20fs-tau/<system>/      MD output, written by md_production
data/models/                            Local model checkpoints
```

Both trees load model checkpoints from `../data/models/`. Trajectories
differ: `updated_configs` resolves those under `../data/` too, whereas the
`paper_configs` scripts still read from a sibling `../ref-trajs/` and write to
`../mlip-trajs/`.

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

RDF and VDOS analyses operate on the MD trajectories already produced above
and are run per-analysis rather than per-model; see the `--help` output of
`rdfs/get-rdf-and-results-by-system-type-same-simulation-length.py` and
`vdos/get_normalized_VDOS.py`. Note that `get_normalized_VDOS.py` imports two
batch/plot helper modules that are not tracked here, and its `--ref-root`
default is an absolute cluster path — pass `--ref-root` explicitly.

Figure-generating scripts (`figure_*.py`, `fig_*.py`) reproduce the plots in
the accompanying paper from the CSVs each pipeline stage produces.

## Requirements

MLIP evaluation depends on the packages named in the `package` field of each
catalog entry (`chgnet`, `fairchem-core`, `tensorpotential`, `mace-torch`,
`mattersim`, `nequip`, `orb-models`, `upet`), plus `ase`, `numpy`, `pandas`,
`scipy`, `matplotlib`, `seaborn`, and `mdtraj` for the analysis/plotting
stages. Models need mutually incompatible dependency sets, so one environment
rarely covers the whole panel — run the panel a slice at a time, one
environment per group of models. A CUDA GPU is expected for MD production,
RMSE, and pressure evaluation.
