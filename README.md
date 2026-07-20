# Finite-Temperature MLIP Benchmark

Benchmarking suite for evaluating machine-learned interatomic potentials
(MLIPs) under finite-temperature molecular dynamics.
For a panel of foundation MLIPs, the pipeline runs NVT
MD on a set of reference systems and compares the resulting trajectories to
AIMD reference trajectories along four axes: energy/force accuracy,
pressure, radial distribution functions (RDFs), and vibrational density of
states (VDOS).

## Benchmark systems

Reference trajectories span several system types (see
`e_f_rmses/compute_mean_rmses_by_system_type.py` for the exact
classification):

- **Pure metals** — bulk Ag, Au, Cu
- **Metal alloys** — CuAu, CuZrAl, LiMgAlZnSn, Pt3Co
- **Metal dichalcogenides** — MoS2, TiSe2
- **Perovskites** — CsSnI3, MAPbBr3
- **Molecular crystals** — anthracene, naphthalene, pentacene, picene,
  tetracene
- **Metal-water interfaces** — Pt(111) with 24 H2O
- **Hydrogen** — H at 1050 K

Per-system simulation settings (temperature, stride, timestep) are recorded
in each analysis directory's `*_settings_ref.csv`.

## Evaluated models

Models are declared declaratively in 
`models_revised.json` (imports + an ASE-calculator-constructing expression),
selected at runtime via the `MODEL_NAME` environment variable. The panel
currently includes: CHGNet, eqV2-M-OMat, eSEN-30M-OAM, GRACE (MP and OAM),
MACE (MP-0, MPA-0, MH-OMat), MatterSim-v1-5M, NequIP-OAM-XL, ORB (v2, v3,
v3-direct), PET (OAM-XL, OMAT-XL), and UMA (s-omat, m-omat).

## Repository layout

```
md_production/   NVT MD production runs (22 ps, Nose-Hoover chain) for every
                 model/system pair; also emits per-step timing.
md_timings/      Standalone timing harness sharing the same MD driver.
e_f_rmses/       Energy/force RMSE of each MLIP against reference AIMD
                 trajectories, plus per-system-type aggregation and figures.
pressures/       Per-frame stress / trajectory-averaged pressure, matched to
                 reference trajectories by simulated time, plus figures.
rdfs/            Radial distribution functions from MLIP vs. reference
                 trajectories (via MDTraj), matched by simulation length.
vdos/            Vibrational density of states via the Fourier transform of
                 the velocity autocorrelation function (Hann-windowed),
                 matched by simulation length, with normalization/plotting.
data/            (empty placeholder for local input data)
```

Each analysis directory is self-contained: it reads trajectories from a
sibling `../ref-trajs/` (and, for MD outputs, `../mlip-trajs-*/`) directory,
and writes results/figures back into its own directory or a `results/`
subfolder. Figure-generating scripts (`figure_*.py`, `fig_*.py`) reproduce
the plots in the accompanying paper from the CSVs each pipeline stage
produces.

## Usage

Each pipeline stage is invoked with the model to evaluate selected via the
`MODEL_NAME` environment variable, e.g.:

```bash
# Run NVT MD production for one model over all reference systems
cd md_production
MODEL_NAME=mace-mpa-0 python md_script-generic.py

# Compute energy/force RMSEs against AIMD reference trajectories
cd ../e_f_rmses
MODEL_NAME=mace-mpa-0 python rmse_script-generic.py

# Compute pressures, matched to the reference simulation length
cd ../pressures
MODEL_NAME=mace-mpa-0 python pressure_script-generic.py
```

RDF and VDOS analyses operate on the MD trajectories already produced above
and are run per-analysis rather than per-model; see the `--help` output of
`rdfs/get-rdf-and-results-by-system-type-same-simulation-length.py` and
`vdos/get_normalized_VDOS.py`.

Valid values for `MODEL_NAME` are the `name` fields listed in the relevant
directory's `model_calculators.json`/`models_revised.json`.

## Requirements

MLIP evaluation depends on the packages named in `model_calculators.json`
(`chgnet`, `fairchem-core`, `tensorpotential`, `mace-torch`, `mattersim`,
`nequip`, `orb-models`, `upet`), plus `ase`, `numpy`, `pandas`, `scipy`,
`matplotlib`, `seaborn`, and `mdtraj` for the analysis/plotting stages. A
CUDA GPU is expected for MD production and RMSE evaluation.
