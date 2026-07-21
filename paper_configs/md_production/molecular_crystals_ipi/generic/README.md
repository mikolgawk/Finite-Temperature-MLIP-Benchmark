# Unified i-PI molecular-crystal runs

Collapses the 85 hand-maintained `<system>/<model>/` directories in the
parent folder into one declarative setup, driven by the same
`../../model_calculators.json` that `md_script-generic.py` uses.

What used to be duplicated, and where it lives now:

| Was | Now |
| --- | --- |
| 85 x `run-ase.py`, differing only in the calculator | `run-ase-generic.py` + `MODEL_NAME` |
| 85 x `submit.sh`, differing only in `conda activate` | `submit-generic.sh` + a `CONDA_ENV` you supply |
| 85 x `input.xml`, 3 unique variants | `input.xml.template` + `ipi_settings_ref.csv` |
| 85 x `init.xyz`, 5 unique | `systems/<system>/init.xyz` |

## Usage

```bash
# One (system, model) pair
sbatch --job-name=naphthalene-mace-mpa-0-ipi \
       --export=ALL,SYSTEM=naphthalene_295K_Sharma_S,MODEL_NAME=mace-mpa-0 \
       submit-generic.sh

# The whole 5 x 17 grid, or a slice of it
./submit_all.sh
./submit_all.sh --model mace-mpa-0
./submit_all.sh --system picene_295K_Sharma_S
./submit_all.sh --dry-run
```

Each job materializes `runs/<system>/<model>/` (rendered `input.xml` plus a
staged `init.xyz`) and runs i-PI against `run-ase-generic.py` there, so
outputs stay separated exactly as they did in the old layout.

Valid `SYSTEM` values are the `system` column of `ipi_settings_ref.csv`;
valid `MODEL_NAME` values are the `name` fields in
`../../model_calculators.json`. A bad `MODEL_NAME` fails immediately with
the list of valid ones, rather than after the job has queued.

## Python environments

Environments are yours to manage -- the catalog describes models, not
machines. Either activate one before submitting, or name it with
`CONDA_ENV` and `submit-generic.sh` will activate it for you:

```bash
CONDA_ENV=my-mace-env ./submit_all.sh --model mace-mpa-0
```

Models need different packages, so one environment rarely covers the whole
panel; the `package` field of each catalog entry says what a given model
requires (`mace-torch`, `fairchem-core`, `tensorpotential`, `chgnet`,
`mattersim`, `nequip`, `orb-models`, `upet`). Submitting a slice per
environment is the usual way to drive the full grid.

## Per-system settings

`ipi_settings_ref.csv` records the settings that genuinely differ between
systems, following the `*_settings_ref.csv` convention used by the `rdfs/`,
`pressures/` and `vdos/` stages. The Langevin `tau` split is real and
preserved from the original inputs:

| System | T (K) | timestep (fs) | tau (fs) |
| --- | --- | --- | --- |
| anthracene | 293 | 0.5 | 10 |
| naphthalene | 295 | 0.5 | 10 |
| pentacene | 295 | 0.5 | 500 |
| picene | 295 | 0.5 | 500 |
| tetracene | 295 | 0.5 | 10 |

`total_steps`, `total_time_s` and `seed` are identical across all systems
and are carried in the same table rather than hard-coded in the template.

## Differences from the old per-directory tree

These are deliberate, and change behaviour relative to the original files:

- **Calculators come from the catalog verbatim.** The old `run-ase.py`
  files had drifted from `model_calculators.json`: `eq-v2-M-omat` loaded
  `eqV2_86M_omat.pt` rather than the catalog's
  `eqV2_86M_omat_mp_salex.pt`, `chgnet` omitted `stress_weight=0.01`, and
  the MatterSim entry loaded the 5M checkpoint while the catalog's
  `mattersim-v1-1M` entry takes the 1M default. The catalog now wins in
  every case.
- **Model paths are relative.** The old files hard-coded two different
  absolute prefixes (`/home/mjgawkowski/...` and `/share/rcif2/...`)
  depending on which cluster the directory was last touched on.
  `run-ase-generic.py` builds the calculator with the working directory set
  to the catalog's own directory, so the catalog's `../data/models/...` paths
  resolve exactly as they do for `md_script-generic.py`.
- **Socket addresses are unique per job**
  (`ipi_<system>_<model>_<jobid>`). Every old `input.xml` used the address
  `driver`, i.e. `/tmp/ipi_driver`, so two concurrent i-PI jobs on one node
  would collide. Relatedly, the old `submit.sh` ran `rm -f /tmp/ipi_*`,
  which would delete *other* running jobs' sockets; the unified script
  removes only its own.
- **`--job-name` is derived from the pair.** Every old `submit.sh` said
  `--job-name=anthracene-ipi`, including the ones under `naphthalene/`,
  `pentacene/`, `picene/` and `tetracene/`.
- **The ORB entries no longer set `TORCH_COMPILE_DISABLE` /
  `torch._dynamo.disable()`.** Those lines ran *after* the ORB imports in
  the old scripts, and `torch._dynamo.disable()` called bare returns a
  wrapper rather than disabling anything, so they were already inert. The
  catalog does not carry them.

- **`grace-mp` and `grace-oam` now run under SLURM like everything else.**
  Their old `submit.sh` files were the two outliers in the tree: no SLURM
  header and no `conda activate`, just `export CUDA_VISIBLE_DEVICES=1` for
  a local run. They now submit through the same path as the other fifteen
  models; both want the `grace_env` environment.
