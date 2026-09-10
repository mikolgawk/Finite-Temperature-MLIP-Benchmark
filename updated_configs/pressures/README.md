# Stress and pressure workflow

The `md/` and `md_accelerated/` trees contain stress-enabled copies of the sibling
MD runners, with the same model choices, precision, thermostats and acceleration
settings. The sibling trees are unchanged. Each tree has `ase-scripts/` and
`torchsim-scripts/`; each runner retains its `uv` dependency metadata.

## Run MD

From this directory, run one model, or a whole backend:

```bash
uv run md/ase-scripts/md_mace_mp_0.py
uv run md/torchsim-scripts/md_mace_mp_0.py
uv run md_accelerated/torchsim-scripts/md_mace_mp_0.py
bash md_accelerated/ase-scripts/run_all_md.sh
```

Runners use `updated_configs/data/ref-trajs/md_metadata.json`. Initial structure
paths are resolved against the repository and then `updated_configs/`. Results
are saved under `updated_configs/data/` in these separate directories:

| Runners | Trajectory root |
| --- | --- |
| `md/` (ASE and TorchSim) | `mlip-trajs-torchsim-matched-stress` |
| `md_accelerated/ase-scripts/` | `mlip-trajs-ase-accelerated-stress` |
| `md_accelerated/torchsim-scripts/` | `mlip-trajs-torchsim-accelerated-stress` |

ASE filenames include `-ase`. Accelerated and eager variants retain their own
identifiers. NequIP uses separate stress-enabled compiled artifacts and cache
configuration; its optional overrides are `NEQUIP_ASE_OEQ_STRESS_MODEL` and
`NEQUIP_OEQ_STRESS_MODEL`. Force/stress derivative heads are retained before
compilation. GRACE retains the virial reduction. MACE and UMA keep their native
stress computation; ORB's stress head/derivatives remain enabled.

Every saved HDF5 frame contains `data/stress` (3 x 3 tensor, eV/Angstrom^3), alongside
positions and velocities. ASE records the time step, stride and initial step as
file attributes. TorchSim records matching `steps/positions` and `steps/stress`.
Its NVT state does not retain stress, so the reporter evaluates the model at each
saved frame to record it. This evaluation is included in the MD timing. The ASE
text logger may still use `stress=False`; the HDF5 writer independently saves
stress. Failed TorchSim systems produce a nonzero process exit and can be retried.

## Run the complete pressure pipeline

```bash
uv run pressure_script-generic.py
```

This searches all three stress-enabled trajectory roots. To select data explicitly:

```bash
uv run pressure_script-generic.py \
  --traj-dir ../data/mlip-trajs-torchsim-matched-stress \
  --ref-dir ../data/ref-trajs \
  --output-dir results
```

`--traj-dir` can be repeated. `--model` (or `MODEL_NAME`) selects an exact trajectory
identifier, including any `-ase`, `-compile`, etc. suffix. Lowercase identifiers
are used in analysis outputs. Duplicate model/system identifiers across roots
are rejected; analyze those roots separately.

References normally come from each system's `traj.extxyz`, with ASE-readable
stress in eV/Angstrom^3. Alternatively use `--reference-file reference.csv`,
containing `trajectory_file`, `frame_index`, and `pressure_GPa` (or
`pressure_ref_GPa`). Include `time_fs` when available. Reference pressures must
use the same potential-only convention as the MLIP pressures.

Time matching uses stored step numbers and the metadata time step. For files
without explicit times it uses `timestep * position_print_stride`; the default
first frame is step zero. Set `--reference-first-step` when references start at a
different step. Explicit extxyz `step`/`time_fs` values take precedence. Comparisons
use the overlapping time window, rather than matching raw frame counts. Saved
sampling grids may differ; histogram comparisons do not pair instantaneous MD
frames. All MLIP stress frames are exported even if the reference is shorter.

Pressure is `-trace(stress)/3 * 160.21766208` GPa. TiSe2 uses the mean of xx and yy
instead. These are tensile-positive potential stresses with **no kinetic/ideal-gas
term**, matching the previous script. Pt111/water interfaces retain the previous
exclusion by default; `--include-interfaces` opts in.

Outputs in `results/`:

- `<model>_stress_per_frame.csv`: all MLIP frames, all six independent stress
  components, pressure, frame index and time.
- `<model>_same-simulation-length_pressure_per_frame.csv` and trajectory summary:
  the matched subset and its statistics.
- `references/<model>.csv`: reference frames matched separately to that model.
- `model_mean_pressure_comparison.csv`: MAE of trajectory mean pressures, with
  equal weight per system (not instantaneous-frame MAE).
- `model_pressure_error_metric.csv` and pair/system/type tables: histogram
  similarity and pressure errors using the existing normalized L1 definition.
- `plots/pressure_model_comparison.{png,pdf}` and `plots/<model>/<system>.png`:
  model-level errors/similarity, matched time series and pressure distributions.
- `pressure_failures.csv`: explicit failures; the command exits nonzero if any
  requested trajectory fails, while retaining successful results.

The automatic plots include all backend variants. The historical `figure_*.py`
publication layouts are retained with their original fixed model tiers. Their
CSV schema remains available, but use the automatic plots for the complete set
of eager/accelerated/ASE variants and per-model matched reference windows.

`get_model_pressure_errors.py` defaults to `results/` and uses each model's matched
reference CSV when present. It can regenerate scores without loading an MLIP.
Stored-stress analysis never initializes model packages. `--recompute --model ...`
explicitly enables the old calculator catalog for extxyz inputs only; install the
corresponding model dependencies separately. Old HDF5 files without stress must
be regenerated with the new runners. `--legacy` retains the historical extxyz
calculator/CSV-settings path; its settings CSV is supplied separately.

## Validation

```bash
python -m unittest -v test_pressure_pipeline
```

CPU tests cover tensor units/sign/order, TiSe2, HDF5 alignment, ASE serialization,
time matching, reference-window rescoring, and end-to-end plot generation.
Production MLIP runs require the model packages/checkpoints, a suitable CUDA
installation, and actual reference structures/trajectories. The local checkout
contains metadata but no reference trajectory files, and the available PyTorch
reports no CUDA device; production model inference has not been validated here.
