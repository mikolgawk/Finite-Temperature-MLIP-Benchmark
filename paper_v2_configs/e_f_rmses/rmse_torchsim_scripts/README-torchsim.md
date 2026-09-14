# TorchSim RMSE scripts

`md/` matches `updated_configs/md/torchsim-scripts/` (16 models).
`md-accelerated/` matches `updated_configs/md_accelerated/torchsim-scripts/`
(16 models, including separate UMA compile/turbo variants).

Each entry point preserves its source MD script's inline uv dependencies,
model implementation, checkpoint, precision, neighbor list and acceleration
settings. Model setup is copied locally; later MD edits do not automatically
update these scripts. No MD integration is performed.

Run from this directory:

```bash
uv run --script md/rmse_mace_mp_0.py
uv run --script md-accelerated/rmse_mace_mp_0.py
bash md/run_all_rmses.sh
bash md-accelerated/run_all_rmses.sh
```

Both launchers support `--dry-run` and retain per-model logs. Individual Python
scripts support `--help`, `--ref-dir`, `--output-dir`, `--max-frames`, `--force`,
`--raw-energies`, `--isolated-atom-dir`, and `--debug`. Relative CLI paths resolve
against the working directory. Default data paths resolve against the repository.
CUDA and the same model access/checkpoints/toolchains as the MD runs are required.

The shared `torchsim_rmse.py` evaluates `traj*.extxyz` recursively under
`updated_configs/data/ref-trajs`. Reference values come from `REF_energy` /
`REF_forces` when present, otherwise the ASE reference calculator. Energy RMSE
is in eV/atom, computed from each frame's energy error divided by its atom count;
force RMSE is over all Cartesian components in eV/Angstrom. Aromatic systems and
hydrogen use the existing benchmark's isolated-atom energy corrections; missing
required isolated-atom files produce explicit failures. `--raw-energies` disables
these corrections. Predictions use the TorchSim model directly, including for
isolated atoms.

Summaries are written to `updated_configs/data/e-f-predictions/md/` and
`updated_configs/data/e-f-predictions/md-accelerated/`, respectively, as
`rmse-results-all_<MD model name>.csv`. Failed frames are excluded from both sides
of the comparison and recorded in `.failures.json` sidecars. Any failure produces
a nonzero exit status. Incomplete runs are retried; completed summaries are skipped
unless `--force` is supplied. `--max-frames` limits reference and evaluated counts
to the selected prefix; use a separate `--output-dir` for smoke tests so their
summaries do not cause full runs to be skipped.
