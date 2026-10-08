# TorchSim RMSE scripts

`md_eager/` matches `paper_v2_configs/md_eager/torchsim-scripts/`.
`md-accelerated/` matches `paper_v2_configs/md_accelerated/torchsim-scripts/`
(including separate UMA compile/turbo variants).

Each entry point preserves its source MD script's inline uv dependencies,
model implementation, checkpoint, precision, neighbor list and acceleration
settings. Model setup is copied locally; later MD edits do not automatically
update these scripts. No MD integration is performed.

Run from this directory:

```bash
uv run --script md_eager/rmse_mace_mp_0.py
uv run --script md-accelerated/rmse_mace_mp_0.py
bash md_eager/run_all_rmses.sh
bash md-accelerated/run_all_rmses.sh
```

Both launchers support `--dry-run` and retain per-model logs. Individual Python
scripts support `--help`, `--ref-dir`, `--output-dir`, `--max-frames`, `--force`,
`--raw-energies`, `--isolated-atom-dir`, `--md-dir`, and `--debug`. Relative CLI paths resolve
against the working directory. Default data paths resolve against the repository.
CUDA and the same model access/checkpoints/toolchains as the MD runs are required.

The shared `torchsim_rmse.py` evaluates `traj*.extxyz` recursively under
`paper_v2_configs/data/ref-trajs`, selecting only systems whose matching TorchSim
MD run has an HDF5 trajectory and a completed timing CSV. The default MD
source matches the eager or accelerated RMSE runner; `--md-dir` overrides it.
Reference values come from `REF_energy` /
`REF_forces` when present, otherwise the ASE reference calculator. Energy RMSE
is in eV/atom, computed from each frame's energy error divided by its atom count;
force RMSE is over all Cartesian components in eV/Angstrom. Aromatic systems and
hydrogen use the existing benchmark's isolated-atom energy corrections; missing
required isolated-atom files produce explicit failures. `--raw-energies` disables
these corrections. Trajectory predictions use the TorchSim model directly.
For eqV2 and eSEN, isolated-atom energies use a native FairChem `OCPCalculator`
loaded from the same checkpoint, on the same device, in float32 with AMP
disabled. This preserves the isolated atoms' nonperiodic flags, which the
periodic TorchSim legacy adapter rejects. The native calculator computes only
energies, with force and stress heads disabled, and is loaded only when a
correction is needed. The native eqV2 instance also handles empty neighbour
graphs: edge rotations are empty and attention projects a zero incoming-message
sum, retaining the learned bias, residuals, and feed-forward layers. This avoids
FairChem 1.10.0's empty-tensor errors without introducing artificial neighbours.
The periodic TorchSim instance is unchanged. Offsets are cached per reference
file. These runners'
CSV `engine` values include `+ase-isolated-atoms` to record the correction
method. Other models also use TorchSim for their isolated-atom predictions.

Summaries are written to `paper_v2_configs/e_f_rmses/data/e-f-predictions/md_eager/`
and `paper_v2_configs/e_f_rmses/data/e-f-predictions/md-accelerated/`, respectively, as
`rmse-results-all_<MD model name>.csv`. Failed frames are excluded from both sides
of the comparison and recorded in `.failures.json` sidecars. Any failure produces
a nonzero exit status. Reruns preserve existing CSV results and retry only the
trajectories listed in a `.failures.json` sidecar. New rows are appended, while
partial rows for retried trajectories are replaced to avoid duplicates. Without
a sidecar, only eligible trajectories missing from the CSV are evaluated.
Results and remaining failures are checkpointed after each trajectory; resolved
failure records are cleared, while failures for unavailable references or systems
without completed MD remain recorded. Saved paths still match after moving the
reference data root. `--force` recomputes all eligible trajectories.
`--max-frames` limits reference and evaluated counts
to the selected prefix; use a separate `--output-dir` for smoke tests so their
summaries do not cause full runs to be skipped.

Both modes include `rmse_orb_v3_omat.py` and `rmse_orb_v3_direct_omat.py`.
These use the conservative/direct OMAT checkpoints and exact MD identifiers.
The conservative OMAT variant uses `float32-highest` with TF32 disabled; the
direct OMAT variant uses `float32-high` with TF32 enabled. Eager runners disable
compilation and accelerated runners enable it, matching the production scripts.
See [the ORB v3 OMAT workflow](../../README-orb-v3-omat.md) for pressure evaluation
and generation of all four metrics.
