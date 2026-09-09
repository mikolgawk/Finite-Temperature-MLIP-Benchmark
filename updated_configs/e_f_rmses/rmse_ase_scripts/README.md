# ASE energy and force RMSE scripts

ASE counterparts to `../rmse_torchsim_scripts`, derived from the calculator
setups in `../../md/ase-scripts` and `../../md_accelerated/ase-scripts`.
`md/` contains 17 standard runners and `md-accelerated/` contains 19 accelerated
runners. Each script retains its source's uv dependency metadata and calculator
configuration. `_mace.py` and `_uma.py` retain the accelerated stress-disabling
helpers. The scripts evaluate reference frames without integrating dynamics.

From this directory:

```bash
uv run --script md/rmse_mace_mp_0.py --max-frames 2
uv run --script md-accelerated/rmse_mace_mp_0.py --max-frames 2
bash md/run_all_rmses.sh --dry-run
bash md/run_all_rmses.sh
bash md-accelerated/run_all_rmses.sh
```

Use each Python runner's `--help` for reference, output and isolated-atom paths,
`--raw-energies`, `--max-frames`, `--force`, and `--debug`. Batch launchers run all
frames with default options and retain individual logs; failures do not stop the
remaining runners.

Defaults read `updated_configs/data/ref-trajs/traj*.extxyz` recursively and write
`updated_configs/data/e-f-predictions-ase/{md,md-accelerated}/rmse-results-all_<model>.csv`.
The CSV schema matches the TorchSim evaluators. Energy RMSE is in eV/atom;
force RMSE is over Cartesian components in eV/Angstrom. Isolated-atom corrections
for hydrogen and the acenes match the TorchSim evaluators. Failed evaluations
produce a `.failures.json` sidecar and a nonzero exit status.

The standard source `md_chgnet.py` contains a self-import rather than a usable
calculator factory. Its RMSE counterpart uses the accelerated source's native
CHGNet energy/force-only calculator with `torch.compile` removed.

Validation: all 36 Python runners pass `--help` without importing model packages;
both batch launchers pass shell syntax and dry-run checks. The shared evaluator
was checked with ASE EMT fixtures for known energy/force errors and isolated-atom
corrections. Full model evaluations require the source dependencies, checkpoints,
reference data and CUDA hardware and have not been run here.
