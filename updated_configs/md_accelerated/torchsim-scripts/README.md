# Accelerated TorchSim MD

Run all `md_*.py` scripts sequentially with their inline uv dependencies:

```bash
./run_all_md.sh
```

The launcher works from any working directory. Preview the discovered scripts with
`./run_all_md.sh --dry-run`, or select a GPU with
`CUDA_VISIBLE_DEVICES=0 ./run_all_md.sh`.

Output is streamed to the terminal and saved to
`logs/md/<timestamp>_<pid>/<script-name>.log`. A failed script does not stop the
remaining models. The launcher exits with status 1 if any script or logging
operation fails; interrupts stop the batch.

Each model catches per-system exceptions during setup and simulation, writes a
**zero-byte** `md_timing_<model-name>.csv` in that system's output directory, and
continues to the next system. Failure-marker write errors are reported and also
allow processing to continue. An existing timing CSV, including an empty failure
marker, skips that model/system pair. Delete an empty CSV to retry the pair.
Partial HDF5 trajectories may remain after failures and are not completed runs.

As in the non-accelerated scripts, missing initial structures are skipped without
a failure marker. Dependency or model-loading failures before the system loop
end that script; the launcher continues with the next model. Handled per-system
failures are reported in logs but do not change the script's exit status.

Runs require uv, the CUDA environment and dependencies declared by each script,
benchmark metadata at `updated_configs/data/ref-trajs/md_metadata.json`, and the
required structures and model checkpoints. Outputs go under
`updated_configs/data/mlip-trajs-torchsim-accelerated/`.
