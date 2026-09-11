"""Shared single-frame TorchSim RMSE evaluation; model setup lives in each runner."""
import argparse
import csv
import json
import re
from pathlib import Path

_ARGS = None
_SCRIPT = None


def early_cli(script):
    """Parse options before importing heavyweight model dependencies."""
    global _ARGS, _SCRIPT
    _SCRIPT = Path(script).resolve()
    data = Path(__file__).resolve().parents[2] / 'data'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref-dir', type=Path, default=data / 'ref-trajs')
    parser.add_argument('--output-dir', type=Path, default=data / 'e-f-predictions' / _SCRIPT.parent.name)
    parser.add_argument('--isolated-atom-dir', type=Path, default=data / 'Hydrogen_E0')
    parser.add_argument('--raw-energies', action='store_true', help='Disable isolated-atom energy corrections.')
    parser.add_argument('--max-frames', type=int, help='Evaluate at most this many frames per trajectory.')
    parser.add_argument('--force', action='store_true', help='Recompute existing results.')
    parser.add_argument('--no-progress', action='store_true', help='Disable trajectory and frame progress bars.')
    parser.add_argument('--debug', action='store_true')
    _ARGS = parser.parse_args()
    if _ARGS.max_frames is not None and _ARGS.max_frames < 1:
        parser.error('--max-frames must be positive')


def reference(atoms):
    """Support explicit REF tags and ASE single-point reference calculators."""
    import numpy as np
    energy = atoms.info['REF_energy'] if 'REF_energy' in atoms.info else atoms.get_potential_energy()
    forces = atoms.arrays['REF_forces'] if 'REF_forces' in atoms.arrays else atoms.get_forces()
    return float(energy), np.asarray(forces, dtype=float)


def predict(model, atoms, device, dtype):
    import numpy as np
    import torch_sim as ts
    state = ts.initialize_state(atoms, device, dtype)
    # Energy-derived forces require autograd; do not wrap in no_grad/inference_mode.
    result = model(state)
    energy = result['energy'].detach().cpu().numpy().copy()
    forces = result['forces'].detach().cpu().numpy().copy()
    if energy.size != 1 or forces.shape != (len(atoms), 3):
        raise ValueError(f'Unexpected prediction shapes: {energy.shape}, {forces.shape}')
    if not np.isfinite(energy).all() or not np.isfinite(forces).all():
        raise ValueError('Non-finite model prediction')
    return float(energy.reshape(-1)[0]), forces


def correction_files(system, args):
    if args.raw_energies:
        return {}
    if system in ('anthracene', 'picene', 'naphthalene', 'pentacene', 'tetracene'):
        root = args.ref_dir / 'naphthalene_295K_Sharma_S'
        return {s: root / f'isolated_atom_{s}.extxyz' for s in ('C', 'H')}
    if system == 'H':
        return {'H': args.isolated_atom_dir / 'isolated_atom_H.extxyz'}
    return {}


def count_extxyz_frames(path, limit=None):
    """Count frames cheaply so the frame progress bar has a useful total."""
    try:
        count = 0
        with path.open() as handle:
            while limit is None or count < limit:
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                n_atoms = int(line)
                if not handle.readline():
                    raise ValueError('missing extended-XYZ comment line')
                for _ in range(n_atoms):
                    if not handle.readline():
                        raise ValueError('truncated extended-XYZ frame')
                count += 1
        return count
    except (OSError, ValueError):
        # ASE will provide the authoritative parse error during evaluation.
        return None


def run_rmse(model_name, model, engine='torchsim', *, state_dtype=None,
             require_stress_disabled=True, warmup=True, validate=None):
    """Evaluate the MD model on reference frames, without integrating dynamics.

    Extra keywords match the source MD entry points. Validation/warmup are not
    needed for untimed RMSE evaluation. Results are cloned before the next call
    because compiled models may reuse CUDA graph output buffers.
    """
    import numpy as np
    import torch
    from ase.io import iread, read
    from tqdm.auto import tqdm
    args = _ARGS
    dtype = state_dtype if state_dtype is not None else torch.float32
    device = torch.device('cuda')
    files = sorted(args.ref_dir.rglob('traj*.extxyz'))
    if not files:
        raise SystemExit(f'No traj*.extxyz files found under {args.ref_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f'rmse-results-all_{model_name}.csv'
    failure_file = output.with_suffix('.failures.json')
    if output.exists() and not failure_file.exists() and not args.force:
        print(f'{output} exists; use --force to recompute.')
        return
    rows, failures, offsets = [], [], {}
    progress_enabled = not args.no_progress
    progress_print = tqdm.write if progress_enabled else print
    file_iterator = tqdm(
        files,
        desc=f'{model_name}: trajectories',
        unit='trajectory',
        dynamic_ncols=True,
        disable=not progress_enabled,
    )
    for path in file_iterator:
        system = path.parent.name.split('_')[0]
        match = re.search(r'(\d+)K', path.parent.name)
        temperature = int(match[1]) if match else 0
        try:
            e0 = {}
            for symbol, atom_path in correction_files(system, args).items():
                key = str(atom_path.resolve())
                if key not in offsets:
                    atom = read(atom_path)
                    ref_e = float(atom.info['REF_energy']) if 'REF_energy' in atom.info else float(atom.get_potential_energy())
                    pred_e, _ = predict(model, atom, device, dtype)
                    offsets[key] = pred_e - ref_e
                e0[symbol] = offsets[key]
            e_squared = f_squared = 0.0
            n_eval = n_ref = n_components = 0
            natoms = None
            iterator = iread(path, index=':' if args.max_frames is None else f':{args.max_frames}')
            frame_iterator = tqdm(
                iterator,
                total=count_extxyz_frames(path, args.max_frames),
                desc=path.parent.name,
                unit='frame',
                leave=False,
                dynamic_ncols=True,
                disable=not progress_enabled,
            )
            for index, atoms in enumerate(frame_iterator):
                n_ref += 1
                try:
                    er, fr = reference(atoms)
                    ep, fp = predict(model, atoms, device, dtype)
                    if fr.shape != fp.shape or not np.isfinite(er) or not np.isfinite(fr).all():
                        raise ValueError('Invalid reference energy/forces')
                    offset = sum(e0.get(s, 0.0) for s in atoms.get_chemical_symbols())
                    e_squared += ((ep - er - offset) / len(atoms)) ** 2
                    f_squared += float(np.sum((fp - fr) ** 2))
                    n_components += fr.size
                    n_eval += 1
                    natoms = len(atoms) if natoms is None else natoms
                except Exception as exc:
                    failures.append({'file': str(path), 'frame': index, 'error': str(exc)})
                    if args.debug:
                        import traceback
                        traceback.print_exc()
            if not n_eval:
                raise ValueError('No successfully evaluated frames')
            rows.append(dict(system=system, temperature_K=temperature,
                             reference_key=f'{system}_{temperature}K', calculator=model_name,
                             natoms=natoms, n_reference_frames=n_ref, n_evaluated_frames=n_eval,
                             energy_rmse=np.sqrt(e_squared / n_eval),
                             force_rmse=np.sqrt(f_squared / n_components),
                             trajectory=str(path), engine=engine))
            progress_print(f'{path.parent.name}: E RMSE={rows[-1]["energy_rmse"]:.6g} eV/atom; '
                           f'F RMSE={rows[-1]["force_rmse"]:.6g} eV/Angstrom '
                           f'({n_eval}/{n_ref} frames)')
        except Exception as exc:
            failures.append({'file': str(path), 'error': str(exc)})
            progress_print(f'FAILED {path}: {exc}')
    # Keep a failure marker so an incomplete summary is never silently skipped.
    if failures:
        failure_file.write_text(json.dumps(failures, indent=2) + '\n')
    if rows:
        temporary = output.with_suffix('.csv.tmp')
        with temporary.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output)
        print(f'Saved {output}')
    if failures:
        raise SystemExit(f'Evaluation incomplete: {len(failures)} failures; see {failure_file}')
    failure_file.unlink(missing_ok=True)
