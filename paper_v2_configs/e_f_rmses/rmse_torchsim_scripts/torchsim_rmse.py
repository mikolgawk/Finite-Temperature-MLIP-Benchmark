"""Shared single-frame TorchSim RMSE evaluation; model setup lives in each runner."""
import argparse
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from md_success import torchsim_md_succeeded
from system_filters import add_molecular_crystal_option, include_system
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rmse_results import RmseResults

_ARGS = None
_SCRIPT = None


def early_cli(script):
    """Parse options before importing heavyweight model dependencies."""
    global _ARGS, _SCRIPT
    _SCRIPT = Path(script).resolve()
    benchmark_data = Path(__file__).resolve().parents[2] / 'data'
    results_data = Path(__file__).resolve().parent.parent / 'data'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref-dir', type=Path, default=benchmark_data / 'ref-trajs')
    md_source = (
        'mlip-trajs-torchsim-accelerated'
        if _SCRIPT.parent.name == 'md-accelerated'
        else 'mlip-trajs-torchsim-eager'
    )
    parser.add_argument('--md-dir', type=Path, default=benchmark_data / md_source,
                        help='matching TorchSim MD trajectories and timing records')
    parser.add_argument(
        '--output-dir', type=Path,
        default=results_data / 'e-f-predictions' / _SCRIPT.parent.name,
    )
    parser.add_argument('--isolated-atom-dir', type=Path, default=benchmark_data / 'Hydrogen_E0')
    parser.add_argument('--raw-energies', action='store_true', help='Disable isolated-atom energy corrections.')
    parser.add_argument('--max-frames', type=int, help='Evaluate at most this many frames per trajectory.')
    parser.add_argument('--force', action='store_true', help='Recompute existing results.')
    parser.add_argument('--no-progress', action='store_true', help='Disable trajectory and frame progress bars.')
    parser.add_argument('--debug', action='store_true')
    add_molecular_crystal_option(parser)
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


def isolated_atom_energy(calculator, atoms):
    """Evaluate only the native ASE calculator's isolated-atom energy."""
    import numpy as np
    atom = atoms.copy()
    atom.calc = calculator
    energy = float(atom.get_potential_energy())
    if not np.isfinite(energy):
        raise ValueError('Non-finite isolated-atom energy prediction')
    return energy


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
             require_stress_disabled=True, warmup=True, validate=None,
             skip_systems=(), make_isolated_atom_calculator=None):
    """Evaluate the MD model on reference frames, without integrating dynamics.

    Extra keywords match the source MD entry points. Validation/warmup are not
    needed for untimed RMSE evaluation. Results are cloned before the next call
    because compiled models may reuse CUDA graph output buffers.
    An optional native ASE calculator is created lazily for isolated-atom
    energies; trajectory predictions always use the supplied TorchSim model.
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
    files = [path for path in files
             if path.parent.name.split('_')[0] not in skip_systems
             and torchsim_md_succeeded(args.md_dir / path.parent.name / f'nvt_{model_name}.h5')]
    files = [path for path in files if include_system(
        path.parent.name, getattr(args, 'include_molecular_crystals', False)
    )]
    print(f'{model_name}: {len(files)} selected reference trajectories with completed TorchSim MD')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f'rmse-results-all_{model_name}.csv'
    results = RmseResults(
        output, files, force=args.force,
        exclude_trajectory=lambda path: not include_system(
            path, getattr(args, 'include_molecular_crystals', False)
        ),
    )
    files = results.files
    if not files:
        print(f'{output}: no pending eligible trajectories; use --force to recompute.')
        results.finish()
        return
    offsets = {}
    isolated_calculator = None
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
        row, failures = None, []
        system = path.parent.name.split('_')[0]
        match = re.search(r'(\d+)K', path.parent.name)
        temperature = int(match[1]) if match else 0
        try:
            e0 = {}
            for symbol, atom_path in correction_files(system, args).items():
                key = str(atom_path.resolve())
                if key not in offsets:
                    try:
                        atom = read(atom_path)
                        ref_e = float(atom.info['REF_energy']) if 'REF_energy' in atom.info else float(atom.get_potential_energy())
                        if not np.isfinite(ref_e):
                            raise ValueError('Non-finite isolated-atom reference energy')
                        if make_isolated_atom_calculator is None:
                            pred_e, _ = predict(model, atom, device, dtype)
                        else:
                            if isolated_calculator is None:
                                isolated_calculator = make_isolated_atom_calculator()
                            pred_e = isolated_atom_energy(isolated_calculator, atom)
                        offsets[key] = pred_e - ref_e
                    except Exception as exc:
                        raise RuntimeError(f'Isolated-atom correction failed for {atom_path}: {exc}') from exc
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
            row = dict(system=system, temperature_K=temperature,
                       reference_key=f'{system}_{temperature}K', calculator=model_name,
                       natoms=natoms, n_reference_frames=n_ref, n_evaluated_frames=n_eval,
                       energy_rmse=np.sqrt(e_squared / n_eval),
                       force_rmse=np.sqrt(f_squared / n_components),
                       trajectory=str(path), engine=engine)
            progress_print(f'{path.parent.name}: E RMSE={row["energy_rmse"]:.6g} eV/atom; '
                           f'F RMSE={row["force_rmse"]:.6g} eV/Angstrom '
                           f'({n_eval}/{n_ref} frames)')
        except Exception as exc:
            failures.append({'file': str(path), 'error': str(exc)})
            progress_print(f'FAILED {path}: {exc}')
        results.record(path, row, failures)
    results.finish()
