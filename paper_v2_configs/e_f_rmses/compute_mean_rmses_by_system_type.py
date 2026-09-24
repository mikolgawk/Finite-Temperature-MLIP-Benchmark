import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from md_success import torchsim_md_succeeded
from metric_sources import SOURCES


DATA_DIR = Path(__file__).resolve().parent / 'data' / 'e-f-predictions'
RESULTS_DIR = Path(__file__).resolve().parent / 'results'
REQUIRED_COLUMNS = [
    'system',
    'energy_rmse',
    'force_rmse',
]

SYSTEM_TYPES = [
    'pure metals',
    'perovskites',
    'metal dichalcogenides',
    'metal alloys',
    'molecular crystals',
    'metal-water interfaces',
    'hydrogen',
]

def infer_system_type(system: str) -> str:
    s = system.lower()

    if (
        s.startswith("bulkcuau")
        or s.startswith("bulkcuzral")
        or s.startswith("bulklimgalznsn")
        or s.startswith("bulkpt3co")
    ):
        return "metal alloys"

    if s.startswith("bulkau") or s.startswith("bulkag") or s.startswith("bulkcu"):
        return "pure metals"
    if s.startswith("cssni3") or s.startswith("mapbbr3"):
        return "perovskites"
    if s.startswith("bulkmos2") or s.startswith("tise2"):
        return "metal dichalcogenides"
    if (
        s.startswith("anthracene")
        or s.startswith("naphthalene")
        or s.startswith("pentacene")
        or s.startswith("picene")
        or s.startswith("tetracene")
    ):
        return "molecular crystals"
    if s.startswith("pt111w24h2o"):
        return "metal-water interfaces"
    if s.startswith("h"):
        return "hydrogen"


def list_rmse_csv_files(data_dir: Path) -> list[Path]:
    csv_files = sorted(data_dir.rglob('rmse-results-all_*.csv'))
    if data_dir.name != 'e-f-predictions':
        return csv_files

    # Older TorchSim runners wrote directly under data/md and data/md-accelerated.
    # Prefer the current summary for each source/model to avoid counting it twice.
    for legacy_name, current_name in (('md', 'md_eager'), ('md-accelerated', 'md-accelerated')):
        for legacy_file in sorted((data_dir.parent / legacy_name).glob('rmse-results-all_*.csv')):
            if not (data_dir / current_name / legacy_file.name).is_file():
                csv_files.append(legacy_file)
    csv_files.extend(sorted((data_dir.parent / 'e-f-predictions-ase').rglob('rmse-results-all_*.csv')))
    return sorted(csv_files)


def extract_model_name(csv_path: Path) -> str:
    stem = csv_path.stem
    if stem.startswith('rmse-results-all_'):
        return stem.replace('rmse-results-all_', '', 1)
    return stem


def canonical_system_key(system_id: str) -> str:
    parts = str(system_id).split('_')
    system_name = parts[0] if parts else str(system_id)
    temperature_match = re.search(r'\d+K', str(system_id))

    if temperature_match:
        return f'{system_name}_{temperature_match.group(0)}'
    return system_name


def csv_source(csv_file: Path) -> str:
    for part in csv_file.parts:
        if part in SOURCES:
            return part
    backend = 'ase' if 'e-f-predictions-ase' in csv_file.parts else 'torchsim'
    accelerated = csv_file.parent.name in {'md-accelerated', 'md_accelerated'}
    return f'mlip-trajs-{backend}' + ('-accelerated' if accelerated else ('-eager' if backend == 'torchsim' else ''))


def load_all_data(
    data_dir: Path,
    models: set[str] | None = None,
    excluded_system_types: set[str] | None = None,
    md_data_dir: Path | None = None,
    sources: set[str] | None = None,
) -> pd.DataFrame:
    csv_files = list_rmse_csv_files(data_dir)
    if sources is not None:
        csv_files = [p for p in csv_files if csv_source(p) in sources]
    if models is not None:
        csv_files = [
            csv_file
            for csv_file in csv_files
            if extract_model_name(csv_file) in models
        ]
    if not csv_files:
        requested = (
            f" for requested model(s): {', '.join(sorted(models))}"
            if models
            else ''
        )
        raise FileNotFoundError(
            f'No rmse-results-all_*.csv files found in {data_dir}{requested}'
        )

    md_data_dir = md_data_dir or Path(__file__).resolve().parent.parent / 'data'
    frames = []
    for csv_file in csv_files:
        frame = pd.read_csv(csv_file)
        model_name = extract_model_name(csv_file)
        source = csv_source(csv_file)
        if 'torchsim' in source and csv_file.parent.name in {'md', 'md_eager', 'md-accelerated', 'md_accelerated', *SOURCES}:
            if 'trajectory' not in frame.columns:
                raise ValueError(f'Missing trajectory column in {csv_file}')
            frame = frame.loc[frame['trajectory'].map(
                lambda ref: torchsim_md_succeeded(
                    md_data_dir / source / Path(ref).parent.name / f'nvt_{model_name}.h5'
                ) if isinstance(ref, str) else False
            )].copy()
        frame['source'] = source
        frame['calculator'] = model_name
        frames.append(frame)

    all_data = pd.concat(frames, ignore_index=True)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in all_data.columns]
    if missing_cols:
        raise ValueError(f'Missing required columns: {missing_cols}')

    all_data['system_type'] = all_data['system'].astype(str).map(infer_system_type)
    if excluded_system_types:
        excluded = {system_type.strip().lower() for system_type in excluded_system_types}
        all_data = all_data[
            ~all_data['system_type'].fillna('').str.lower().isin(excluded)
        ].copy()
    return all_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Aggregate energy and force RMSEs by system type.'
    )
    parser.add_argument(
        '--model',
        action='append',
        dest='models',
        help='process only this model (repeatable; default: all discovered models)',
    )
    parser.add_argument(
        '--exclude-system-type',
        action='append',
        dest='excluded_system_types',
        help='exclude this system type from aggregation (repeatable)',
    )
    parser.add_argument('--source', action='append', choices=SOURCES, dest='sources')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def write_source_results(all_data: pd.DataFrame, results_dir: Path, excluded_system_types: set[str]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    all_data.to_csv(results_dir / 'rmse_per_system.csv', index=False)
    all_data.groupby('calculator', as_index=False)[['energy_rmse', 'force_rmse']].mean().to_csv(
        results_dir / 'mean_metrics_by_model.csv', index=False)
    metrics = ['energy_rmse', 'force_rmse']

    mapped_data = all_data[all_data['system_type'].notna()].copy()

    means_by_system_type = (
        mapped_data.groupby('system_type', as_index=False)[metrics]
        .mean(numeric_only=True)
    )
    excluded_normalized = {
        value.strip().lower() for value in excluded_system_types
    }
    included_system_types = [
        system_type for system_type in SYSTEM_TYPES
        if system_type.lower() not in excluded_normalized
    ]
    means_by_system_type = means_by_system_type.set_index('system_type').reindex(included_system_types).reset_index()

    means_by_system_type_and_model = (
        mapped_data.groupby(['system_type', 'calculator'], as_index=False)[metrics]
        .mean(numeric_only=True)
        .sort_values(['system_type', 'calculator'], key=lambda c: c.map({t: i for i, t in enumerate(SYSTEM_TYPES)}) if c.name == 'system_type' else c)
    )

    overall_output = results_dir / 'mean_metrics_by_system_type.csv'
    by_model_output = results_dir / 'mean_metrics_by_system_type_and_model.csv'

    means_by_system_type.to_csv(overall_output, index=False)
    means_by_system_type_and_model.to_csv(by_model_output, index=False)

    print(f'Saved {overall_output}')
    print(f'Saved {by_model_output}')
    print('Mean metrics by system type:')
    print(means_by_system_type.to_string(index=False))


def main() -> None:
    args = parse_args()
    excluded = set(args.excluded_system_types or ())
    all_data = load_all_data(args.data_dir, set(args.models) if args.models else None,
                             excluded, sources=set(args.sources) if args.sources else None)
    if all_data.empty:
        raise SystemExit('No RMSE rows remain after source, model, MD-completion and system-type filtering.')
    for source, frame in all_data.groupby('source'):
        write_source_results(frame, args.results_dir.resolve() / source, excluded)


if __name__ == '__main__':
    main()
