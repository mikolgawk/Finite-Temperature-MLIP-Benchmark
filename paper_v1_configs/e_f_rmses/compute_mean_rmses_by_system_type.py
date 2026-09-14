from pathlib import Path
import re

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / 'data'
RESULTS_DIR = Path(__file__).resolve().parent / 'results'
REQUIRED_COLUMNS = [
    'system',
    'energy_rmse',
    'force_rmse',
    'mlip_force_eval_time_per_call_per_atom_s',
]

SYSTEM_TYPES = [
    'pure metals',
    'perovskites',
    'metal dichalcogenides',
    'metal alloys',
    'molecular crystals',

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



def list_rmse_csv_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob('rmse-results-all_*.csv'))


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


def load_all_data(data_dir: Path) -> pd.DataFrame:
    csv_files = list_rmse_csv_files(data_dir)
    if not csv_files:
        raise FileNotFoundError(f'No rmse-results-all_*.csv files found in {data_dir}')

    frames = []
    for csv_file in csv_files:
        frame = pd.read_csv(csv_file)
        model_name = extract_model_name(csv_file)
        frame['calculator'] = model_name
        frames.append(frame)

    all_data = pd.concat(frames, ignore_index=True)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in all_data.columns]
    if missing_cols:
        raise ValueError(f'Missing required columns: {missing_cols}')

    all_data['system_type'] = all_data['system'].astype(str).map(infer_system_type)
    return all_data


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_data = load_all_data(DATA_DIR)

    metrics = ['energy_rmse', 'force_rmse', 'mlip_force_eval_time_per_call_per_atom_s']

    mapped_data = all_data[all_data['system_type'].notna()].copy()

    means_by_system_type = (
        mapped_data.groupby('system_type', as_index=False)[metrics]
        .mean(numeric_only=True)
        .rename(columns={
            'mlip_force_eval_time_per_call_per_atom_s': 'mean_force_eval_time_per_atom_s'
        })
    )
    means_by_system_type = means_by_system_type.set_index('system_type').reindex(SYSTEM_TYPES).reset_index()

    means_by_system_type_and_model = (
        mapped_data.groupby(['system_type', 'calculator'], as_index=False)[metrics]
        .mean(numeric_only=True)
        .rename(columns={
            'mlip_force_eval_time_per_call_per_atom_s': 'mean_force_eval_time_per_atom_s'
        })
        .sort_values(['system_type', 'calculator'], key=lambda c: c.map({t: i for i, t in enumerate(SYSTEM_TYPES)}) if c.name == 'system_type' else c)
    )

    overall_output = RESULTS_DIR / 'mean_metrics_by_system_type.csv'
    by_model_output = RESULTS_DIR / 'mean_metrics_by_system_type_and_model.csv'

    means_by_system_type.to_csv(overall_output, index=False)
    means_by_system_type_and_model.to_csv(by_model_output, index=False)

    print(f'Saved {overall_output}')
    print(f'Saved {by_model_output}')
    print('Mean metrics by system type:')
    print(means_by_system_type.to_string(index=False))


if __name__ == '__main__':
    main()
