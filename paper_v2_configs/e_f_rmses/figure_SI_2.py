from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys

PAPER_V2_CONFIG_DIR = Path(__file__).resolve().parents[1]
if str(PAPER_V2_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_V2_CONFIG_DIR))
from model_display_names import MODEL_DISPLAY_NAMES

FONT_SIZE = 12
PANEL_HEIGHT = 3.53 * 1.2

plt.rcParams.update({
    'lines.markersize': 4,
    'lines.linewidth': 1.5,
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
    'figure.titlesize': FONT_SIZE,
    'axes.grid': True,
    'grid.linewidth': 0.5,
    'grid.alpha': 1.0,
})

palette = sns.color_palette("deep")

CALCULATOR_DISPLAY_NAMES = MODEL_DISPLAY_NAMES


def normalize_calculator_name(name):
    text = str(name).strip()
    key = text.lower().replace("-force-only", "").replace("-stress", "")
    if key.endswith("-eager"):
        key = key.removesuffix("-eager")
    key = {"nequip-oam-l": "nequip"}.get(key, key)
    return CALCULATOR_DISPLAY_NAMES.get(key, text)

BASE_DIR = Path(__file__).resolve().parent
from _plot_inputs import plot_args, read_metric_csv
RESULTS_DIR, PLOTS_DIR = plot_args()
INPUT_CSV = RESULTS_DIR / 'mean_metrics_by_system_type_and_model.csv'
OUTPUT_PDF = PLOTS_DIR / 'figure_SI_1.pdf'

SYSTEM_TYPE_ORDER = [
    'pure metals',
    'perovskites',
    'metal dichalcogenides',
    'metal alloys',
    'molecular crystals',
    'metal-water interfaces',
    'hydrogen'
]

tier_1 = [normalize_calculator_name(model) for model in ["chgnet", "mace-mp-0", "mace-mp-0-compile", "grace-mp"]]
tier_2 = [normalize_calculator_name(model) for model in ["mace-mpa-0", "mace-mpa-0-compile", "orb-v2"]]
tier_3 = [normalize_calculator_name(model) for model in ["mattersim-v1-5M", "grace-oam", "orb-v3-omat", "orb-v3-direct-omat", "eSEN-30M-OAM", "nequip", "eq-v2-M-omat", "pet-oam-xl", "pet-omat-xl", "grace-oam-compiled", "mattersim-v1-5M-compile", "pet-oam-xl-torchscript", "pet-omat-xl-torchscript"]]
tier_4 = [normalize_calculator_name(model) for model in ["mace-mh-omat", "mace-mh-omat-compile", "uma-s-omat", "uma-s-omat-compile", "uma-s-omat-turbo", "uma-m-omat", "uma-m-omat-compile", "uma-m-omat-turbo"]]

tier_defs = [
    ("Tier 1", tier_1, palette[2]),
    ("Tier 2", tier_2, palette[1]),
    ("Tier 3", tier_3, palette[3]),
    ("Tier 4", tier_4, palette[0]),
]

def annotate_median(ax, x_center, y_value, y_text, fmt="{:.3f}", color="black"):
    ax.annotate(
        fmt.format(y_value),
        xy=(x_center, y_value),
        xytext=(x_center, y_text),
        textcoords="data",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE,
        fontweight="bold",
        color=color,
        zorder=10,
        annotation_clip=False,
        arrowprops=dict(
            arrowstyle="-",
            connectionstyle="arc3,rad=0",
            relpos=(0.5, 0.0),
            color=color,
            linewidth=0.7,
            alpha=0.8,
            shrinkA=0,
            shrinkB=0,
        ),
    )


def tier_center(start_idx, end_idx):
    return (start_idx + end_idx) / 2


def tier_label_y(y_max):
    return y_max * 1.12


def median_value_label_y(y_max):
    # Leave more space below the 12-point tier headings, above the tallest bar.
    return y_max * 1.01


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f'Missing input CSV: {INPUT_CSV}')

    df = read_metric_csv(INPUT_CSV)

    required_columns = {'system_type', 'calculator', 'energy_rmse'}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in {INPUT_CSV}: {sorted(missing)}')

    df = df.copy()
    df['system_type'] = df['system_type'].astype(str).str.strip().str.lower()
    df['calculator'] = df['calculator'].map(normalize_calculator_name)
    df['energy_rmse'] = pd.to_numeric(df['energy_rmse'], errors='coerce')
    df = df.dropna(subset=['energy_rmse'])

    system_types_to_plot = [
        system_type
        for system_type in SYSTEM_TYPE_ORDER
        if not df[df['system_type'] == system_type].empty
    ]

    n_panels = len(system_types_to_plot)
    if not n_panels:
        raise ValueError(f'No system types with finite RMSE data in {INPUT_CSV}')
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig = plt.figure(figsize=(3.53 * ncols, PANEL_HEIGHT * nrows), layout='constrained')
    gs = fig.add_gridspec(nrows, ncols * 2)
    axes_flat = []
    for idx in range(n_panels):
        panels_in_row = min(ncols, n_panels - (idx // ncols) * ncols)
        start_col = ncols - panels_in_row + (idx % ncols) * 2
        axes_flat.append(fig.add_subplot(gs[idx // ncols, start_col:start_col + 2]))

    for idx, system_type in enumerate(system_types_to_plot):
        ax = axes_flat[idx]
        subset = df[df['system_type'] == system_type].copy()
        # Keep panel labels outside the axes, clear of the centered system titles.
        ax.annotate(
            f'({chr(97 + idx)})',
            xy=(0, 1),
            xycoords='axes fraction',
            xytext=(-6, 6),
            textcoords='offset points',
            ha='right',
            va='bottom',
            fontsize=FONT_SIZE,
            annotation_clip=False,
        )

        selected_rows: list[pd.Series] = []
        selected_labels: list[str] = []
        selected_colors: list[tuple[float, float, float]] = []
        tier_medians: list[tuple[int, int, float, tuple[float, float, float]]] = []
        tier_label_ranges: list[tuple[str, int, int, tuple[float, float, float]]] = []
        current_pos = 0

        for tier_name, tier_models, tier_color in tier_defs:
            tier_sub = subset[subset['calculator'].isin(tier_models)].copy()
            if tier_sub.empty:
                continue

            tier_sub['_tier_order'] = pd.Categorical(tier_sub['calculator'], categories=list(dict.fromkeys(tier_models)), ordered=True)
            tier_sub = tier_sub.sort_values('_tier_order').drop(columns=['_tier_order']).reset_index(drop=True)

            best_idx = tier_sub['energy_rmse'].idxmin()
            worst_idx = tier_sub['energy_rmse'].idxmax()
            best_row = tier_sub.loc[best_idx]
            worst_row = tier_sub.loc[worst_idx]
            tier_median = float(tier_sub['energy_rmse'].median())

            selected_rows.extend([best_row, worst_row])
            selected_labels.extend([best_row['calculator'], worst_row['calculator']])
            selected_colors.extend([tier_color, tier_color])

            tier_start = current_pos
            tier_end = current_pos + 1
            tier_medians.append((tier_start, tier_end, tier_median, tier_color))
            tier_label_ranges.append((tier_name, tier_start, tier_end, tier_color))
            current_pos += 2

        if not selected_rows:
            ax.text(0.5, 0.5, f'No tier data for {system_type}', ha='center', va='center')
            ax.axis('off')
            continue

        selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)
        x = np.arange(len(selected_df))

        ax.bar(
            x,
            selected_df['energy_rmse'],
            width=0.65,
            color=selected_colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5,
        )

        ax.set_title(system_type.title())
        ax.set_xlabel('Model')
        show_ylabel = (idx % ncols == 0)
        ax.set_ylabel('Energy RMSE [eV/atom]' if show_ylabel else '')
        ax.set_xticks(x)
        ax.set_xticklabels(selected_labels, rotation=45, ha='right', fontsize=FONT_SIZE)
        ax.grid(axis='y')
        y_max = float(selected_df['energy_rmse'].max())
        ax.set_ylim(0, y_max * 1.25)

        for start_idx, end_idx, median_val, tier_color in tier_medians:
            ax.hlines(
                y=median_val,
                xmin=start_idx - 0.35,
                xmax=end_idx + 0.35,
                colors=tier_color,
                linestyles='--',
                linewidth=2,
                zorder=3,
            )

        for start_idx, end_idx, median_val, tier_color in tier_medians:
            annotate_median(
                ax,
                tier_center(start_idx, end_idx),
                median_val,
                median_value_label_y(y_max),
                color=tier_color,
            )

        if len(selected_df) >= 4:
            for vline_x in np.arange(1.5, len(selected_df) - 0.5, 2):
                ax.axvline(x=vline_x, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

        for tier_name, start_idx, end_idx, tier_color in tier_label_ranges:
            ax.text(
                tier_center(start_idx, end_idx),
                tier_label_y(y_max),
                tier_name,
                ha='center',
                fontsize=FONT_SIZE,
                color=tier_color,
            )

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)
    print(f'Saved {OUTPUT_PDF}')


if __name__ == '__main__':
    main()
