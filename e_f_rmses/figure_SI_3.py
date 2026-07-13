from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    'lines.markersize': 4,
    'lines.linewidth': 1.5,
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 8,
    'axes.grid': True,
    'grid.linewidth': 0.5,
    'grid.alpha': 1.0,
})

palette = sns.color_palette("deep")

CALCULATOR_DISPLAY_NAMES = {
    'chgnet': 'CHGNet',
    'mace-mp-0': 'MACE-MP-0',
    'grace-mp': 'GRACE-2L-MPtrj',
    'mace-mpa-0': 'MACE-MPA-0',
    'orb-v2': 'orb-v2',
    'eq-v2-m-omat': 'EquiformerV2',
    'mattersim-v1-5m': 'MatterSim-v1.0.0-5M',
    'orb-v3': 'orb-v3-conservative-inf-mpa',
    'grace-oam': 'GRACE-2L-OAM',
    'nequip': 'NequIP-OAM-XL',
    'pet-oam-xl': 'PET-OAM-XL',
    'esen-30m-oam': 'eSEN-30M-OAM',
    'mace-mh-omat': 'MACE-MH-1-OMAT',
    'uma-s-omat': 'UMA-S-P1',
    'uma-m-omat': 'UMA-M-P1',
}


def normalize_calculator_name(name):
    text = str(name).strip()
    return CALCULATOR_DISPLAY_NAMES.get(text.lower(), text)

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / 'results' / 'mean_metrics_by_system_type_and_model.csv'
OUTPUT_PDF = BASE_DIR / 'plots' / 'plot_SI_force_rmse_by_system_type.pdf'

SYSTEM_TYPE_ORDER = [
    'pure metals',
    'perovskites',
    'metal dichalcogenides',
    'metal alloys',
    'molecular crystals',
    'metal-water interfaces',
]

tier_1 = [normalize_calculator_name(model) for model in ["chgnet", "mace-mp-0", "grace-mp"]]
tier_2 = [normalize_calculator_name(model) for model in ["mace-mpa-0", "orb-v2"]]
tier_3 = [normalize_calculator_name(model) for model in ["mattersim-v1-5M", "grace-oam", "orb-v3", "eSEN-30M-OAM", "nequip", "eq-v2-M-omat", "pet-oam-xl"]]
tier_4 = [normalize_calculator_name(model) for model in ["mace-mh-omat", "uma-s-omat", "uma-m-omat"]]

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
        fontsize=8,
        fontweight="bold",
        color=color,
        zorder=10,
        annotation_clip=False,
        arrowprops=dict(
            arrowstyle="-",
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
    return y_max * 1.04


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f'Missing input CSV: {INPUT_CSV}')

    df = pd.read_csv(INPUT_CSV)

    required_columns = {'system_type', 'calculator', 'force_rmse'}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in {INPUT_CSV}: {sorted(missing)}')

    df = df.copy()
    df['system_type'] = df['system_type'].astype(str).str.strip().str.lower()
    df['calculator'] = df['calculator'].map(normalize_calculator_name)
    df['force_rmse'] = pd.to_numeric(df['force_rmse'], errors='coerce')
    df = df.dropna(subset=['force_rmse'])

    system_types_to_plot = [
        system_type
        for system_type in SYSTEM_TYPE_ORDER
        if not df[df['system_type'] == system_type].empty
    ]

    n_panels = len(system_types_to_plot)
    custom_layout_5 = (n_panels == 5)

    if custom_layout_5:
        fig = plt.figure(figsize=(3.53 * 2, 3.53 * 2))
        gs = fig.add_gridspec(2, 6)
        axes_flat = [
            fig.add_subplot(gs[0, 0:2]),
            fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[0, 4:6]),
            fig.add_subplot(gs[1, 1:3]),
            fig.add_subplot(gs[1, 3:5]),
        ]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(3.53 * 3, 3.53 * 2), squeeze=False)
        axes_flat = axes.flatten().tolist()

    for idx, system_type in enumerate(system_types_to_plot):
        ax = axes_flat[idx]
        subset = df[df['system_type'] == system_type].copy()
        ax.text(0.02, 0.99, f'({chr(97 + idx)})', transform=ax.transAxes, ha='left', va='top')

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

            tier_sub['_tier_order'] = pd.Categorical(tier_sub['calculator'], categories=tier_models, ordered=True)
            tier_sub = tier_sub.sort_values('_tier_order').drop(columns=['_tier_order']).reset_index(drop=True)

            best_idx = tier_sub['force_rmse'].idxmin()
            worst_idx = tier_sub['force_rmse'].idxmax()
            best_row = tier_sub.loc[best_idx]
            worst_row = tier_sub.loc[worst_idx]
            tier_median = float(tier_sub['force_rmse'].median())

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
            selected_df['force_rmse'],
            width=0.65,
            color=selected_colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5,
        )

        ax.set_title(system_type.title())
        ax.set_xlabel('Model')
        if custom_layout_5:
            show_ylabel = idx in (0, 3)
        else:
            show_ylabel = (idx % 2 == 0)
        ax.set_ylabel(r'Force RMSE [eV/$\AA$]' if show_ylabel else '')
        ax.set_xticks(x)
        ax.set_xticklabels(selected_labels, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y')
        y_max = float(selected_df['force_rmse'].max())
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
                fontsize=8,
                color=tier_color,
            )

    if not custom_layout_5:
        for idx in range(len(system_types_to_plot), len(axes_flat)):
            axes_flat[idx].axis('off')

    plt.tight_layout()
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PDF)
    plt.show()
    plt.close(fig)
    print(f'Saved {OUTPUT_PDF}')


if __name__ == '__main__':
    main()
