import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

import sys

PAPER_V2_CONFIG_DIR = Path(__file__).resolve().parents[1]
if str(PAPER_V2_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_V2_CONFIG_DIR))
from model_display_names import MODEL_DISPLAY_NAMES

FONT_SIZE = 8

BASE_DIR = Path(__file__).resolve().parent

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

CALCULATOR_RAW_NAMES = {display_name: raw_name for raw_name, display_name in CALCULATOR_DISPLAY_NAMES.items()}


def normalize_calculator_name(name):
    text = str(name).strip()
    key = text.lower().replace("-force-only", "").replace("-stress", "")
    if key.endswith("-eager"):
        key = key.removesuffix("-eager")
    key = {"nequip-oam-l": "nequip"}.get(key, key)
    return CALCULATOR_DISPLAY_NAMES.get(key, text)


def is_molecular_crystal(system):
    system_name = str(system).lower()
    return (
        system_name.startswith("anthracene")
        or system_name.startswith("naphthalene")
        or system_name.startswith("pentacene")
        or system_name.startswith("picene")
        or system_name.startswith("tetracene")
    )

# Read and aggregate data from per-model RMSE CSV files
from _plot_inputs import plot_args
results_dir, plots_dir = plot_args()
all_data = pd.read_csv(results_dir / 'rmse_per_system.csv')
all_data['calculator'] = all_data['calculator'].map(normalize_calculator_name)

if 'system' not in all_data.columns:
    raise ValueError("Missing required column: system")

before_filter_count = len(all_data)
all_data = all_data[~all_data['system'].map(is_molecular_crystal)].copy()
excluded_count = before_filter_count - len(all_data)

df = (
    all_data.groupby('calculator', as_index=False)[
        ['energy_rmse', 'force_rmse']
    ]
    .mean(numeric_only=True)
)

mean_summary = df[[
    'calculator',
    'energy_rmse',
    'force_rmse',
]]
mean_summary['calculator'] = mean_summary['calculator'].map(lambda name: CALCULATOR_RAW_NAMES.get(name, name))
results_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
mean_summary.to_csv(results_dir / 'mean_metrics_by_model_no_molecular_crystals.csv', index=False)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(3.53 * 2, 3.53))


tier_1 = [normalize_calculator_name(model) for model in ["chgnet", "mace-mp-0", "mace-mp-0-compile", "grace-mp"]]
tier_2 = [normalize_calculator_name(model) for model in ["mace-mpa-0", "mace-mpa-0-compile", "orb-v2"]]
tier_3 = [normalize_calculator_name(model) for model in ["mattersim-v1-5M", "grace-oam", "orb-v3", "orb-v3-direct", "eSEN-30M-OAM", "nequip", "eq-v2-M-omat", "pet-oam-xl", "pet-omat-xl", "grace-oam-compiled", "mattersim-v1-5M-compile", "pet-oam-xl-torchscript", "pet-omat-xl-torchscript"]]
tier_4 = [normalize_calculator_name(model) for model in ["mace-mh-omat", "mace-mh-omat-compile", "uma-s-omat", "uma-s-omat-compile", "uma-s-omat-turbo", "uma-m-omat", "uma-m-omat-compile", "uma-m-omat-turbo"]]



# Reorder dataframe by tiers (include tier 4)
tier_order = tier_1 + tier_2 + tier_3 + tier_4
df['tier_order'] = df['calculator'].apply(lambda x: tier_order.index(x) if x in tier_order else len(tier_order))
df = df.sort_values('tier_order')

# Assign colors based on tier
tier_colors = {
    'tier_1': palette[2],
    'tier_2': palette[1],
    'tier_3': palette[3],
    'tier_4': palette[0]
}

def get_tier_colors(models):
    colors = []
    for model in models:
        if model in tier_1:
            colors.append(tier_colors['tier_1'])
        elif model in tier_2:
            colors.append(tier_colors['tier_2'])
        elif model in tier_3:
            colors.append(tier_colors['tier_3'])
        elif model in tier_4:
            colors.append(tier_colors['tier_4'])
        else:
            colors.append('#757575')  # Gray for unclassified
    return colors

def get_tier_counts(models):
    t1_count = sum(model in tier_1 for model in models)
    t2_count = sum(model in tier_2 for model in models)
    t3_count = sum(model in tier_3 for model in models)
    t4_count = sum(model in tier_4 for model in models)
    return t1_count, t2_count, t3_count, t4_count


def catalog_positions(models, catalog):
    """Place available models at their positions in the complete tier catalog."""
    next_other_position = len(catalog)
    positions = []
    for model in models:
        if model in catalog:
            positions.append(catalog.index(model))
        else:
            positions.append(next_other_position)
            next_other_position += 1
    return np.asarray(positions, dtype=float), next_other_position

def annotate_median(ax, x_center, y_value, y_text, color, fmt="{:.3f}"):
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
    return y_max * 1.17


def median_value_label_y(y_max):
    return y_max * 1.08

# Separate data for each subplot
energy_df = df[df['calculator'] != 'EquiformerV2'].copy()
force_df = df.copy()

energy_models = energy_df['calculator'].values
force_models = force_df['calculator'].values

energy_catalog = [model for model in tier_order if model != 'EquiformerV2']
force_catalog = tier_order
energy_x_pos, energy_axis_size = catalog_positions(energy_models, energy_catalog)
force_x_pos, force_axis_size = catalog_positions(force_models, force_catalog)

energy_colors = get_tier_colors(energy_models)
force_colors = get_tier_colors(force_models)

energy_t1_count = len(tier_1)
energy_t2_count = len(tier_2)
energy_t3_count = len([model for model in tier_3 if model != 'EquiformerV2'])
energy_t4_count = len(tier_4)
force_t1_count = len(tier_1)
force_t2_count = len(tier_2)
force_t3_count = len(tier_3)
force_t4_count = len(tier_4)

# Plot 1: Energy RMSE
axes[0].bar(energy_x_pos, energy_df['energy_rmse'], color=energy_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('Model')
axes[0].set_ylabel('Energy RMSE [eV/atom]')
# axes[0].set_title('Energy RMSE by Model (Grouped by Tier)', fontsize=13, fontweight='bold')
axes[0].set_xticks(energy_x_pos)
axes[0].set_xticklabels(energy_models, rotation=45, ha='right', fontsize=FONT_SIZE)
axes[0].set_xlim(-0.5, energy_axis_size - 0.5)
axes[0].grid(axis='y')
energy_ymax = float(energy_df['energy_rmse'].max())
axes[0].set_ylim(0, energy_ymax * 1.25)

# Add vertical separators between tiers
energy_tier1_end = energy_t1_count - 0.5
energy_tier2_end = energy_t1_count + energy_t2_count - 0.5
energy_tier3_end = energy_t1_count + energy_t2_count + energy_t3_count - 0.5
axes[0].axvline(x=energy_tier1_end, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0].axvline(x=energy_tier2_end, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0].axvline(x=energy_tier3_end, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

# Draw median (dashed) lines for each tier on energy plot
t1_energy_values = energy_df[energy_df['calculator'].isin(tier_1)]['energy_rmse']
t2_energy_values = energy_df[energy_df['calculator'].isin(tier_2)]['energy_rmse']
t3_energy_values = energy_df[energy_df['calculator'].isin(tier_3)]['energy_rmse']
t4_energy_values = energy_df[energy_df['calculator'].isin(tier_4)]['energy_rmse']
t1_med_energy = t1_energy_values.median()
t2_med_energy = t2_energy_values.median()
t3_med_energy = t3_energy_values.median()
t4_med_energy = t4_energy_values.median()

axes[0].hlines(t1_med_energy, xmin=-0.5, xmax=energy_tier1_end, colors=tier_colors['tier_1'], linestyles='--', linewidth=2, alpha=0.9, zorder=5)
axes[0].hlines(t2_med_energy, xmin=energy_tier1_end, xmax=energy_tier2_end, colors=tier_colors['tier_2'], linestyles='--', linewidth=2, alpha=0.9, zorder=5)
axes[0].hlines(t3_med_energy, xmin=energy_tier2_end, xmax=energy_tier3_end, colors=tier_colors['tier_3'], linestyles='--', linewidth=2, alpha=0.9, zorder=5)
axes[0].hlines(t4_med_energy, xmin=energy_tier3_end, xmax=len(energy_catalog)-0.5, colors=tier_colors['tier_4'], linestyles='--', linewidth=2, alpha=0.9, zorder=5)

annotate_median(
    axes[0],
    tier_center(0, energy_t1_count - 1),
    t1_med_energy,
    median_value_label_y(energy_ymax),
    tier_colors['tier_1'],
)
annotate_median(
    axes[0],
    tier_center(energy_t1_count, energy_t1_count + energy_t2_count - 1),
    t2_med_energy,
    median_value_label_y(energy_ymax),
    tier_colors['tier_2'],
)
annotate_median(
    axes[0],
    tier_center(
        energy_t1_count + energy_t2_count,
        energy_t1_count + energy_t2_count + energy_t3_count - 1,
    ),
    t3_med_energy,
    median_value_label_y(energy_ymax),
    tier_colors['tier_3'],
)
annotate_median(
    axes[0],
    tier_center(
        energy_t1_count + energy_t2_count + energy_t3_count,
        len(energy_catalog) - 1,
    ),
    t4_med_energy,
    median_value_label_y(energy_ymax),
    tier_colors['tier_4'],
)

# Add tier labels
axes[0].text(tier_center(0, energy_t1_count - 1), tier_label_y(energy_ymax), 'Tier 1',
             ha='center', fontsize=FONT_SIZE, color=tier_colors['tier_1'])
axes[0].text(tier_center(energy_t1_count, energy_t1_count + energy_t2_count - 1), tier_label_y(energy_ymax), 'Tier 2',
             ha='center', fontsize=FONT_SIZE, color=tier_colors['tier_2'])
axes[0].text(tier_center(energy_t1_count + energy_t2_count, energy_t1_count + energy_t2_count + energy_t3_count - 1), tier_label_y(energy_ymax), 'Tier 3',
             ha='center', fontsize=FONT_SIZE, color=tier_colors['tier_3'])
axes[0].text(tier_center(energy_t1_count + energy_t2_count + energy_t3_count, len(energy_catalog) - 1), tier_label_y(energy_ymax), 'Tier 4',
             ha='center', fontsize=FONT_SIZE, color=tier_colors['tier_4'])

# Plot 2: Force RMSE
axes[1].bar(force_x_pos, force_df['force_rmse'], color=force_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
axes[1].set_xlabel('Model')
axes[1].set_ylabel('Force RMSE [eV/Å]')
# axes[1].set_title('Force RMSE by Model (Grouped by Tier)', fontsize=13, fontweight='bold')
axes[1].set_xticks(force_x_pos)
axes[1].set_xticklabels(force_models, rotation=45, ha='right', fontsize=FONT_SIZE)
axes[1].set_xlim(-0.5, force_axis_size - 0.5)
axes[1].grid(axis='y')
force_ymax = float(force_df['force_rmse'].max())
axes[1].set_ylim(0, force_ymax * 1.25)

# Add vertical separators between tiers
force_tier1_end = force_t1_count - 0.5
force_tier2_end = force_t1_count + force_t2_count - 0.5
force_tier3_end = force_t1_count + force_t2_count + force_t3_count - 0.5
axes[1].axvline(x=force_tier1_end, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].axvline(x=force_tier2_end, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].axvline(x=force_tier3_end, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

# Draw median (dashed) lines for each tier on force plot
t1_force_values = force_df[force_df['calculator'].isin(tier_1)]['force_rmse']
t2_force_values = force_df[force_df['calculator'].isin(tier_2)]['force_rmse']
t3_force_values = force_df[force_df['calculator'].isin(tier_3)]['force_rmse']
t4_force_values = force_df[force_df['calculator'].isin(tier_4)]['force_rmse']
t1_med_force = t1_force_values.median()
t2_med_force = t2_force_values.median()
t3_med_force = t3_force_values.median()
t4_med_force = t4_force_values.median()

axes[1].hlines(t1_med_force, xmin=-0.5, xmax=force_tier1_end, colors=tier_colors['tier_1'], linestyles='--', linewidth=2, alpha=0.9, zorder=5)
axes[1].hlines(t2_med_force, xmin=force_tier1_end, xmax=force_tier2_end, colors=tier_colors['tier_2'], linestyles='--', linewidth=2, alpha=0.9, zorder=5)
axes[1].hlines(t3_med_force, xmin=force_tier2_end, xmax=force_tier3_end, colors=tier_colors['tier_3'], linestyles='--', linewidth=2, alpha=0.9, zorder=5)
axes[1].hlines(t4_med_force, xmin=force_tier3_end, xmax=len(force_catalog)-0.5, colors=tier_colors['tier_4'], linestyles='--', linewidth=2, alpha=0.9, zorder=5)

annotate_median(
    axes[1],
    tier_center(0, force_t1_count - 1),
    t1_med_force,
    median_value_label_y(force_ymax),
    tier_colors['tier_1'],
)
annotate_median(
    axes[1],
    tier_center(force_t1_count, force_t1_count + force_t2_count - 1),
    t2_med_force,
    median_value_label_y(force_ymax),
    tier_colors['tier_2'],
)
annotate_median(
    axes[1],
    tier_center(
        force_t1_count + force_t2_count,
        force_t1_count + force_t2_count + force_t3_count - 1,
    ),
    t3_med_force,
    median_value_label_y(force_ymax),
    tier_colors['tier_3'],
)
annotate_median(
    axes[1],
    tier_center(
        force_t1_count + force_t2_count + force_t3_count,
        len(force_catalog) - 1,
    ),
    t4_med_force,
    median_value_label_y(force_ymax),
    tier_colors['tier_4'],
)

# Add tier labels
axes[1].text(tier_center(0, force_t1_count - 1), tier_label_y(force_ymax), 'Tier 1',
             ha='center', fontsize=FONT_SIZE, color=tier_colors['tier_1'])
axes[1].text(tier_center(force_t1_count, force_t1_count + force_t2_count - 1), tier_label_y(force_ymax), 'Tier 2',
             ha='center', fontsize=FONT_SIZE, color=tier_colors['tier_2'])
axes[1].text(tier_center(force_t1_count + force_t2_count, force_t1_count + force_t2_count + force_t3_count - 1), tier_label_y(force_ymax), 'Tier 3',
             ha='center', fontsize=FONT_SIZE, color=tier_colors['tier_3'])
axes[1].text(tier_center(force_t1_count + force_t2_count + force_t3_count, len(force_catalog) - 1), tier_label_y(force_ymax), 'Tier 4',
             ha='center', fontsize=FONT_SIZE, color=tier_colors['tier_4'])

# Subplot labels
axes[0].text(0.02, 0.96, '(a)', transform=axes[0].transAxes, ha='left', va='top', fontsize=FONT_SIZE)
axes[1].text(0.02, 0.96, '(b)', transform=axes[1].transAxes, ha='left', va='top', fontsize=FONT_SIZE)

plt.tight_layout()
plot_path = plots_dir / 'plot_e_f_rmses_no_molecular_crystals.pdf'
plt.savefig(plot_path)
print(f"Plot saved as {plot_path}")
print(
    "Saved mean_metrics_by_model_no_molecular_crystals.csv with per-model means for "
    "energy and force RMSE"
)
print(f"Excluded {excluded_count} molecular-crystal rows before averaging")
print(
    "Energy RMSE medians -> "
    + ", ".join(
        f"Tier {index}: {value:.8f}" if np.isfinite(value) else f"Tier {index}: N/A"
        for index, value in enumerate(
            (t1_med_energy, t2_med_energy, t3_med_energy, t4_med_energy), start=1
        )
    )
)
print(
    "Force RMSE medians -> "
    + ", ".join(
        f"Tier {index}: {value:.8f}" if np.isfinite(value) else f"Tier {index}: N/A"
        for index, value in enumerate(
            (t1_med_force, t2_med_force, t3_med_force, t4_med_force), start=1
        )
    )
)
plt.close(fig)
