import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

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

CALCULATOR_DISPLAY_NAMES = {
    'chgnet': 'CHGNet',
    'mace-mp-0': 'MACE-MP-0',
    'grace-mp': 'GRACE-2L-MPtrj',
    'mace-mpa-0': 'MACE-MPA-0',
    'orb-v2': 'orb-v2',
    'eq-v2-m-omat': 'EquiformerV2',
    'mattersim-v1-5m': 'MatterSim-v1.0.0-5M',
    'grace-oam': 'GRACE-2L-OAM',
    'orb-v3': 'orb-v3-conservative-inf-mpa',
    'orb-v3-direct': 'orb-v3-direct-20-mpa',
    'nequip': 'NequIP-OAM-XL',
    'esen-30m-oam': 'eSEN-30M-OAM',
    'pet-oam-xl': 'PET-OAM-XL',
    'pet-omat-xl': 'PET-OMAT-XL',
    'mace-mh-omat': 'MACE-MH-1-OMAT',
    'uma-s-omat': 'UMA-S-P1',
    'uma-m-omat': 'UMA-M-P1',
}

CALCULATOR_RAW_NAMES = {display_name: raw_name for raw_name, display_name in CALCULATOR_DISPLAY_NAMES.items()}


def normalize_calculator_name(name):
    text = str(name).strip()
    return CALCULATOR_DISPLAY_NAMES.get(text.lower(), text)

# Read and aggregate data from per-model RMSE CSV files
data_dir = BASE_DIR.parent / 'data'
csv_files = sorted(data_dir.glob('rmse-*.csv'))

if not csv_files:
    raise FileNotFoundError(f'No rmse-*.csv files found in {data_dir}')

all_frames = []
for csv_file in csv_files:
    frame = pd.read_csv(csv_file)
    model_name = csv_file.stem
    if model_name.startswith('rmse-results-all_'):
        model_name = model_name.replace('rmse-results-all_', '', 1)
    elif model_name.startswith('rmse-'):
        model_name = model_name.replace('rmse-', '', 1)
    frame['calculator'] = normalize_calculator_name(model_name)
    all_frames.append(frame)

all_data = pd.concat(all_frames, ignore_index=True)

df = (
    all_data.groupby('calculator', as_index=False)[
        ['energy_rmse', 'force_rmse', 'mlip_force_eval_time_per_call_per_atom_s']
    ]
    .mean(numeric_only=True)
)

mean_summary = df[[
    'calculator',
    'energy_rmse',
    'force_rmse',
    'mlip_force_eval_time_per_call_per_atom_s',
]].rename(columns={
    'mlip_force_eval_time_per_call_per_atom_s': 'mean_force_eval_time_per_atom_s'
})
mean_summary['calculator'] = mean_summary['calculator'].map(lambda name: CALCULATOR_RAW_NAMES.get(name, name))
results_dir = BASE_DIR / 'results'
plots_dir = BASE_DIR / 'plots'
results_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
mean_summary.to_csv(results_dir / 'mean_metrics_by_model.csv', index=False)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(3.53 * 2, 3.53))

tier_1 = [normalize_calculator_name(model) for model in ["chgnet", "mace-mp-0", "grace-mp"]]
tier_2 = [normalize_calculator_name(model) for model in ["mace-mpa-0", "orb-v2"]]
tier_3 = [normalize_calculator_name(model) for model in ["mattersim-v1-5M", "grace-oam", "orb-v3", "orb-v3-direct", "eSEN-30M-OAM", "nequip", "eq-v2-M-omat", "pet-oam-xl", "pet-omat-xl"]]
tier_4 = [normalize_calculator_name(model) for model in ["mace-mh-omat", "uma-s-omat", "uma-m-omat"]]



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

energy_x_pos = np.arange(len(energy_models))
force_x_pos = np.arange(len(force_models))

energy_colors = get_tier_colors(energy_models)
force_colors = get_tier_colors(force_models)

energy_t1_count, energy_t2_count, energy_t3_count, energy_t4_count = get_tier_counts(energy_models)
force_t1_count, force_t2_count, force_t3_count, force_t4_count = get_tier_counts(force_models)

# Plot 1: Energy RMSE
axes[0].bar(energy_x_pos, energy_df['energy_rmse'], color=energy_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('Model')
axes[0].set_ylabel('Energy RMSE [eV/atom]')
# axes[0].set_title('Energy RMSE by Model (Grouped by Tier)', fontsize=13, fontweight='bold')
axes[0].set_xticks(energy_x_pos)
axes[0].set_xticklabels(energy_models, rotation=45, ha='right', fontsize=FONT_SIZE)
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
axes[0].hlines(t4_med_energy, xmin=energy_tier3_end, xmax=len(energy_models)-0.5, colors=tier_colors['tier_4'], linestyles='--', linewidth=2, alpha=0.9, zorder=5)

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
        len(energy_models) - 1,
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
axes[0].text(tier_center(energy_t1_count + energy_t2_count + energy_t3_count, len(energy_models) - 1), tier_label_y(energy_ymax), 'Tier 4',
             ha='center', fontsize=FONT_SIZE, color=tier_colors['tier_4'])

# Plot 2: Force RMSE
axes[1].bar(force_x_pos, force_df['force_rmse'], color=force_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
axes[1].set_xlabel('Model')
axes[1].set_ylabel('Force RMSE [eV/Å]')
# axes[1].set_title('Force RMSE by Model (Grouped by Tier)', fontsize=13, fontweight='bold')
axes[1].set_xticks(force_x_pos)
axes[1].set_xticklabels(force_models, rotation=45, ha='right', fontsize=FONT_SIZE)
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
axes[1].hlines(t4_med_force, xmin=force_tier3_end, xmax=len(force_models)-0.5, colors=tier_colors['tier_4'], linestyles='--', linewidth=2, alpha=0.9, zorder=5)

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
        len(force_models) - 1,
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
axes[1].text(tier_center(force_t1_count + force_t2_count + force_t3_count, len(force_models) - 1), tier_label_y(force_ymax), 'Tier 4',
             ha='center', fontsize=FONT_SIZE, color=tier_colors['tier_4'])

# Subplot labels
axes[0].text(0.02, 0.96, '(a)', transform=axes[0].transAxes, ha='left', va='top', fontsize=FONT_SIZE)
axes[1].text(0.02, 0.96, '(b)', transform=axes[1].transAxes, ha='left', va='top', fontsize=FONT_SIZE)

plt.tight_layout()
plot_path = plots_dir / 'plot_e_f_rmses.pdf'
plt.savefig(plot_path)
print(f"Plot saved as {plot_path}")
print(
    "Saved mean_metrics_by_model.csv with per-model means for energy RMSE, force RMSE, "
    "and force eval time per atom"
)
print(
    f"Energy RMSE medians -> Tier 1: {t1_med_energy:.8f}, "
    f"Tier 2: {t2_med_energy:.8f}, Tier 3: {t3_med_energy:.8f}, Tier 4: {t4_med_energy:.8f}"
)
print(
    f"Force RMSE medians -> Tier 1: {t1_med_force:.8f}, "
    f"Tier 2: {t2_med_force:.8f}, Tier 3: {t3_med_force:.8f}, Tier 4: {t4_med_force:.8f}"
)
print("Mean force eval time per atom (s):")
for _, row in mean_summary.sort_values('mean_force_eval_time_per_atom_s').iterrows():
    print(f"  {row['calculator']}: {row['mean_force_eval_time_per_atom_s']:.6e}")
plt.show()
