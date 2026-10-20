#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["ase>=3.26", "matplotlib>=3.9", "numpy>=1.26", "pandas>=2.2"]
# ///
"""Plot VASP cutoff and k-point convergence from the supplied raw data.

Energy differences use each sweep's highest-resolution reference. Force errors
are maximum absolute Cartesian-component differences, available for Pt/water
only. FHI-aims datasets and stale Pt CSV exports are deliberately not loaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil

from ase.cell import Cell
from ase.io import read
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem
PAPER_FIGURE = HERE.parent / "paper" / "v2" / "figs_v2_torchsim" / "dft_settings" / f"{STEM}.pdf"
FONT_SIZE = 12
LEGEND_FONT_SIZE = 8
# Use Figure 6's source width so fitting both PDFs to the manuscript's linewidth
# scales the fonts, markers and strokes equally (12 pt becomes about 8 pt).
FIGURE_SIZE = (3.53 * 3, 7.5)
PT_FOLDER = "Pt111w24H2O_380K_Hendrik"
MAP_FOLDER = "MAPbBr3 in 2x2x2 cell (96 atoms)"
LABELS = {
    "MAPbBr3": r"MAPbBr$_3$",
    "Pt111+24H2O": r"Pt(111) + 24 H$_2$O",
    "Cu3Au": r"Cu$_3$Au",
}
COLORS = {"MAPbBr3": "#4C72B0", "Pt111+24H2O": "#DD8452", "Cu3Au": "#55A868"}


def k_spacing(cell: Cell, mesh: str, dimensions: int = 3) -> float:
    """Maximum reciprocal spacing; slab sweeps sample only the in-plane mesh."""
    counts = np.array([int(value) for value in re.split(r"[_x]", mesh)])
    if len(counts) != 3 or np.any(counts <= 0):
        raise ValueError(f"Invalid k-point mesh: {mesh}")
    reciprocal_lengths = np.linalg.norm(2 * np.pi * cell.reciprocal(), axis=1)
    return float(np.max(reciprocal_lengths[:dimensions] / counts[:dimensions]))


def raw_meshes(path: Path) -> list[str]:
    """Preserve underscores before ASE can interpret e.g. 22_19_1 as 22191."""
    meshes = []
    with path.open() as handle:
        while line := handle.readline():
            if not line.strip():
                continue
            count = int(line)
            header = handle.readline()
            match = re.search(r'\bkpoints=(?:"([^\"]+)"|(\S+))', header)
            if match is None:
                raise ValueError(f"Missing k-point mesh in {path}")
            meshes.append(match.group(1) or match.group(2))
            for _ in range(count):
                if not handle.readline():
                    raise ValueError(f"Truncated atom records in {path}")
    return meshes


def curve_rows(
    system: str, sweep: str, configurations: list[dict], reference: int,
    n_atoms: int, source_file: str, atom_count_source: str,
    forces: list[np.ndarray] | None = None,
) -> list[dict]:
    ref = configurations[reference]
    rows = []
    for index, configuration in enumerate(configurations):
        difference = (configuration["energy_value_ev"] - ref["energy_value_ev"]) * 1000 / n_atoms
        force_error = (float(np.max(np.abs(forces[index] - forces[reference]))) * 1000
                       if forces is not None else np.nan)
        rows.append({
            "system": system, "sweep": sweep, **configuration,
            "n_atoms": n_atoms, "atom_count_source": atom_count_source,
            "energy_error_mev_per_atom": difference,
            "abs_energy_error_mev_per_atom": abs(difference),
            "max_force_component_error_mev_per_angstrom": force_error,
            "reference_cutoff_ev": ref["cutoff_ev"],
            "reference_k_grid": ref["k_grid"],
            "reference_k_spacing_inv_angstrom": ref["k_spacing_inv_angstrom"],
            "is_reference": index == reference, "source_file": source_file,
        })
    return rows


def pt_rows(data_dir: Path, filename: str, sweep: str) -> list[dict]:
    path = data_dir / PT_FOLDER / filename
    frames = read(path, index=":")
    meshes = raw_meshes(path)
    if len(frames) != len(meshes) or len(frames) < 2:
        raise ValueError(f"Insufficient or mismatched reference configurations in {path}")
    initial = frames[0]
    configurations, forces = [], []
    for atoms, mesh in zip(frames, meshes, strict=True):
        if (not np.array_equal(atoms.numbers, initial.numbers)
                or not np.allclose(atoms.positions, initial.positions, rtol=0, atol=1e-10)
                or not np.allclose(atoms.cell, initial.cell, rtol=0, atol=1e-10)):
            raise ValueError(f"Convergence configurations have different geometries in {path}")
        spacing = k_spacing(atoms.cell, mesh, dimensions=2)
        configurations.append({
            "cutoff_ev": float(atoms.info["encut"]), "k_grid": mesh,
            "k_spacing_inv_angstrom": spacing, "inverse_k_spacing_angstrom": 1 / spacing,
            "energy_value_ev": float(atoms.get_potential_energy()),
            "energy_value_kind": "total energy", "sampling_dimensions": "xy",
        })
        forces.append(atoms.get_forces())
    if sweep == "encut" and len(set(meshes)) != 1:
        raise ValueError(f"Cutoff sweep changes its k-point mesh in {path}")
    if sweep == "k_grid" and len({row["cutoff_ev"] for row in configurations}) != 1:
        raise ValueError(f"K-point sweep changes its cutoff in {path}")
    reference = (int(np.argmax([row["cutoff_ev"] for row in configurations])) if sweep == "encut"
                 else int(np.argmin([row["k_spacing_inv_angstrom"] for row in configurations])))
    return curve_rows("Pt111+24H2O", sweep, configurations, reference, len(initial),
                      str(path.relative_to(data_dir)), "raw extXYZ", forces)


def load_convergence_data(data_dir: Path, cu3au_atoms: int = 128) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return auditable point and reference tables for VASP datasets only."""
    data_dir = Path(data_dir)
    if cu3au_atoms <= 0:
        raise ValueError("Cu3Au atom count must be positive")
    rows = pt_rows(data_dir, "encut_convergence.extxyz", "encut")
    rows += pt_rows(data_dir, "kpoints_convergence.extxyz", "k_grid")

    folder = data_dir / MAP_FOLDER
    geometry = (folder / "MAPbBr3_300K_Ivor_VASP.xyz").read_text().splitlines()
    header_index = 1 if geometry[0].strip().isdigit() else 0
    lattice = re.search(r'Lattice="([^\"]+)"', geometry[header_index])
    if lattice is None:
        raise ValueError("MAPbBr3 geometry is missing lattice vectors")
    cell = Cell(np.fromstring(lattice.group(1), sep=" ").reshape(3, 3))
    n_atoms = sum(bool(re.match(r"^[A-Z][a-z]?\s+[-+0-9.]", line))
                  for line in geometry[header_index + 1:])
    if header_index and n_atoms != int(geometry[0]):
        raise ValueError("MAPbBr3 declared atom count does not match its atom records")
    if n_atoms != 96:
        raise ValueError(f"Expected 96 MAPbBr3 atoms, found {n_atoms}")
    table_path = folder / "gemini-code-1779112066944.txt"
    table = pd.read_csv(table_path)
    for sweep, subset in (("encut", table.loc[table.k_grid.eq("3x3x3")]),
                          ("k_grid", table.loc[table.ENCUT.eq(700)])):
        configurations = []
        for row in subset.itertuples():
            mesh = row.k_grid.replace("x", "_")
            spacing = k_spacing(cell, mesh)
            configurations.append({
                "cutoff_ev": float(row.ENCUT), "k_grid": mesh,
                "k_spacing_inv_angstrom": spacing, "inverse_k_spacing_angstrom": 1 / spacing,
                "energy_value_ev": float(row.E_minus_E0_meV) / 1000,
                "energy_value_kind": "relative to 520 eV, 2x2x2", "sampling_dimensions": "xyz",
            })
        reference = (int(np.argmax([row["cutoff_ev"] for row in configurations])) if sweep == "encut"
                     else int(np.argmin([row["k_spacing_inv_angstrom"] for row in configurations])))
        rows += curve_rows("MAPbBr3", sweep, configurations, reference, n_atoms,
                           str(table_path.relative_to(data_dir)), "supplied XYZ")

    cu_path = data_dir / "Cu3Au" / "Cu3Au_Ecut_total_energy.csv"
    values = np.loadtxt(cu_path, delimiter=",")
    configurations = [{
        "cutoff_ev": float(cutoff), "k_grid": "",
        "k_spacing_inv_angstrom": np.nan, "inverse_k_spacing_angstrom": np.nan,
        "energy_value_ev": float(energy), "energy_value_kind": "total energy",
        "sampling_dimensions": "not supplied",
    } for cutoff, energy in values]
    rows += curve_rows("Cu3Au", "encut", configurations,
                       int(np.argmax(values[:, 0])), cu3au_atoms, str(cu_path.relative_to(data_dir)),
                       "supplied legacy plotting script (CLI override supported)")
    points = pd.DataFrame(rows).sort_values(["sweep", "system", "cutoff_ev", "inverse_k_spacing_angstrom"])
    points = points.reset_index(drop=True)
    if not np.isfinite(points[["cutoff_ev", "energy_value_ev", "abs_energy_error_mev_per_atom"]]).all().all():
        raise ValueError("Convergence data contain invalid energies or cutoffs")
    references = points.loc[points.is_reference].reset_index(drop=True)
    return points, references


def plot_convergence(points: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    # Match the larger v2 RDF panels, retaining the paper's convergence ranges.
    style = {"font.family": "DejaVu Sans", "font.size": FONT_SIZE,
             "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE,
             "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
             "legend.fontsize": LEGEND_FONT_SIZE, "figure.titlesize": FONT_SIZE,
             "pdf.fonttype": 42, "axes.grid": True, "grid.color": "#B0B0B0",
             "grid.alpha": 1, "grid.linewidth": 0.5, "axes.axisbelow": True,
             "axes.linewidth": 0.8, "lines.linewidth": 1.5, "lines.markersize": 4}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, sharex="col", sharey="row",
                                 layout="constrained")
        for column, sweep in enumerate(("encut", "k_grid")):
            x = "cutoff_ev" if sweep == "encut" else "inverse_k_spacing_angstrom"
            for system in LABELS:
                data = points.loc[points.sweep.eq(sweep) & points.system.eq(system)].sort_values(x)
                if data.empty:
                    continue
                axes[0, column].plot(data[x], data.energy_error_mev_per_atom, "o-",
                                     color=COLORS[system], label=LABELS[system])
                force = data.dropna(subset=["max_force_component_error_mev_per_angstrom"])
                if not force.empty:
                    axes[1, column].plot(force[x], force.max_force_component_error_mev_per_angstrom,
                                         "o-", color=COLORS[system], label=LABELS[system])
            for row in range(2):
                ax = axes[row, column]
                ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
                ax.legend(loc="upper right", framealpha=0.8, handlelength=1.0,
                          handletextpad=0.3, borderpad=0.2, labelspacing=0.2,
                          borderaxespad=0.2)
        axes[0, 0].set_xlim(250, 1050)
        axes[0, 0].set_xticks([400, 600, 800, 1000])
        axes[0, 1].set_xlim(0, 25)
        axes[0, 1].set_xticks([0, 5, 10, 15, 20, 25])
        axes[0, 0].set_ylim(-5, 5)
        axes[0, 0].set_yticks([-4, -2, 0, 2, 4])
        axes[1, 0].set_ylim(-100, 100)
        axes[1, 0].set_yticks([-100, -50, 0, 50, 100])
        axes[0, 0].set_ylabel("Energy Error [meV/atom]")
        axes[1, 0].set_ylabel(r"Max Force Error [meV/$\AA$]")
        axes[1, 0].set_xlabel("ENCUT [eV]")
        # The original paper label has inverse units; 1 / delta-k is a length.
        axes[1, 1].set_xlabel(r"Inverse k-point spacing [$\AA$]")
        for label, ax in zip("abcd", axes.flat, strict=True):
            ax.text(0.02, 0.95, f"({label})", transform=ax.transAxes, ha="left", va="top")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "convergence_tests")
    parser.add_argument("--plots-dir", type=Path, default=HERE / "plots")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--cu3au-atoms", type=int, default=128,
                        help="Cu3Au atom count (128 in the supplied legacy plotting script).")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--update-paper", action=argparse.BooleanOptionalAction, default=True,
                        help="Refresh the manuscript's PDF copy (default: enabled).")
    args = parser.parse_args()
    points, references = load_convergence_data(args.data_dir, args.cu3au_atoms)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for suffix, table in (("points", points), ("references", references)):
        path = args.results_dir / f"{STEM}_{suffix}.csv"
        table.to_csv(path, index=False)
        print(f"Saved: {path}")
    print(references[["system", "sweep", "n_atoms", "reference_cutoff_ev", "reference_k_grid"]].to_string(index=False))
    if not args.data_only:
        output = args.plots_dir / f"{STEM}.pdf"
        plot_convergence(points, output)
        print(f"Saved: {output}")
        if args.update_paper:
            PAPER_FIGURE.parent.mkdir(parents=True, exist_ok=True)
            if output.resolve() != PAPER_FIGURE.resolve():
                shutil.copyfile(output, PAPER_FIGURE)
            print(f"Updated manuscript figure: {PAPER_FIGURE}")


if __name__ == "__main__":
    main()
