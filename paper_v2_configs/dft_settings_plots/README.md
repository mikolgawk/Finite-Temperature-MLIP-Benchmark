# VASP convergence plots

Generate a four-panel energy/force convergence figure from the supplied VASP
data, without running new electronic-structure calculations:

```bash
uv run --script paper_v2_configs/dft_settings_plots/plot_vasp_convergence.py
```

The PDF and PNG are saved as `plots/plot_vasp_convergence.pdf` and
`plots/plot_vasp_convergence.png`. The point-level values and selected reference
configurations are exported to `results/plot_vasp_convergence_points.csv` and
`results/plot_vasp_convergence_references.csv`. Pass `--data-only` to export only
the tables. `--data-dir`, `--plots-dir` and `--results-dir` support separate inputs
and outputs. The original data and legacy figures are preserved.

## Data and definitions

| System | Energy data | Atom count | Cutoff-sweep reference | K-grid-sweep reference |
|---|---|---:|---|---|
| MAPbBr3 | Original nine-row table and supplied XYZ | 96 | 1000 eV, 3x3x3 | 700 eV, 4x4x4 |
| Pt(111) + 24 H2O | Raw extXYZ energy and force records | 108 | 900 eV, 22x19x1 | 700 eV, 22x19x1 |
| Cu3Au | Supplied cutoff/total-energy CSV | 128 | Highest supplied cutoff: 740 eV | No supplied sweep |

Each plotted energy error is the signed difference `(E - E_reference) / N`,
converted to meV/atom, as in the paper's convergence figure. Both signed and
absolute differences are retained in the exported point table.
The MAPbBr3 table contains total-energy offsets relative to one common baseline;
subtracting the selected reference removes that baseline. The Cu3Au atom count
is inherited from the supplied legacy plotting script and cannot be checked
against a structure in this folder. Override it with `--cu3au-atoms N` if needed.
The highest supplied cutoff is a comparison reference, not proof of absolute
convergence beyond the sampled range. In particular, the Cu3Au sweep is
nonmonotonic and has no 1000 eV reference.

Force errors are available only for Pt/H2O. They are the maximum absolute
Cartesian-component difference `max(abs(F - F_reference))`, in meV/angstrom.
The structures and cells must remain identical throughout a sweep. The scalar
MAPbBr3 force errors use a different baseline and cannot be re-referenced from
their maxima alone, so they are not plotted as converged-reference errors.

The raw Pt sweeps each contain seven frames. The pre-existing Pt CSVs contain
only six and omit the final reference points, so the new script does not use
those exports. It also preserves k-point mesh strings directly from the raw
headers: ASE otherwise parses an unquoted value such as `22_19_1` as `22191`.

The k-point coordinate is the inverse of the largest sampled reciprocal spacing:
`1 / max(2*pi*norm(b_i) / N_i)`, where ASE's reciprocal vectors `b_i` omit `2*pi`.
All three directions are used for bulk MAPbBr3; the two in-plane directions are
used for the Pt slab, whose out-of-plane mesh is fixed at one. This coordinate
has units of angstrom, not inverse angstrom. The production setting
`KSPACING = 0.22 / angstrom` corresponds to `1 / 0.22` angstrom.
The maximum spacing along a specific reciprocal direction need not equal that
input value exactly because the number of mesh points is an integer.

The figure follows the larger v2 RDF panel style: 12-point axis labels, ticks
and panel labels; 8-point legends; circular markers; 1.5-point solid curves;
and a grey grid with 0.5-point lines. The 10.59 x 7.5 inch canvas uses Figure 6's
source width so scaling both PDFs to the same manuscript width scales their
fonts, markers and strokes equally. At a 7.06-inch paper width, labels and ticks
are about 8 points and legends about 5.3 points, consistent with the other v2
figures. Legends use the compact RDF panel spacing. Each system keeps the same
deep-palette color in all panels.

The convergence figure in `paper/v2/main_v2.tex` uses the manuscript copy at
`paper/v2/figs_v2_torchsim/dft_settings/plot_vasp_convergence.pdf`. The script
refreshes that copy automatically after plotting, so the paper folder remains
self-contained and the next manuscript compilation uses the latest figure.
Pass `--no-update-paper` for isolated plotting outputs. `--data-only` also leaves
the manuscript figure unchanged. The original convergence PDF under
`paper/figs_si/convergence_tests` is retained as a legacy figure.

The 2x2 layout retains shared axes and the ranges of the convergence figure
included in `paper/v2/main_v2.tex`. Dashed lines mark zero. All axes are linear.
Cutoff panels use 250--1050 eV and
inverse-spacing panels use 0--25 angstrom. Energy panels use -5--5 meV/atom;
force panels use -100--100 meV/angstrom. These paper-matched limits clip the
largest low-cutoff errors and place the finest Pt k-point reference (about
29.14 angstrom) outside the displayed range. All input points remain in the
exported tables.
The inverse-spacing label retains the correct angstrom unit rather than the
inverse-angstrom unit printed in the original paper figure. Reference details
are documented above rather than added as a separate strip below the panels.

`Cu_bulk` and `naphthalene_bulk` contain FHI-aims data and are excluded from this
VASP-only figure.
