# Refined VDOS analysis

This directory provides the default VDOS analysis for the v2 paper. The
top-level `generate_metrics.py` command computes these refined results, and the
standard Pareto scripts read them by default. It reads the same reference and
MLIP trajectories and writes its spectra, metric tables, comparisons and figures
here. Historical results in `../vdos_legacy` and manuscript figure copies are not
overwritten.

## Method

1. Unwrap positions using the recorded periodic cell and periodic-boundary
   flags. The present analysis is for fixed-cell NVT trajectories; changing
   cells are rejected explicitly. Unwrapping assumes the saved frames resolve
   the atomic motion across periodic boundaries.
2. Reconstruct velocities in both reference and MLIP trajectories using the
   same central difference, `(r[t+1] - r[t-1]) / (2 * dt)`. The first and last
   position samples are omitted. Stored MLIP velocities are not mixed with
   numerical reference velocities. Central differences have a frequency
   response of their own; matching the procedure avoids applying this filter
   to only one side of a comparison.
3. Estimate the one-sided velocity power spectral density using **Hann-windowed
   Welch averaging with 50% segment overlap**. Each segment's temporal mean is
   removed before windowing. The requested segment duration is **2000 fs**.
   For a short matched reference/MLIP pair it is shortened, identically on both
   sides, to allow at least **four overlapping segments**. The requested
   duration is a physical time, not a fixed frame count, so systems with
   different timesteps use the same target frequency resolution. The actual
   duration, segment count and native frequency resolution are recorded with
   each spectrum. Extremely short records that cannot fit the requested
   number of segments are flagged in the metadata.
4. Multiply each atom's Cartesian velocity powers by its ASE standard atomic mass, sum
   over atoms and components, and normalize the total spectrum to unit area.
   This **mass-weighted classical VDOS** convention removes the inverse-mass
   weighting of unweighted thermal velocity power: under equipartition,
   vibrational modes contribute equally in the classical harmonic limit. It is distinct from the previous
   unweighted velocity-power observable for mixed-element systems. No
   independent normalization of each atom or Cartesian component is applied.
   The VACF Fourier transform is **not** squared again. The Hann window acts
   on velocity segments rather than suppressing the zero-lag VACF.
5. Compare matched-duration reference and MLIP spectra with the existing
   normalized L1 error, bounded by 0--100%. Failure penalties and model/system
   populations retain the existing benchmark definitions. The plotted curves
   and error calculation use the same spectrum; no plot-only smoothing is
   applied.
6. Validate spectrum caches against a method version, analyzed input content,
   input provenance, timestep, frame count, processing/FFT settings and cached
   output integrity. The signature also includes the estimator, atomic
   masses/weighting, requested and actual Welch segment settings and overlap.
   Missing or inconsistent metadata causes recalculation; `--overwrite` still
   forces recalculation.

The estimator follows the standard [SciPy Welch
definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).
The relation between the vibrational velocity spectrum and the Fourier
transform of the VACF is described in the [GROMACS
documentation](https://manual.gromacs.org/current/onlinehelp/gmx-velacc.html).
Welch averaging reduces spectral sampling variance at the cost of frequency
resolution. Its overlapping segments are not independent replicas, and it
does not eliminate finite-trajectory uncertainty. Short systems use shorter
segments and consequently have coarser resolution. Zero padding interpolates
frequency bins without improving the native resolution. The refinement
checks requested windows of 1, 2 and 4 ps and a minimum of eight segments for
short records; sensitivity to those settings should be considered when
interpreting small model differences. These checks do not establish a
statistical confidence interval or convergence with trajectory length.
Per-segment mean removal attenuates near-zero-frequency and diffusive power;
these spectra assess vibrational shape rather than diffusion. Central
differences also attenuate high-frequency velocity power, consistently on
both sides of each comparison.

The **full-record Hann periodogram** and **unweighted velocity-power**
convention remain available as explicit diagnostics through
`--estimator periodogram --weighting velocity`. They are not the new default.
The original corrected outputs, before Welch/mass weighting, are preserved
under `archive/periodogram_velocity_v1/`, including their results, figures,
deliverables, summary and archive manifest. The original `../vdos_legacy` analysis
also remains untouched.

Molecular crystals remain excluded by default. Hydrogen and Pt/H2O remain in
the VDOS analysis. Failed, missing, incomplete and audited unstable MD runs
receive 100% VDOS error in benchmark aggregates. Figure 6 and SI7--9 retain
the model-specific stable-MD subset policy; a stable run with a genuine 100%
error is retained. The hybrid accelerated cohort still includes eager EQ/eSEN
and ASE-eager CHGNet using the existing source mappings.

## Run

Compute the corrected TorchSim eager and accelerated results and create their
plots:

```bash
uv run --script paper_v2_configs/vdos/run_vdos_pipeline.py \
  --source mlip-trajs-torchsim-eager \
  --source mlip-trajs-torchsim-accelerated
```

This command defaults to `--estimator welch --weighting mass --window-fs 2000
--min-segments 4 --pad-factor 4`. Zero padding provides denser frequency bins
for plotting and integration without changing physical resolution. For a
diagnostic comparison without replacing those
outputs:

```bash
uv run --script paper_v2_configs/vdos/run_vdos_pipeline.py \
  --estimator periodogram --weighting velocity --pad-factor 1 \
  --results-dir paper_v2_configs/vdos/diagnostics/periodogram_velocity/results \
  --plots-dir paper_v2_configs/vdos/diagnostics/periodogram_velocity/plots
```

For window sensitivity, use `--window-fs 1000` or `--window-fs 4000` with a
separate results directory. Use `--min-segments 8` to test stronger averaging
for short records. Changing these settings invalidates the relevant spectrum
cache; distinct output directories retain both metric tables for comparison.
`--window-fs` and `--min-segments` affect Welch only.

Use `--compute-only` to skip plots. Recreate plots from the corrected CSVs with:

```bash
uv run --script paper_v2_configs/vdos/plot_all.py \
  --source mlip-trajs-torchsim-eager \
  --source mlip-trajs-torchsim-accelerated
```

Outputs are organized by trajectory source under `results/` and `plots/`.
The plotting set includes the model overview, Figure 5, Figure 6 and its
all-label companion, SI7--9, and SI16. It also includes updated combined
Pareto Figures 7/SI13 and observable-correlation Figures SI11/SI12 via
`plot_dependent.py`. Only their VDOS input changes; RDF, pressure, RMSE,
Matbench scores and timing inputs continue to come from the original analysis
directories. The extra no-hydrogen/no-Pt+H2O plot sets remain outside the
default plotting pipeline.

To reproduce the accelerated ASE/TorchSim comparison with the same refined
VDOS method for both runners, first compute the ASE summaries, then plot SI18:

```bash
uv run --script paper_v2_configs/vdos/run_vdos_pipeline.py \
  --source mlip-trajs-ase-accelerated --compute-only
uv run --script paper_v2_configs/vdos/figure_SI_18.py
```

SI18 reads both accelerated VDOS summaries from `vdos/results/`;
RDF, pressure and timing inputs retain their runner-specific original paths.
The TorchSim cohort keeps its existing eager substitutions. Each runner has
an independent Pareto frontier. For configurations shown in both runners,
timings are averaged over identical successful system workloads using SI17's
validation, including matching step counts, time steps and recording intervals.
The observable errors keep their existing failed-run penalties. Configurations
shown for only one runner retain their usual cohort timing. The exported
metrics include the matched system names and counts (`timing_systems` and
`n_timing_systems`), plus `timing_basis`; shared configurations with no matching
workloads are omitted.
The ASE UMA-S-P1 (turbo) and UMA-M-P1 (turbo) configurations are excluded from
SI18 only; their TorchSim counterparts and all benchmark metrics are retained.
The comparison is saved under
`plots/torchsim_vs_ase_accelerated/figure_SI_18.pdf`, with metrics and frontier
CSVs under `results/torchsim_vs_ase_accelerated/`. Pass `--data-only` to export
those tables without plotting. `--results-dir` and `--plots-dir` support
isolated alternative runs. The wrapper requires both refined VDOS summaries
and never falls back to legacy VDOS metrics. `plot_dependent.py` includes
SI18 once in its TorchSim accelerated stage when both summaries are available;
if ASE has not been computed, the existing per-cohort plotting stages still
run without SI18. The standard Pareto entry points also default to these refined
VDOS inputs; their outputs remain in `pareto_plots/`. Manuscript copies are
updated separately.

When an ASE source is passed explicitly to `plot_all.py`, the ordinary VDOS
overview, spectra, correlations and VDOS Pareto plot are generated. The
TorchSim-only dependent plotting stage is skipped. Use the separate SI18
wrapper above for the accelerated runner comparison.

For a historical comparison, the standard Pareto plotting pipeline accepts
`--vdos-results-dir paper_v2_configs/vdos_legacy/results`. Use separate output paths
for that comparison. Neither default pipeline falls back to historical metrics
when a refined input is missing.

The numerical tests in this directory check frequency and power recovery,
mass weighting, Welch averaging, periodic unwrapping, consistent velocity
handling, cache invalidation, CLI forwarding and output path isolation.

`analysis_summary.md` records the completed run and its implications. Recreate
the original/corrected metric comparison with
`uv run --script paper_v2_configs/vdos/compare_results.py`.

Run the six-setting accelerated sensitivity sweep without replacing the default
spectra or plots:

```bash
uv run --script paper_v2_configs/vdos/sensitivity_analysis.py
uv run --script paper_v2_configs/vdos/sensitivity_analysis.py --impact-only
uv run --script paper_v2_configs/vdos/summarize_analysis.py
```

The sweep preserves all model/system pairs and failure penalties. It compares
the 1/2/4 ps targets, eight-segment averaging, unweighted velocity power and
eightfold padding with the default. The summary verifies that its baseline
reproduces the active per-system errors. Results are saved under
`results/sensitivity/`; the report includes score and ranking changes.
The impact-only command uses the existing sensitivity scores to assess the
combined arithmetic Pareto frontier and stable-MD correlations, holding all
other metric and timing inputs fixed.

After plotting, `python paper_v2_configs/vdos/package_plots.py` creates
12-page TorchSim PDF collections and a ZIP in `deliverables/`. The ZIP includes
the individual figures, metric and diagnostic summaries, native ASE accelerated
summaries, and the standalone SI18 PDF and comparison tables when available.
SI18 is optional when none of its exports exist; a partial figure/metrics/frontier
set must be completed before packaging. Raw trajectories and spectrum caches
remain outside the bundle.

Creating the PDF collections requires Poppler's `pdfunite`. To refresh only the
ZIP and index from existing collections, without changing any PDF or invoking
PDF tools, run:

```bash
python paper_v2_configs/vdos/package_plots.py --archive-only
```

The regular analysis and plotting commands do not require `pdfunite`.
