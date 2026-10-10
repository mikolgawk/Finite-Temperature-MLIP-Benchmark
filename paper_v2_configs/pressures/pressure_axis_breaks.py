"""Plotting-only pressure axis breaks; never alter data used for metrics."""

from __future__ import annotations

import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator


def finite_values(values):
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def pressure_tick(value, _position=None):
    """Keep scientific multipliers away from neighbouring axes and headings."""
    precision = 3 if abs(value) < 1000 else 4 if abs(value) < 1e4 else 2
    label = f"{value:.{precision}g}"
    return label.replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")


def mae_axis_windows(values):
    """Separate widely spaced MAEs, retaining every bar and tier median."""
    values = np.unique(np.r_[0.0, finite_values(values)])
    if values.size == 1:
        return [(0.0, 1.0)]
    floor = max(values[values > 0].min(), 1e-9)
    gaps = np.diff(values) / np.maximum(values[:-1], floor)
    cuts = np.flatnonzero(gaps > 5)
    # Also separate errors of tens of GPa from those close to one GPa.
    cuts = sorted(cuts[np.argsort(gaps[cuts])[-3:]] + 1)
    groups = np.split(values, cuts)
    windows = []
    for group in groups:
        span = max(np.ptp(group), abs(group[-1]) * 0.15, floor)
        windows.append((max(0.0, group[0] - span * 0.12), group[-1] + span * 0.25))
    return windows


def distribution_axis_windows(arrays, reference_arrays):
    """Give reference-scale pressures space alongside signed extreme tails.

    Extreme tails get separate ranges when the full range is over twenty
    times the reference-scale core. Widely separated reference clusters also
    get a break. The outer ranges retain each distribution's extrema.
    """
    arrays = [finite_values(values) for values in arrays]
    arrays = [values for values in arrays if values.size]
    references = [finite_values(values) for values in reference_arrays]
    references = [values for values in references if values.size]
    if not arrays or not references:
        raise ValueError("Pressure axes require finite model and reference values.")
    ref_lo = min(values.min() for values in references)
    ref_hi = max(values.max() for values in references)
    ref_span = max(ref_hi - ref_lo, abs(ref_hi) * 0.01, 1e-6)
    core_lo, core_hi = ref_lo, ref_hi
    samples = []
    for values in arrays:
        quantiles = np.quantile(values, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1])
        samples.extend(quantiles)
        lo, hi = quantiles[1], quantiles[-2]
        if (hi - lo <= 10 * ref_span
                and lo >= ref_lo - 10 * ref_span
                and hi <= ref_hi + 10 * ref_span):
            core_lo, core_hi = min(core_lo, lo), max(core_hi, hi)
    core_span = max(core_hi - core_lo, ref_span)
    lo = min(values.min() for values in arrays)
    hi = max(values.max() for values in arrays)
    # Pooled references can themselves contain separate scales: ordinary
    # systems near zero and hydrogen near 140 GPa. Keep both readable.
    ref_samples = np.unique(np.concatenate([
        np.quantile(values, np.linspace(0, 1, min(values.size, 1001)))
        for values in references
    ]))
    reference_floor = max(np.subtract(*np.quantile(ref_samples, [0.75, 0.25])), 1e-6)
    gaps = np.diff(ref_samples)
    relative_gaps = gaps / np.maximum(
        np.minimum(np.abs(ref_samples[:-1]), np.abs(ref_samples[1:])), reference_floor
    )
    cuts = np.flatnonzero(relative_gaps > 5)
    core_break = int(cuts[np.argmax(relative_gaps[cuts])]) if cuts.size else None
    if hi - lo <= 20 * core_span and core_break is None:
        padding = max(hi - lo, ref_span) * 0.06
        return [(lo - padding, hi + padding)]

    core = (core_lo - core_span * 0.1, core_hi + core_span * 0.1)
    samples = np.asarray(samples)
    windows = []
    negative_tail = samples[samples < core[0] - core_span]
    if negative_tail.size:
        tail_hi = negative_tail.max()
        padding = max(tail_hi - lo, ref_span) * 0.04
        windows.append((lo - padding, min(tail_hi + padding, core[0] - core_span * 0.2)))
    if core_break is None:
        windows.append(core)
    else:
        lower, upper = ref_samples[core_break:core_break + 2]
        gap = upper - lower
        windows.extend([(core[0], lower + gap * 0.06),
                        (upper - gap * 0.06, core[1])])
    positive_tail = samples[samples > core[1] + core_span]
    if positive_tail.size:
        tail_lo = positive_tail.min()
        padding = max(hi - tail_lo, ref_span) * 0.04
        windows.append((max(tail_lo - padding, core[1] + core_span * 0.2), hi + padding))
    return windows


def pressure_axes(fig, subplot_spec, windows, *, orientation, core_index=0, header=False):
    """Return axes in increasing value order, with conventional break marks."""
    count = len(windows)
    ratios = [2.8 if i == core_index else 1.0 for i in range(count)]
    if orientation == "y" and header:
        # Keep the overview's tier headings legible within its top segment.
        ratios[-1] = max(ratios[-1], 2.8)
    if orientation == "x":
        grid = subplot_spec.subgridspec(1, count, width_ratios=ratios, wspace=0.10)
    else:
        grid = subplot_spec.subgridspec(count, 1, height_ratios=ratios[::-1], hspace=0.10)
    axes = []
    for index, limits in enumerate(windows):
        position = grid[0, index] if orientation == "x" else grid[count - 1 - index, 0]
        shared = {} if not axes else {"sharey" if orientation == "x" else "sharex": axes[0]}
        ax = fig.add_subplot(position, **shared)
        getattr(ax, f"set_{orientation}lim")(limits)
        axis = getattr(ax, f"{orientation}axis")
        if orientation == "x" and count > 1 and index != core_index:
            axis.set_major_locator(FixedLocator([np.mean(limits)]))
        else:
            axis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2, prune="both"))
        axis.set_major_formatter(FuncFormatter(pressure_tick))
        axes.append(ax)
    for index, ax in enumerate(axes):
        if orientation == "x":
            if index:
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", left=False, labelleft=False)
                ax.yaxis.get_offset_text().set_visible(False)
            if index < count - 1:
                ax.spines["right"].set_visible(False)
            locations = ([0] if index else []) + ([1] if index < count - 1 else [])
            for x in locations:
                ax.plot([x, x], [0, 1], transform=ax.transAxes, linestyle="none",
                        marker=[(-1, -1), (1, 1)], markersize=5, color="black",
                        clip_on=False, markeredgewidth=0.8)
        else:
            if index:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            if index < count - 1:
                ax.spines["top"].set_visible(False)
            locations = ([0] if index else []) + ([1] if index < count - 1 else [])
            for y in locations:
                ax.plot([0, 1], [y, y], transform=ax.transAxes, linestyle="none",
                        marker=[(-1, -1), (1, 1)], markersize=5, color="black",
                        clip_on=False, markeredgewidth=0.8)
    return axes


def histogram_segments(values, windows, bins):
    """Resolve each visible range using counts normalized by ALL frames."""
    values = finite_values(values)
    segments = []
    for lo, hi in windows:
        edges = np.linspace(lo, hi, bins + 1)
        counts, _ = np.histogram(values, bins=edges)
        density = counts / (max(values.size, 1) * np.diff(edges))
        segments.append((density, edges))
    return segments
