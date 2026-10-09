"""Shared appearance and data-driven axes for the SI 14-16 Pareto figures."""

from matplotlib.axes import Axes
import numpy as np


FIGURE_SIZE = (3.53 * 1.5, 3.53 * 1.5)
MODEL_LABEL_STYLE = {
    "fontsize": 6,
    "bbox": {
        "boxstyle": "round,pad=0.08",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.7,
    },
    "zorder": 5,
}


def apply_pareto_style(
    ax: Axes, times: np.ndarray, errors: np.ndarray, *, ylabel: str | None = None,
) -> None:
    """Fit all timing points and use the pressure Pareto figure's formatting."""
    times = np.asarray(times, dtype=float)
    errors = np.asarray(errors, dtype=float)
    times = times[np.isfinite(times)]
    errors = errors[np.isfinite(errors)]
    if times.size == 0 or errors.size == 0:
        raise ValueError("Pareto axes require finite timing and error values.")

    time_min, time_max = float(times.min()), float(times.max())
    time_padding = (time_max - time_min) * 0.12
    ax.set_xlim(left=min(0.0, time_min - time_padding), right=max(1.0, time_max * 1.1))
    lower, upper = float(errors.min()), float(errors.max())
    padding = max((upper - lower) * 0.05, 1e-6)
    ax.set_ylim(lower - padding, upper + padding)
    ax.set_xlabel("Mean time per step [ms]")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=True, fontsize=6)
