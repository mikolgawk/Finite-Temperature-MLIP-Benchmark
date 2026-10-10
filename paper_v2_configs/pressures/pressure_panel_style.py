"""Annotations shared by the pressure panel figures."""


def annotate_tier_medians(axes, tiers, *, maximum, font_size, format_value):
    """Keep tier headings inside the box and connect medians vertically.

    Axes are ordered from the lowest to the highest visible pressure range.
    Connector segments stop at each axis break instead of crossing the gap.
    Tier tuples contain the label, first index, exclusive end, median, and color.
    """
    top = axes[-1]
    height_points = top.get_position().height * top.figure.get_figheight() * 72
    padding = 3 / height_points
    tier_y = 1 - padding
    median_y = tier_y - 1.35 * font_size / height_points
    connector_y = median_y - 1.2 * font_size / height_points
    bar_y = min(0.74, connector_y - padding)
    if bar_y <= 0:
        raise ValueError("Pressure overview needs enough height for tier and median labels.")

    lo, hi = top.get_ylim()
    # Reserve annotation space above every bar without changing any metric.
    if maximum > lo:
        hi = lo + (maximum - lo) / bar_y
    elif len(axes) == 1:
        lo, hi = 0.0, 1.0
    top.set_ylim(lo, hi)
    connector_top = lo + (hi - lo) * connector_y

    for label, first, end, median, color in tiers:
        center = (first + end - 1) / 2
        top.text(center, tier_y, label, transform=top.get_xaxis_transform(),
                 ha="center", va="top", color=color, fontsize=font_size)
        top.text(center, median_y, format_value(median), transform=top.get_xaxis_transform(),
                 ha="center", va="top", color=color, fontsize=font_size, fontweight="bold")
        for ax in axes:
            segment_lo, segment_hi = ax.get_ylim()
            start, stop = max(median, segment_lo), min(connector_top, segment_hi)
            if stop > start:
                ax.vlines(center, start, stop, colors=color, linewidth=0.7, alpha=0.8)
