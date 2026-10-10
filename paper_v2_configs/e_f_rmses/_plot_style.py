"""Keep tier labels readable on the compact v1 RMSE canvas."""

from matplotlib.collections import LineCollection
from matplotlib.text import Annotation, Text


def space_tier_labels(fig):
    """Space tier labels while keeping median connectors strictly vertical."""
    groups_by_axis = []
    for ax in fig.axes:
        groups = []
        for header in ax.texts:
            if not header.get_text().startswith('Tier '):
                continue
            median_labels = [text for text in ax.texts
                             if isinstance(text, Annotation)
                             and text.get_color() == header.get_color()]
            groups.append((header.get_position()[0], [header, *median_labels]))
        groups_by_axis.append((ax, groups))

    # Draw twice to account for constrained layout updating the axes margins.
    for _ in range(2):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        gap = 3 * fig.dpi / 72
        for ax, groups in groups_by_axis:
            if not groups:
                continue
            half_widths = [max(Text.get_window_extent(text, renderer).width
                               for text in texts) / 2 for _, texts in groups]
            bounds = ax.get_window_extent(renderer)
            centers = []
            for index, (anchor, _) in enumerate(groups):
                preferred = ax.transData.transform((anchor, 0))[0]
                lower = (centers[-1] + half_widths[index - 1] + gap + half_widths[index]
                         if index else bounds.x0 + half_widths[index])
                centers.append(max(preferred, lower))
            centers[-1] = min(centers[-1], bounds.x1 - half_widths[-1])
            for index in range(len(centers) - 2, -1, -1):
                centers[index] = min(centers[index], centers[index + 1]
                                     - half_widths[index + 1] - gap - half_widths[index])
            for center, (anchor, texts) in zip(centers, groups):
                x = ax.transData.inverted().transform((center, 0))[0]
                # Keep the connector attached to its horizontal median segment.
                for text in texts:
                    if not isinstance(text, Annotation):
                        continue
                    for collection in ax.collections:
                        if not isinstance(collection, LineCollection):
                            continue
                        for segment in collection.get_segments():
                            if (len(segment) == 2
                                    and segment[0, 1] == segment[1, 1] == text.xy[1]
                                    and min(segment[:, 0]) <= anchor <= max(segment[:, 0])):
                                x = min(max(x, min(segment[:, 0])), max(segment[:, 0]))
                for text in texts:
                    text.set_position((x, text.get_position()[1]))
                    if isinstance(text, Annotation):
                        text.xy = (x, text.xy[1])
