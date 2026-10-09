"""Place model names beside plotted points without connectors."""

from collections.abc import Iterable

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


def inward_label_offset(
    ax: plt.Axes,
    x_value: float,
) -> tuple[tuple[int, int], str, str]:
    x_min, x_max = ax.get_xlim()

    x_fraction = 0.5
    if x_max > x_min:
        x_fraction = (x_value - x_min) / (x_max - x_min)

    x_offset = 3
    horizontal_alignment = "left"
    if x_fraction >= 0.65:
        x_offset = -3
        horizontal_alignment = "right"

    return (x_offset, 3), horizontal_alignment, "bottom"


def position_model_labels(
    fig: plt.Figure, axes: Iterable[plt.Axes], *, max_offset_points: int = 3,
    max_distance_points: float | None = None,
) -> None:
    """Choose nearby labels, optionally bounding their distance from the dot.

    ``max_distance_points`` measures the gap from the dot to the nearest text
    edge, including when labels sit above or beside their dot.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def overlap_area(first: Bbox, second: Bbox) -> float:
        width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
        height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
        return width * height

    for ax in axes:
        axis_bbox = ax.get_window_extent(renderer)
        label_texts = [
            text for text in ax.texts if getattr(text, "_model_point_label", False)
        ]
        fixed_boxes = [
            text.get_window_extent(renderer)
            for text in ax.texts
            if text not in label_texts
        ]
        marker_radius = 4 * fig.dpi / 72
        for collection in ax.collections:
            for x_pixel, y_pixel in collection.get_offset_transform().transform(
                collection.get_offsets()
            ):
                fixed_boxes.append(
                    Bbox.from_extents(
                        x_pixel - marker_radius,
                        y_pixel - marker_radius,
                        x_pixel + marker_radius,
                        y_pixel + marker_radius,
                    )
                )

        passes = 6 if max_distance_points is not None else 3
        for pass_index in range(passes):
            texts = (label_texts[::-1] if max_distance_points is not None
                     and pass_index % 2 else label_texts)
            for text in texts:
                preferred = inward_label_offset(ax, float(text.xy[0]))
                candidates = [preferred, ((preferred[0][0], -3), preferred[1], "top")]
                candidates.extend(
                    [
                        ((3, 3), "left", "bottom"),
                        ((3, -3), "left", "top"),
                        ((-3, 3), "right", "bottom"),
                        ((-3, -3), "right", "top"),
                        ((0, 3), "center", "bottom"),
                        ((0, -3), "center", "top"),
                    ]
                )
                for distance in range(6, max_offset_points + 1, 3):
                    candidates.extend(
                        [
                            ((3, distance), "left", "bottom"),
                            ((3, -distance), "left", "top"),
                            ((-3, distance), "right", "bottom"),
                            ((-3, -distance), "right", "top"),
                            ((0, distance), "center", "bottom"),
                            ((0, -distance), "center", "top"),
                        ]
                    )
                if max_distance_points is not None:
                    candidates.extend(
                        [
                            ((3, 0), "left", "center"),
                            ((-3, 0), "right", "center"),
                            ((0, 5), "center", "bottom"),
                            ((0, -5), "center", "top"),
                        ]
                    )
                other_boxes = fixed_boxes + [
                    other.get_window_extent(renderer)
                    for other in label_texts
                    if other is not text
                ]
                if max_offset_points > 3:
                    other_boxes = [box.padded(2 * fig.dpi / 72) for box in other_boxes]
                best_position, best_score = candidates[0], (float("inf"), float("inf"))
                for offset, horizontal, vertical in candidates:
                    text.set_position(offset)
                    text.set_horizontalalignment(horizontal)
                    text.set_verticalalignment(vertical)
                    box = text.get_window_extent(renderer)
                    if max_distance_points is not None:
                        point_x, point_y = ax.transData.transform(text.xy)
                        dx = max(box.x0 - point_x, point_x - box.x1, 0.0)
                        dy = max(box.y0 - point_y, point_y - box.y1, 0.0)
                        distance = (dx * dx + dy * dy) ** 0.5 * 72 / fig.dpi
                        if distance > max_distance_points:
                            continue
                    score = (
                        max(0.0, box.width * box.height - overlap_area(box, axis_bbox)),
                        sum(overlap_area(box, other) for other in other_boxes),
                    )
                    if score < best_score:
                        best_position, best_score = (
                            (offset, horizontal, vertical),
                            score,
                        )
                offset, horizontal, vertical = best_position
                text.set_position(offset)
                text.set_horizontalalignment(horizontal)
                text.set_verticalalignment(vertical)
