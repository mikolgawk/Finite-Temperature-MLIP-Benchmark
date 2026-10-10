"""Presentation filters for the small RDF, pressure, and VDOS panels."""

from math import isclose, isfinite


def display_error_percent(error: float) -> bool:
    """Keep finite errors below 100% as candidates for model curves only."""
    return isfinite(error) and error < 100.0 and not isclose(
        error, 100.0, rel_tol=0.0, abs_tol=1e-8
    )


def display_full_error_legend(error: float, *, has_data: bool) -> bool:
    """Show a legend-only 100% score when valid data exists for a successful run."""
    return has_data and isfinite(error) and isclose(
        error, 100.0, rel_tol=0.0, abs_tol=1e-8
    )


def format_subplot_error_percent(error: float) -> str:
    """Keep a rounded, near-100% curve label distinct from an exact full error."""
    label = f"{error:.1f}%"
    return "<100.0%" if label == "100.0%" and display_error_percent(error) else label


def fit_subplot_legends(fig, grid, *, first_panel_row: int) -> None:
    """Wrap oversized legend labels and give crowded panel rows enough height."""
    for iteration in range(2):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ratios = list(grid.get_height_ratios())
        scales = [1.0] * len(ratios)
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend is None:
                continue
            row = ax.get_subplotspec().get_topmost_subplotspec().rowspan.start
            if row < first_panel_row:
                continue
            if iteration == 0:
                texts = legend.get_texts()
                padding = legend.get_window_extent(renderer).width - max(
                    text.get_window_extent(renderer).width for text in texts
                )
                panel_width = ax.get_subplotspec().get_topmost_subplotspec().get_position(fig).width
                available_width = panel_width * fig.bbox.width - padding - 4
                for text in texts:
                    lines, line = [], ""
                    for word in text.get_text().split():
                        candidate = f"{line} {word}".strip()
                        width = renderer.get_text_width_height_descent(
                            candidate, text.get_fontproperties(), False
                        )[0]
                        if line and width > available_width:
                            lines.append(line)
                            line = word
                        else:
                            line = candidate
                    text.set_text("\n".join(lines + [line]))
                fig.canvas.draw()
            legend_height = legend.get_window_extent(renderer).height
            panel_height = ax.get_window_extent(renderer).height
            scales[row] = max(scales[row], (legend_height + 4) / panel_height)
        if max(scales) <= 1.0:
            return
        expanded = [ratio * scale for ratio, scale in zip(ratios, scales)]
        fig.set_size_inches(fig.get_figwidth(), fig.get_figheight() * sum(expanded) / sum(ratios))
        grid.set_height_ratios(expanded)
        if fig.get_layout_engine() is None:
            fig.subplots_adjust()
