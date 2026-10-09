"""Presentation filters for the small RDF, pressure, and VDOS panels."""

from math import isclose, isfinite


def display_error_percent(error: float) -> bool:
    """Keep finite errors below 100% as candidates for model curves only."""
    return isfinite(error) and error < 100.0 and not isclose(
        error, 100.0, rel_tol=0.0, abs_tol=1e-8
    )
