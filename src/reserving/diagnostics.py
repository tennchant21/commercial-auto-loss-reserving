"""Exploratory diagnostics on triangles."""

import chainladder as cl
import pandas as pd

from reserving.data import origin_series

AVERAGES: dict[str, dict[str, object]] = {
    "simple": {"average": "simple"},
    "volume": {"average": "volume"},
    "volume_5yr": {"average": "volume", "n_periods": 5},
}


def link_ratios(tri: cl.Triangle) -> pd.DataFrame:
    """Return age-to-age factors for a single-column triangle.

    Args:
        tri: Cumulative triangle.

    Returns:
        Accident year by development interval.
    """
    return tri.link_ratio.to_frame()


def development_summary(tri: cl.Triangle) -> pd.DataFrame:
    """Compare selected link-ratio averages side by side.

    Args:
        tri: Cumulative triangle.

    Returns:
        One row per averaging method, one column per development interval.
    """
    rows = {
        name: cl.Development(**kwargs).fit(tri).ldf_.to_frame().iloc[0]
        for name, kwargs in AVERAGES.items()
    }
    return pd.DataFrame(rows).T


def latest_loss_ratios(losses: cl.Triangle, premium: cl.Triangle) -> pd.DataFrame:
    """Loss-to-date over earned premium by accident year.

    Args:
        losses: Cumulative loss triangle.
        premium: Earned premium triangle on the same origins.

    Returns:
        Columns ``losses``, ``premium`` and ``loss_ratio`` by accident year.
    """
    table = pd.DataFrame(
        {
            "losses": origin_series(losses.latest_diagonal),
            "premium": origin_series(premium.latest_diagonal),
        }
    )
    return table.assign(loss_ratio=table["losses"] / table["premium"])
