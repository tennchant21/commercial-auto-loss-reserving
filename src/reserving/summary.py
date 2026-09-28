"""Turn fitted chainladder models into tidy tables."""

from collections.abc import Mapping

import chainladder as cl
import pandas as pd
from chainladder.methods.base import MethodBase

from reserving.data import origin_series


def ultimates_table(
    models: Mapping[str, MethodBase], latest: cl.Triangle
) -> pd.DataFrame:
    """Ultimate losses by accident year for each method.

    Args:
        models: Fitted models keyed by name.
        latest: Latest diagonal of the loss triangle.

    Returns:
        A ``latest`` column followed by one column per method.
    """
    columns = {"latest": origin_series(latest)}
    columns |= {name: origin_series(m.ultimate_) for name, m in models.items()}
    return pd.DataFrame(columns)


def ibnr_table(models: Mapping[str, MethodBase]) -> pd.DataFrame:
    """IBNR by accident year for each method, with a total row.

    Args:
        models: Fitted models keyed by name.

    Returns:
        One column per method, accident years plus ``Total``.
    """
    table = pd.DataFrame({name: origin_series(m.ibnr_) for name, m in models.items()})
    table.loc["Total"] = table.sum()
    return table


def loss_ratio_table(ultimates: pd.DataFrame, premium: pd.Series) -> pd.DataFrame:
    """Divide every ultimate column by earned premium.

    Args:
        ultimates: Output of :func:`ultimates_table`.
        premium: Earned premium by accident year.

    Returns:
        Loss ratios with the same shape as ``ultimates``.
    """
    return ultimates.div(premium, axis=0)


def mack_summary(model: cl.MackChainladder) -> pd.DataFrame:
    """Mack reserve summary with coefficient of variation.

    Args:
        model: Fitted Mack model.

    Returns:
        Latest, IBNR, ultimate, Mack standard error and CV by accident year.
    """
    table = model.summary_.to_frame()
    table.index = table.index.year
    return table.assign(CV=table["Mack Std Err"] / table["IBNR"])


def total_mack_std_err(model: cl.MackChainladder) -> float:
    """Standard error of the total reserve across all accident years.

    Args:
        model: Fitted Mack model.

    Returns:
        Total Mack standard error.
    """
    return float(model.total_mack_std_err_.iloc[0, 0])
