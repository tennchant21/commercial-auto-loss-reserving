"""Figures built on the object-oriented matplotlib API (no global state)."""

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


def plot_development(cumulative: pd.DataFrame) -> Figure:
    """Cumulative losses by development age, one line per accident year.

    Args:
        cumulative: Accident year by development age, as from ``to_frame()``.

    Returns:
        Figure with one line per accident year.
    """
    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    for year, row in cumulative.iterrows():
        ax.plot(row.index.astype(int), row.to_numpy(), marker="o", label=year.year)
    ax.set(xlabel="Development age (months)", ylabel="Cumulative paid losses")
    ax.legend(title="Accident year", fontsize="small", ncols=2)
    return fig


def plot_loss_ratios(loss_ratios: pd.DataFrame, apriori: float) -> Figure:
    """Ultimate loss ratio by accident year for each method.

    Args:
        loss_ratios: Output of ``summary.loss_ratio_table``.
        apriori: Expected loss ratio, drawn as a reference line.

    Returns:
        Figure with one line per method.
    """
    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    for name, series in loss_ratios.items():
        ax.plot(series.index, series.to_numpy(), marker="o", label=name)
    ax.axhline(apriori, color="grey", linestyle="--", label="a priori ELR")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set(xlabel="Accident year", ylabel="Loss ratio")
    ax.legend()
    return fig


def plot_mack_ibnr(mack: pd.DataFrame) -> Figure:
    """Mack IBNR with a two-standard-error band by accident year.

    Args:
        mack: Output of ``summary.mack_summary``.

    Returns:
        Bar chart with error bars.
    """
    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    ax.bar(mack.index, mack["IBNR"], yerr=2 * mack["Mack Std Err"], capsize=3)
    ax.set(xlabel="Accident year", ylabel="IBNR (±2 standard errors)")
    return fig
