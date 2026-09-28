"""End-to-end run: data to tables and figures in ``outputs/``."""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from reserving import data, diagnostics, methods, plots, summary
from reserving.config import Config


def build_tables(config: Config) -> dict[str, pd.DataFrame]:
    """Load data, fit every method and return the result tables.

    Args:
        config: Run configuration.

    Returns:
        Result tables keyed by output file stem.
    """
    columns = [config.loss_column, config.premium_column]
    portfolio = data.aggregate_portfolio(data.load_claims(config.data_path), columns)
    data.validate_claims(portfolio, columns)
    tri = data.to_triangle(portfolio, columns)
    losses, premium = tri[config.loss_column], tri[config.premium_column]

    developed = methods.develop(
        losses, average=config.development_average, curve=config.tail_curve
    )
    models = methods.fit_methods(developed, premium.latest_diagonal, config.apriori_elr)
    ultimates = summary.ultimates_table(models, losses.latest_diagonal)
    return {
        "triangle": losses.to_frame(),
        "link_ratios": diagnostics.link_ratios(losses),
        "development_summary": diagnostics.development_summary(losses),
        "selected_cdfs": developed.cdf_.to_frame(),
        "loss_ratios_to_date": diagnostics.latest_loss_ratios(losses, premium),
        "ultimates": ultimates,
        "ibnr": summary.ibnr_table(models),
        "ultimate_loss_ratios": summary.loss_ratio_table(
            ultimates, data.origin_series(premium.latest_diagonal)
        ),
        "mack_summary": summary.mack_summary(models["mack"]),
    }


def build_figures(
    tables: Mapping[str, pd.DataFrame], apriori: float
) -> dict[str, Figure]:
    """Draw the report figures from result tables.

    Args:
        tables: Output of :func:`build_tables`.
        apriori: Expected loss ratio for the reference line.

    Returns:
        Figures keyed by output file stem.
    """
    return {
        "development": plots.plot_development(tables["triangle"]),
        "ultimate_loss_ratios": plots.plot_loss_ratios(
            # Mack ultimates equal chain ladder by construction; plot once.
            tables["ultimate_loss_ratios"].drop(columns=["latest", "mack"]),
            apriori,
        ),
        "mack_ibnr": plots.plot_mack_ibnr(tables["mack_summary"]),
    }


def write_outputs(
    tables: Mapping[str, pd.DataFrame], figures: Mapping[str, Figure], out: Path
) -> None:
    """Write tables as CSV and figures as PNG.

    Args:
        tables: DataFrames keyed by file stem.
        figures: Figures keyed by file stem.
        out: Destination directory, created if missing.
    """
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv")
    for name, fig in figures.items():
        fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")


def run(config: Config) -> dict[str, pd.DataFrame]:
    """Run the full pipeline and write all outputs.

    Args:
        config: Run configuration.

    Returns:
        Result tables keyed by output file stem.
    """
    tables = build_tables(config)
    write_outputs(tables, build_figures(tables, config.apriori_elr), config.output_dir)
    return tables
