import chainladder as cl
import numpy as np
import pytest

from reserving import diagnostics, summary
from reserving.data import origin_series


def test_ultimates_equal_latest_plus_ibnr(models: dict, losses: cl.Triangle) -> None:
    ultimates = summary.ultimates_table(models, losses.latest_diagonal)
    ibnr = summary.ibnr_table(models).drop(index="Total")
    for name in models:
        np.testing.assert_allclose(
            ultimates[name], ultimates["latest"] + ibnr[name], rtol=1e-10
        )


def test_ibnr_total_row(models: dict) -> None:
    table = summary.ibnr_table(models)
    np.testing.assert_allclose(
        table.loc["Total"], table.drop(index="Total").sum(), rtol=1e-12
    )


def test_loss_ratio_table(
    models: dict, losses: cl.Triangle, premium: cl.Triangle
) -> None:
    ultimates = summary.ultimates_table(models, losses.latest_diagonal)
    ratios = summary.loss_ratio_table(ultimates, origin_series(premium.latest_diagonal))
    assert ratios.shape == ultimates.shape
    assert ratios["bf"].between(0.5, 0.9).all()


def test_mack_summary(models: dict) -> None:
    table = summary.mack_summary(models["mack"])
    assert {"Latest", "IBNR", "Ultimate", "Mack Std Err", "CV"} <= set(table.columns)
    assert table.index.tolist() == list(range(1988, 1998))
    assert summary.total_mack_std_err(models["mack"]) > table["Mack Std Err"].max()


def test_latest_loss_ratios(losses: cl.Triangle, premium: cl.Triangle) -> None:
    table = diagnostics.latest_loss_ratios(losses, premium)
    assert table.loc[1988, "loss_ratio"] == pytest.approx(626_097 / 913_636)


def test_development_summary(losses: cl.Triangle) -> None:
    table = diagnostics.development_summary(losses)
    assert table.index.tolist() == ["simple", "volume", "volume_5yr"]
    assert (table > 1).all().all()
