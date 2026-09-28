import chainladder as cl
import numpy as np
import pytest

from reserving import methods, summary
from reserving.config import Config
from reserving.data import origin_series


def _tail(losses: cl.Triangle, curve: str) -> float:
    dev = methods.fit_development(losses).transform(losses)
    return float(methods.fit_tail(dev, curve).tail_.iloc[0, 0])


def test_registry_names(models: dict) -> None:
    assert set(models) == {"chainladder", "mack", "bf"}


@pytest.mark.parametrize("name", ["chainladder", "mack", "bf"])
def test_ultimate_at_least_latest(models: dict, losses: cl.Triangle, name: str) -> None:
    latest = origin_series(losses.latest_diagonal)
    assert (origin_series(models[name].ultimate_) >= latest).all()


def test_exponential_tail_is_small_and_positive(losses: cl.Triangle) -> None:
    assert 1.0 < _tail(losses, "exponential") < 1.01


def test_inverse_power_tail_is_heavier(losses: cl.Triangle) -> None:
    assert _tail(losses, "inverse_power") > _tail(losses, "exponential")


def test_mack_ultimates_equal_chainladder(models: dict) -> None:
    np.testing.assert_allclose(
        origin_series(models["mack"].ultimate_),
        origin_series(models["chainladder"].ultimate_),
    )


def test_mack_std_err_grows_with_accident_year(models: dict) -> None:
    std_err = summary.mack_summary(models["mack"])["Mack Std Err"]
    assert (std_err > 0).all()
    assert std_err.is_monotonic_increasing


def test_bf_ibnr_proportional_to_apriori(
    developed: cl.Triangle, premium: cl.Triangle, config: Config
) -> None:
    apriori = config.apriori_elr
    exposure = premium.latest_diagonal
    base = origin_series(methods.fit_bf(developed, exposure, apriori).ibnr_)
    doubled = origin_series(methods.fit_bf(developed, exposure, 2 * apriori).ibnr_)
    np.testing.assert_allclose(doubled, 2 * base)


def test_bf_tempers_latest_year(models: dict) -> None:
    cl_ibnr = origin_series(models["chainladder"].ibnr_)
    bf_ibnr = origin_series(models["bf"].ibnr_)
    assert bf_ibnr.loc[1997] < cl_ibnr.loc[1997]
