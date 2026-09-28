"""The hand-built chain ladder must agree with chainladder."""

import chainladder as cl
import numpy as np
import pytest

from reserving import manual, methods
from reserving.data import origin_series


def test_small_triangle_by_hand(small_array: np.ndarray) -> None:
    ldfs = manual.volume_weighted_ldfs(small_array)
    np.testing.assert_allclose(ldfs, [318 / 210, 1.1])
    expected = [165.0, 168 * 1.1, 120 * 318 / 210 * 1.1]
    np.testing.assert_allclose(manual.chainladder_ultimates(small_array), expected)


def test_cdfs_apply_tail() -> None:
    cdfs = manual.cumulative_dev_factors(np.array([2.0, 1.5]), tail=1.1)
    np.testing.assert_allclose(cdfs, [3.3, 1.65, 1.1])


def test_small_triangle_matches_chainladder(
    small_array: np.ndarray, small_triangle: cl.Triangle
) -> None:
    ultimates = origin_series(cl.Chainladder().fit(small_triangle).ultimate_)
    np.testing.assert_allclose(manual.chainladder_ultimates(small_array), ultimates)


def test_ldfs_match_chainladder(losses: cl.Triangle) -> None:
    expected = methods.fit_development(losses).ldf_.values.ravel()
    np.testing.assert_allclose(
        manual.volume_weighted_ldfs(losses.values[0, 0]), expected, rtol=1e-10
    )


def test_ultimates_match_chainladder_without_tail(losses: cl.Triangle) -> None:
    model = cl.Chainladder().fit(losses)
    np.testing.assert_allclose(
        manual.chainladder_ultimates(losses.values[0, 0]),
        origin_series(model.ultimate_),
        rtol=1e-10,
    )


@pytest.mark.parametrize("curve", ["exponential", "inverse_power"])
def test_ultimates_match_chainladder_with_tail(losses: cl.Triangle, curve: str) -> None:
    dev = methods.fit_development(losses).transform(losses)
    tail = float(methods.fit_tail(dev, curve).tail_.iloc[0, 0])
    model = methods.fit_chainladder(methods.develop(losses, curve=curve))
    np.testing.assert_allclose(
        manual.chainladder_ultimates(losses.values[0, 0], tail=tail),
        origin_series(model.ultimate_),
        rtol=1e-10,
    )
