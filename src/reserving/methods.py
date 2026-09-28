"""Thin wrappers over chainladder development and reserving estimators.

Every reserving method takes a triangle that already carries development
factors (see :func:`develop`) and returns a fitted chainladder estimator
exposing ``ultimate_`` and ``ibnr_``. Adding a method such as Cape Cod means
writing one ``fit_*`` function and registering it in :func:`fit_methods`.
"""

import chainladder as cl
from chainladder.methods.base import MethodBase


def fit_development(
    tri: cl.Triangle, average: str = "volume", n_periods: int = -1
) -> cl.Development:
    """Fit age-to-age development factors.

    Args:
        tri: Cumulative loss triangle.
        average: ``"volume"``, ``"simple"`` or ``"regression"``.
        n_periods: Most recent diagonals to use; ``-1`` uses all.

    Returns:
        Fitted development estimator.
    """
    return cl.Development(average=average, n_periods=n_periods).fit(tri)


def fit_tail(tri: cl.Triangle, curve: str = "exponential") -> cl.TailCurve:
    """Fit a curve to the development factors to extrapolate a tail.

    Args:
        tri: Triangle already transformed by a development estimator.
        curve: ``"exponential"`` or ``"inverse_power"``.

    Returns:
        Fitted tail estimator.
    """
    return cl.TailCurve(curve=curve).fit(tri)


def develop(
    tri: cl.Triangle, average: str = "volume", curve: str = "exponential"
) -> cl.Triangle:
    """Attach development factors and a curve-fitted tail to a triangle.

    Args:
        tri: Cumulative loss triangle.
        average: Link-ratio averaging method.
        curve: Tail curve.

    Returns:
        Triangle carrying ``ldf_`` and ``cdf_`` including the tail.
    """
    developed = fit_development(tri, average=average).transform(tri)
    return fit_tail(developed, curve=curve).transform(developed)


def fit_chainladder(developed: cl.Triangle) -> cl.Chainladder:
    """Fit the deterministic chain ladder.

    Args:
        developed: Output of :func:`develop`.

    Returns:
        Fitted chain ladder.
    """
    return cl.Chainladder().fit(developed)


def fit_mack(developed: cl.Triangle) -> cl.MackChainladder:
    """Fit Mack's chain ladder with standard errors.

    Args:
        developed: Output of :func:`develop`.

    Returns:
        Fitted Mack model.
    """
    return cl.MackChainladder().fit(developed)


def fit_bf(
    developed: cl.Triangle, exposure: cl.Triangle, apriori: float
) -> cl.BornhuetterFerguson:
    """Fit Bornhuetter-Ferguson with an expected loss ratio on exposure.

    Args:
        developed: Output of :func:`develop`.
        exposure: Earned premium by origin, e.g. a ``latest_diagonal``.
        apriori: Expected loss ratio applied to ``exposure``.

    Returns:
        Fitted Bornhuetter-Ferguson model.
    """
    return cl.BornhuetterFerguson(apriori=apriori).fit(
        developed, sample_weight=exposure
    )


def fit_methods(
    developed: cl.Triangle, exposure: cl.Triangle, apriori: float
) -> dict[str, MethodBase]:
    """Fit every configured reserving method.

    Args:
        developed: Output of :func:`develop`.
        exposure: Earned premium by origin.
        apriori: Expected loss ratio for exposure-based methods.

    Returns:
        Fitted models keyed by display name.
    """
    return {
        "chainladder": fit_chainladder(developed),
        "mack": fit_mack(developed),
        "bf": fit_bf(developed, exposure, apriori),
    }
