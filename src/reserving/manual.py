"""Hand-built chain ladder used only to cross-check chainladder.

Operates on a plain numpy array: rows are accident years, columns are
development ages, and unobserved cells are ``NaN``.
"""

import numpy as np


def volume_weighted_ldfs(cumulative: np.ndarray) -> np.ndarray:
    """Volume-weighted age-to-age factors.

    Args:
        cumulative: Cumulative triangle with ``NaN`` in the unobserved corner.

    Returns:
        One factor per development interval (``n_cols - 1`` values).
    """
    current, following = cumulative[:, :-1], cumulative[:, 1:]
    observed = ~np.isnan(following)
    numerator = np.where(observed, following, 0.0).sum(axis=0)
    denominator = np.where(observed, current, 0.0).sum(axis=0)
    return numerator / denominator


def cumulative_dev_factors(ldfs: np.ndarray, tail: float = 1.0) -> np.ndarray:
    """Age-to-ultimate factors from age-to-age factors.

    Args:
        ldfs: Age-to-age factors.
        tail: Factor from the last observed age to ultimate.

    Returns:
        One factor per development age (``len(ldfs) + 1`` values).
    """
    with_tail = np.append(ldfs, tail)
    return np.cumprod(with_tail[::-1])[::-1]


def latest_diagonal(cumulative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Latest observed value and its development column for each row.

    Args:
        cumulative: Cumulative triangle with ``NaN`` in the unobserved corner.

    Returns:
        Latest values and their column indices.
    """
    ages = (~np.isnan(cumulative)).sum(axis=1) - 1
    return cumulative[np.arange(len(cumulative)), ages], ages


def chainladder_ultimates(cumulative: np.ndarray, tail: float = 1.0) -> np.ndarray:
    """Chain ladder ultimates: latest diagonal times age-to-ultimate factor.

    Args:
        cumulative: Cumulative triangle with ``NaN`` in the unobserved corner.
        tail: Factor from the last observed age to ultimate.

    Returns:
        Ultimate losses by accident year.
    """
    cdfs = cumulative_dev_factors(volume_weighted_ldfs(cumulative), tail)
    latest, ages = latest_diagonal(cumulative)
    return latest * cdfs[ages]
