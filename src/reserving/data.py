"""Loading, validating and shaping claims data into triangles."""

from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

import chainladder as cl
import pandas as pd

ORIGIN = "AccidentYear"
VALUATION = "DevelopmentYear"


def export_sample(lob: str, path: Path) -> None:
    """Write one line of business from chainladder's bundled CAS data to CSV.

    Args:
        lob: Line-of-business code in the ``LOB`` column, e.g. ``"comauto"``.
        path: Destination CSV path.
    """
    source = files("chainladder") / "utils" / "data" / "clrd.csv"
    raw = pd.read_csv(str(source))
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.loc[raw["LOB"] == lob].to_csv(path, index=False)


def load_claims(path: Path) -> pd.DataFrame:
    """Read a long-format claims CSV.

    Args:
        path: CSV with one row per company, accident year and valuation year.

    Returns:
        Raw claims records.
    """
    return pd.read_csv(path)


def aggregate_portfolio(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Sum company-level records to one portfolio row per triangle cell.

    Args:
        df: Raw claims records.
        columns: Value columns to sum.

    Returns:
        One row per accident year and valuation year.
    """
    return df.groupby([ORIGIN, VALUATION], as_index=False)[list(columns)].sum()


def validate_claims(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Check that portfolio records can form a valid cumulative triangle.

    Args:
        df: Aggregated claims records.
        columns: Value columns that must be present and non-negative.

    Raises:
        ValueError: If columns are missing, values are negative, cells are
            duplicated, or a valuation precedes its accident year.
    """
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if (df[list(columns)] < 0).any().any():
        raise ValueError("negative values in portfolio totals")
    if df.duplicated([ORIGIN, VALUATION]).any():
        raise ValueError("duplicate accident/valuation cells")
    if (df[VALUATION] < df[ORIGIN]).any():
        raise ValueError("valuation year before accident year")


def to_triangle(df: pd.DataFrame, columns: Sequence[str]) -> cl.Triangle:
    """Convert aggregated records into a cumulative annual triangle.

    Args:
        df: Aggregated, validated claims records.
        columns: Value columns to carry into the triangle.

    Returns:
        Triangle with one column per entry in ``columns``.
    """
    return cl.Triangle(
        df,
        origin=ORIGIN,
        development=VALUATION,
        columns=list(columns),
        cumulative=True,
    )


def origin_series(vector: cl.Triangle) -> pd.Series:
    """Flatten a single-column origin vector to a Series keyed by accident year.

    Args:
        vector: Triangle with one column and one development period, such as
            ``latest_diagonal`` or ``ultimate_``.

    Returns:
        Values indexed by integer accident year.
    """
    series = vector.to_frame().iloc[:, 0]
    series.index = series.index.year
    series.index.name = ORIGIN
    return series
