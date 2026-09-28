from pathlib import Path

import chainladder as cl
import pandas as pd
import pytest

from reserving import data
from reserving.config import Config

COLUMNS = ["CumPaidLoss", "EarnedPremNet"]


def _cells(**overrides: list[float]) -> pd.DataFrame:
    base = {
        "AccidentYear": [2000, 2000, 2001],
        "DevelopmentYear": [2000, 2001, 2001],
        "CumPaidLoss": [10.0, 15.0, 12.0],
        "EarnedPremNet": [20.0, 20.0, 22.0],
    }
    return pd.DataFrame(base | overrides)


def test_export_sample_writes_only_requested_lob(tmp_path: Path) -> None:
    path = tmp_path / "comauto.csv"
    data.export_sample("comauto", path)
    exported = pd.read_csv(path)
    assert set(exported["LOB"]) == {"comauto"}
    assert len(exported) == 8690


def test_raw_data_has_expected_columns(config: Config) -> None:
    raw = data.load_claims(config.data_path)
    assert {"GRCODE", "AccidentYear", "DevelopmentYear", *COLUMNS} <= set(raw.columns)


def test_aggregate_sums_companies() -> None:
    raw = pd.DataFrame(
        {
            "AccidentYear": [2000, 2000],
            "DevelopmentYear": [2000, 2000],
            "CumPaidLoss": [1.0, 2.0],
            "EarnedPremNet": [5.0, 6.0],
        }
    )
    result = data.aggregate_portfolio(raw, COLUMNS)
    assert result[COLUMNS].iloc[0].tolist() == [3.0, 11.0]


def test_portfolio_is_valid(portfolio: pd.DataFrame) -> None:
    data.validate_claims(portfolio, COLUMNS)
    assert len(portfolio) == 55


@pytest.mark.parametrize(
    ("df", "message"),
    [
        (_cells().drop(columns="EarnedPremNet"), "missing columns"),
        (_cells(CumPaidLoss=[10.0, -1.0, 12.0]), "negative"),
        (_cells(DevelopmentYear=[2000, 2000, 2001]), "duplicate"),
        (_cells(DevelopmentYear=[2000, 2001, 2000]), "before accident year"),
    ],
)
def test_validate_rejects_bad_data(df: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        data.validate_claims(df, COLUMNS)


def test_triangle_shape(triangle: cl.Triangle) -> None:
    assert triangle.shape == (1, 2, 10, 10)


def test_origin_series_keyed_by_year(losses: cl.Triangle) -> None:
    series = data.origin_series(losses.latest_diagonal)
    assert series.index.tolist() == list(range(1988, 1998))
    assert series.loc[1988] == 626_097
