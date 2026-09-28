from pathlib import Path

import chainladder as cl
import numpy as np
import pandas as pd
import pytest

from reserving import data, methods
from reserving.config import Config, default_config

ROOT = Path(__file__).resolve().parents[1]

# 3x3 cumulative triangle small enough to work by hand:
#   ldf 12-24 = (150 + 168) / (100 + 110) = 1.514285...
#   ldf 24-36 = 165 / 150 = 1.1
SMALL = np.array(
    [
        [100.0, 150.0, 165.0],
        [110.0, 168.0, np.nan],
        [120.0, np.nan, np.nan],
    ]
)


@pytest.fixture
def small_array() -> np.ndarray:
    return SMALL.copy()


@pytest.fixture
def small_triangle() -> cl.Triangle:
    rows = [
        {"AccidentYear": 2000 + i, "DevelopmentYear": 2000 + i + j, "paid": v}
        for i, row in enumerate(SMALL)
        for j, v in enumerate(row)
        if not np.isnan(v)
    ]
    return data.to_triangle(pd.DataFrame(rows), ["paid"])


@pytest.fixture(scope="session")
def config() -> Config:
    return default_config(ROOT)


@pytest.fixture(scope="session")
def portfolio(config: Config) -> pd.DataFrame:
    raw = data.load_claims(config.data_path)
    return data.aggregate_portfolio(raw, [config.loss_column, config.premium_column])


@pytest.fixture(scope="session")
def triangle(portfolio: pd.DataFrame, config: Config) -> cl.Triangle:
    return data.to_triangle(portfolio, [config.loss_column, config.premium_column])


@pytest.fixture(scope="session")
def losses(triangle: cl.Triangle, config: Config) -> cl.Triangle:
    return triangle[config.loss_column]


@pytest.fixture(scope="session")
def premium(triangle: cl.Triangle, config: Config) -> cl.Triangle:
    return triangle[config.premium_column]


@pytest.fixture(scope="session")
def developed(losses: cl.Triangle) -> cl.Triangle:
    return methods.develop(losses)


@pytest.fixture(scope="session")
def models(developed: cl.Triangle, premium: cl.Triangle, config: Config) -> dict:
    return methods.fit_methods(developed, premium.latest_diagonal, config.apriori_elr)
