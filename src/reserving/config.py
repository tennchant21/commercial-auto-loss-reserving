"""Run configuration for the reserving pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Settings for a single reserving run.

    Attributes:
        data_path: CSV of Schedule P claims in long format.
        output_dir: Directory that receives tables and figures.
        loss_column: Cumulative loss column to reserve.
        premium_column: Earned premium column used as BF exposure.
        apriori_elr: Expected loss ratio for Bornhuetter-Ferguson. 0.68 is the
            paid-to-date loss ratio on the matured 1988-1990 accident years and
            the incurred-to-date loss ratio across all years; see README.
        development_average: Link-ratio averaging passed to ``cl.Development``.
        tail_curve: Curve passed to ``cl.TailCurve``.
    """

    data_path: Path
    output_dir: Path
    loss_column: str = "CumPaidLoss"
    premium_column: str = "EarnedPremNet"
    apriori_elr: float = 0.68
    development_average: str = "volume"
    tail_curve: str = "exponential"


def default_config(root: Path) -> Config:
    """Build the default configuration relative to a project root.

    Args:
        root: Project root containing ``data/`` and ``outputs/``.

    Returns:
        Config pointing at the bundled Commercial Auto extract.
    """
    return Config(
        data_path=root / "data" / "clrd_comauto.csv",
        output_dir=root / "outputs",
    )
