from dataclasses import replace
from pathlib import Path

from reserving.config import Config
from reserving.pipeline import run


def test_run_writes_all_outputs(config: Config, tmp_path: Path) -> None:
    tables = run(replace(config, output_dir=tmp_path))
    written = {p.name for p in tmp_path.iterdir()}
    assert {f"{name}.csv" for name in tables} <= written
    assert {"development.png", "ultimate_loss_ratios.png", "mack_ibnr.png"} <= written
