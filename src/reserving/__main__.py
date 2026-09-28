"""Entry point: ``python -m reserving`` from the project root."""

import warnings
from pathlib import Path

from reserving.config import default_config
from reserving.pipeline import run


def main() -> None:
    """Run the pipeline with default settings and print headline figures."""
    # Upstream chainladder 0.10.1 / numpy 2.5 deprecation; see pyproject.toml.
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="chainladder")
    config = default_config(Path.cwd())
    tables = run(config)
    print(tables["ibnr"].loc["Total"].round(0).to_string())
    print(f"Mack CV by accident year:\n{tables['mack_summary']['CV'].round(3)}")
    print(f"Outputs written to {config.output_dir}")


if __name__ == "__main__":
    main()
