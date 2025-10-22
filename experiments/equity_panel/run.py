from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path | str) -> dict[str, Any]:
    """Load the equity experiment configuration from disk."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping.")
    return data


def run_experiment(config_path: Path | str | None = None) -> None:
    """Execute the equity panel portfolio experiment."""
    path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).with_name("config.yaml")
    )
    config = load_config(path)
    raise NotImplementedError(
        "Equity panel de-aliasing experiment is not implemented yet. "
        f"Loaded config keys: {', '.join(sorted(config))}."
    )


def main() -> None:
    """Entry point for CLI execution."""
    run_experiment()


if __name__ == "__main__":
    main()
