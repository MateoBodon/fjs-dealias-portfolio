"""
Daily data loader scaffolding for replicated group estimation.

The final implementation will provide daily returns pulled from on-disk sources,
winsorisation, and balanced-universe enforcement.  For now we expose a typed
interface that other modules can build upon during the sprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

__all__ = [
    "DailyLoaderConfig",
    "DailyPanel",
    "load_daily_panel",
]


@dataclass(frozen=True)
class DailyLoaderConfig:
    """Configuration knobs for the daily loader."""

    winsor_lower: float = 0.005
    winsor_upper: float = 0.995
    min_history: int = 252
    forward_fill: bool = False
    required_symbols: Sequence[str] | None = None


@dataclass(frozen=True)
class DailyPanel:
    """Container for the balanced daily panel data."""

    returns: pd.DataFrame
    meta: dict[str, object]


def load_daily_panel(
    source: str | Path | Iterable[Path],
    *,
    config: DailyLoaderConfig | None = None,
) -> DailyPanel:
    """
    Load and clean daily return data, enforcing a balanced universe.

    Parameters
    ----------
    source
        File path or iterable of paths containing raw price/return data.
    config
        Loader configuration; when omitted, defaults tuned for equity universes
        are used.

    Returns
    -------
    DailyPanel
        Cleaned daily returns along with loader metadata.
    """

    raise NotImplementedError("Daily loader will be implemented in Sprint 1.")
