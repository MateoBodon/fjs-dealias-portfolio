"""
Memo builder scaffolding.

The final tool will assemble evaluation fragments into an RC memo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def build_memo(fragments: Iterable[Path], *, output: Path) -> None:
    """
    Combine memo fragments into a single markdown document (placeholder).
    """

    raise NotImplementedError("Memo builder will be implemented in Sprint 2.")
