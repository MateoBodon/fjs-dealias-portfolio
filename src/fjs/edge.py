"""
Robust Marčenko–Pastur edge estimation scaffolding.

The Sprint 1 implementation will provide Tyler/Huber scatter defaults with
buffering logic.  This stub captures the API and diagnostics containers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EdgeMode",
    "EdgeConfig",
    "EdgeEstimate",
    "compute_edge",
]


class EdgeMode(str, Enum):
    """Supported edge estimation back-ends."""

    SCM = "scm"
    TYLER = "tyler"
    HUBER = "huber"


@dataclass(frozen=True)
class EdgeConfig:
    """Configuration parameters for edge estimation."""

    mode: EdgeMode = EdgeMode.TYLER
    huber_c: float = 2.5
    buffer: float = 0.0
    buffer_frac: float = 0.05


@dataclass(frozen=True)
class EdgeEstimate:
    """Container for an estimated edge and accompanying diagnostics."""

    edge: float
    scale: float
    mode: EdgeMode


def compute_edge(
    observations: Iterable[Iterable[float]] | NDArray[np.float64],
    *,
    config: EdgeConfig | None = None,
) -> EdgeEstimate:
    """
    Placeholder interface for edge estimation.

    Returns
    -------
    EdgeEstimate
        Estimated edge location and diagnostics (to be implemented in Sprint 1).
    """

    raise NotImplementedError("Edge estimation will be implemented in Sprint 1.")
