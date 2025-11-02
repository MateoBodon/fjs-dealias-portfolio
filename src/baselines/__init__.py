"""
Baseline covariance estimators used for comparison in the de-aliasing overlay project.

The full implementations are populated across sprint milestones.  This package
initially exposes stubs so that import sites and tests can be scaffolded before
the numerical routines are wired up.
"""

from __future__ import annotations

__all__ = [
    "RIEConfig",
    "rie_covariance",
    "EWMAConfig",
    "ewma_covariance",
]

from .rie import RIEConfig, rie_covariance
from .ewma import EWMAConfig, ewma_covariance
