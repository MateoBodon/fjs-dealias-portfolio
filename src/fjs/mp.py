from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MarchenkoPasturModel:
    """Summary statistics for a Marchenko–Pastur limiting law."""

    n_samples: int
    n_features: int

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if self.n_features <= 0:
            raise ValueError("n_features must be positive.")

    @property
    def aspect_ratio(self) -> float:
        """Return the sample-to-feature aspect ratio."""
        return self.n_features / self.n_samples


def marchenko_pastur_edges(model: MarchenkoPasturModel) -> tuple[float, float]:
    """
    Compute the theoretical support edges for a Marchenko–Pastur distribution.

    Parameters
    ----------
    model:
        MarchenkoPasturModel describing the observation regime.

    Returns
    -------
    tuple[float, float]
        Lower and upper spectral edges.
    """
    raise NotImplementedError("Marchenko–Pastur edge computation is not implemented.")


def marchenko_pastur_pdf(
    model: MarchenkoPasturModel,
    grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Evaluate the Marchenko–Pastur density over a grid.

    Parameters
    ----------
    model:
        MarchenkoPasturModel describing the observation regime.
    grid:
        Points at which to evaluate the density.

    Returns
    -------
    numpy.ndarray
        Density values matching the grid shape.
    """
    raise NotImplementedError("Marchenko–Pastur PDF evaluation is not implemented.")
