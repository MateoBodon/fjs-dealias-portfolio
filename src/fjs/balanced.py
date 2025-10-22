from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BalancedConfig:
    """Configuration for the balanced risk contribution solver."""

    regularization: float
    max_iter: int
    tolerance: float = 1e-6
    mode: Literal["long-only", "long-short"] = "long-only"

    def __post_init__(self) -> None:
        if self.regularization <= 0:
            raise ValueError("regularization must be positive.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if self.mode not in {"long-only", "long-short"}:
            raise ValueError("mode must be either 'long-only' or 'long-short'.")


def compute_balanced_weights(
    returns: NDArray[np.float64],
    config: BalancedConfig,
) -> NDArray[np.float64]:
    """
    Compute portfolio weights that balance the contribution of estimated MANOVA spikes.

    Parameters
    ----------
    returns:
        Matrix of asset returns shaped `(n_samples, n_assets)`.
    config:
        Hyper-parameters controlling convergence and regularisation.

    Returns
    -------
    numpy.ndarray
        Array of weights shaped `(n_assets,)`.

    Raises
    ------
    NotImplementedError
        Always raised until the solver is implemented in a subsequent milestone.
    """
    raise NotImplementedError("Balanced weight computation is not implemented yet.")
