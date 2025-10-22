from __future__ import annotations

import numpy as np
import pytest

from fjs.balanced import BalancedConfig, compute_balanced_weights


def test_compute_balanced_weights_is_stub() -> None:
    config = BalancedConfig(regularization=0.1, max_iter=10)
    with pytest.raises(NotImplementedError):
        compute_balanced_weights(np.ones((2, 2)), config)
