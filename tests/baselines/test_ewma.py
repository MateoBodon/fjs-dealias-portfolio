from __future__ import annotations

import numpy as np
import pytest

from src.baselines.ewma import EWMAConfig, ewma_covariance


def test_ewma_config_defaults() -> None:
    config = EWMAConfig()
    assert 0.0 < config.lambda_ < 1.0


def test_ewma_stub_raises_not_implemented() -> None:
    data = np.zeros((5, 3))
    with pytest.raises(NotImplementedError):
        ewma_covariance(data)
