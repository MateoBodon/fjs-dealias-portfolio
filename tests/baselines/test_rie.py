from __future__ import annotations

import numpy as np
import pytest

from src.baselines.rie import RIEConfig, ries_covariance


def test_rie_config_defaults() -> None:
    config = RIEConfig()
    assert config.min_eigenvalue > 0.0


def test_rie_stub_raises_not_implemented() -> None:
    data = np.zeros((5, 3))
    with pytest.raises(NotImplementedError):
        ries_covariance(data)
