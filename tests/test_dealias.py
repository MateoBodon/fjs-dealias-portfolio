from __future__ import annotations

import numpy as np
import pytest

from fjs.dealias import DealiasingResult, dealias_covariance


def test_dealiasing_result_structure() -> None:
    fields = getattr(DealiasingResult, "__dataclass_fields__", None)
    assert fields is not None


def test_dealias_covariance_is_stub() -> None:
    covariance = np.eye(3)
    spectrum = np.ones(3)
    with pytest.raises(NotImplementedError):
        dealias_covariance(covariance, spectrum)
