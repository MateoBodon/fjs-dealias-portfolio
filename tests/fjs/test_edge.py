from __future__ import annotations

import numpy as np
import pytest

from src.fjs.edge import EdgeConfig, EdgeMode, compute_edge


def test_edge_stub_exposes_enum() -> None:
    assert EdgeMode.TYLER.value == "tyler"


def test_edge_stub_raises_not_implemented() -> None:
    data = np.zeros((5, 3))
    config = EdgeConfig()
    with pytest.raises(NotImplementedError):
        compute_edge(data, config=config)
