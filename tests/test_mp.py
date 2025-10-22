from __future__ import annotations

import numpy as np
import pytest

from fjs.mp import MarchenkoPasturModel, marchenko_pastur_edges, marchenko_pastur_pdf


def test_marchenko_pastur_edges_is_stub() -> None:
    model = MarchenkoPasturModel(n_samples=100, n_features=50)
    with pytest.raises(NotImplementedError):
        marchenko_pastur_edges(model)


def test_marchenko_pastur_pdf_is_stub() -> None:
    model = MarchenkoPasturModel(n_samples=100, n_features=50)
    grid = np.linspace(0.0, 1.0, 5)
    with pytest.raises(NotImplementedError):
        marchenko_pastur_pdf(model, grid)
