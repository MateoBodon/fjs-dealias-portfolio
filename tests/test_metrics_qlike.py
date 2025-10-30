from __future__ import annotations

import numpy as np

from evaluation.evaluate import qlike


def test_qlike_matches_manual_computation() -> None:
    forecasts = np.array([0.5, 1.0, 2.0])
    realised = np.array([0.4, 1.5, 1.0])
    expected = np.log(forecasts) + realised / forecasts
    result = qlike(forecasts, realised)
    np.testing.assert_allclose(result, expected)
