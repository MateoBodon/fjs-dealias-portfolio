from __future__ import annotations

import pandas as pd
import pytest

from evaluation import check_dealiased_applied

pytestmark = pytest.mark.unit


def test_check_dealiased_applied_raises_on_identical_forecasts() -> None:
    df = pd.DataFrame(
        {
            "n_detections": [0, 1, 2],
            "eq_aliased_forecast": [0.1, 0.2, 0.3],
            "eq_dealiased_forecast": [0.1, 0.2, 0.3],  # identical
        }
    )
    try:
        check_dealiased_applied(df)
    except AssertionError:
        pass
    else:  # pragma: no cover - ensure failure in case assertion not triggered
        raise AssertionError("Expected an AssertionError for identical forecasts.")


def test_check_dealiased_applied_passes_when_different() -> None:
    df = pd.DataFrame(
        {
            "n_detections": [0, 1, 0],
            "eq_aliased_forecast": [0.1, 0.2, 0.3],
            "eq_dealiased_forecast": [0.1, 0.25, 0.3],  # differs when n_detections>0
        }
    )
    check_dealiased_applied(df)  # should not raise
