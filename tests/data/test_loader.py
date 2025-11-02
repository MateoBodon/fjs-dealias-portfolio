from __future__ import annotations

import pytest

from src.data.loader import DailyLoaderConfig, load_daily_panel


def test_daily_loader_stub_raises_not_implemented() -> None:
    config = DailyLoaderConfig()
    with pytest.raises(NotImplementedError):
        load_daily_panel("dummy.csv", config=config)
