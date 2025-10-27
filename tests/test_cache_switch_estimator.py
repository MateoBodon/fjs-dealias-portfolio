from __future__ import annotations

import numpy as np
import pytest

from data.panels import PanelManifest
from meta.cache import load_window, save_window, window_key


@pytest.mark.unit
def test_window_cache_key_changes_with_estimator(tmp_path) -> None:
    manifest = PanelManifest(
        asset_count=10,
        weeks=4,
        days_per_week=5,
        dropped_weeks=0,
        imputed_weeks=0,
        partial_week_policy="drop",
        start_week="2024-01-01",
        end_week="2024-01-29",
        data_hash="hash-data",
        universe_hash="hash-universe",
        preprocess_flags={"winsorize": "0.01"},
    )

    week_list = ["2024-01-01", "2024-01-08"]
    tickers = ["A", "B", "C"]
    replicates = 5
    base_kwargs = {
        "code_signature": "deadbeef",
        "design": "oneway",
        "nested_replicates": None,
        "oneway_a_solver": "auto",
        "preprocess_flags": manifest.preprocess_flags,
    }

    key_lw = window_key(
        manifest,
        week_list,
        tickers,
        replicates,
        estimator="lw",
        **base_kwargs,
    )
    key_oas = window_key(
        manifest,
        week_list,
        tickers,
        replicates,
        estimator="oas",
        **base_kwargs,
    )

    assert key_lw != key_oas

    payload = {"alpha": np.arange(3, dtype=np.float64), "beta": 1.23}
    cache_dir = tmp_path / "cache"
    save_window(cache_dir, key_lw, payload)

    cached_same = load_window(cache_dir, key_lw)
    assert cached_same is not None
    assert np.allclose(cached_same["alpha"], payload["alpha"])
    assert cached_same["beta"] == payload["beta"]

    cached_other = load_window(cache_dir, key_oas)
    assert cached_other is None
