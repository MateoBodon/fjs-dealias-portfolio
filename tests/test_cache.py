from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import meta.cache as meta_cache
from experiments.equity_panel import run as equity_run

pytestmark = pytest.mark.integration


def _write_returns_csv(path: Path, *, seed: int = 1) -> Path:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=30)
    tickers = [f"U{idx:02d}" for idx in range(6)]
    records: list[dict[str, object]] = []
    for date in dates:
        for ticker in tickers:
            records.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "ret": float(rng.normal(scale=0.012)),
                }
            )
    frame = pd.DataFrame(records)
    out_path = path / "returns_cache.csv"
    frame.to_csv(out_path, index=False)
    return out_path


def test_window_cache_resume(monkeypatch, tmp_path: Path) -> None:
    returns_path = _write_returns_csv(tmp_path)
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    config_payload = {
        "data_path": str(returns_path),
        "start_date": "2023-01-02",
        "end_date": "2023-02-24",
        "window_weeks": 4,
        "horizon_weeks": 1,
        "output_dir": str(output_dir),
        "dealias_delta": 0.0,
        "dealias_delta_frac": 0.02,
        "dealias_eps": 0.03,
        "stability_eta_deg": 0.4,
        "signed_a": True,
        "cs_drop_top_frac": 0.05,
        "a_grid": 72,
        "partial_week_policy": "drop",
    }
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_payload, handle)

    cache_dir = tmp_path / "cache"

    save_calls = {"count": 0}
    load_calls = {"count": 0}

    orig_save = meta_cache.save_window
    orig_load = meta_cache.load_window

    def counting_save(cache_path: Path, key: str, payload: dict[str, object]) -> None:
        save_calls["count"] += 1
        orig_save(cache_path, key, payload)

    def counting_load(cache_path: Path, key: str):
        load_calls["count"] += 1
        return orig_load(cache_path, key)

    monkeypatch.setattr(meta_cache, "save_window", counting_save)
    monkeypatch.setattr(meta_cache, "load_window", counting_load)
    monkeypatch.setattr(equity_run, "save_window", counting_save)
    monkeypatch.setattr(equity_run, "load_window", counting_load)

    equity_run.run_experiment(
        config_path=config_path,
        progress_override=False,
        precompute_panel=True,
        cache_dir_override=str(cache_dir),
        resume_cache=False,
    )

    assert save_calls["count"] > 0, "Expected cache writes during initial run"
    assert any(cache_dir.glob("**/*.json")), "Cache JSON files not found"

    save_calls["count"] = 0
    load_calls["count"] = 0

    equity_run.run_experiment(
        config_path=config_path,
        progress_override=False,
        precompute_panel=True,
        cache_dir_override=str(cache_dir),
        resume_cache=True,
    )

    assert load_calls["count"] > 0, "Expected cache loads when resuming"
    assert save_calls["count"] == 0, "Unexpected cache writes during resume"
