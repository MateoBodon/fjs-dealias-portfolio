from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.equity_panel import run as equity_run
from tools.summarize_run import summarize_run


def _write_returns_csv(path: Path, *, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=30)
    tickers = [f"T{idx:02d}" for idx in range(8)]
    records: list[dict[str, object]] = []
    for date in dates:
        for ticker in tickers:
            records.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "ret": float(rng.normal(scale=0.01)),
                }
            )
    frame = pd.DataFrame(records)
    out_path = path / "returns.csv"
    frame.to_csv(out_path, index=False)
    return out_path


def test_window_artifacts_expose_edge_margin_and_admissible_root(tmp_path: Path) -> None:
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
    equity_run.run_experiment(
        config_path=config_path,
        progress_override=False,
        precompute_panel=True,
        cache_dir_override=str(cache_dir),
        resume_cache=False,
    )

    results_path = output_dir / "rolling_results.csv"
    assert results_path.exists(), "rolling_results.csv not produced"
    results = pd.read_csv(results_path)
    assert "top_edge_margin" in results.columns
    assert "top_admissible_root" in results.columns
    assert "detections_detail" in results.columns
    if not results.empty and results["detections_detail"].notna().any():
        first_detail = json.loads(results["detections_detail"].dropna().iloc[0])
        if first_detail:
            assert "edge_margin" in first_detail[0]
            assert "admissible_root" in first_detail[0]

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        summarize_run(output_dir)
    output_text = buffer.getvalue()
    assert "Rejection reasons:" in output_text
    rejection_lines = [line for line in output_text.splitlines() if line.strip().startswith("- ")]
    assert rejection_lines, "Expected non-empty rejection breakdown"
