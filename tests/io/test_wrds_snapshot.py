from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def test_wrds_returns_snapshot_columns() -> None:
    path = Path("data/wrds/returns_daily.parquet")
    if not path.exists():
        unpack = Path("scripts/data/unpack_wrds_snapshot.sh")
        if unpack.exists():
            subprocess.run(["bash", str(unpack)], check=True)
    assert path.exists(), "WRDS returns parquet missing; run Step 1 snapshot."
    columns = ["date", "permno", "ticker", "ret", "price", "volume", "shares_out", "market_cap", "exchcd", "shrcd"]
    frame = pd.read_parquet(path, columns=columns)
    assert not frame.empty, "WRDS returns snapshot unexpectedly empty."
    for column in columns:
        assert column in frame.columns, f"returns parquet missing column: {column}"


def test_wrds_labels_snapshot_columns() -> None:
    path = Path("data/wrds/labels.parquet")
    assert path.exists(), "WRDS labels parquet missing; derive labels before running tests."
    columns = ["date", "vol_signal", "dow", "dow_label", "vol_state", "calm_threshold", "crisis_threshold", "ewma_span"]
    frame = pd.read_parquet(path, columns=columns)
    assert not frame.empty, "WRDS labels snapshot unexpectedly empty."
    for column in columns:
        assert column in frame.columns, f"labels parquet missing column: {column}"
