from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.daily.grouping import (
    GroupingError,
    group_by_day_of_week,
    group_by_vol_state,
)
from experiments.daily.run import main as daily_main


def _make_returns_frame(start: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="B")
    rng = np.random.default_rng(42)
    data = rng.normal(scale=0.01, size=(periods, 4))
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(4)])


def test_group_by_day_of_week_balances() -> None:
    frame = _make_returns_frame("2024-01-01", 25)  # spans multiple weeks
    trimmed, labels = group_by_day_of_week(frame, min_weeks=4)
    # Drops partial boundary weeks, leaving four complete weeks (20 rows)
    assert trimmed.shape[0] == 20
    assert set(labels) == {0, 1, 2, 3, 4}
    counts = {label: int(np.sum(labels == label)) for label in set(labels)}
    assert len({count for count in counts.values()}) == 1
    assert all(count == 4 for count in counts.values())


def test_group_by_day_of_week_requires_full_weeks() -> None:
    frame = _make_returns_frame("2024-01-01", 23)  # missing two days
    with pytest.raises(GroupingError):
        group_by_day_of_week(frame, min_weeks=5)


def test_group_by_vol_state_balances() -> None:
    frame = _make_returns_frame("2024-02-01", 45)
    vol_values = np.concatenate([
        np.full(15, 0.4),
        np.full(15, 0.9),
        np.full(15, 1.8),
    ])
    vol_proxy = pd.Series(vol_values, index=frame.index)
    trimmed, labels = group_by_vol_state(
        frame,
        vol_proxy=vol_proxy,
        calm_threshold=0.6,
        crisis_threshold=1.4,
        min_replicates=5,
    )
    assert trimmed.shape[0] == 45
    assert set(labels) == {0, 1, 2}
    counts = {label: int(np.sum(labels == label)) for label in set(labels)}
    assert len(set(counts.values())) == 1
    assert counts[0] == counts[1] == counts[2] == 15


def test_group_by_vol_state_enforces_min_replicates() -> None:
    frame = _make_returns_frame("2024-02-01", 45)
    vol_values = np.concatenate([
        np.full(15, 0.35),
        np.full(15, 0.95),
        np.full(15, 1.75),
    ])
    vol_proxy = pd.Series(vol_values, index=frame.index)
    with pytest.raises(GroupingError):
        group_by_vol_state(
            frame,
            vol_proxy=vol_proxy,
            calm_threshold=0.6,
            crisis_threshold=1.4,
            min_replicates=16,
        )


def test_daily_cli_forwards_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    returns_csv = tmp_path / "returns.csv"
    frame = _make_returns_frame("2024-03-01", 30)
    frame.reset_index().rename(columns={"index": "date"}).to_csv(returns_csv, index=False)

    captured: dict[str, list[str]] = {}

    def fake_eval_main(args: list[str]) -> None:
        captured["args"] = args

    monkeypatch.setattr("experiments.daily.run.eval_main", fake_eval_main)

    daily_main([
        "--returns-csv",
        str(returns_csv),
        "--design",
        "dow",
        "--rc-date",
        "20251103",
    ])

    forwarded = captured["args"]
    assert "--group-design" in forwarded and forwarded[forwarded.index("--group-design") + 1] == "dow"
    assert "--edge-mode" in forwarded and forwarded[forwarded.index("--edge-mode") + 1] == "tyler"
    assert "--out" in forwarded
    out_path = Path(forwarded[forwarded.index("--out") + 1])
    assert out_path.parts[-3:] == ("reports", "rc-20251103", "dow")


def test_daily_cli_group_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    returns_csv = tmp_path / "returns.csv"
    frame = _make_returns_frame("2024-05-01", 30)
    frame.reset_index().rename(columns={"index": "date"}).to_csv(returns_csv, index=False)

    captured: dict[str, list[str]] = {}

    def fake_eval_main(args: list[str]) -> None:
        captured["args"] = args

    monkeypatch.setattr("experiments.daily.run.eval_main", fake_eval_main)

    daily_main([
        "--returns-csv",
        str(returns_csv),
        "--group",
        "vol",
        "--rc-date",
        "20251103",
    ])

    forwarded = captured["args"]
    assert "--group-design" in forwarded
    assert forwarded[forwarded.index("--group-design") + 1] == "vol"


def test_daily_cli_group_conflict(tmp_path: Path) -> None:
    returns_csv = tmp_path / "returns.csv"
    frame = _make_returns_frame("2024-06-01", 30)
    frame.reset_index().rename(columns={"index": "date"}).to_csv(returns_csv, index=False)

    with pytest.raises(SystemExit):
        daily_main([
            "--returns-csv",
            str(returns_csv),
            "--design",
            "dow",
            "--group",
            "vol",
        ])


def test_daily_cli_forwards_prewhiten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    returns_csv = tmp_path / "returns.csv"
    frame = _make_returns_frame("2024-04-01", 30)
    frame.reset_index().rename(columns={"index": "date"}).to_csv(returns_csv, index=False)

    captured: dict[str, list[str]] = {}

    def fake_eval_main(args: list[str]) -> None:
        captured["args"] = args

    monkeypatch.setattr("experiments.daily.run.eval_main", fake_eval_main)

    daily_main([
        "--returns-csv",
        str(returns_csv),
        "--design",
        "vol",
        "--rc-date",
        "20251103",
        "--prewhiten",
        "off",
    ])

    forwarded = captured["args"]
    assert "--prewhiten" in forwarded
    assert forwarded[forwarded.index("--prewhiten") + 1] == "off"


def test_group_by_day_of_week_three_year_slice() -> None:
    frame = _make_returns_frame("2019-01-01", 3 * 252)
    trimmed, labels = group_by_day_of_week(frame, min_weeks=10)
    assert trimmed.shape[0] >= 500
    unique_labels = np.unique(labels)
    assert unique_labels.size == 5
    counts = [int(np.sum(labels == label)) for label in unique_labels]
    assert min(counts) == max(counts)
