from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from report.gather import collect_estimator_panel, find_runs, load_run

pytestmark = pytest.mark.unit

FIXTURE_RUN = Path(__file__).parent / "report_fixtures" / "sample_run"


def test_load_run_frames() -> None:
    frames = load_run(FIXTURE_RUN)
    assert not frames["metrics"].empty
    assert frames["metrics"].shape[0] == 5
    assert pytest.approx(frames["summary"]["detection_rate"].iloc[0], rel=1e-6) == 0.75
    keys = set(frames.keys())
    assert {"metrics", "rolling", "summary", "run_path"}.issubset(keys)


def test_find_runs_prefers_tagged(tmp_path: Path) -> None:
    (tmp_path / "plain_run").mkdir()
    tagged = tmp_path / "oneway_J5_solver-auto_est-lw_prep-none"
    tagged.mkdir()
    (tmp_path / "another").mkdir()

    discovered = find_runs(tmp_path)
    assert discovered == [tagged.resolve()]

    pattern_runs = find_runs(tmp_path, pattern="oneway*")
    assert pattern_runs == [tagged.resolve()]


def test_collect_estimator_panel() -> None:
    df = collect_estimator_panel([FIXTURE_RUN])
    assert not df.empty
    lw_eq = df[(df["estimator"] == "Ledoit-Wolf") & (df["strategy"] == "Equal Weight")].iloc[0]
    assert pytest.approx(lw_eq["delta_mse_vs_de"], rel=1e-6) == 0.1
    assert pytest.approx(lw_eq["detection_rate"], rel=1e-6) == 0.75
    assert pytest.approx(lw_eq["edge_margin_median"], rel=1e-6) == 0.002
    assert lw_eq["crisis_label"] == "sample"

    oas_eq = df[(df["estimator"] == "OAS") & (df["strategy"] == "Equal Weight")].iloc[0]
    assert pytest.approx(oas_eq["dm_p"], rel=1e-6) == 0.32

    mv_lw = df[(df["estimator"] == "Ledoit-Wolf") & (df["strategy"] == "Min-Variance (long-only)")].iloc[0]
    assert pytest.approx(mv_lw["delta_mse_vs_de"], rel=1e-6) == 0.09999999999999998
    assert mv_lw["n_windows"] == 4
