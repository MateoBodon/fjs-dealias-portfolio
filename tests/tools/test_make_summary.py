from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from tools.make_summary import summarise_rc_directory, write_summaries


def _copy_sample_rc(tmp_path: Path) -> Path:
    src = Path("reports/rc-20251103")
    if not src.exists():
        raise RuntimeError("Sample RC artifacts missing under reports/rc-20251103.")
    dest = tmp_path / "reports" / src.name
    shutil.copytree(src, dest)
    return dest


def test_summarise_rc_directory(tmp_path: Path) -> None:
    rc_dir = _copy_sample_rc(tmp_path)
    artifacts = summarise_rc_directory(rc_dir)

    perf_df = artifacts.performance
    assert not perf_df.empty
    required_perf_cols = {
        "delta_mse_vs_baseline",
        "var95_overlay",
        "var95_baseline",
        "dm_stat",
        "dm_p_value",
        "n_effective",
    }
    assert required_perf_cols.issubset(perf_df.columns)
    assert ((perf_df["regime"] == "full") & (perf_df["portfolio"] == "ew")).any()

    detection_df = artifacts.detection
    assert not detection_df.empty
    required_det_cols = {
        "windows",
        "detection_rate_mean",
        "isolation_share_mean",
        "edge_margin_mean",
        "stability_margin_mean",
        "alignment_cos_mean",
        "reason_code_mode",
    }
    assert required_det_cols.issubset(detection_df.columns)
    assert set(detection_df["regime"]) == {"full", "calm", "crisis"}

    write_summaries([rc_dir])
    summary_dir = rc_dir / "summary"
    perf_path = summary_dir / "summary_perf.csv"
    det_path = summary_dir / "summary_detection.csv"
    assert perf_path.exists()
    assert det_path.exists()

    kill_path = summary_dir / "kill_criteria.json"
    limits_path = summary_dir / "limitations.md"
    assert kill_path.exists()
    assert limits_path.exists()
    kill_data = json.loads(kill_path.read_text(encoding="utf-8"))
    assert "criteria" in kill_data

    loaded_det = pd.read_csv(det_path)
    assert "regime" in loaded_det.columns
    assert set(loaded_det["regime"]) == {"full", "calm", "crisis"}
