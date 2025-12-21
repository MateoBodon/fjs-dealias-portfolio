from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tools.paper_v1_ablation import build_ablation_table


def _write_summary_perf(path: Path) -> None:
    records = []
    for rc_run, overlay_delta in ("scm_off", 0.0), ("scm_on", 0.123):
        for portfolio in ("ew", "mv"):
            records.append(
                {
                    "rc_run": rc_run,
                    "regime": "full",
                    "portfolio": portfolio,
                    "delta_mse_vs_baseline": overlay_delta,
                    "delta_qlike_vs_baseline": overlay_delta + 0.01,
                    "n_effective": 12,
                    "n_effective_mse": 12,
                    "n_effective_qlike": 12,
                    "comparison_valid_mse": 1,
                    "comparison_valid_qlike": 1,
                    "comparison_valid_dm": 1,
                    "comparison_valid_delta": 1,
                    "cap_active": False,
                    "cap_sources": "",
                    "window_coverage": 1.0,
                }
            )
    pd.DataFrame(records).to_csv(path, index=False)


def _write_summary_detection(path: Path) -> None:
    records = []
    for rc_run, rate in ("scm_off", 0.0), ("scm_on", 0.05):
        records.append(
            {
                "rc_run": rc_run,
                "regime": "full",
                "windows": 20,
                "detection_windows": 2,
                "detection_rate_mean": rate,
                "detection_rate_median": rate,
                "isolation_share_mean": 0.2,
                "edge_margin_mean": 0.3,
                "stability_margin_mean": 0.4,
                "alignment_cos_mean": 0.9,
                "reason_code_mode": "accepted",
                "cap_active": False,
                "cap_sources": "",
                "window_coverage": 1.0,
            }
        )
    pd.DataFrame(records).to_csv(path, index=False)


def test_build_ablation_table(tmp_path: Path) -> None:
    rc_dir = tmp_path / "rc-ablate"
    summary_dir = rc_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    perf_path = summary_dir / "summary_perf.csv"
    det_path = summary_dir / "summary_detection.csv"
    _write_summary_perf(perf_path)
    _write_summary_detection(det_path)

    out_path = build_ablation_table(rc_dir)
    assert out_path.exists()

    out_df = pd.read_csv(out_path)
    assert set(out_df["shrinker"]) == {"scm"}
    assert set(out_df["overlay_flag"]) == {"off", "on"}
    assert out_df.shape[0] == 2

    expected_cols = {
        "shrinker",
        "overlay_flag",
        "cap_active",
        "cap_sources",
        "windows_evaluated",
        "window_coverage",
        "detection_rate_mean",
        "changed_share",
        "delta_mse_ew",
        "delta_qlike_ew",
        "n_effective_mse_ew",
        "n_effective_qlike_ew",
        "comparison_valid_mse_ew",
        "comparison_valid_qlike_ew",
    }
    assert expected_cols.issubset(set(out_df.columns))

    off_row = out_df[out_df["overlay_flag"] == "off"].iloc[0]
    on_row = out_df[out_df["overlay_flag"] == "on"].iloc[0]

    assert np.isnan(off_row["delta_mse_ew"])
    assert np.isfinite(on_row["delta_mse_ew"])
