from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from tools.make_summary import summarise_rc_directory, write_summaries


def _copy_sample_rc(tmp_path: Path) -> Path:
    src = Path("reports/rc-test")
    if not src.exists():
        raise RuntimeError("Sample RC artifacts missing under reports/rc-test.")
    dest = tmp_path / "reports" / src.name
    shutil.copytree(src, dest)
    return dest


def _write_minimal_design_run(
    run_dir: Path,
    *,
    cap_active: bool,
    cap_sources: list[str],
    mv_skip_on_missing_solver: bool,
    overlay_delta: float,
    detection_rate: float,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_payload = {
        "config": {"mv_skip_on_missing_solver": mv_skip_on_missing_solver},
        "windows": {
            "cap_active": cap_active,
            "cap_sources": cap_sources,
            "windows_requested": 10,
            "windows_after_caps": 10,
            "windows_evaluated": 10,
            "window_coverage": 1.0,
        },
    }
    (run_dir / "run.json").write_text(json.dumps(run_payload), encoding="utf-8")
    full_dir = run_dir / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(
        [
            {
                "regime": "full",
                "estimator": "overlay",
                "portfolio": "ew",
                "delta_mse_vs_baseline": overlay_delta,
                "delta_mse_ci_lower": overlay_delta - 0.1,
                "delta_mse_ci_upper": overlay_delta + 0.1,
                "delta_es_vs_baseline": overlay_delta,
                "var95": 0.1,
                "es95": 0.2,
                "realised_var": 0.3,
                "realised_es": 0.4,
                "n_effective_mse": 5,
                "n_effective_es": 5,
                "n_effective_qlike": 5,
                "comparison_valid": 1,
            },
            {
                "regime": "full",
                "estimator": "baseline",
                "portfolio": "ew",
                "var95": 0.05,
                "es95": 0.15,
                "realised_var": 0.25,
                "realised_es": 0.35,
            },
        ]
    )
    metrics_df.to_csv(full_dir / "metrics.csv", index=False)
    dm_df = pd.DataFrame(
        [
            {
                "regime": "full",
                "portfolio": "ew",
                "baseline": "baseline",
                "dm_stat": overlay_delta,
                "p_value": 0.5,
                "n_effective": 5,
                "comparison_valid": 1,
            }
        ]
    )
    dm_df.to_csv(full_dir / "dm.csv", index=False)
    diag_df = pd.DataFrame(
        [
            {
                "regime": "full",
                "detections": 1,
                "detection_rate": detection_rate,
                "isolation_share": 0.2,
                "edge_margin_mean": 0.3,
                "stability_margin_mean": 0.4,
                "alignment_cos_mean": 0.5,
                "alignment_angle_mean": 0.6,
            }
        ]
    )
    diag_df.to_csv(full_dir / "diagnostics.csv", index=False)
    detail_df = pd.DataFrame(
        [
            {
                "regime": "full",
                "detections": 1,
                "detection_rate": detection_rate,
                "isolation_share": 0.2,
                "edge_margin_mean": 0.3,
                "stability_margin_mean": 0.4,
                "alignment_cos_mean": 0.5,
                "alignment_angle_mean": 0.6,
            }
        ]
    )
    detail_df.to_csv(full_dir / "diagnostics_detail.csv", index=False)
    pd.DataFrame(
        columns=[
            "regime",
            "portfolio",
            "estimator",
            "skip_reason",
            "windows",
            "skip_count",
            "skip_share",
        ]
    ).to_csv(full_dir / "skip_stats.csv", index=False)


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
    overlay_path = summary_dir / "overlay_forensics.csv"
    assert perf_path.exists()
    assert det_path.exists()
    assert overlay_path.exists()

    kill_path = summary_dir / "kill_criteria.json"
    limits_path = summary_dir / "limitations.md"
    assert kill_path.exists()
    assert limits_path.exists()
    kill_data = json.loads(kill_path.read_text(encoding="utf-8"))
    assert "criteria" in kill_data

    loaded_det = pd.read_csv(det_path)
    assert "regime" in loaded_det.columns
    assert set(loaded_det["regime"]) == {"full", "calm", "crisis"}

    overlay_df = pd.read_csv(overlay_path)
    required_overlay_cols = {
        "window_end",
        "window_id",
        "design",
        "shrinker",
        "edge_mode",
        "changed",
        "skip_reason_primary",
        "skip_reason_detail",
        "gate_mode",
        "delta_frac_used",
        "lambda1_base",
        "lambda1_treat",
        "delta_lambda1",
        "mp_edge",
        "edge_margin",
        "realized_var",
        "mse_base",
        "mse_treat",
        "qlike_base",
        "qlike_treat",
    }
    assert required_overlay_cols.issubset(overlay_df.columns)


def test_make_summary_excludes_capped_design_runs(tmp_path: Path) -> None:
    rc_root = tmp_path / "reports" / "rc-caps"
    uncapped_dir = rc_root / "design-uncapped"
    capped_dir = rc_root / "design-capped"
    _write_minimal_design_run(
        uncapped_dir,
        cap_active=False,
        cap_sources=[],
        mv_skip_on_missing_solver=False,
        overlay_delta=1.0,
        detection_rate=0.1,
    )
    _write_minimal_design_run(
        capped_dir,
        cap_active=True,
        cap_sources=["max_windows"],
        mv_skip_on_missing_solver=False,
        overlay_delta=100.0,
        detection_rate=0.9,
    )

    write_summaries([rc_root])
    summary_dir = rc_root / "summary"
    perf_df = pd.read_csv(summary_dir / "summary_perf.csv")
    det_df = pd.read_csv(summary_dir / "summary_detection.csv")

    perf_row = perf_df[(perf_df["regime"] == "full") & (perf_df["portfolio"] == "ew")].iloc[0]
    det_row = det_df[(det_df["regime"] == "full")].iloc[0]

    assert perf_row["delta_mse_vs_baseline"] == 1.0
    assert det_row["detection_rate_mean"] == 0.1

    limits = (summary_dir / "limitations.md").read_text(encoding="utf-8")
    assert str(capped_dir.resolve()) in limits
    assert "max_windows" in limits
