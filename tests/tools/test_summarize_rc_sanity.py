from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from meta.completeness import evaluate_eval_run

pytestmark = pytest.mark.unit


def _write_daily_run(run_dir: Path, *, detection_rate: float, coverage: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cap_active": False,
        "windows_total": 2,
        "windows_completed": int(round(coverage * 2)),
        "windows_evaluated": int(round(coverage * 2)),
        "window_coverage": coverage,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    diag_csv = textwrap.dedent(
        f"""\
        detection_rate,percent_changed,alignment_cos_mean,reason_code
        {detection_rate},0.25,0.9,accepted
        """
    )
    (run_dir / "diagnostics.csv").write_text(diag_csv, encoding="utf-8")

    metrics_csv = textwrap.dedent(
        """\
        regime,estimator,portfolio,sq_error
        full,baseline,ew,0.10
        full,overlay,ew,0.12
        full,baseline,mv,0.05
        full,overlay,mv,0.07
        """
    )
    (run_dir / "metrics_detail.csv").write_text(metrics_csv, encoding="utf-8")

    full_dir = run_dir / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    (full_dir / "metrics.csv").write_text(metrics_csv, encoding="utf-8")
    (full_dir / "diagnostics.csv").write_text(diag_csv, encoding="utf-8")


def test_eval_completeness_flags_partial_dir(tmp_path: Path) -> None:
    partial = tmp_path / "partial_run"
    partial.mkdir()
    (partial / "resolved_config.json").write_text("{}", encoding="utf-8")

    comp = evaluate_eval_run(partial, label="partial")
    assert not comp.is_complete
    assert comp.excluded_from_aggregate
    assert {"metrics.csv", "diagnostics.csv"}.issubset(set(comp.missing_files))
    assert comp.incomplete_reason


def test_summarizer_marks_missing_sections_and_excludes_incomplete(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    rc_dir = tmp_path / "rc-lite"
    rc_dir.mkdir()

    dow_dir = tmp_path / "dow"
    _write_daily_run(dow_dir, detection_rate=0.5, coverage=1.0)

    vol_dir = tmp_path / "vol"
    _write_daily_run(vol_dir, detection_rate=0.2, coverage=0.5)

    weekly_dir = tmp_path / "weekly"
    nested_dir = tmp_path / "nested"
    # Leave weekly/nested absent to exercise missing sections.

    cmd = [
        sys.executable,
        "tools/summarize_rc_sanity.py",
        "--rc-dir",
        str(rc_dir),
        "--dow-dir",
        str(dow_dir),
        "--vol-dir",
        str(vol_dir),
        "--weekly-dow-dir",
        str(weekly_dir),
        "--nested-dir",
        str(nested_dir),
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)

    summary_path = rc_dir / "summary_sanity.json"
    assert summary_path.exists(), "summary_sanity.json should be emitted"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    entries = summary["entries"]
    assert entries["daily_dow"]["status"] == "complete"
    assert entries["daily_vol"]["status"] == "incomplete"
    assert entries["weekly_dow"]["status"] == "missing"
    assert entries["nested_weekly"]["status"] == "missing"
    assert entries["daily_vol"]["excluded_from_aggregate"] is True

    aggregate = summary["aggregate"]
    assert aggregate["included"] == ["daily_dow"]
    assert pytest.approx(0.5) == aggregate["detection_rate_mean"]

    incomplete_labels = {item["label"] for item in summary["incomplete_runs"]}
    assert {"daily_vol", "weekly_dow", "nested_weekly"}.issubset(incomplete_labels)
