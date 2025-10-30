from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from meta.run_meta import write_run_meta

pytestmark = pytest.mark.unit


def test_write_run_meta_creates_file_with_expected_fields(tmp_path: Path) -> None:
    out = tmp_path

    # Minimal summary to provide shape info
    summary = {
        "n_assets": 12,
        "window_weeks": 20,
        "horizon_weeks": 4,
        "replicates_per_week": 5,
        "rolling_windows_evaluated": 3,
    }
    (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    # Simple detection summary: 3 windows, with total detections 1+0+2=3 and L_max=2
    det = pd.DataFrame({"n_detections": [1, 0, 2]})
    det.to_csv(out / "detection_summary.csv", index=False)

    # Dummy figure to hash
    pdf_path = out / "figure.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\n")

    manifest = {
        "asset_count": 12,
        "weeks": 3,
        "days_per_week": 5,
        "dropped_weeks": 0,
        "imputed_weeks": 0,
        "partial_week_policy": "drop",
        "start_week": "2023-01-02",
        "end_week": "2023-01-30",
        "data_hash": "hash",
        "universe_hash": "universe123",
        "preprocess_flags": {"winsorize": "0.01"},
    }
    (out / "panel_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    cfg = {
        "dealias_delta": 0.2,
        "dealias_delta_frac": 0.05,
        "a_grid": 120,
        "signed_a": True,
        "cs_drop_top_frac": 0.1,
    }

    meta_path = write_run_meta(
        out,
        config=cfg,
        delta=0.2,
        delta_frac=0.05,
        a_grid=120,
        signed_a=True,
        sigma2_plugin="Cs_from_MS_drop_top_frac=0.1",
        code_signature_hash="deadbeef",
    )
    assert meta_path.exists()
    data = json.loads(meta_path.read_text(encoding="utf-8"))

    # Basic field checks
    assert data["n_assets"] == 12
    assert data["window_weeks"] == 20
    assert data["horizon_weeks"] == 4
    assert data["replicates_per_week"] == 5
    assert data["delta"] == 0.2
    assert data["delta_frac"] == 0.05
    assert data["a_grid"] == 120
    assert data["signed_a"] is True
    assert data["sigma2_plugin"] == "Cs_from_MS_drop_top_frac=0.1"
    assert data["code_signature"] == "deadbeef"
    assert data["estimator"] is None
    assert data["crisis_label"] is None
    assert data["detections_total"] == 3
    assert data["L"] == 2
    assert isinstance(data["figure_sha256"], dict)
    assert "figure.pdf" in data["figure_sha256"]
    assert data["panel_universe_hash"] == "universe123"
    assert data["panel_preprocess_flags"] == {"winsorize": "0.01"}
