from __future__ import annotations

import pandas as pd

from tools.gallery import build_gallery
from tools.memo_builder import build_memo


def _write_artifacts(tmp_path):
    metrics = pd.DataFrame(
        [
            {"estimator": "overlay", "portfolio": "ew", "mse": 0.8, "delta_mse_vs_rie": -0.05},
            {"estimator": "rie", "portfolio": "ew", "mse": 0.85, "delta_mse_vs_rie": 0.0},
            {"estimator": "overlay", "portfolio": "mv", "mse": 0.9, "delta_mse_vs_rie": -0.04},
            {"estimator": "rie", "portfolio": "mv", "mse": 0.94, "delta_mse_vs_rie": 0.0},
        ]
    )
    dm = pd.DataFrame([{"portfolio": "ew", "baseline": "rie", "dm_stat": 1.2, "p_value": 0.23}])
    var = pd.DataFrame(
        [
            {"estimator": "overlay", "portfolio": "ew", "violation_rate": 0.04, "kupiec_p": 0.8, "christoffersen_p": 0.7},
            {"estimator": "rie", "portfolio": "ew", "violation_rate": 0.06, "kupiec_p": 0.5, "christoffersen_p": 0.6},
        ]
    )
    metrics_path = tmp_path / "metrics.csv"
    dm_path = tmp_path / "dm.csv"
    var_path = tmp_path / "var.csv"
    metrics.to_csv(metrics_path, index=False)
    dm.to_csv(dm_path, index=False)
    var.to_csv(var_path, index=False)
    return metrics_path, dm_path, var_path


def test_build_memo_creates_markdown(tmp_path) -> None:
    metrics_path, dm_path, var_path = _write_artifacts(tmp_path)
    memo_path = tmp_path / "memo.md"
    build_memo(
        metrics_path=metrics_path,
        dm_path=dm_path,
        var_path=var_path,
        output_path=memo_path,
        regime="full",
    )
    assert memo_path.exists()
    content = memo_path.read_text()
    assert "ΔMSE" in content


def test_build_gallery_outputs_assets(tmp_path) -> None:
    metrics_path, _, var_path = _write_artifacts(tmp_path)
    gallery_dir = tmp_path / "gallery"
    build_gallery(
        metrics_path=metrics_path,
        var_path=var_path,
        output_dir=gallery_dir,
        regime="full",
    )
    files = list(gallery_dir.iterdir())
    assert files, "gallery builder should create at least one artifact"
