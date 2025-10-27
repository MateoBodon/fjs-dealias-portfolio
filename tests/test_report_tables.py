from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from report.gather import collect_estimator_panel
from report.tables import table_ablation, table_estimators, table_rejections

FIXTURE_RUN = Path(__file__).parent / "report_fixtures" / "sample_run"

pytestmark = pytest.mark.unit


def test_table_estimators(tmp_path: Path) -> None:
    panel = collect_estimator_panel([FIXTURE_RUN])
    csv_path, md_path, tex_path = table_estimators(panel, root=tmp_path)
    assert csv_path.exists() and md_path.exists() and tex_path.exists()
    csv_df = pd.read_csv(csv_path)
    assert "delta_mse_ew" in csv_df.columns
    assert "Ledoit-Wolf" in csv_df["estimator"].values


def test_table_rejections(tmp_path: Path) -> None:
    rejection_df = pd.DataFrame(
        {
            "run": ["sample_run", "sample_run"],
            "rejection_reason": ["other", "edge_buffer"],
            "count": [10, 1],
        }
    )
    csv_path, md_path, tex_path = table_rejections(rejection_df, root=tmp_path)
    assert csv_path.exists() and md_path.exists() and tex_path.exists()
    pivot = pd.read_csv(csv_path)
    assert "other" in pivot.columns


def test_table_ablation(tmp_path: Path) -> None:
    ablation_df = pd.DataFrame(
        {
            "run": ["sample_run", "sample_run"],
            "delta_frac": [0.02, 0.05],
            "eps": [0.02, 0.03],
            "a_grid": [72, 120],
            "detection_rate": [0.5, 0.8],
            "mse_alias": [0.9, 1.1],
            "mse_de": [0.8, 1.0],
        }
    )
    csv_path, md_path, tex_path = table_ablation(ablation_df, root=tmp_path)
    assert csv_path.exists() and md_path.exists() and tex_path.exists()
    csv_df = pd.read_csv(csv_path)
    assert "mse_gain" in csv_df.columns
