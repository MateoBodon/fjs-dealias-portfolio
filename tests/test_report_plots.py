from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from report.gather import collect_estimator_panel
from report.plots import (
    plot_ablation_heatmap,
    plot_detection_rate,
    plot_dm_bars,
    plot_edge_margin_hist,
)

FIXTURE_RUN = Path(__file__).parent / "report_fixtures" / "sample_run"

pytestmark = pytest.mark.unit


def test_plot_dm_and_detection(tmp_path: Path) -> None:
    panel = collect_estimator_panel([FIXTURE_RUN])
    dm_path = plot_dm_bars(panel, root=tmp_path)
    detect_path = plot_detection_rate(panel, root=tmp_path)
    assert dm_path.exists()
    assert detect_path.exists()


def test_plot_edge_margin(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "run": ["sample_run"] * 5,
            "edge_margin": [0.1, 0.2, 0.15, 0.05, 0.3],
        }
    )
    path = plot_edge_margin_hist(df, root=tmp_path)
    assert path.exists()


def test_plot_ablation_heatmap(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "run": ["sample_run", "sample_run", "sample_run", "sample_run"],
            "delta_frac": [0.02, 0.02, 0.05, 0.05],
            "eps": [0.02, 0.03, 0.02, 0.03],
            "mse_alias": [0.9, 1.0, 1.1, 1.2],
            "mse_de": [0.8, 0.9, 1.0, 1.1],
        }
    )
    path = plot_ablation_heatmap(df, root=tmp_path)
    assert path.exists()
