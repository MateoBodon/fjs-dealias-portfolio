from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.equity_panel.run import _prepare_window_stats
from fjs.dealias import dealias_search


pytestmark = pytest.mark.unit


def _synthetic_nested_blocks(
    *,
    years: list[int],
    weeks: list[int],
    replicates: int,
    assets: int,
    rng: np.random.Generator,
) -> list[pd.DataFrame]:
    """Generate a small balanced Year⊃Week panel with a Σ₁ spike."""

    vec = np.array([0.7, 0.3, 0.1, 0.0, 0.0, 0.0], dtype=np.float64)[:assets]
    sigma1 = np.diag(np.linspace(0.5, 0.02, assets)) + 3.5 * np.outer(vec, vec)
    sigma2 = 0.005 * np.eye(assets)
    sigma3 = 0.002 * np.eye(assets)

    blocks: list[pd.DataFrame] = []
    columns = [f"T{idx:02d}" for idx in range(assets)]
    for year in years:
        g_i = rng.multivariate_normal(np.zeros(assets), sigma1)
        for week in weeks:
            h_ij = rng.multivariate_normal(np.zeros(assets), sigma2)
            residual = rng.multivariate_normal(np.zeros(assets), sigma3, size=replicates)
            observations = g_i + h_ij + residual
            dates = [
                pd.Timestamp.fromisocalendar(year, week, day)
                for day in range(1, replicates + 1)
            ]
            blocks.append(
                pd.DataFrame(
                    observations,
                    index=pd.DatetimeIndex(dates),
                    columns=columns,
                )
            )
    return blocks


def test_nested_smoke_detection_positive_stability() -> None:
    rng = np.random.default_rng(42)
    replicates = 5
    weeks = [10, 11, 12]
    years = [2022, 2023]
    assets = 6

    fit_blocks = _synthetic_nested_blocks(
        years=years,
        weeks=weeks,
        replicates=replicates,
        assets=assets,
        rng=rng,
    )

    prepared, reason = _prepare_window_stats(
        "nested",
        fit_blocks,
        replicates,
        nested_replicates=replicates,
    )
    assert reason is None
    assert prepared is not None

    diagnostics: dict[str, int] = {}
    # Relax eps to accommodate small-sample t-vector fluctuations.
    detections = dealias_search(
        prepared.y_fit,
        prepared.groups,
        target_r=0,
        delta=0.0,
        delta_frac=0.02,
        eps=6.0,
        stability_eta_deg=1.0,
        use_tvector=True,
        nonnegative_a=False,
        a_grid=60,
        diagnostics=diagnostics,
        stats=prepared.stats,
        design=prepared.design_override,
        oneway_a_solver="auto",
    )

    assert detections, "Expected at least one accepted detection"
    assert any(det.get("stability_margin", 0.0) > 0.0 for det in detections)
