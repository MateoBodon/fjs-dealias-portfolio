from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plotting import (
    e1_plot_spectrum_with_mp,
    e2_plot_spike_timeseries,
    e3_plot_var_mse,
    e4_plot_var_coverage,
    s4_plot_guardrails_from_csv,
)
try:  # optional nicety: show quick progress in tests
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


@pytest.mark.parametrize(
    "run_name",
    [
        "unit_plots",
    ],
)
def test_e1_e2_e3_e4_and_s4_create_pdfs(tmp_path: Path, run_name: str) -> None:
    # Use a unique run name per test worker if xdist is active
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    run = f"{run_name}_{worker}"

    # E1: spectrum with MP edge
    eig = np.linspace(0.5, 3.0, 50)
    mp_edges = (1.0, 1.8)

    tasks = [
        ("E1", lambda: e1_plot_spectrum_with_mp(eig, mp_edges, run=run, title="Test spectrum")),
        (
            "E2",
            lambda: e2_plot_spike_timeseries(
                np.arange(20), np.sin(np.arange(20) / 3.0) + 2, (np.sin(np.arange(20) / 3.0) + 2) * 0.9, run=run, title="Spikes"
            ),
        ),
        (
            "E3",
            lambda: e3_plot_var_mse(
                {
                    "Aliased": (np.random.default_rng(0).random(12) ** 2).tolist(),
                    "De-aliased": (np.random.default_rng(1).random(12) ** 2).tolist(),
                    "Ledoit-Wolf": (np.random.default_rng(2).random(12) ** 2).tolist(),
                },
                run=run,
            ),
        ),
        ("E4", lambda: e4_plot_var_coverage({"Aliased": -0.02, "De-aliased": 0.01, "Ledoit-Wolf": 0.005}, run=run)),
    ]

    for _, fn in tqdm(tasks, desc="Plots", unit="plot"):
        p = fn()
        assert p.exists() and p.stat().st_size > 0

    # S4: guardrails from CSV
    csv_path = tmp_path / "s4_guardrails.csv"
    pd.DataFrame(
        [
            {"setting": "default", "false_positive_rate": 0.01, "detections": 1, "trials": 100},
            {"setting": "delta=0, no stability", "false_positive_rate": 0.08, "detections": 8, "trials": 100},
        ]
    ).to_csv(csv_path, index=False)
    p_s4 = s4_plot_guardrails_from_csv(csv_path, run=run)
    assert p_s4.exists() and p_s4.stat().st_size > 0

    # All PDFs should be under experiments/<run>/figures
    expected_dir = Path("experiments") / run / "figures"
    assert expected_dir.exists()
