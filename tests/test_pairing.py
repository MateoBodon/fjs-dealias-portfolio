from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from experiments.synthetic_oneway import run as synth_run

pytestmark = pytest.mark.slow


def test_pairing_alignment_improves_median_bias(tmp_path: Path) -> None:
    # Configure a compact but informative rank-5 scenario
    config = {
        "n_assets": 40,
        "n_groups": 40,
        "replicates": 2,
        "noise_variance": 1.0,
        "signal_to_noise": 0.5,
        "multi_spike_strengths": [7.0, 6.0, 5.0, 4.0, 3.0],
        "multi_spike_trials": 16,
        "delta": 0.3,
        "eps": 0.05,
        "stability_eta_deg": 0.5,
        "output_dir": str(tmp_path),
        "progress": False,
    }
    rng = np.random.default_rng(321)

    _ = synth_run.s5_multi_spike_bias(config, rng)

    comp_path = Path(tmp_path) / "s5_pairing_comparison.csv"
    assert comp_path.exists(), "Expected pairing comparison CSV to be emitted."
    df = pd.read_csv(comp_path)
    assert {"dealiased_bias_naive", "dealiased_bias_aligned"}.issubset(df.columns)

    naive_abs = df["dealiased_bias_naive"].astype(float).abs()
    aligned_abs = df["dealiased_bias_aligned"].astype(float).abs()
    # Alignment should be no worse, and typically better, in median across spikes
    assert aligned_abs.median() <= naive_abs.median() + 1e-6
