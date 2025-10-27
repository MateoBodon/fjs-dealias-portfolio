from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from experiments.synthetic_oneway import run as synth_run

pytestmark = pytest.mark.slow


def test_s5_alignment_pairing_comparison(tmp_path: Path) -> None:
    # Configure a light-weight multi-spike run
    config = {
        "n_assets": 40,
        "n_groups": 40,
        "replicates": 2,
        "noise_variance": 1.0,
        "signal_to_noise": 0.5,
        "multi_spike_strengths": [6.5, 5.0, 3.5],
        "multi_spike_trials": 40,
        "delta": 0.3,
        "eps": 0.05,
        "stability_eta_deg": 0.4,
        "output_dir": str(tmp_path),
        "progress": False,
    }
    rng = np.random.default_rng(123)
    # Run only the S5 component directly to keep test fast
    _ = synth_run.s5_multi_spike_bias(config, rng)

    comp_path = Path(tmp_path) / "s5_pairing_comparison.csv"
    assert comp_path.exists(), "Expected pairing comparison CSV to be emitted."
    df = pd.read_csv(comp_path)
    assert {"aliased_bias", "dealiased_bias_naive", "dealiased_bias_aligned"}.issubset(
        set(df.columns)
    )
    # Alignment should not be worse than naive on average
    naive = df["dealiased_bias_naive"].astype(float).abs().mean()
    aligned = df["dealiased_bias_aligned"].astype(float).abs().mean()
    # Allow small numerical slack
    assert aligned <= naive + 1e-6
