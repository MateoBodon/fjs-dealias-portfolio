from __future__ import annotations

import numpy as np
import pytest

from experiments.synthetic.harness_utils import HarnessConfig, roc_table, select_energy_floor, simulate_scores

pytestmark = pytest.mark.unit


def _small_config(edge_modes: tuple[str, ...] = ("scm",)) -> HarnessConfig:
    return HarnessConfig(
        n_assets=12,
        n_groups=16,
        replicates=2,
        noise_variance=1.0,
        signal_to_noise=0.35,
        edge_modes=edge_modes,
        trials=48,
        seed=42,
    )


def test_simulate_scores_increases_with_signal() -> None:
    config = _small_config()
    null_scores = simulate_scores(config, mu_values=[0.0]).scores
    signal_scores = simulate_scores(config, mu_values=[4.0]).scores

    mean_null = float(np.mean(null_scores["score"].to_numpy(dtype=np.float64)))
    mean_signal = float(np.mean(signal_scores["score"].to_numpy(dtype=np.float64)))
    assert mean_signal > mean_null


def test_select_energy_floor_respects_target_fpr() -> None:
    config = _small_config(("scm", "tyler"))
    null_df = simulate_scores(config, mu_values=[0.0]).scores
    power = {
        4.0: simulate_scores(config, mu_values=[4.0]).scores,
        6.0: simulate_scores(config, mu_values=[6.0]).scores,
    }
    selection = select_energy_floor(null_df, power, target_fpr=0.5)
    assert selection is not None
    assert selection.fpr <= 0.5 + 1e-9
    assert selection.threshold >= 0.0
    assert selection.edge_mode in {"scm", "tyler"}


def test_roc_table_emits_entries_for_each_mu() -> None:
    config = _small_config()
    null_df = simulate_scores(config, mu_values=[0.0]).scores
    power = {
        4.0: simulate_scores(config, mu_values=[4.0]).scores,
        8.0: simulate_scores(config, mu_values=[8.0]).scores,
    }
    roc = roc_table(null_df, power)
    assert not roc.empty
    assert set(np.unique(roc["edge_mode"])) == {"scm"}
    assert set(np.round(roc["mu"].unique(), 1)) == {4.0, 8.0}
