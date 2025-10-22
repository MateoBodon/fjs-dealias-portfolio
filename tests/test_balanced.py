from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fjs.balanced import (
    BalancedConfig,
    compute_balanced_weights,
    group_means,
    mean_squares,
)


def _balanced_panel() -> tuple[np.ndarray, np.ndarray, int, int, int]:
    rng = np.random.default_rng(42)
    n_groups, replicates, n_features = 3, 2, 4
    base = (np.arange(n_groups, dtype=np.float64)[:, None] + 1.0) * np.linspace(
        1.0, 1.8, n_features
    )
    noise = rng.normal(scale=0.1, size=(n_groups, replicates, n_features))
    panel = base[:, None, :] + noise
    observations = panel.reshape(n_groups * replicates, n_features)
    groups = np.repeat(np.arange(n_groups), replicates)
    return observations, groups, n_groups, replicates, n_features


def test_group_means_returns_expected_statistics() -> None:
    observations, groups, n_groups, _, n_features = _balanced_panel()
    group_mean_matrix, overall_mean = group_means(observations, groups)

    assert group_mean_matrix.shape == (n_groups, n_features)
    assert overall_mean.shape == (n_features,)

    manual_group_means = np.vstack(
        [observations[groups == label].mean(axis=0) for label in np.unique(groups)]
    )
    assert_allclose(group_mean_matrix, manual_group_means)
    assert_allclose(overall_mean, observations.mean(axis=0))


def test_mean_squares_matrices_and_identities() -> None:
    observations, groups, n_groups, replicates, n_features = _balanced_panel()
    stats = mean_squares(observations, groups)

    assert stats["I"] == n_groups
    assert stats["J"] == replicates
    assert stats["n"] == observations.shape[0]
    assert stats["p"] == n_features

    for key in ("MS1", "MS2", "Sigma1_hat", "Sigma2_hat"):
        matrix = stats[key]
        assert matrix.shape == (n_features, n_features)
        assert_allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12)

    manual_within = np.zeros((n_features, n_features), dtype=np.float64)
    for label in np.unique(groups):
        group_obs = observations[groups == label]
        centered = group_obs - group_obs.mean(axis=0, keepdims=True)
        manual_within += centered.T @ centered
    manual_within /= observations.shape[0] - n_groups

    assert_allclose(stats["Sigma2_hat"], manual_within)
    assert_allclose(
        stats["Sigma1_hat"] + stats["Sigma2_hat"] / replicates,
        stats["MS1"] / replicates,
    )


def test_mean_squares_rejects_unbalanced_design() -> None:
    observations, groups, *_ = _balanced_panel()
    unbalanced_groups = np.concatenate([groups, [0]])
    unbalanced_observations = np.vstack([observations, observations[:1]])

    with pytest.raises(ValueError):
        group_means(unbalanced_observations, unbalanced_groups)
    with pytest.raises(ValueError):
        mean_squares(unbalanced_observations, unbalanced_groups)


def test_compute_balanced_weights_is_stub() -> None:
    config = BalancedConfig(regularization=0.1, max_iter=10)
    with pytest.raises(NotImplementedError):
        compute_balanced_weights(np.ones((2, 2)), config)
