from __future__ import annotations

import numpy as np

from evaluation.evaluate import (
    christoffersen_independence_test,
    expected_shortfall_test,
    kupiec_pof_test,
)


def test_kupiec_pof_rates_relative_misspecification() -> None:
    violations_good = np.array([False] * 95 + [True] * 5, dtype=bool)
    violations_bad = np.array([False] * 90 + [True] * 10, dtype=bool)
    p_good = kupiec_pof_test(violations_good, alpha=0.05)
    p_bad = kupiec_pof_test(violations_bad, alpha=0.05)
    assert p_good > 0.1
    assert p_bad < 0.05


def test_christoffersen_independence_flags_clustering() -> None:
    rng = np.random.default_rng(11)
    violations_random = rng.random(200) < 0.05
    violations_clustered = np.array([False] * 190 + [True] * 10, dtype=bool)
    p_random = christoffersen_independence_test(violations_random)
    p_clustered = christoffersen_independence_test(violations_clustered)
    assert p_clustered < p_random


def test_expected_shortfall_test_reacts_to_bias() -> None:
    losses = np.full(80, -0.05)
    es_good = np.full(80, -0.05)
    es_bad = np.full(80, -0.03)
    violations = np.ones(80, dtype=bool)
    p_good = expected_shortfall_test(losses, es_good, violations)
    p_bad = expected_shortfall_test(losses, es_bad, violations)
    assert p_good > 0.5
    assert p_bad < 0.05
