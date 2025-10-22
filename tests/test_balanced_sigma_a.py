import numpy as np

from fjs.dealias import _sigma_of_a_from_MS


def test_sigma_of_a_balanced_design_identity() -> None:
    group_count = 3
    replicate_count = 2
    feature_count = 4
    n_obs = group_count * replicate_count

    rng = np.random.default_rng(0)
    observations = rng.standard_normal((n_obs, feature_count))

    group_labels = np.repeat(np.arange(group_count), replicate_count)
    indicator = np.zeros((n_obs, group_count), dtype=np.float64)
    indicator[np.arange(n_obs), group_labels] = 1.0

    projection_groups = indicator @ np.linalg.inv(indicator.T @ indicator) @ indicator.T
    projection_mean = np.full((n_obs, n_obs), 1.0 / n_obs, dtype=np.float64)

    pi1 = projection_groups - projection_mean
    pi2 = np.eye(n_obs, dtype=np.float64) - projection_groups

    d1 = float(group_count - 1)
    d2 = float(n_obs - group_count)

    ms1 = observations.T @ (pi1 / d1) @ observations
    ms2 = observations.T @ (pi2 / d2) @ observations

    a = rng.standard_normal(2)
    a /= np.linalg.norm(a)

    design_operator = a[0] * pi1 / d1 + a[1] * pi2 / d2

    lhs = observations.T @ design_operator @ observations
    rhs = a[0] * ms1 + a[1] * ms2
    rhs_helper = _sigma_of_a_from_MS(a, [ms1, ms2])

    difference_norm = np.linalg.norm(lhs - rhs, "fro")
    baseline_norm = np.linalg.norm(lhs, "fro")

    assert np.allclose(rhs, rhs_helper)
    if baseline_norm == 0.0:
        assert difference_norm == 0.0
    else:
        assert difference_norm / baseline_norm < 1e-12
