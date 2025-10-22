import numpy as np


def test_weekly_covariance_identity() -> None:
    rng = np.random.default_rng(42)
    asset_dim = 6
    theta = 0.8
    sigma = 0.5
    replicates = 5
    weeks = 5000

    direction = rng.standard_normal(asset_dim)
    direction /= np.linalg.norm(direction)
    sigma1 = theta * np.outer(direction, direction)
    sigma2 = (sigma**2) * np.eye(asset_dim, dtype=np.float64)

    group_effects = rng.multivariate_normal(
        mean=np.zeros(asset_dim, dtype=np.float64),
        cov=sigma1,
        size=weeks,
    )
    noise = rng.multivariate_normal(
        mean=np.zeros(asset_dim, dtype=np.float64),
        cov=sigma2,
        size=(weeks, replicates),
    )
    weekly_sums = replicates * group_effects + noise.sum(axis=1)

    empirical_cov = np.cov(weekly_sums, rowvar=False, ddof=1)
    analytic_cov = (replicates**2) * sigma1 + replicates * sigma2

    numerator = np.linalg.norm(empirical_cov - analytic_cov, ord="fro")
    denominator = np.linalg.norm(analytic_cov, ord="fro")
    assert numerator / denominator < 0.05
