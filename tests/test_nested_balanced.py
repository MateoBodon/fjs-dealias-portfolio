from __future__ import annotations

import numpy as np
import numpy.testing as npt

from fjs.balanced_nested import mean_squares_nested
from fjs.dealias import dealias_search


def _generate_nested_sample(
    I: int,
    J: int,
    R: int,
    p: int,
    *,
    seed: int = 0,
    year_scale: float = 1.0,
    week_scale: float = 0.1,
    residual_scale: float = 0.05,
    base_vectors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    year_labels: list[int] = []
    week_labels: list[int] = []
    for year_idx in range(I):
        if base_vectors is not None:
            year_effect = np.asarray(base_vectors[year_idx], dtype=np.float64)
        else:
            year_effect = rng.normal(loc=0.0, scale=year_scale, size=p)
        for week_idx in range(J):
            week_effect = rng.normal(loc=0.0, scale=week_scale, size=p)
            for _ in range(R):
                residual = rng.normal(loc=0.0, scale=residual_scale, size=p)
                observations.append(year_effect + week_effect + residual)
                year_labels.append(2010 + year_idx)
                week_labels.append(week_idx)
    return (
        np.asarray(observations, dtype=np.float64),
        np.asarray(year_labels, dtype=np.int64),
        np.asarray(week_labels, dtype=np.int64),
    )


def test_mean_squares_nested_balanced() -> None:
    I, J, R, p = 3, 4, 5, 3
    y, years, weeks = _generate_nested_sample(
        I,
        J,
        R,
        p,
        seed=42,
        year_scale=2.5,
        week_scale=0.4,
        residual_scale=0.1,
    )
    (ms1, ms2, ms3), metadata = mean_squares_nested(y, years, weeks, R)

    assert ms1.shape == (p, p)
    assert ms2.shape == (p, p)
    assert ms3.shape == (p, p)
    assert metadata.I == I
    assert metadata.J == J
    assert metadata.replicates == R
    assert metadata.n == y.shape[0]
    assert metadata.p == p

    expected_d = np.array([I - 1, I * (J - 1), I * J * (R - 1)], dtype=np.float64)
    expected_c = np.array([J * R, R, 1.0], dtype=np.float64)
    npt.assert_allclose(metadata.d, expected_d)
    npt.assert_allclose(metadata.c, expected_c)
    assert metadata.N == float(R)

    overall_mean = y.mean(axis=0)
    centered = y - overall_mean
    ss_total = centered.T @ centered
    ss1 = ms1 * float(metadata.d[0])
    ss2 = ms2 * float(metadata.d[1])
    ss3 = ms3 * float(metadata.d[2])
    npt.assert_allclose(ss1 + ss2 + ss3, ss_total, rtol=1e-10, atol=1e-10)


def test_nested_dealias_smoke_run() -> None:
    I, J, R, p = 2, 3, 4, 3
    base_vectors = np.array(
        [
            [0.0, 0.0, 0.0],
            [15.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )
    y, years, weeks = _generate_nested_sample(
        I,
        J,
        R,
        p,
        seed=7,
        week_scale=0.1,
        residual_scale=0.02,
        base_vectors=base_vectors,
    )
    (ms1, ms2, ms3), metadata = mean_squares_nested(y, years, weeks, R)

    sigma1 = (ms1 - ms2) / float(metadata.J * metadata.replicates)
    sigma2 = (ms2 - ms3) / float(metadata.replicates)
    sigma3 = ms3.copy()

    stats_local = {
        "MS1": ms1,
        "MS2": ms2,
        "MS3": ms3,
        "Sigma1_hat": sigma1,
        "Sigma2_hat": sigma2,
        "Sigma3_hat": sigma3,
        "I": metadata.I,
        "J": metadata.J,
        "n": metadata.n,
    }
    design = {
        "c": metadata.c,
        "C": np.ones_like(metadata.c),
        "d": metadata.d,
        "N": metadata.N,
        "order": [[1, 2, 3], [2, 3], [3]],
    }

    detections = dealias_search(
        y,
        np.arange(y.shape[0]),
        target_r=0,
        delta=0.0,
        delta_frac=None,
        eps=0.0,
        energy_min_abs=None,
        stability_eta_deg=0.0,
        use_tvector=False,
        nonnegative_a=False,
        a_grid=60,
        cs_drop_top_frac=0.0,
        cs_sensitivity_frac=0.0,
        scan_basis="ms",
        diagnostics={},
        stats=stats_local,
        design=design,
    )
    assert detections, "expected at least one nested detection"
