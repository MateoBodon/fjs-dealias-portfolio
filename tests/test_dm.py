from __future__ import annotations

import numpy as np
import pytest

from evaluation.dm import dm_test

pytestmark = pytest.mark.unit


def test_dm_test_detects_mean_difference() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=256)
    err1 = base**2
    noise = rng.normal(loc=0.2, scale=0.05, size=256)
    err2 = err1 + noise

    stat, pval = dm_test(err1, err2, h=1)
    assert stat < 0.0
    assert pval < 0.05

    stat_no_nw, pval_no_nw = dm_test(err1, err2, h=1, use_nw=False)
    assert stat_no_nw < 0.0
    assert pval_no_nw < 0.05


def test_dm_test_handles_identical_or_missing_losses() -> None:
    err = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stat, pval = dm_test(err, err)
    assert np.isnan(stat)
    assert np.isnan(pval)

    err2 = err.copy()
    err2[::2] = np.nan
    stat2, pval2 = dm_test(err, err2 + 0.1)
    assert np.isnan(stat2) and np.isnan(pval2)
