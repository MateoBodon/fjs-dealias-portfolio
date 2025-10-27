from __future__ import annotations

import math
from typing import Dict

import numpy as np

from evaluation.evaluate import (
    iqr,
    sign_test_pvalue,
    block_bootstrap_ci_median,
    build_metrics_summary,
)


def test_sign_test_pvalue_strong_asymmetry() -> None:
    # 9 positives, 1 negative -> p ~ 0.0215 (two-sided)
    diffs = [1.0] * 9 + [-0.1]
    p = sign_test_pvalue(diffs)
    assert p < 0.05


def test_block_bootstrap_ci_median_contains_sample_median() -> None:
    rng = np.random.default_rng(123)
    # Synthetic AR(1)-like series to induce dependence
    n = 60
    x = np.empty(n)
    x[0] = 0.0
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal(scale=0.3)
    med = float(np.median(x))
    lo, hi = block_bootstrap_ci_median(x, block_len=12, n_boot=300, rng=rng)
    assert math.isfinite(lo) and math.isfinite(hi)
    assert lo <= med <= hi


def test_iqr_matches_percentile_difference() -> None:
    arr = np.array([1, 2, 3, 4, 5], dtype=float)
    assert iqr(arr) == 4.0 - 2.0


def test_build_metrics_summary_dealiased_vs_baselines() -> None:
    # Create a small consistent window-level error map
    windows = list(range(20))
    err_de = {i: 1.0 + 0.01 * i + 0.02 * np.sin(i / 3.0) for i in windows}
    err_lw = {i: err_de[i] + 0.02 + 0.005 * np.cos(i) for i in windows}
    err_al = {i: err_de[i] + 0.05 + 0.01 * np.sin(i / 5.0) for i in windows}
    errors_by_combo: Dict[str, Dict[int, float]] = {
        "Equal Weight::De-aliased": err_de,
        "Equal Weight::Ledoit-Wolf": err_lw,
        "Equal Weight::Aliased": err_al,
    }
    coverage = {k: 0.0 for k in errors_by_combo}

    summary = build_metrics_summary(
        errors_by_combo=errors_by_combo,
        coverage_errors=coverage,
        label="unit",
        block_len=6,
        n_boot=200,
    )
    # Extract De-aliased row
    row = summary[(summary["strategy"] == "Equal Weight") & (summary["estimator"] == "De-aliased")]
    assert not row.empty
    med_delta_lw = float(row["delta_median_de_minus_lw"].iloc[0])
    med_delta_al = float(row["delta_median_de_minus_alias"].iloc[0])
    p_lw = float(row["sign_test_p_de_vs_lw"].iloc[0])
    p_al = float(row["sign_test_p_de_vs_alias"].iloc[0])
    # De-aliased strictly better (lower MSE) than LW/Aliased in this synthetic setup
    assert med_delta_lw < 0.0
    assert med_delta_al < 0.0
    assert p_lw < 0.05
    assert p_al < 0.05
    dm_p_lw = float(row["dm_p_de_vs_lw"].iloc[0])
    dm_stat_lw = float(row["dm_stat_de_vs_lw"].iloc[0])
    assert dm_stat_lw < 0.0
    assert dm_p_lw < 0.05
