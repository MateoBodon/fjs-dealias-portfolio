from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from experiments.equity_panel import run
from experiments.equity_panel.reasons import DIAGNOSTIC_GUARD_KEYS, SkipReasonPrimary
import pytest


def test_gating_diagnostics_artifact(tmp_path: Path) -> None:
    dates = pd.date_range("2023-01-02", periods=60, freq="B")
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        rng.normal(scale=0.01, size=(len(dates), 6)),
        index=dates,
        columns=[f"T{idx:02d}" for idx in range(6)],
    )

    out_dir = tmp_path / "weekly_diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    run._run_single_period(
        daily_returns=returns,
        start=dates[0],
        end=dates[-1],
        output_dir=out_dir,
        window_weeks=2,
        horizon_weeks=1,
        delta=0.0,
        delta_frac=0.01,
        eps=0.01,
        stability_eta=0.5,
        signed_a=True,
        target_component=0,
        partial_week_policy="drop",
        precompute_panel=False,
        cache_dir=None,
        resume_cache=False,
        cs_drop_top_frac=0.05,
        cs_sensitivity_frac=0.0,
        off_component_leak_cap=10.0,
        sigma_ablation=False,
        label="test",
        design_mode="oneway",
        nested_replicates=5,
        oneway_a_solver="auto",
        estimator="dealias",
        progress=False,
        a_grid=32,
        energy_min_abs=1e-6,
        factor_returns=None,
        prewhiten_meta=None,
        minvar_ridge=1e-4,
        minvar_box=(0.0, 0.1),
        turnover_cost_bps=1.0,
        minvar_condition_cap=1e6,
        preprocess_flags={},
        gating={"enable": True, "q_max": 1, "require_isolated": True},
        alignment_top_p=2,
        edge_mode="scm",
        edge_huber_c=1.5,
        use_tvector=True,
        diagnostics={"gating_trace": True},
    )

    diag_path = out_dir / "gating_diagnostics.csv"
    assert diag_path.exists()
    diag_df = pd.read_csv(diag_path)
    required = {
        "window_index",
        "p",
        "t",
        "design",
        "edge_mode",
        "estimator",
        "raw_detections",
        "candidate_pool",
        "accepted",
        "skip_reason",
        "skip_reason_primary",
        "skip_reason_detail",
        "exception_type",
        "delta_frac_used",
        "edge_used",
        "lambda_top_over_edge",
    }
    assert required.issubset(set(diag_df.columns))
    for guard_key in DIAGNOSTIC_GUARD_KEYS:
        assert f"guard_{guard_key}" in diag_df.columns
    assert "guard_other" not in diag_df.columns

    rejected = diag_df[diag_df["accepted"] == False]
    if not rejected.empty:
        assert rejected["skip_reason"].replace("", pd.NA).notna().all()
        assert rejected["skip_reason_primary"].replace("", pd.NA).notna().all()
        diag_failures = rejected[rejected["skip_reason_primary"] == str(SkipReasonPrimary.DIAGNOSTIC_FAILURE)]
        if not diag_failures.empty:
            assert diag_failures["skip_reason_detail"].replace("", pd.NA).notna().all()
            assert diag_failures["exception_type"].replace("", pd.NA).notna().all()


def test_gating_diagnostics_records_exception_detail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2023-01-02", periods=90, freq="B")
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(
        rng.normal(scale=0.01, size=(len(dates), 4)),
        index=dates,
        columns=[f"E{idx:02d}" for idx in range(4)],
    )

    def boom(*args, **kwargs):
        raise RuntimeError("boom diagnostic")

    monkeypatch.setattr(run, "dealias_search", boom)

    out_dir = tmp_path / "weekly_diag_failure"
    out_dir.mkdir(parents=True, exist_ok=True)

    run._run_single_period(
        daily_returns=returns,
        start=dates[0],
        end=dates[-1],
        output_dir=out_dir,
        window_weeks=2,
        horizon_weeks=1,
        delta=0.0,
        delta_frac=0.01,
        eps=0.01,
        stability_eta=0.5,
        signed_a=True,
        target_component=0,
        partial_week_policy="drop",
        precompute_panel=False,
        cache_dir=None,
        resume_cache=False,
        cs_drop_top_frac=0.05,
        cs_sensitivity_frac=0.0,
        off_component_leak_cap=10.0,
        sigma_ablation=False,
        label="test",
        design_mode="oneway",
        nested_replicates=3,
        oneway_a_solver="auto",
        estimator="dealias",
        progress=False,
        a_grid=32,
        energy_min_abs=1e-6,
        factor_returns=None,
        prewhiten_meta=None,
        minvar_ridge=1e-4,
        minvar_box=(0.0, 0.1),
        turnover_cost_bps=1.0,
        minvar_condition_cap=1e6,
        preprocess_flags={},
        gating={"enable": True, "q_max": 1, "require_isolated": True},
        alignment_top_p=2,
        edge_mode="scm",
        edge_huber_c=1.5,
        use_tvector=True,
        diagnostics={"gating_trace": True},
    )

    diag_df = pd.read_csv(out_dir / "gating_diagnostics.csv")
    failures = diag_df[diag_df["skip_reason_primary"] == str(SkipReasonPrimary.DIAGNOSTIC_FAILURE)]
    assert not failures.empty
    assert failures["skip_reason_detail"].replace("", pd.NA).notna().all()
    assert failures["exception_type"].replace("", pd.NA).notna().all()
