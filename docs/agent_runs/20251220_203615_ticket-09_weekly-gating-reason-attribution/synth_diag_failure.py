from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.equity_panel import run


RUN_NAME = os.environ.get(
    "RUN_NAME", "20251220_203615_ticket-09_weekly-gating-reason-attribution"
)
OUTPUT_DIR = (
    Path("experiments/equity_panel")
    / f"outputs_ticket-09_synth_failure_{RUN_NAME}"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    dates = pd.date_range("2023-01-02", periods=60, freq="B")
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(
        rng.normal(scale=0.01, size=(len(dates), 6)),
        index=dates,
        columns=[f"S{idx:02d}" for idx in range(6)],
    )

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic diagnostic failure for ticket-09")

    run.dealias_search = boom  # type: ignore

    run._run_single_period(
        daily_returns=returns,
        start=dates[0],
        end=dates[-1],
        output_dir=OUTPUT_DIR,
        window_weeks=2,
        horizon_weeks=1,
        max_windows=None,
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
        label="synthetic_diag_failure",
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


if __name__ == "__main__":
    main()
