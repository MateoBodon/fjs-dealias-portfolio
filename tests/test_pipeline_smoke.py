from __future__ import annotations

from inspect import isclass, isfunction
from pathlib import Path

from experiments.equity_panel import run as equity_run
from experiments.synthetic_oneway import run as synthetic_run

from finance import (
    OptimizationResult,
    build_design_matrix,
    compute_log_returns,
    evaluate_portfolio,
    ledoit_wolf_shrinkage,
    load_market_data,
    optimize_portfolio,
)
from fjs import (
    BalancedConfig,
    DealiasingResult,
    MarchenkoPasturModel,
    compute_balanced_weights,
    dealias_covariance,
    estimate_spectrum,
    marchenko_pastur_edges,
    marchenko_pastur_pdf,
)


def test_core_types_are_accessible() -> None:
    assert isclass(BalancedConfig)
    assert isclass(DealiasingResult)
    assert isclass(MarchenkoPasturModel)
    assert isclass(OptimizationResult)


def test_core_functions_are_callable() -> None:
    for candidate in [
        compute_balanced_weights,
        dealias_covariance,
        estimate_spectrum,
        marchenko_pastur_edges,
        marchenko_pastur_pdf,
        load_market_data,
        compute_log_returns,
        build_design_matrix,
        ledoit_wolf_shrinkage,
        optimize_portfolio,
        evaluate_portfolio,
    ]:
        assert isfunction(candidate)


def test_experiment_entry_points_are_callable() -> None:
    assert isfunction(synthetic_run.run_experiment)
    assert isfunction(equity_run.run_experiment)


def test_experiment_configs_load() -> None:
    synth_config = synthetic_run.load_config(
        Path("experiments/synthetic_oneway/config.yaml")
    )
    equity_config = equity_run.load_config(Path("experiments/equity_panel/config.yaml"))
    assert isinstance(synth_config, dict)
    assert isinstance(equity_config, dict)
