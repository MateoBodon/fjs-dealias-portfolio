"""Rolling evaluation harness for the de-aliasing overlay."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.baselines import EWMAConfig, ewma_covariance, rie_covariance
from src.data.loader import DailyLoaderConfig, load_daily_panel
from src.evaluation.dm import dm_test
from src.evaluation.evaluate import christoffersen_independence_test, kupiec_pof_test
from src.fjs.overlay import DetectionResult, OverlayConfig, apply_overlay, detect_spikes


@dataclass(slots=True, frozen=True)
class EvalConfig:
    regime: str = "full"
    output_dir: Path = Path("reports/rc-latest")
    returns_csv: Path | None = None
    start: str | None = None
    end: str | None = None
    window: int = 126
    horizon: int = 21
    ridge: float = 5e-4
    ewma_lambda: float = 0.97


@dataclass(slots=True)
class EvaluationResult:
    metrics: pd.DataFrame
    dm: pd.DataFrame
    var: pd.DataFrame
    window_errors_eq: Dict[str, list[float]]
    window_errors_mv: Dict[str, list[float]]


def parse_args(argv: Sequence[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(description="Rolling evaluation for de-aliasing overlay.")
    parser.add_argument("--regime", default="full", help="Regime label for outputs (full/calm/crisis).")
    parser.add_argument("--returns-csv", type=Path, dest="returns_csv", help="Path to returns CSV (date,ticker,ret).", default=None)
    parser.add_argument("--start", type=str, default=None, help="Optional start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="Optional end date (YYYY-MM-DD).")
    parser.add_argument("--window", type=int, default=126, help="Estimation window length (days).")
    parser.add_argument("--horizon", type=int, default=21, help="Out-of-sample horizon length (days).")
    parser.add_argument("--ridge", type=float, default=5e-4, help="Ridge added to covariance for min-var portfolios.")
    parser.add_argument("--ewma-lambda", type=float, default=0.97, help="EWMA decay parameter.")
    parser.add_argument("--out", type=Path, default=Path("reports/rc-latest"), help="Output directory for artifacts.")
    args = parser.parse_args(argv)
    return EvalConfig(
        regime=args.regime,
        output_dir=args.out,
        returns_csv=args.returns_csv,
        start=args.start,
        end=args.end,
        window=args.window,
        horizon=args.horizon,
        ridge=args.ridge,
        ewma_lambda=args.ewma_lambda,
    )


def run_evaluation(returns: pd.DataFrame, config: EvalConfig) -> EvaluationResult:
    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")
    returns = returns.sort_index()
    windows = _rolling_windows(returns, window=config.window, horizon=config.horizon)
    if not windows:
        raise ValueError("Not enough observations for the supplied window/horizon configuration.")

    estimators = ["overlay", "rie", "ewma", "sample"]
    window_errors_eq: Dict[str, list[float]] = {name: [] for name in estimators}
    window_errors_mv: Dict[str, list[float]] = {name: [] for name in estimators}
    violations_eq: Dict[str, list[bool]] = {name: [] for name in estimators}
    violations_mv: Dict[str, list[bool]] = {name: [] for name in estimators}

    for fit, hold in windows:
        covariances = _compute_covariances(fit, config)
        realised_cov = np.cov(hold.to_numpy(dtype=np.float64), rowvar=False, ddof=1)
        weights_eq = np.full(fit.shape[1], 1.0 / fit.shape[1], dtype=np.float64)

        for name, cov in covariances.items():
            weights_mv = _min_variance_weights(cov, ridge=config.ridge)
            mse_eq, vio_eq = _window_loss(hold, cov, weights_eq)
            mse_mv, vio_mv = _window_loss(hold, cov, weights_mv)
            window_errors_eq[name].append(mse_eq)
            window_errors_mv[name].append(mse_mv)
            violations_eq[name].extend(vio_eq)
            violations_mv[name].extend(vio_mv)

    metrics_rows = []
    var_rows = []
    alpha = 0.05

    baseline = "rie"
    for name in estimators:
        mse_eq = float(np.mean(window_errors_eq[name]))
        mse_mv = float(np.mean(window_errors_mv[name]))
        delta_eq = mse_eq - float(np.mean(window_errors_eq[baseline]))
        delta_mv = mse_mv - float(np.mean(window_errors_mv[baseline]))
        metrics_rows.append({
            "estimator": name,
            "portfolio": "ew",
            "mse": mse_eq,
            "delta_mse_vs_rie": delta_eq,
        })
        metrics_rows.append({
            "estimator": name,
            "portfolio": "mv",
            "mse": mse_mv,
            "delta_mse_vs_rie": delta_mv,
        })

        violations_eq_arr = np.asarray(violations_eq[name], dtype=bool)
        violations_mv_arr = np.asarray(violations_mv[name], dtype=bool)
        coverage_eq = float(np.mean(violations_eq_arr)) if violations_eq_arr.size else float("nan")
        coverage_mv = float(np.mean(violations_mv_arr)) if violations_mv_arr.size else float("nan")
        var_rows.append({
            "estimator": name,
            "portfolio": "ew",
            "violation_rate": coverage_eq,
            "kupiec_p": kupiec_pof_test(violations_eq_arr, alpha=alpha),
            "christoffersen_p": christoffersen_independence_test(violations_eq_arr),
        })
        var_rows.append({
            "estimator": name,
            "portfolio": "mv",
            "violation_rate": coverage_mv,
            "kupiec_p": kupiec_pof_test(violations_mv_arr, alpha=alpha),
            "christoffersen_p": christoffersen_independence_test(violations_mv_arr),
        })

    dm_rows = []
    for portfolio, errors in (("ew", window_errors_eq), ("mv", window_errors_mv)):
        if baseline in errors:
            base_errors = errors[baseline]
            overlay_errors = errors["overlay"]
            stat, pval = dm_test(overlay_errors, base_errors)
            dm_rows.append({"portfolio": portfolio, "baseline": baseline, "dm_stat": stat, "p_value": pval})

    metrics_df = pd.DataFrame(metrics_rows)
    dm_df = pd.DataFrame(dm_rows)
    var_df = pd.DataFrame(var_rows)
    return EvaluationResult(metrics=metrics_df, dm=dm_df, var=var_df, window_errors_eq=window_errors_eq, window_errors_mv=window_errors_mv)


def _rolling_windows(
    returns: pd.DataFrame,
    *,
    window: int,
    horizon: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    total = returns.shape[0]
    if window <= 0 or horizon <= 0:
        raise ValueError("window and horizon must be positive.")
    windows: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for start in range(0, total - window - horizon + 1):
        fit = returns.iloc[start : start + window]
        hold = returns.iloc[start + window : start + window + horizon]
        windows.append((fit, hold))
    return windows


def _compute_covariances(fit: pd.DataFrame, config: EvalConfig) -> Dict[str, np.ndarray]:
    observations = fit.to_numpy(dtype=np.float64)
    sample = np.cov(observations, rowvar=False, ddof=1)
    overlay_cfg = OverlayConfig(min_margin=0.1, min_isolation=0.05, max_detections=3, shrinkage=0.05, sample_count=fit.shape[0])
    detection: DetectionResult = detect_spikes(sample, samples=observations, config=overlay_cfg)
    overlay_cov = apply_overlay(sample, detection, config=overlay_cfg)
    rie_cov = rie_covariance(observations)
    ewma_cov = ewma_covariance(observations, config=EWMAConfig(lambda_=config.ewma_lambda))
    return {
        "overlay": overlay_cov,
        "rie": rie_cov,
        "ewma": ewma_cov,
        "sample": sample,
    }


def _min_variance_weights(cov: np.ndarray, *, ridge: float) -> np.ndarray:
    p = cov.shape[0]
    adjusted = cov + float(ridge) * np.eye(p)
    inv = np.linalg.pinv(adjusted, rcond=1e-8)
    ones = np.ones(p, dtype=np.float64)
    weights = inv @ ones
    denom = float(np.sum(weights))
    if abs(denom) < 1e-12:
        return np.full(p, 1.0 / p)
    return weights / denom


def _window_loss(
    hold: pd.DataFrame,
    covariance: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, list[bool]]:
    sigma_hat = float(weights @ covariance @ weights)
    sigma_hat = max(sigma_hat, 1e-10)
    projected = hold.to_numpy(dtype=np.float64) @ weights
    squared = projected**2
    mse = float(np.mean((squared - sigma_hat) ** 2))
    sigma = float(np.sqrt(sigma_hat))
    alpha = 0.05
    z_alpha = stats.norm.ppf(alpha)
    var_forecast = sigma * z_alpha
    violations = (projected < var_forecast).tolist()
    return mse, violations


def save_results(result: EvaluationResult, config: EvalConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / f"metrics_{config.regime}.csv"
    dm_path = config.output_dir / f"dm_{config.regime}.csv"
    var_path = config.output_dir / f"var_{config.regime}.csv"
    result.metrics.to_csv(metrics_path, index=False)
    result.dm.to_csv(dm_path, index=False)
    result.var.to_csv(var_path, index=False)
    _maybe_plot_delta_mse(result.metrics, config)


def _maybe_plot_delta_mse(metrics: pd.DataFrame, config: EvalConfig) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        return
    pivot = metrics.pivot(index="estimator", columns="portfolio", values="delta_mse_vs_rie")
    fig, ax = plt.subplots(figsize=(6, 4))
    pivot.plot.bar(ax=ax, rot=0)
    ax.set_ylabel("ΔMSE vs RIE")
    ax.set_title(f"ΔMSE by estimator ({config.regime})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.output_dir / f"delta_mse_{config.regime}.png")
    plt.close(fig)


def load_returns_from_csv(
    path: Path,
    *,
    start: str | None = None,
    end: str | None = None,
    min_history: int = 252,
) -> pd.DataFrame:
    panel = load_daily_panel(path, config=DailyLoaderConfig(min_history=min_history))
    returns = panel.returns
    if start is not None:
        returns = returns.loc[returns.index >= pd.to_datetime(start)]
    if end is not None:
        returns = returns.loc[returns.index <= pd.to_datetime(end)]
    return returns


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    min_history = max(config.window + config.horizon, 30)
    if config.returns_csv is None:
        default_path = Path("data/returns_daily.csv")
        if not default_path.exists():
            raise FileNotFoundError("--returns-csv not provided and default data/returns_daily.csv missing.")
        returns = load_returns_from_csv(default_path, start=config.start, end=config.end, min_history=min_history)
    else:
        returns = load_returns_from_csv(config.returns_csv, start=config.start, end=config.end, min_history=min_history)
    result = run_evaluation(returns, config)
    save_results(result, config)


if __name__ == "__main__":
    main()
