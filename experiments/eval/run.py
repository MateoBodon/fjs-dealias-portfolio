from __future__ import annotations

import argparse
import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

try:  # pragma: no cover - optional plotting dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - fallback when matplotlib missing
    plt = None

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines import load_observed_factors, prewhiten_returns
from baselines.factors import PrewhitenResult
from evaluation.dm import dm_test
from fjs.overlay import OverlayConfig, apply_overlay, detect_spikes
from experiments.eval.config import resolve_eval_config
from experiments.eval.diagnostics import DiagnosticReason

try:
    from data.loader import DailyLoaderConfig, DailyPanel, load_daily_panel
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    @dataclass(slots=True, frozen=True)
    class DailyLoaderConfig:
        winsor_lower: float = 0.005
        winsor_upper: float = 0.995
        min_history: int = 252
        forward_fill: bool = False

    @dataclass(slots=True, frozen=True)
    class DailyPanel:
        returns: pd.DataFrame
        meta: dict[str, object]

    def load_daily_panel(
        source: str | Path,
        *,
        config: DailyLoaderConfig | None = None,
    ) -> DailyPanel:
        cfg = config or DailyLoaderConfig()
        frame = pd.read_csv(source)
        if {"date", "ticker", "ret"}.issubset(frame.columns):
            frame["date"] = pd.to_datetime(frame["date"])
            frame = frame.dropna(subset=["date", "ticker", "ret"])
            frame = frame.sort_values(["date", "ticker"])
            frame = frame.drop_duplicates(subset=["date", "ticker"], keep="last")
            wide = (
                frame.pivot(index="date", columns="ticker", values="ret")
                .sort_index()
                .astype(float)
            )
        else:
            if "date" in frame.columns:
                frame["date"] = pd.to_datetime(frame["date"])
                frame = frame.set_index("date")
            else:
                frame = frame.rename(columns={frame.columns[0]: "date"})
                frame["date"] = pd.to_datetime(frame["date"])
                frame = frame.set_index("date")
            wide = frame.sort_index().astype(float)
        if cfg.forward_fill:
            wide = wide.ffill()
        wide = wide.dropna(axis=0, how="all")
        if wide.shape[0] < cfg.min_history:
            raise ValueError("Insufficient history for requested window.")
        meta = {
            "start": wide.index.min(),
            "end": wide.index.max(),
            "symbols": list(wide.columns),
            "n_days": int(wide.shape[0]),
        }
        return DailyPanel(returns=wide, meta=meta)


@dataclass(slots=True, frozen=True)
class EvalConfig:
    returns_csv: Path
    factors_csv: Path | None = None
    window: int = 126
    horizon: int = 21
    out_dir: Path = Path("reports/eval-latest")
    start: str | None = None
    end: str | None = None
    shrinker: str = "rie"
    seed: int = 0
    calm_quantile: float = 0.2
    crisis_quantile: float = 0.8
    vol_ewma_span: int = 21
    config_path: Path | None = None
    thresholds_path: Path | None = None
    echo_config: bool = True
    reason_codes: bool = True
    workers: int | None = None
    overlay_a_grid: int = 60
    overlay_seed: int | None = None


@dataclass(slots=True, frozen=True)
class EvalOutputs:
    metrics: dict[str, Path]
    risk: dict[str, Path]
    dm: dict[str, Path]
    diagnostics: dict[str, Path]
    plots: dict[str, Path]


_REGIMES = ("full", "calm", "crisis")


def _serialise_config(config: EvalConfig) -> dict[str, Any]:
    return {
        "returns_csv": str(config.returns_csv),
        "factors_csv": str(config.factors_csv) if config.factors_csv else None,
        "window": config.window,
        "horizon": config.horizon,
        "out_dir": str(config.out_dir),
        "start": config.start,
        "end": config.end,
        "shrinker": config.shrinker,
        "seed": config.seed,
        "calm_quantile": config.calm_quantile,
        "crisis_quantile": config.crisis_quantile,
        "vol_ewma_span": config.vol_ewma_span,
        "config_path": str(config.config_path) if config.config_path else None,
        "thresholds_path": str(config.thresholds_path) if config.thresholds_path else None,
        "echo_config": config.echo_config,
        "reason_codes": config.reason_codes,
        "workers": config.workers,
        "overlay_a_grid": config.overlay_a_grid,
        "overlay_seed": config.overlay_seed,
    }


def _mode_string(values: pd.Series) -> str:
    valid = values.dropna()
    if valid.empty:
        return ""
    try:
        mode_values = valid.mode(dropna=True)  # type: ignore[call-arg]
    except TypeError:
        mode_values = valid.astype(str).mode(dropna=True)  # type: ignore[call-arg]
    if mode_values.empty:
        return ""
    return str(mode_values.iloc[0])


def _vol_thresholds(
    vol_proxy: pd.Series,
    train_end: pd.Timestamp,
    config: EvalConfig,
) -> tuple[float, float]:
    window_slice = vol_proxy.loc[:train_end].dropna()
    if window_slice.empty:
        return float("inf"), float("-inf")
    calm_cut = float(window_slice.quantile(config.calm_quantile))
    crisis_cut = float(window_slice.quantile(config.crisis_quantile))
    if calm_cut > crisis_cut:
        calm_cut, crisis_cut = crisis_cut, calm_cut
    return calm_cut, crisis_cut


def parse_args(argv: Sequence[str] | None = None) -> tuple[EvalConfig, dict[str, Any]]:
    parser = argparse.ArgumentParser(description="Daily overlay evaluation with diagnostics.")
    parser.add_argument("--returns-csv", type=Path, required=True, help="Path to daily returns CSV.")
    parser.add_argument(
        "--factors-csv",
        type=Path,
        default=None,
        help="Optional FF5+MOM factor CSV (falls back to MKT proxy when absent).",
    )
    parser.add_argument("--window", type=int, default=None, help="Estimation window (days).")
    parser.add_argument("--horizon", type=int, default=None, help="Holdout horizon (days).")
    parser.add_argument("--start", type=str, default=None, help="Optional start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="Optional end date (YYYY-MM-DD).")
    parser.add_argument("--out", type=Path, default=None, help="Output directory.")
    parser.add_argument(
        "--shrinker",
        type=str,
        default=None,
        choices=["rie", "lw", "oas", "sample"],
        help="Baseline shrinker for non-detected directions.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed for gating utilities.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML configuration file applied after thresholds and before CLI overrides.",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="Optional JSON thresholds file applied after defaults and before YAML config.",
    )
    parser.add_argument(
        "--echo-config",
        action="store_true",
        help="Print the resolved configuration dictionary before running evaluation.",
    )
    parser.add_argument(
        "--calm-quantile",
        dest="calm_quantile",
        type=float,
        default=None,
        help="Override calm-volatility quantile (default from config layers).",
    )
    parser.add_argument(
        "--crisis-quantile",
        dest="crisis_quantile",
        type=float,
        default=None,
        help="Override crisis-volatility quantile (default from config layers).",
    )
    parser.add_argument(
        "--vol-ewma-span",
        dest="vol_ewma_span",
        type=int,
        default=None,
        help="EWMA span (in days) for volatility regime proxy.",
    )
    parser.add_argument(
        "--overlay-a-grid",
        dest="overlay_a_grid",
        type=int,
        default=None,
        help="Angular grid resolution for overlay t-vector search.",
    )
    parser.add_argument(
        "--overlay-seed",
        dest="overlay_seed",
        type=int,
        default=None,
        help="Optional seed override specifically for overlay search.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional worker count for experimental parallel evaluation.",
    )
    parser.add_argument(
        "--no-reason-codes",
        dest="reason_codes",
        action="store_false",
        help="Disable reason-code annotation in diagnostics.",
    )
    args = parser.parse_args(argv)
    resolved = resolve_eval_config(vars(args))
    config = resolved.config
    if config.echo_config or args.echo_config:
        print(json.dumps(resolved.resolved, indent=2, sort_keys=True))
    return config, resolved.resolved


def _compute_vol_proxy(returns: pd.DataFrame, span: int = 21) -> pd.Series:
    squared = returns.pow(2).mean(axis=1)
    ewma = squared.ewm(span=max(span, 2), adjust=False, min_periods=5).mean()
    proxy = np.sqrt(ewma)
    proxy.name = "vol_proxy"
    return proxy.dropna()


def _balanced_window(frame: pd.DataFrame, replicates: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
    week_ids = frame.index.to_period("W-MON")
    balanced: list[pd.DataFrame] = []
    for _, block in frame.groupby(week_ids):
        if block.shape[0] == replicates:
            balanced.append(block)
    if not balanced:
        raise ValueError("Window does not contain any fully populated weeks.")
    trimmed = pd.concat(balanced)
    groups = np.repeat(np.arange(len(balanced)), replicates)
    return trimmed, groups


def _min_variance_weights(covariance: np.ndarray, ridge: float = 5e-4) -> np.ndarray:
    p = covariance.shape[0]
    adjusted = covariance + float(ridge) * np.eye(p, dtype=np.float64)
    try:
        inv = np.linalg.pinv(adjusted, rcond=1e-8)
    except np.linalg.LinAlgError:
        return np.full(p, 1.0 / p, dtype=np.float64)
    ones = np.ones(p, dtype=np.float64)
    weights = inv @ ones
    denom = float(weights.sum())
    if abs(denom) <= 1e-12:
        return np.full(p, 1.0 / p, dtype=np.float64)
    return weights / denom


def _expected_shortfall(sigma: float, alpha: float = 0.05) -> float:
    z = stats.norm.ppf(alpha)
    return float(-(sigma * stats.norm.pdf(z) / alpha))


def _realised_tail_mean(returns: np.ndarray, var_threshold: float) -> float:
    tail = returns[returns < var_threshold]
    if tail.size == 0:
        return float("nan")
    return float(np.mean(tail))


def _window_regime(
    vol_proxy: pd.Series,
    date: pd.Timestamp,
    calm_cut: float,
    crisis_cut: float,
    *,
    fallback: float | None = None,
) -> str:
    if date not in vol_proxy.index:
        proxy_value = vol_proxy.reindex(vol_proxy.index.union([date])).sort_index().ffill().loc[date]
    else:
        proxy_value = vol_proxy.loc[date]
    if np.isnan(proxy_value) and fallback is not None:
        proxy_value = fallback
    if np.isnan(proxy_value):
        return "full"
    if proxy_value <= calm_cut:
        return "calm"
    if proxy_value >= crisis_cut:
        return "crisis"
    return "full"


def _prepare_returns(config: EvalConfig) -> tuple[DailyPanel, pd.DataFrame, PrewhitenResult]:
    loader_cfg = DailyLoaderConfig(min_history=config.window + config.horizon + 10)
    panel = load_daily_panel(config.returns_csv, config=loader_cfg)
    returns = panel.returns
    if config.start:
        returns = returns.loc[returns.index >= pd.to_datetime(config.start)]
    if config.end:
        returns = returns.loc[returns.index <= pd.to_datetime(config.end)]
    if returns.shape[0] < config.window + config.horizon + 5:
        raise ValueError("Not enough observations for requested window and horizon.")

    if config.factors_csv is not None:
        factors = load_observed_factors(path=config.factors_csv)
    else:
        try:
            factors = load_observed_factors(path=None, returns=returns)
        except FileNotFoundError:
            factors = load_observed_factors(returns=returns)
    try:
        whitening = prewhiten_returns(returns, factors)
        residuals = whitening.residuals
    except ValueError:
        residuals = returns.copy()
        betas = pd.DataFrame(
            np.zeros((residuals.shape[1], factors.shape[1] if hasattr(factors, "shape") else 0)),
            index=residuals.columns,
        )
        intercept = pd.Series(np.zeros(residuals.shape[1]), index=residuals.columns, name="intercept")
        r2 = pd.Series(np.zeros(residuals.shape[1]), index=residuals.columns, name="r_squared")
        whitening = PrewhitenResult(
            residuals=residuals,
            betas=betas,
            intercept=intercept,
            r_squared=r2,
            fitted=pd.DataFrame(np.zeros_like(residuals.to_numpy()), index=residuals.index, columns=residuals.columns),
            factors=pd.DataFrame(index=residuals.index, data=np.zeros((residuals.shape[0], 0))),
        )
    return panel, returns, whitening


def run_evaluation(
    config: EvalConfig,
    *,
    resolved_config: Mapping[str, Any] | None = None,
) -> EvalOutputs:
    panel, returns, whitening = _prepare_returns(config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    residuals = whitening.residuals
    vol_proxy_full = _compute_vol_proxy(residuals, span=config.vol_ewma_span)
    vol_proxy_past = vol_proxy_full.shift(1)

    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    regime_dirs = {reg: out_dir / reg for reg in _REGIMES}
    for dir_path in regime_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    resolved_payload = dict(resolved_config) if resolved_config is not None else _serialise_config(config)
    resolved_path = out_dir / "resolved_config.json"
    resolved_path.write_text(json.dumps(resolved_payload, indent=2, sort_keys=True))
    resolved_path_str = str(resolved_path)

    overlay_cfg = OverlayConfig(
        shrinker=config.shrinker,
        q_max=1,
        max_detections=1,
        edge_mode="tyler",
        seed=config.overlay_seed if config.overlay_seed is not None else config.seed,
        a_grid=int(config.overlay_a_grid),
    )

    def _evaluate_window(start: int) -> tuple[list[dict[str, object]], dict[str, object] | None]:
        fit = residuals.iloc[start : start + config.window]
        hold = residuals.iloc[start + config.window : start + config.window + config.horizon]
        if hold.empty:
            return [], None
        train_end = pd.to_datetime(fit.index[-1])
        calm_cut, crisis_cut = _vol_thresholds(vol_proxy_past, train_end, config)
        train_vol_slice = vol_proxy_past.loc[:train_end].dropna()
        fallback_vol = float(train_vol_slice.iloc[-1]) if not train_vol_slice.empty else float("nan")
        hold_start = pd.to_datetime(hold.index[0])
        regime = _window_regime(
            vol_proxy_past,
            hold_start,
            calm_cut,
            crisis_cut,
            fallback=fallback_vol,
        )
        vol_signal_series = vol_proxy_past.reindex(vol_proxy_past.index.union([hold_start])).sort_index().ffill()
        vol_signal_value = float(vol_signal_series.loc[hold_start]) if hold_start in vol_signal_series.index else float("nan")
        if np.isnan(vol_signal_value):
            vol_signal_value = fallback_vol

        try:
            fit_balanced, group_labels = _balanced_window(fit)
        except ValueError:
            reason_value = (
                DiagnosticReason.BALANCE_FAILURE.value if config.reason_codes else ""
            )
            diag_record = {
                "regime": regime,
                "detections": 0,
                "detection_rate": 0.0,
                "edge_margin_mean": 0.0,
                "stability_margin_mean": 0.0,
                "isolation_share": 0.0,
                "reason_code": reason_value,
                "resolved_config_path": resolved_path_str,
                "calm_threshold": float(calm_cut),
                "crisis_threshold": float(crisis_cut),
                "vol_signal": float(vol_signal_value),
            }
            return [], diag_record

        fit_matrix = fit_balanced.to_numpy(dtype=np.float64)
        p_assets = fit_matrix.shape[1]
        sample_cov = np.cov(fit_matrix, rowvar=False, ddof=1)
        group_count = int(np.unique(group_labels).size)

        reason = DiagnosticReason.NO_DETECTIONS
        if group_count < 5:
            detections: list[dict[str, object]] = []
            reason = DiagnosticReason.INSUFFICIENT_GROUPS
        else:
            try:
                detections = detect_spikes(fit_matrix, group_labels, config=overlay_cfg)
                reason = DiagnosticReason.ACCEPTED if detections else DiagnosticReason.NO_DETECTIONS
            except Exception:
                detections = []
                reason = DiagnosticReason.DETECTION_ERROR

        overlay_cov = apply_overlay(sample_cov, detections, observations=fit_matrix, config=overlay_cfg)
        baseline_cov = apply_overlay(sample_cov, [], observations=fit_matrix, config=overlay_cfg)
        covariances = {
            "overlay": overlay_cov,
            "baseline": baseline_cov,
            "sample": sample_cov,
        }

        hold_matrix = hold.to_numpy(dtype=np.float64)
        eq_weights = np.full(p_assets, 1.0 / p_assets, dtype=np.float64)
        mv_weights = _min_variance_weights(baseline_cov)
        weights_map = {"ew": eq_weights, "mv": mv_weights}

        metrics_block: list[dict[str, object]] = []
        for estimator, cov in covariances.items():
            for portfolio, weights in weights_map.items():
                forecast_var = float(weights.T @ cov @ weights)
                sigma = float(np.sqrt(max(forecast_var, 1e-12)))
                realised_returns = hold_matrix @ weights
                realised_var = float(np.var(realised_returns, ddof=1)) if realised_returns.size > 1 else float("nan")
                var95 = float(stats.norm.ppf(0.05) * sigma)
                es95 = _expected_shortfall(sigma)
                violations = realised_returns < var95
                violation_rate = float(np.mean(violations)) if violations.size else float("nan")
                realised_es = _realised_tail_mean(realised_returns, var95)
                mse = (forecast_var - realised_var) ** 2 if np.isfinite(realised_var) else float("nan")
                es_error = (es95 - realised_es) ** 2 if np.isfinite(realised_es) else float("nan")

                metrics_block.append(
                    {
                        "regime": regime,
                        "estimator": estimator,
                        "portfolio": portfolio,
                        "forecast_var": forecast_var,
                        "realised_var": realised_var,
                        "vaR95": var95,
                        "es95": es95,
                        "violation_rate": violation_rate,
                        "realised_es": realised_es,
                        "sq_error": mse,
                        "sq_error_es": es_error,
                    }
                )

        if detections:
            edge_margins = [float(det.get("edge_margin", 0.0) or 0.0) for det in detections]
            stability = [float(det.get("stability_margin", 0.0) or 0.0) for det in detections]
            isolation = [
                1.0 - min(1.0, float(det.get("off_component_ratio", 1.0) or 1.0))
                for det in detections
            ]
        else:
            edge_margins = []
            stability = []
            isolation = []

        reason_value = reason.value if config.reason_codes else ""
        diag_record = {
            "regime": regime,
            "detections": len(detections),
            "detection_rate": len(detections) / float(p_assets) if p_assets else 0.0,
            "edge_margin_mean": float(np.mean(edge_margins)) if edge_margins else 0.0,
            "stability_margin_mean": float(np.mean(stability)) if stability else 0.0,
            "isolation_share": float(np.mean(isolation)) if isolation else 0.0,
            "reason_code": reason_value,
            "resolved_config_path": resolved_path_str,
            "calm_threshold": float(calm_cut),
            "crisis_threshold": float(crisis_cut),
            "vol_signal": float(vol_signal_value),
        }
        return metrics_block, diag_record

    window_records: list[dict[str, object]] = []
    diagnostics_records: list[dict[str, object]] = []

    total_days = residuals.shape[0]
    start_indices = range(0, total_days - config.window - config.horizon + 1)
    if config.workers and config.workers > 1:
        max_workers = max(1, int(config.workers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for metrics_block, diag_record in executor.map(_evaluate_window, start_indices):
                if metrics_block:
                    window_records.extend(metrics_block)
                if diag_record is not None:
                    diagnostics_records.append(diag_record)
    else:
        for start in start_indices:
            metrics_block, diag_record = _evaluate_window(start)
            if metrics_block:
                window_records.extend(metrics_block)
            if diag_record is not None:
                diagnostics_records.append(diag_record)

    metrics_df = pd.DataFrame(window_records)
    diagnostics_df = pd.DataFrame(diagnostics_records)
    expected_diag_columns = [
        "regime",
        "detections",
        "detection_rate",
        "edge_margin_mean",
        "stability_margin_mean",
        "isolation_share",
        "reason_code",
        "resolved_config_path",
        "calm_threshold",
        "crisis_threshold",
        "vol_signal",
    ]
    if diagnostics_df.empty:
        diagnostics_df = pd.DataFrame(columns=expected_diag_columns)
    else:
        for column in expected_diag_columns:
            if column not in diagnostics_df.columns:
                diagnostics_df[column] = np.nan
        diagnostics_df = diagnostics_df[expected_diag_columns]

    outputs_metrics: dict[str, Path] = {}
    outputs_risk: dict[str, Path] = {}
    outputs_dm: dict[str, Path] = {}
    outputs_diag: dict[str, Path] = {}
    outputs_plots: dict[str, Path] = {}

    if metrics_df.empty:
        for regime, path in regime_dirs.items():
            empty = pd.DataFrame(
                columns=[
                    "regime",
                    "estimator",
                    "portfolio",
                    "forecast_var",
                    "realised_var",
                    "vaR95",
                    "es95",
                    "violation_rate",
                    "realised_es",
                    "sq_error",
                    "sq_error_es",
                ]
            )
            metrics_path = path / "metrics.csv"
            risk_path = path / "risk.csv"
            dm_path = path / "dm.csv"
            diag_path = path / "diagnostics.csv"
            empty.to_csv(metrics_path, index=False)
            empty.to_csv(risk_path, index=False)
            pd.DataFrame(columns=["portfolio", "baseline", "dm_stat", "p_value"]).to_csv(
                dm_path, index=False
            )
            diagnostics_df.to_csv(diag_path, index=False)
            outputs_metrics[regime] = metrics_path
            outputs_risk[regime] = risk_path
            outputs_dm[regime] = dm_path
            outputs_diag[regime] = diag_path
            outputs_plots[regime] = path / "delta_mse.png"
        return EvalOutputs(outputs_metrics, outputs_risk, outputs_dm, outputs_diag, outputs_plots)

    summaries = []
    for regime in _REGIMES:
        subset = metrics_df[metrics_df["regime"] == regime] if regime != "full" else metrics_df
        if subset.empty:
            continue
        summary = (
            subset.groupby(["estimator", "portfolio"])
            .agg(
                mse_mean=("sq_error", "mean"),
                es_mse_mean=("sq_error_es", "mean"),
                var95=("vaR95", "mean"),
                es95=("es95", "mean"),
                realised_var=("realised_var", "mean"),
                realised_es=("realised_es", "mean"),
                violation_rate=("violation_rate", "mean"),
                count=("sq_error", "count"),
            )
            .reset_index()
        )
        summary.insert(0, "regime", regime)
        summaries.append(summary)
    summary_df = pd.concat(summaries, ignore_index=True)

    baseline_mask = summary_df["estimator"] == "baseline"
    baseline_df = summary_df[baseline_mask].rename(
        columns={"mse_mean": "baseline_mse", "es_mse_mean": "baseline_es_mse"}
    )[["regime", "portfolio", "baseline_mse", "baseline_es_mse"]]
    summary_df = summary_df.merge(baseline_df, on=["regime", "portfolio"], how="left")
    summary_df["delta_mse_vs_baseline"] = summary_df["mse_mean"] - summary_df["baseline_mse"]
    summary_df["delta_es_vs_baseline"] = summary_df["es_mse_mean"] - summary_df["baseline_es_mse"]

    diagnostics_summary = (
        diagnostics_df.groupby("regime")
        .agg(
            detections=("detections", "mean"),
            detection_rate=("detection_rate", "mean"),
            edge_margin_mean=("edge_margin_mean", "mean"),
            stability_margin_mean=("stability_margin_mean", "mean"),
            isolation_share=("isolation_share", "mean"),
            calm_threshold=("calm_threshold", "mean"),
            crisis_threshold=("crisis_threshold", "mean"),
            vol_signal=("vol_signal", "mean"),
        )
        .reset_index()
    )
    if diagnostics_summary.empty:
        diagnostics_summary["reason_code"] = ""
        diagnostics_summary["resolved_config_path"] = resolved_path_str
        diagnostics_summary["calm_threshold"] = np.nan
        diagnostics_summary["crisis_threshold"] = np.nan
        diagnostics_summary["vol_signal"] = np.nan
    else:
        reason_summary = (
            diagnostics_df.groupby("regime")["reason_code"]
            .agg(_mode_string)
            .reset_index()
        )
        path_summary = (
            diagnostics_df.groupby("regime")["resolved_config_path"]
            .agg(
                lambda col: (
                    col.dropna().iloc[0] if not col.dropna().empty else resolved_path_str
                )
            )
            .reset_index()
        )
        diagnostics_summary = diagnostics_summary.merge(reason_summary, on="regime", how="left")
        diagnostics_summary = diagnostics_summary.merge(path_summary, on="regime", how="left")

    for regime, path in regime_dirs.items():
        subset_metrics = summary_df[summary_df["regime"] == regime]
        metrics_path = path / "metrics.csv"
        subset_metrics.to_csv(metrics_path, index=False)
        outputs_metrics[regime] = metrics_path

        risk_subset = metrics_df if regime == "full" else metrics_df[metrics_df["regime"] == regime]
        risk_path = path / "risk.csv"
        risk_columns = [
            "estimator",
            "portfolio",
            "regime",
            "vaR95",
            "es95",
            "violation_rate",
            "realised_es",
        ]
        risk_subset[risk_columns].to_csv(risk_path, index=False)
        outputs_risk[regime] = risk_path

        dm_rows = []
        for portfolio in ("ew", "mv"):
            overlay_errors = metrics_df[
                (metrics_df["regime"].eq(regime) if regime != "full" else True)
                & metrics_df["portfolio"].eq(portfolio)
                & metrics_df["estimator"].eq("overlay")
            ]["sq_error"].to_numpy()
            baseline_errors = metrics_df[
                (metrics_df["regime"].eq(regime) if regime != "full" else True)
                & metrics_df["portfolio"].eq(portfolio)
                & metrics_df["estimator"].eq("baseline")
            ]["sq_error"].to_numpy()
            if overlay_errors.size and baseline_errors.size:
                dm_stat, p_value = dm_test(overlay_errors, baseline_errors)
            else:
                dm_stat, p_value = float("nan"), float("nan")
            dm_rows.append(
                {
                    "portfolio": portfolio,
                    "baseline": "baseline",
                    "dm_stat": dm_stat,
                    "p_value": p_value,
                }
            )
        dm_path = path / "dm.csv"
        pd.DataFrame(dm_rows).to_csv(dm_path, index=False)
        outputs_dm[regime] = dm_path

        diag_path = path / "diagnostics.csv"
        diag_subset = diagnostics_summary[diagnostics_summary["regime"] == regime]
        diag_subset.to_csv(diag_path, index=False)
        outputs_diag[regime] = diag_path

        if plt is not None and not subset_metrics.empty:  # pragma: no cover - plotting smoke
            fig, ax = plt.subplots(figsize=(6, 4))
            pivot = subset_metrics.pivot(index="estimator", columns="portfolio", values="delta_mse_vs_baseline")
            pivot.plot(kind="bar", ax=ax, rot=0)
            ax.set_ylabel("ΔMSE vs baseline")
            ax.set_title(f"ΔMSE by estimator ({regime})")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            plot_path = path / "delta_mse.png"
            fig.savefig(plot_path)
            plt.close(fig)
            outputs_plots[regime] = plot_path
        else:
            outputs_plots[regime] = path / "delta_mse.png"

    return EvalOutputs(
        metrics=outputs_metrics,
        risk=outputs_risk,
        dm=outputs_dm,
        diagnostics=outputs_diag,
        plots=outputs_plots,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config, resolved = parse_args(argv)
    run_evaluation(config, resolved_config=resolved)


if __name__ == "__main__":  # pragma: no cover
    main()
