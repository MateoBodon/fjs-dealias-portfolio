from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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


@dataclass(slots=True, frozen=True)
class EvalOutputs:
    metrics: dict[str, Path]
    risk: dict[str, Path]
    dm: dict[str, Path]
    diagnostics: dict[str, Path]
    plots: dict[str, Path]


_REGIMES = ("full", "calm", "crisis")


def parse_args(argv: Sequence[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(description="Daily overlay evaluation with diagnostics.")
    parser.add_argument("--returns-csv", type=Path, required=True, help="Path to daily returns CSV.")
    parser.add_argument(
        "--factors-csv",
        type=Path,
        default=None,
        help="Optional FF5+MOM factor CSV (falls back to MKT proxy when absent).",
    )
    parser.add_argument("--window", type=int, default=126, help="Estimation window (days).")
    parser.add_argument("--horizon", type=int, default=21, help="Holdout horizon (days).")
    parser.add_argument("--start", type=str, default=None, help="Optional start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="Optional end date (YYYY-MM-DD).")
    parser.add_argument("--out", type=Path, default=Path("reports/eval-latest"), help="Output directory.")
    parser.add_argument(
        "--shrinker",
        type=str,
        default="rie",
        choices=["rie", "lw", "oas", "sample"],
        help="Baseline shrinker for non-detected directions.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for gating utilities.")
    args = parser.parse_args(argv)
    return EvalConfig(
        returns_csv=args.returns_csv,
        factors_csv=args.factors_csv,
        window=args.window,
        horizon=args.horizon,
        out_dir=args.out,
        start=args.start,
        end=args.end,
        shrinker=args.shrinker,
        seed=args.seed,
    )


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
) -> str:
    if date not in vol_proxy.index:
        proxy_value = vol_proxy.reindex(vol_proxy.index.union([date])).sort_index().ffill().loc[date]
    else:
        proxy_value = vol_proxy.loc[date]
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


def run_evaluation(config: EvalConfig) -> EvalOutputs:
    panel, returns, whitening = _prepare_returns(config)
    residuals = whitening.residuals
    vol_proxy = _compute_vol_proxy(residuals)
    calm_cut = float(vol_proxy.quantile(0.2)) if not vol_proxy.empty else float("inf")
    crisis_cut = float(vol_proxy.quantile(0.8)) if not vol_proxy.empty else float("-inf")

    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    regime_dirs = {reg: out_dir / reg for reg in _REGIMES}
    for dir_path in regime_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    overlay_cfg = OverlayConfig(
        shrinker=config.shrinker,
        q_max=1,
        max_detections=1,
        edge_mode="tyler",
        seed=config.seed,
        a_grid=60,
    )

    window_records: list[dict[str, object]] = []
    diagnostics_records: list[dict[str, object]] = []

    total_days = residuals.shape[0]
    for start in range(0, total_days - config.window - config.horizon + 1):
        fit = residuals.iloc[start : start + config.window]
        hold = residuals.iloc[start + config.window : start + config.window + config.horizon]
        if hold.empty:
            continue
        hold_start = pd.to_datetime(hold.index[0])
        regime = _window_regime(vol_proxy, hold_start, calm_cut, crisis_cut)
        try:
            fit_balanced, group_labels = _balanced_window(fit)
        except ValueError:
            continue
        fit_matrix = fit_balanced.to_numpy(dtype=np.float64)
        p_assets = fit_matrix.shape[1]
        sample_cov = np.cov(fit_matrix, rowvar=False, ddof=1)
        group_count = int(np.unique(group_labels).size)
        try:
            detections = (
                detect_spikes(fit_matrix, group_labels, config=overlay_cfg)
                if group_count >= 5
                else []
            )
        except Exception:
            detections = []
        overlay_cov = apply_overlay(
            sample_cov,
            detections,
            observations=fit_matrix,
            config=overlay_cfg,
        )
        baseline_cov = apply_overlay(
            sample_cov,
            [],
            observations=fit_matrix,
            config=overlay_cfg,
        )
        covariances = {
            "overlay": overlay_cov,
            "baseline": baseline_cov,
            "sample": sample_cov,
        }

        hold_matrix = hold.to_numpy(dtype=np.float64)
        eq_weights = np.full(p_assets, 1.0 / p_assets, dtype=np.float64)
        mv_weights = _min_variance_weights(baseline_cov)
        weights_map = {"ew": eq_weights, "mv": mv_weights}

        for estimator, cov in covariances.items():
            for portfolio, weights in weights_map.items():
                forecast_var = float(weights.T @ cov @ weights)
                sigma = float(np.sqrt(max(forecast_var, 1e-12)))
                realised_returns = hold_matrix @ weights
                if realised_returns.size > 1:
                    realised_var = float(np.var(realised_returns, ddof=1))
                else:
                    realised_var = float("nan")
                var95 = float(stats.norm.ppf(0.05) * sigma)
                es95 = _expected_shortfall(sigma)
                violations = realised_returns < var95
                violation_rate = float(np.mean(violations)) if violations.size else float("nan")
                realised_es = _realised_tail_mean(realised_returns, var95)
                mse = (forecast_var - realised_var) ** 2 if np.isfinite(realised_var) else float("nan")
                es_error = (es95 - realised_es) ** 2 if np.isfinite(realised_es) else float("nan")

                window_records.append(
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
        diagnostics_records.append(
            {
                "regime": regime,
                "detections": len(detections),
                "detection_rate": len(detections) / float(p_assets),
                "edge_margin_mean": float(np.mean(edge_margins)) if edge_margins else 0.0,
                "stability_margin_mean": float(np.mean(stability)) if stability else 0.0,
                "isolation_share": float(np.mean(isolation)) if isolation else 0.0,
            }
        )

    metrics_df = pd.DataFrame(window_records)
    diagnostics_df = pd.DataFrame(diagnostics_records)

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
        )
        .reset_index()
    )

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
    config = parse_args(argv)
    run_evaluation(config)


if __name__ == "__main__":  # pragma: no cover
    main()
