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
from evaluation.evaluate import alignment_diagnostics
from fjs.overlay import OverlayConfig, apply_overlay, detect_spikes
from experiments.daily.grouping import (
    GroupingError,
    group_by_day_of_week,
    group_by_vol_state,
    group_by_week,
)
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
    mv_gamma: float = 5e-4
    mv_tau: float = 0.0
    bootstrap_samples: int = 0
    require_isolated: bool = True
    q_max: int = 1
    edge_mode: str = "tyler"
    angle_min_cos: float | None = None
    alignment_top_p: int = 3
    cs_drop_top_frac: float | None = None
    prewhiten: str = "ff5mom"
    calm_window_sample: int | None = None
    crisis_window_top_k: int | None = None
    group_design: str = "week"
    group_min_count: int = 5
    group_min_replicates: int = 3
    ewma_halflife: float = 30.0


@dataclass(slots=True, frozen=True)
class PrewhitenTelemetry:
    mode_requested: str
    mode_effective: str
    factor_columns: tuple[str, ...]
    beta_abs_mean: float
    beta_abs_std: float
    beta_abs_median: float
    r2_mean: float
    r2_median: float


@dataclass(slots=True, frozen=True)
class EvalOutputs:
    metrics: dict[str, Path]
    risk: dict[str, Path]
    dm: dict[str, Path]
    diagnostics: dict[str, Path]
    plots: dict[str, Path]
    diagnostics_detail: dict[str, Path]


_REGIMES = ("full", "calm", "crisis")

_DOW_LABELS = {
    0: "mon",
    1: "tue",
    2: "wed",
    3: "thu",
    4: "fri",
}

_VOL_LABELS = {
    0: "calm",
    1: "mid",
    2: "crisis",
}

_FACTOR_SETS: dict[str, tuple[str, ...]] = {
    "ff5mom": ("MKT", "SMB", "HML", "RMW", "CMA", "MOM"),
    "ff5": ("MKT", "SMB", "HML", "RMW", "CMA"),
    "mkt": ("MKT",),
}

_FACTOR_FALLBACKS: dict[str, tuple[str, ...]] = {
    "ff5mom": ("ff5mom", "ff5", "mkt"),
    "ff5": ("ff5", "mkt"),
}


def _format_group_label_counts(labels: np.ndarray, design: str) -> tuple[str, dict[int, int]]:
    if labels.size == 0:
        return "", {}
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size == 0:
        return "", {}
    design_key = design.lower()
    label_map = _DOW_LABELS if design_key == "dow" else _VOL_LABELS if design_key == "vol" else {}
    entries: list[str] = []
    counts_map: dict[int, int] = {}
    pairs = [(int(raw_label), int(raw_count)) for raw_label, raw_count in zip(unique.tolist(), counts.tolist())]
    for label, count in sorted(pairs, key=lambda pair: pair[0]):
        counts_map[label] = count
        label_name = label_map.get(label, str(label))
        entries.append(f"{label_name}:{count}")
    return "|".join(entries), counts_map


def _vol_state_label(value: float, calm_cut: float, crisis_cut: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= calm_cut:
        return "calm"
    if value >= crisis_cut:
        return "crisis"
    return "mid"


def _identity_prewhiten_result(
    returns: pd.DataFrame,
    factor_cols: Sequence[str] | None = None,
) -> PrewhitenResult:
    columns = list(factor_cols or [])
    assets = list(returns.columns)
    betas = pd.DataFrame(
        np.zeros((len(assets), len(columns)), dtype=np.float64),
        index=assets,
        columns=columns,
    )
    intercept = pd.Series(np.zeros(len(assets), dtype=np.float64), index=assets, name="intercept")
    r_squared = pd.Series(np.zeros(len(assets), dtype=np.float64), index=assets, name="r_squared")
    fitted = pd.DataFrame(
        np.zeros_like(returns.to_numpy(dtype=np.float64, copy=True)),
        index=returns.index,
        columns=assets,
    )
    factor_frame = pd.DataFrame(
        np.zeros((returns.shape[0], len(columns)), dtype=np.float64),
        index=returns.index,
        columns=columns,
    )
    return PrewhitenResult(
        residuals=returns.copy(),
        betas=betas,
        intercept=intercept,
        r_squared=r_squared,
        fitted=fitted,
        factors=factor_frame,
    )


def _select_prewhiten_factors(
    factors: pd.DataFrame | None,
    requested: str,
) -> tuple[str, pd.DataFrame | None]:
    if factors is None or factors.empty:
        return "off", None
    requested_key = requested.lower()
    if requested_key == "off":
        return "off", None
    candidate_modes = _FACTOR_FALLBACKS.get(requested_key, ())
    for mode in candidate_modes:
        required = _FACTOR_SETS.get(mode, ())
        if all(col in factors.columns for col in required):
            subset = factors.loc[:, list(required)].copy()
            return mode, subset
    if "MKT" in factors.columns:
        return "mkt", factors.loc[:, ["MKT"]].copy()
    return "off", None


def _beta_abs_stats(betas: pd.DataFrame) -> tuple[float, float, float]:
    if betas.empty:
        return 0.0, 0.0, 0.0
    numeric = betas.select_dtypes(include=["number"])
    if numeric.empty:
        return 0.0, 0.0, 0.0
    values = np.abs(numeric.to_numpy(dtype=np.float64, copy=True))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    median = float(np.median(values))
    return mean, std, median


def _compute_prewhiten_telemetry(
    whitening: PrewhitenResult,
    *,
    requested_mode: str,
    effective_mode: str,
) -> PrewhitenTelemetry:
    r2_series = whitening.r_squared if not whitening.r_squared.empty else pd.Series(dtype=np.float64)
    r2_vals = r2_series.to_numpy(dtype=np.float64, copy=True) if not r2_series.empty else np.array([], dtype=np.float64)
    r2_vals = r2_vals[np.isfinite(r2_vals)] if r2_vals.size else r2_vals
    r2_mean = float(np.mean(r2_vals)) if r2_vals.size else 0.0
    r2_median = float(np.median(r2_vals)) if r2_vals.size else 0.0
    beta_mean, beta_std, beta_median = _beta_abs_stats(whitening.betas)
    factor_columns = tuple(whitening.betas.columns.tolist())
    return PrewhitenTelemetry(
        mode_requested=requested_mode,
        mode_effective=effective_mode,
        factor_columns=factor_columns,
        beta_abs_mean=beta_mean,
        beta_abs_std=beta_std,
        beta_abs_median=beta_median,
        r2_mean=r2_mean,
        r2_median=r2_median,
    )


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
        "mv_gamma": config.mv_gamma,
        "mv_tau": config.mv_tau,
        "bootstrap_samples": config.bootstrap_samples,
        "require_isolated": config.require_isolated,
        "q_max": config.q_max,
        "edge_mode": config.edge_mode,
        "angle_min_cos": config.angle_min_cos,
        "alignment_top_p": config.alignment_top_p,
        "cs_drop_top_frac": config.cs_drop_top_frac,
        "prewhiten": config.prewhiten,
        "calm_window_sample": config.calm_window_sample,
        "crisis_window_top_k": config.crisis_window_top_k,
        "group_design": config.group_design,
        "group_min_count": config.group_min_count,
        "group_min_replicates": config.group_min_replicates,
        "ewma_halflife": config.ewma_halflife,
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


def _aligned_error_table(
    metrics: pd.DataFrame,
    regime: str,
    portfolio: str,
) -> pd.DataFrame:
    mask = metrics["portfolio"].eq(portfolio)
    if regime != "full":
        mask &= metrics["regime"].eq(regime)
    subset = metrics.loc[mask, ["window_id", "estimator", "sq_error"]]
    if subset.empty:
        return pd.DataFrame(columns=["overlay", "baseline"])
    pivot = subset.pivot_table(
        index="window_id",
        columns="estimator",
        values="sq_error",
        aggfunc="first",
    )
    if "overlay" not in pivot.columns or "baseline" not in pivot.columns:
        return pd.DataFrame(columns=["overlay", "baseline"])
    return pivot[["overlay", "baseline"]].dropna()


def _aligned_dm_stat(
    metrics: pd.DataFrame,
    regime: str,
    portfolio: str,
) -> tuple[float, float, int]:
    aligned = _aligned_error_table(metrics, regime, portfolio)
    n_eff = int(aligned.shape[0])
    if n_eff < 2:
        return float("nan"), float("nan"), n_eff
    dm_stat, p_value = dm_test(
        aligned["overlay"].to_numpy(),
        aligned["baseline"].to_numpy(),
    )
    return dm_stat, p_value, n_eff


def _bootstrap_delta_mse(
    diffs: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
    block_size: int | None = None,
) -> tuple[float, float]:
    n = diffs.size
    if n == 0 or resamples <= 0:
        return float("nan"), float("nan")
    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))
    block_size = max(1, min(block_size, n))
    num_blocks = int(np.ceil(n / block_size))
    stats = np.empty(resamples, dtype=np.float64)
    for idx in range(resamples):
        sample = np.empty(n, dtype=np.float64)
        cursor = 0
        for _ in range(num_blocks):
            start = int(rng.integers(0, n - block_size + 1))
            block = diffs[start : start + block_size]
            end = min(cursor + block.size, n)
            sample[cursor:end] = block[: end - cursor]
            cursor = end
            if cursor >= n:
                break
        if cursor < n:
            remainder = diffs[: n - cursor]
            sample[cursor:] = remainder
        stats[idx] = float(np.mean(sample))
    lower = float(np.quantile(stats, 0.025))
    upper = float(np.quantile(stats, 0.975))
    return lower, upper


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
    parser.set_defaults(require_isolated=None, prewhiten=None)
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
        choices=["rie", "lw", "oas", "sample", "quest", "ewma"],
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
        "--ewma-halflife",
        dest="ewma_halflife",
        type=float,
        default=None,
        help="Half-life (in days) for EWMA shrinkage baseline (requires --shrinker ewma).",
    )
    parser.add_argument(
        "--edge-mode",
        dest="edge_mode",
        type=str,
        default=None,
        choices=["tyler", "scm", "huber"],
        help="Edge estimation mode for dealias search (default sourced from config layers).",
    )
    parser.add_argument(
        "--q-max",
        dest="q_max",
        type=int,
        default=None,
        help="Maximum detections retained per window.",
    )
    parser.add_argument(
        "--angle-min-cos",
        dest="angle_min_cos",
        type=float,
        default=None,
        help="Minimum cosine alignment with PCA subspace required to accept detections.",
    )
    parser.add_argument(
        "--alignment-top-p",
        dest="alignment_top_p",
        type=int,
        default=None,
        help="Number of principal components used for alignment diagnostics.",
    )
    parser.add_argument(
        "--calm-window-sample",
        dest="calm_window_sample",
        type=int,
        default=None,
        help="Uniformly sample this many calm windows (omit or set <=0 to disable).",
    )
    parser.add_argument(
        "--crisis-window-topk",
        dest="crisis_window_top_k",
        type=int,
        default=None,
        help="Select top-K crisis windows by edge margin (omit or set <=0 to disable).",
    )
    parser.add_argument(
        "--cs-drop-top-frac",
        dest="cs_drop_top_frac",
        type=float,
        default=None,
        help="Fraction of leading eigenvalues dropped when estimating Cs (0 disables).",
    )
    parser.add_argument(
        "--require-isolated",
        dest="require_isolated",
        action="store_true",
        help="Require MP-isolated spikes for substitution (default true).",
    )
    parser.add_argument(
        "--allow-non-isolated",
        dest="require_isolated",
        action="store_false",
        help="Permit non-isolated detections to substitute eigenvalues.",
    )
    parser.add_argument(
        "--mv-gamma",
        dest="mv_gamma",
        type=float,
        default=None,
        help="Ridge regulariser for minimum-variance weights (default sourced from config layers).",
    )
    parser.add_argument(
        "--mv-tau",
        dest="mv_tau",
        type=float,
        default=None,
        help="Turnover penalty for minimum-variance weights; set to 0 for no penalty.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        dest="bootstrap_samples",
        type=int,
        default=None,
        help="Number of block bootstrap resamples for ΔMSE confidence bands (0 disables).",
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
    parser.add_argument(
        "--prewhiten",
        type=str,
        choices=["off", "ff5", "ff5mom"],
        default=None,
        help="Observed-factor prewhitening mode (default sourced from config layers).",
    )
    parser.add_argument(
        "--group-design",
        dest="group_design",
        type=str,
        default=None,
        choices=["week", "dow", "vol"],
        help="Replicate grouping design for the detection window.",
    )
    parser.add_argument(
        "--group-min-count",
        dest="group_min_count",
        type=int,
        default=None,
        help="Minimum number of groups required to attempt detection.",
    )
    parser.add_argument(
        "--group-min-replicates",
        dest="group_min_replicates",
        type=int,
        default=None,
        help="Minimum replicates per group for balancing.",
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


def _write_prewhiten_diagnostics(
    out_dir: Path,
    whitening: PrewhitenResult,
    telemetry: PrewhitenTelemetry,
) -> None:
    diag_path = out_dir / "prewhiten_diagnostics.csv"
    summary_path = out_dir / "prewhiten_summary.json"
    betas = whitening.betas.copy()
    betas["intercept"] = whitening.intercept
    betas["r_squared"] = whitening.r_squared
    betas.to_csv(diag_path)
    summary = {
        "asset_count": int(whitening.r_squared.shape[0]) if not whitening.r_squared.empty else 0,
        "mean_r_squared": telemetry.r2_mean,
        "median_r_squared": telemetry.r2_median,
        "mode_requested": telemetry.mode_requested,
        "mode_effective": telemetry.mode_effective,
        "beta_abs_mean": telemetry.beta_abs_mean,
        "beta_abs_std": telemetry.beta_abs_std,
        "beta_abs_median": telemetry.beta_abs_median,
        "factor_columns": list(telemetry.factor_columns),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_overlay_toggle(path: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        content = ["# Overlay Toggle", "", "No detection telemetry available."]
    else:
        columns = [
            ("detection_rate", "Detection Rate"),
            ("edge_margin_mean", "Edge Margin"),
            ("isolation_share", "Isolation Share"),
            ("alignment_cos_mean", "Alignment Cos"),
            ("prewhiten_r2_mean", "Prewhiten R²"),
            ("prewhiten_beta_abs_mean", "|β| Mean"),
            ("residual_energy_mean", "Residual Energy"),
            ("acceptance_delta", "Acceptance Δ"),
        ]
        header = "| Regime | " + " | ".join(title for _, title in columns) + " |"
        separator = "|" + " --- |" * (len(columns) + 1)
        rows = ["# Overlay Toggle", "", header, separator]
        for _, row in summary.iterrows():
            values: list[str] = []
            for col, _ in columns:
                value = row.get(col)
                if pd.isna(value):
                    values.append("n/a")
                else:
                    values.append(f"{float(value):.3f}")
            rows.append(f"| {row['regime']} | " + " | ".join(values) + " |")
        content = rows
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def _build_grouped_window(
    frame: pd.DataFrame,
    *,
    config: EvalConfig,
    calm_threshold: float,
    crisis_threshold: float,
    vol_proxy: pd.Series,
) -> tuple[pd.DataFrame, np.ndarray]:
    design = (config.group_design or "week").lower()
    min_replicates = max(1, int(config.group_min_replicates))
    if design == "dow":
        return group_by_day_of_week(frame, min_weeks=max(min_replicates, 3))
    if design == "vol":
        return group_by_vol_state(
            frame,
            vol_proxy=vol_proxy,
            calm_threshold=float(calm_threshold),
            crisis_threshold=float(crisis_threshold),
            min_replicates=min_replicates,
        )
    return group_by_week(frame, replicates=5)


def _min_variance_weights(
    covariance: np.ndarray,
    *,
    gamma: float,
    tau: float,
    prev_weights: np.ndarray | None,
) -> np.ndarray:
    p = covariance.shape[0]
    identity = np.eye(p, dtype=np.float64)
    adjusted = np.asarray(covariance, dtype=np.float64) + float(gamma + tau) * identity
    rhs = np.zeros(p + 1, dtype=np.float64)
    if tau > 0.0 and prev_weights is not None and prev_weights.shape[0] == p:
        rhs[:p] = float(tau) * prev_weights
    K = np.zeros((p + 1, p + 1), dtype=np.float64)
    K[:p, :p] = adjusted
    K[:p, -1] = 1.0
    K[-1, :p] = 1.0
    rhs[-1] = 1.0
    try:
        solution = np.linalg.solve(K, rhs)
        weights = solution[:p]
    except np.linalg.LinAlgError:
        try:
            solution, *_ = np.linalg.lstsq(K, rhs, rcond=None)
            weights = solution[:p]
        except np.linalg.LinAlgError:
            return np.full(p, 1.0 / p, dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        return np.full(p, 1.0 / p, dtype=np.float64)
    total = float(weights.sum())
    if abs(total) <= 1e-12:
        return np.full(p, 1.0 / p, dtype=np.float64)
    return weights / total


def _expected_shortfall(sigma: float, alpha: float = 0.05) -> float:
    z = stats.norm.ppf(alpha)
    return float(-(sigma * stats.norm.pdf(z) / alpha))


def _realised_tail_mean(returns: np.ndarray, var_threshold: float) -> float:
    tail = returns[returns < var_threshold]
    if tail.size == 0:
        return float("nan")
    return float(np.mean(tail))


def _limit_windows_by_regime(
    metrics_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    *,
    calm_limit: int | None,
    crisis_limit: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics_df.empty or diagnostics_df.empty:
        return metrics_df, diagnostics_df
    if "window_id" not in diagnostics_df.columns or "window_id" not in metrics_df.columns:
        return metrics_df, diagnostics_df

    calm_limit = int(calm_limit) if calm_limit is not None and calm_limit > 0 else None
    crisis_limit = int(crisis_limit) if crisis_limit is not None and crisis_limit > 0 else None
    if calm_limit is None and crisis_limit is None:
        return metrics_df, diagnostics_df

    keep_ids: set[int] = set()
    rng = np.random.default_rng(seed + 211)

    for regime, frame in diagnostics_df.groupby("regime"):
        window_ids = frame["window_id"].dropna().to_numpy(dtype=int, copy=False)
        if window_ids.size == 0:
            continue
        if regime == "calm" and calm_limit is not None and window_ids.size > calm_limit:
            chosen = rng.choice(window_ids, size=calm_limit, replace=False)
        elif regime == "crisis" and crisis_limit is not None:
            edge_sorted = frame.assign(edge_sort=frame["edge_margin_mean"].fillna(-np.inf))
            edge_sorted = edge_sorted.sort_values(["edge_sort", "window_id"], ascending=[False, True])
            limited = edge_sorted["window_id"].to_numpy(dtype=int)
            if crisis_limit < limited.size:
                chosen = limited[:crisis_limit]
            else:
                chosen = limited
        else:
            chosen = window_ids
        keep_ids.update(int(x) for x in np.atleast_1d(chosen))

    if not keep_ids:
        return metrics_df, diagnostics_df

    metrics_filtered = metrics_df[metrics_df["window_id"].isin(keep_ids)].reset_index(drop=True)
    diagnostics_filtered = diagnostics_df[diagnostics_df["window_id"].isin(keep_ids)].reset_index(drop=True)
    return metrics_filtered, diagnostics_filtered


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


def _prepare_returns(config: EvalConfig) -> tuple[DailyPanel, pd.DataFrame, PrewhitenResult, PrewhitenTelemetry]:
    loader_cfg = DailyLoaderConfig(min_history=config.window + config.horizon + 10)
    panel = load_daily_panel(config.returns_csv, config=loader_cfg)
    returns = panel.returns
    if config.start:
        returns = returns.loc[returns.index >= pd.to_datetime(config.start)]
    if config.end:
        returns = returns.loc[returns.index <= pd.to_datetime(config.end)]
    if returns.shape[0] < config.window + config.horizon + 5:
        raise ValueError("Not enough observations for requested window and horizon.")

    requested_mode = str(config.prewhiten or "off").lower()
    if requested_mode not in {"off", "ff5", "ff5mom"}:
        requested_mode = "off"

    factors_source: pd.DataFrame | None = None
    if requested_mode != "off":
        if config.factors_csv is not None:
            try:
                factors_source = load_observed_factors(path=config.factors_csv)
            except FileNotFoundError:
                factors_source = None
        if factors_source is None:
            try:
                factors_source = load_observed_factors(path=None, returns=returns)
            except FileNotFoundError:
                try:
                    factors_source = load_observed_factors(returns=returns)
                except Exception:  # pragma: no cover - defensive fallback
                    factors_source = None

    effective_mode, factor_subset = _select_prewhiten_factors(factors_source, requested_mode)

    if effective_mode != "off" and factor_subset is not None:
        try:
            whitening = prewhiten_returns(returns, factor_subset)
            if whitening.residuals.empty:
                whitening = _identity_prewhiten_result(returns)
                effective_mode = "off"
        except ValueError:
            whitening = _identity_prewhiten_result(returns)
            effective_mode = "off"
    else:
        whitening = _identity_prewhiten_result(returns)
        effective_mode = "off"

    telemetry = _compute_prewhiten_telemetry(
        whitening,
        requested_mode=requested_mode,
        effective_mode=effective_mode,
    )
    return panel, returns, whitening, telemetry


def run_evaluation(
    config: EvalConfig,
    *,
    resolved_config: Mapping[str, Any] | None = None,
) -> EvalOutputs:
    panel, returns, whitening, prewhiten_meta = _prepare_returns(config)
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
    _write_prewhiten_diagnostics(out_dir, whitening, prewhiten_meta)
    prewhiten_r2_mean = prewhiten_meta.r2_mean
    resolved_payload["prewhiten_r2_mean"] = prewhiten_meta.r2_mean
    resolved_payload["prewhiten_r2_median"] = prewhiten_meta.r2_median
    resolved_payload["prewhiten_mode_requested"] = prewhiten_meta.mode_requested
    resolved_payload["prewhiten_mode_effective"] = prewhiten_meta.mode_effective
    resolved_payload["prewhiten_factor_columns"] = list(prewhiten_meta.factor_columns)
    resolved_payload["prewhiten_beta_abs_mean"] = prewhiten_meta.beta_abs_mean
    resolved_payload["prewhiten_beta_abs_std"] = prewhiten_meta.beta_abs_std
    resolved_payload["prewhiten_beta_abs_median"] = prewhiten_meta.beta_abs_median
    resolved_path.write_text(json.dumps(resolved_payload, indent=2, sort_keys=True))
    resolved_path_str = str(resolved_path)

    overlay_cfg = OverlayConfig(
        shrinker=config.shrinker,
        q_max=int(config.q_max) if config.q_max is not None else None,
        max_detections=int(config.q_max) if config.q_max is not None else None,
        edge_mode=str(config.edge_mode),
        seed=config.overlay_seed if config.overlay_seed is not None else config.seed,
        a_grid=int(config.overlay_a_grid),
        require_isolated=bool(config.require_isolated),
        cs_drop_top_frac=config.cs_drop_top_frac,
        ewma_halflife=float(config.ewma_halflife),
    )

    mv_gamma = float(config.mv_gamma)
    mv_tau = float(config.mv_tau)
    prev_mv_weights: dict[tuple[str, ...], np.ndarray] = {}

    def _evaluate_window(start: int) -> tuple[list[dict[str, object]], dict[str, object] | None]:
        fit = residuals.iloc[start : start + config.window]
        hold = residuals.iloc[start + config.window : start + config.window + config.horizon]
        design = (config.group_design or "week").lower()
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

        hold_vol_state = _vol_state_label(vol_signal_value, calm_cut, crisis_cut)

        prewhiten_factor_count = len(prewhiten_meta.factor_columns)
        prewhiten_factors_str = ",".join(prewhiten_meta.factor_columns)

        try:
            fit_balanced, group_labels = _build_grouped_window(
                fit,
                config=config,
                calm_threshold=calm_cut,
                crisis_threshold=crisis_cut,
                vol_proxy=vol_proxy_past,
            )
        except GroupingError:
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
                "alignment_cos_mean": float("nan"),
                "alignment_angle_mean": float("nan"),
                "group_design": config.group_design,
                "group_count": 0,
                "group_replicates": 0,
                "prewhiten_r2_mean": prewhiten_r2_mean,
                "prewhiten_r2_median": prewhiten_meta.r2_median,
                "prewhiten_mode_requested": prewhiten_meta.mode_requested,
                "prewhiten_mode_effective": prewhiten_meta.mode_effective,
                "prewhiten_factor_count": prewhiten_factor_count,
                "prewhiten_beta_abs_mean": prewhiten_meta.beta_abs_mean,
                "prewhiten_beta_abs_std": prewhiten_meta.beta_abs_std,
                "prewhiten_beta_abs_median": prewhiten_meta.beta_abs_median,
                "prewhiten_factors": prewhiten_factors_str,
                "residual_energy_mean": 0.0,
                "acceptance_delta": 0.0,
                "group_label_counts": "",
                "group_observations": 0,
                "vol_state_label": hold_vol_state,
            }
            return [], diag_record

        asset_key = tuple(str(col) for col in fit_balanced.columns)
        fit_matrix = fit_balanced.to_numpy(dtype=np.float64)
        p_assets = fit_matrix.shape[1]
        sample_cov = np.cov(fit_matrix, rowvar=False, ddof=1)
        group_ids, group_counts = np.unique(group_labels, return_counts=True)
        group_count = int(group_ids.size)
        replicates_per_group = int(group_counts.min()) if group_counts.size else 0

        group_label_counts, counts_map = _format_group_label_counts(group_labels, design)
        group_observations = int(sum(counts_map.values()))

        reason = DiagnosticReason.NO_DETECTIONS
        if group_count < int(config.group_min_count):
            detections: list[dict[str, object]] = []
            reason = DiagnosticReason.INSUFFICIENT_GROUPS
        else:
            try:
                detections = detect_spikes(fit_matrix, group_labels, config=overlay_cfg)
                reason = DiagnosticReason.ACCEPTED if detections else DiagnosticReason.NO_DETECTIONS
            except Exception:
                detections = []
                reason = DiagnosticReason.DETECTION_ERROR

        alignment_cos_values: list[float] = []
        alignment_angle_values: list[float] = []
        raw_detection_count = len(detections)
        if detections:
            filtered: list[dict[str, object]] = []
            threshold = float(config.angle_min_cos) if config.angle_min_cos is not None else None
            top_p = max(1, int(config.alignment_top_p))
            for det in detections:
                eigvec = det.get("eigvec")
                angle_deg = float("nan")
                cos_val = float("nan")
                energy_mu = float("nan")
                if isinstance(eigvec, np.ndarray):
                    try:
                        angle_deg, energy_mu = alignment_diagnostics(
                            sample_cov,
                            np.asarray(eigvec, dtype=np.float64),
                            top_p=top_p,
                        )
                        cos_val = float(np.cos(np.deg2rad(angle_deg)))
                    except Exception:
                        angle_deg = float("nan")
                        cos_val = float("nan")
                        energy_mu = float("nan")
                det_copy = dict(det)
                det_copy["alignment_angle_deg"] = angle_deg
                det_copy["alignment_cos"] = cos_val
                det_copy["alignment_energy_mu"] = energy_mu
                if threshold is not None:
                    if not np.isfinite(cos_val) or cos_val < threshold:
                        continue
                filtered.append(det_copy)
                if np.isfinite(cos_val):
                    alignment_cos_values.append(cos_val)
                if np.isfinite(angle_deg):
                    alignment_angle_values.append(angle_deg)
            detections = filtered
            if not detections and reason == DiagnosticReason.ACCEPTED:
                reason = DiagnosticReason.ALIGNMENT_REJECTED
        else:
            alignment_cos_values = []
            alignment_angle_values = []

        energy_values = [
            float(det.get("target_energy", det.get("lambda_hat", 0.0)) or 0.0)
            for det in detections
        ]
        energy_values = [val for val in energy_values if np.isfinite(val)]
        residual_energy_mean = float(np.mean(energy_values)) if energy_values else 0.0
        acceptance_delta = float(max(0, raw_detection_count - len(detections)))

        overlay_cov = apply_overlay(sample_cov, detections, observations=fit_matrix, config=overlay_cfg)
        baseline_cov = apply_overlay(sample_cov, [], observations=fit_matrix, config=overlay_cfg)
        covariances = {
            "overlay": overlay_cov,
            "baseline": baseline_cov,
            "sample": sample_cov,
        }

        hold_matrix = hold.to_numpy(dtype=np.float64)
        eq_weights = np.full(p_assets, 1.0 / p_assets, dtype=np.float64)
        prev_weights = prev_mv_weights.get(asset_key) if mv_tau > 0.0 else None
        mv_weights = _min_variance_weights(
            baseline_cov,
            gamma=mv_gamma,
            tau=mv_tau,
            prev_weights=prev_weights,
        )
        if mv_tau > 0.0:
            prev_mv_weights[asset_key] = mv_weights
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
                        "window_id": start,
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
        alignment_cos_mean = float(np.mean(alignment_cos_values)) if alignment_cos_values else float("nan")
        alignment_angle_mean = float(np.mean(alignment_angle_values)) if alignment_angle_values else float("nan")

        reason_value = reason.value if config.reason_codes else ""
        diag_record = {
            "window_id": start,
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
            "alignment_cos_mean": alignment_cos_mean,
            "alignment_angle_mean": alignment_angle_mean,
            "group_design": config.group_design,
            "group_count": group_count,
            "group_replicates": replicates_per_group,
            "prewhiten_r2_mean": prewhiten_r2_mean,
            "prewhiten_r2_median": prewhiten_meta.r2_median,
            "prewhiten_mode_requested": prewhiten_meta.mode_requested,
            "prewhiten_mode_effective": prewhiten_meta.mode_effective,
            "prewhiten_factor_count": prewhiten_factor_count,
            "prewhiten_beta_abs_mean": prewhiten_meta.beta_abs_mean,
            "prewhiten_beta_abs_std": prewhiten_meta.beta_abs_std,
            "prewhiten_beta_abs_median": prewhiten_meta.beta_abs_median,
            "prewhiten_factors": prewhiten_factors_str,
            "residual_energy_mean": residual_energy_mean,
            "acceptance_delta": acceptance_delta,
            "group_label_counts": group_label_counts,
            "group_observations": group_observations,
            "vol_state_label": hold_vol_state,
        }
        return metrics_block, diag_record

    window_records: list[dict[str, object]] = []
    diagnostics_records: list[dict[str, object]] = []

    total_days = residuals.shape[0]
    start_indices = range(0, total_days - config.window - config.horizon + 1)
    worker_setting = config.workers
    if mv_tau > 0.0:
        worker_setting = 1
    if worker_setting and worker_setting > 1:
        max_workers = max(1, int(worker_setting))
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
    if "group_label_counts" not in diagnostics_df.columns:
        diagnostics_df["group_label_counts"] = ""
    if "group_observations" not in diagnostics_df.columns:
        diagnostics_df["group_observations"] = np.nan
    if "vol_state_label" not in diagnostics_df.columns:
        diagnostics_df["vol_state_label"] = ""
    if "prewhiten_mode_requested" not in diagnostics_df.columns:
        diagnostics_df["prewhiten_mode_requested"] = prewhiten_meta.mode_requested if diagnostics_records else ""
    if "prewhiten_mode_effective" not in diagnostics_df.columns:
        diagnostics_df["prewhiten_mode_effective"] = prewhiten_meta.mode_effective if diagnostics_records else ""
    if "prewhiten_factor_count" not in diagnostics_df.columns:
        diagnostics_df["prewhiten_factor_count"] = np.nan
    if "prewhiten_beta_abs_mean" not in diagnostics_df.columns:
        diagnostics_df["prewhiten_beta_abs_mean"] = np.nan
    if "prewhiten_beta_abs_std" not in diagnostics_df.columns:
        diagnostics_df["prewhiten_beta_abs_std"] = np.nan
    if "prewhiten_beta_abs_median" not in diagnostics_df.columns:
        diagnostics_df["prewhiten_beta_abs_median"] = np.nan
    default_factor_str = ",".join(prewhiten_meta.factor_columns)
    if "prewhiten_factors" not in diagnostics_df.columns:
        diagnostics_df["prewhiten_factors"] = default_factor_str if diagnostics_records else ""
    if "prewhiten_r2_median" not in diagnostics_df.columns:
        diagnostics_df["prewhiten_r2_median"] = np.nan
    if "residual_energy_mean" not in diagnostics_df.columns:
        diagnostics_df["residual_energy_mean"] = np.nan
    if "acceptance_delta" not in diagnostics_df.columns:
        diagnostics_df["acceptance_delta"] = np.nan
    metrics_df, diagnostics_df = _limit_windows_by_regime(
        metrics_df,
        diagnostics_df,
        calm_limit=config.calm_window_sample,
        crisis_limit=config.crisis_window_top_k,
        seed=config.seed,
    )
    bootstrap_samples = max(0, int(config.bootstrap_samples))
    rng_bootstrap = np.random.default_rng(config.seed + 97)
    bootstrap_bands: dict[tuple[str, str], tuple[float, float]] = {}
    expected_diag_columns = [
        "window_id",
        "regime",
        "detections",
        "detection_rate",
        "edge_margin_mean",
        "stability_margin_mean",
        "isolation_share",
        "alignment_cos_mean",
        "alignment_angle_mean",
        "reason_code",
        "resolved_config_path",
        "calm_threshold",
        "crisis_threshold",
        "vol_signal",
        "group_design",
        "group_count",
        "group_replicates",
        "prewhiten_r2_mean",
        "prewhiten_r2_median",
        "prewhiten_mode_requested",
        "prewhiten_mode_effective",
        "prewhiten_factor_count",
        "prewhiten_beta_abs_mean",
        "prewhiten_beta_abs_std",
        "prewhiten_beta_abs_median",
        "prewhiten_factors",
        "residual_energy_mean",
        "acceptance_delta",
        "group_label_counts",
        "group_observations",
        "vol_state_label",
    ]
    if diagnostics_df.empty:
        diagnostics_df = pd.DataFrame(columns=expected_diag_columns)
    else:
        for column in expected_diag_columns:
            if column not in diagnostics_df.columns:
                if column in {
                    "group_label_counts",
                    "vol_state_label",
                    "prewhiten_mode_requested",
                    "prewhiten_mode_effective",
                    "prewhiten_factors",
                }:
                    diagnostics_df[column] = ""
                else:
                    diagnostics_df[column] = np.nan
        diagnostics_df = diagnostics_df[expected_diag_columns]

    outputs_metrics: dict[str, Path] = {}
    outputs_risk: dict[str, Path] = {}
    outputs_dm: dict[str, Path] = {}
    outputs_diag: dict[str, Path] = {}
    outputs_plots: dict[str, Path] = {}
    outputs_diag_detail: dict[str, Path] = {}

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
            detail_path = path / "diagnostics_detail.csv"
            diagnostics_df.to_csv(detail_path, index=False)
            outputs_metrics[regime] = metrics_path
            outputs_risk[regime] = risk_path
            outputs_dm[regime] = dm_path
            outputs_diag[regime] = diag_path
            outputs_plots[regime] = path / "delta_mse.png"
            outputs_diag_detail[regime] = detail_path
        detail_root_path = out_dir / "diagnostics_detail.csv"
        diagnostics_df.to_csv(detail_root_path, index=False)
        outputs_diag_detail["all"] = detail_root_path
        return EvalOutputs(outputs_metrics, outputs_risk, outputs_dm, outputs_diag, outputs_plots, outputs_diag_detail)

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
    summary_df["delta_mse_ci_lower"] = np.nan
    summary_df["delta_mse_ci_upper"] = np.nan

    if bootstrap_samples > 0:
        for regime_key in _REGIMES:
            for portfolio in ("ew", "mv"):
                aligned_table = _aligned_error_table(metrics_df, regime_key, portfolio)
                if aligned_table.empty:
                    continue
                diffs = aligned_table["overlay"].to_numpy() - aligned_table["baseline"].to_numpy()
                if diffs.size < 2:
                    continue
                lower, upper = _bootstrap_delta_mse(diffs, bootstrap_samples, rng_bootstrap)
                bootstrap_bands[(regime_key, portfolio)] = (lower, upper)
        for (regime_key, portfolio), (lower, upper) in bootstrap_bands.items():
            mask_overlay = (
                summary_df["regime"].eq(regime_key)
                & summary_df["portfolio"].eq(portfolio)
                & summary_df["estimator"].eq("overlay")
            )
            summary_df.loc[mask_overlay, "delta_mse_ci_lower"] = lower
            summary_df.loc[mask_overlay, "delta_mse_ci_upper"] = upper

    agg_spec = {
        "detections": ("detections", "mean"),
        "detection_rate": ("detection_rate", "mean"),
        "edge_margin_mean": ("edge_margin_mean", "mean"),
        "stability_margin_mean": ("stability_margin_mean", "mean"),
        "isolation_share": ("isolation_share", "mean"),
        "alignment_cos_mean": ("alignment_cos_mean", "mean"),
        "alignment_angle_mean": ("alignment_angle_mean", "mean"),
        "calm_threshold": ("calm_threshold", "mean"),
        "crisis_threshold": ("crisis_threshold", "mean"),
        "vol_signal": ("vol_signal", "mean"),
        "group_count": ("group_count", "mean"),
        "group_replicates": ("group_replicates", "mean"),
        "prewhiten_r2_mean": ("prewhiten_r2_mean", "mean"),
        "prewhiten_r2_median": ("prewhiten_r2_median", "mean"),
        "prewhiten_factor_count": ("prewhiten_factor_count", "mean"),
        "prewhiten_beta_abs_mean": ("prewhiten_beta_abs_mean", "mean"),
        "prewhiten_beta_abs_std": ("prewhiten_beta_abs_std", "mean"),
        "prewhiten_beta_abs_median": ("prewhiten_beta_abs_median", "mean"),
        "residual_energy_mean": ("residual_energy_mean", "mean"),
        "acceptance_delta": ("acceptance_delta", "mean"),
        "group_observations": ("group_observations", "mean"),
    }
    available_spec = {
        key: value for key, value in agg_spec.items() if value[0] in diagnostics_df.columns
    }
    if available_spec:
        diagnostics_summary = (
            diagnostics_df.groupby("regime")
            .agg(**available_spec)
            .reset_index()
        )
    else:
        diagnostics_summary = pd.DataFrame(columns=["regime"])
    for output_col in agg_spec:
        if output_col not in diagnostics_summary.columns:
            diagnostics_summary[output_col] = np.nan
    if diagnostics_summary.empty:
        diagnostics_summary["reason_code"] = ""
        diagnostics_summary["resolved_config_path"] = resolved_path_str
        diagnostics_summary["calm_threshold"] = np.nan
        diagnostics_summary["crisis_threshold"] = np.nan
        diagnostics_summary["vol_signal"] = np.nan
        diagnostics_summary["alignment_cos_mean"] = np.nan
        diagnostics_summary["alignment_angle_mean"] = np.nan
        diagnostics_summary["group_design"] = ""
        diagnostics_summary["group_count"] = np.nan
        diagnostics_summary["group_replicates"] = np.nan
        diagnostics_summary["prewhiten_r2_mean"] = np.nan
        diagnostics_summary["prewhiten_r2_median"] = np.nan
        diagnostics_summary["prewhiten_factor_count"] = np.nan
        diagnostics_summary["prewhiten_beta_abs_mean"] = np.nan
        diagnostics_summary["prewhiten_beta_abs_std"] = np.nan
        diagnostics_summary["prewhiten_beta_abs_median"] = np.nan
        diagnostics_summary["residual_energy_mean"] = np.nan
        diagnostics_summary["acceptance_delta"] = np.nan
        diagnostics_summary["group_observations"] = np.nan
        diagnostics_summary["group_label_counts"] = ""
        diagnostics_summary["vol_state_label"] = ""
        diagnostics_summary["prewhiten_mode_requested"] = ""
        diagnostics_summary["prewhiten_mode_effective"] = ""
        diagnostics_summary["prewhiten_factors"] = ""
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
        design_summary = (
            diagnostics_df.groupby("regime")["group_design"]
            .agg(_mode_string)
            .reset_index()
        )
        diagnostics_summary = diagnostics_summary.merge(reason_summary, on="regime", how="left")
        diagnostics_summary = diagnostics_summary.merge(path_summary, on="regime", how="left")
        diagnostics_summary = diagnostics_summary.merge(design_summary, on="regime", how="left")
        if "group_label_counts" in diagnostics_df.columns:
            label_counts_summary = (
                diagnostics_df.groupby("regime")["group_label_counts"]
                .agg(_mode_string)
                .reset_index()
            )
            diagnostics_summary = diagnostics_summary.merge(label_counts_summary, on="regime", how="left")
        else:
            diagnostics_summary["group_label_counts"] = ""
        if "vol_state_label" in diagnostics_df.columns:
            vol_state_summary = (
                diagnostics_df.groupby("regime")["vol_state_label"]
                .agg(_mode_string)
                .reset_index()
            )
            diagnostics_summary = diagnostics_summary.merge(vol_state_summary, on="regime", how="left")
        else:
            diagnostics_summary["vol_state_label"] = ""
        if "prewhiten_mode_requested" in diagnostics_df.columns:
            req_summary = (
                diagnostics_df.groupby("regime")["prewhiten_mode_requested"]
                .agg(_mode_string)
                .reset_index()
            )
            diagnostics_summary = diagnostics_summary.merge(req_summary, on="regime", how="left")
        else:
            diagnostics_summary["prewhiten_mode_requested"] = ""
        if "prewhiten_mode_effective" in diagnostics_df.columns:
            eff_summary = (
                diagnostics_df.groupby("regime")["prewhiten_mode_effective"]
                .agg(_mode_string)
                .reset_index()
            )
            diagnostics_summary = diagnostics_summary.merge(eff_summary, on="regime", how="left")
        else:
            diagnostics_summary["prewhiten_mode_effective"] = ""
        if "prewhiten_factors" in diagnostics_df.columns:
            factors_summary = (
                diagnostics_df.groupby("regime")["prewhiten_factors"]
                .agg(_mode_string)
                .reset_index()
            )
            diagnostics_summary = diagnostics_summary.merge(factors_summary, on="regime", how="left")
        else:
            diagnostics_summary["prewhiten_factors"] = ""

    overlay_toggle_path = out_dir / "overlay_toggle.md"
    _write_overlay_toggle(overlay_toggle_path, diagnostics_summary)

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
            dm_stat, p_value, n_eff = _aligned_dm_stat(metrics_df, regime, portfolio)
            dm_rows.append(
                {
                    "portfolio": portfolio,
                    "baseline": "baseline",
                    "dm_stat": dm_stat,
                    "p_value": p_value,
                    "n_effective": n_eff,
                }
            )
        dm_path = path / "dm.csv"
        pd.DataFrame(dm_rows).to_csv(dm_path, index=False)
        outputs_dm[regime] = dm_path

        diag_path = path / "diagnostics.csv"
        diag_subset = diagnostics_summary[diagnostics_summary["regime"] == regime]
        diag_subset.to_csv(diag_path, index=False)
        outputs_diag[regime] = diag_path

        detail_path = path / "diagnostics_detail.csv"
        detail_subset = diagnostics_df[diagnostics_df["regime"] == regime]
        detail_subset.to_csv(detail_path, index=False)
        outputs_diag_detail[regime] = detail_path

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

    detail_root_path = out_dir / "diagnostics_detail.csv"
    diagnostics_df.to_csv(detail_root_path, index=False)
    outputs_diag_detail["all"] = detail_root_path

    return EvalOutputs(
        metrics=outputs_metrics,
        risk=outputs_risk,
        dm=outputs_dm,
        diagnostics=outputs_diag,
        plots=outputs_plots,
        diagnostics_detail=outputs_diag_detail,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config, resolved = parse_args(argv)
    run_evaluation(config, resolved_config=resolved)


if __name__ == "__main__":  # pragma: no cover
    main()
