from __future__ import annotations

import argparse
import random
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

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

from baselines import load_observed_factors
from baselines.covariance import (
    cc_covariance as baseline_cc_covariance,
    ewma_covariance as baseline_ewma_covariance,
    lw_covariance as baseline_lw_covariance,
    oas_covariance as baseline_oas_covariance,
    quest_covariance as baseline_quest_covariance,
    rie_covariance,
)
from data.factors import FactorRegistryEntry, FactorRegistryError, load_registered_factors
from baselines.factors import PrewhitenResult
from eval.balance import build_balanced_window
from eval.clean import apply_nan_policy
from evaluation.factor import observed_factor_covariance, poet_lite_covariance
from evaluation.dm import dm_test
from evaluation.evaluate import alignment_diagnostics
from finance import MinVarMemo, minvar_ridge_box, turnover
from fjs.overlay import OverlayConfig, apply_overlay, detect_spikes
from experiments.daily.grouping import (
    GroupingError,
    group_by_day_of_week,
    group_by_dow_month,
    group_by_dow_vol,
    group_by_vol_state,
    group_by_week,
)
from experiments.eval.config import resolve_eval_config
from experiments.eval.diagnostics import DiagnosticReason
from experiments.prewhiten import (
    FACTOR_FALLBACKS,
    FACTOR_SETS,
    PrewhitenTelemetry,
    apply_prewhitening,
)
from meta import runtime


@dataclass(slots=True, frozen=True)
class EvalOutputs:
    metrics: dict[str, Path]
    risk: dict[str, Path]
    dm: dict[str, Path]
    diagnostics: dict[str, Path]
    plots: dict[str, Path]
    diagnostics_detail: dict[str, Path]


_REGIMES = ("full", "calm", "crisis")


def _plot_regime_histograms(
    diagnostics_df: pd.DataFrame,
    column: str,
    *,
    out_dir: Path,
    xlabel: str,
    title_prefix: str,
) -> dict[str, Path]:
    if plt is None or diagnostics_df.empty or column not in diagnostics_df.columns:
        return {}
    outputs: dict[str, Path] = {}
    slug = column.replace("/", "_").replace(" ", "_")
    for regime in _REGIMES:
        subset = diagnostics_df[diagnostics_df["regime"] == regime]
        if subset.empty:
            continue
        series = pd.to_numeric(subset[column], errors="coerce")
        path = out_dir / f"{slug}_{regime}_hist.png"
        plotted = _plot_histogram(
            series,
            path,
            xlabel=xlabel,
            title=f"{title_prefix} ({regime})",
        )
        if plotted is not None:
            outputs[f"{column}_{regime}"] = plotted
    return outputs


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

_MONTH_LABELS = {
    1: "jan",
    2: "feb",
    3: "mar",
    4: "apr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "aug",
    9: "sep",
    10: "oct",
    11: "nov",
    12: "dec",
}


try:
    from data.loader import DailyLoaderConfig, DailyPanel, load_daily_panel
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    @dataclass(slots=True, frozen=True)
    class DailyLoaderConfig:
        winsor_lower: float = 0.005
        winsor_upper: float = 0.995
        min_history: int = 252
        forward_fill: bool = False
        assets_top: int | None = None

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
        path = Path(source)
        if path.suffix.lower() in {".parquet", ".parq"}:
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
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
        if cfg.assets_top is not None:
            if cfg.assets_top <= 0:
                raise ValueError("assets_top must be positive when provided.")
            ordered_columns = sorted(wide.columns)
            capped = ordered_columns[: int(cfg.assets_top)]
            wide = wide.loc[:, capped]
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
    assets_top: int | None = None
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
    overlay_delta_frac: float | None = None
    mv_gamma: float = 1e-4
    mv_tau: float = 0.0
    mv_box_lo: float = 0.0
    mv_box_hi: float = 0.1
    mv_turnover_bps: float = 5.0
    mv_condition_cap: float = 1e6
    bootstrap_samples: int = 0
    require_isolated: bool = True
    q_max: int = 1
    edge_mode: str = "tyler"
    angle_min_cos: float | None = None
    alignment_top_p: int = 3
    cs_drop_top_frac: float | None = None
    prewhiten: str = "ff5mom"
    use_factor_prewhiten: bool = True
    calm_window_sample: int | None = None
    crisis_window_top_k: int | None = None
    group_design: str = "week"
    group_min_count: int = 5
    group_min_replicates: int = 3
    min_reps_dow: int = 20
    min_reps_vol: int = 15
    max_missing_asset: float = 0.05
    max_missing_group_row: float = 0.0
    ewma_halflife: float = 30.0
    gate_mode: str = "strict"
    gate_soft_max: int | None = None
    gate_delta_calibration: Path | None = None
    gate_delta_frac_min: float | None = None
    gate_delta_frac_max: float | None = None
    gate_stability_min: float | None = None
    gate_alignment_min: float | None = None
    gate_accept_nonisolated: bool = False
    coarse_candidate: bool = False
def _format_group_label_counts(labels: np.ndarray, design: str) -> tuple[str, dict[int, int]]:
    if labels.size == 0:
        return "", {}
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size == 0:
        return "", {}
    design_key = design.lower()
    entries: list[str] = []
    counts_map: dict[int, int] = {}
    pairs = [(int(raw_label), int(raw_count)) for raw_label, raw_count in zip(unique.tolist(), counts.tolist())]
    for label, count in sorted(pairs, key=lambda pair: pair[0]):
        counts_map[label] = count
        if design_key == "dow":
            label_name = _DOW_LABELS.get(label, str(label))
        elif design_key == "vol":
            label_name = _VOL_LABELS.get(label, str(label))
        elif design_key == "week":
            label_name = f"wk{label}"
        elif design_key == "dowxvol":
            dow_code = label % 10
            vol_code = label // 10
            dow_name = _DOW_LABELS.get(dow_code, str(dow_code))
            vol_name = _VOL_LABELS.get(vol_code, str(vol_code))
            label_name = f"{dow_name}@{vol_name}"
        elif design_key == "dowxmonth":
            dow_code = label % 10
            month_code = label // 10
            month_name = _MONTH_LABELS.get(month_code, str(month_code))
            dow_name = _DOW_LABELS.get(dow_code, str(dow_code))
            label_name = f"{month_name}@{dow_name}"
        else:
            label_name = str(label)
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


def _serialise_config(config: EvalConfig) -> dict[str, Any]:
    return {
        "returns_csv": str(config.returns_csv),
        "factors_csv": str(config.factors_csv) if config.factors_csv else None,
        "window": config.window,
        "horizon": config.horizon,
        "out_dir": str(config.out_dir),
        "assets_top": config.assets_top,
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
        "overlay_delta_frac": config.overlay_delta_frac,
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
        "use_factor_prewhiten": config.use_factor_prewhiten,
        "calm_window_sample": config.calm_window_sample,
        "crisis_window_top_k": config.crisis_window_top_k,
        "group_design": config.group_design,
        "group_min_count": config.group_min_count,
        "group_min_replicates": config.group_min_replicates,
        "ewma_halflife": config.ewma_halflife,
        "gate_mode": config.gate_mode,
        "gate_soft_max": config.gate_soft_max,
        "gate_delta_calibration": str(config.gate_delta_calibration) if config.gate_delta_calibration else None,
        "gate_delta_frac_min": config.gate_delta_frac_min,
        "gate_delta_frac_max": config.gate_delta_frac_max,
        "gate_stability_min": config.gate_stability_min,
        "gate_alignment_min": config.gate_alignment_min,
        "gate_accept_nonisolated": config.gate_accept_nonisolated,
    }


def _current_git_sha() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return output.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _write_run_metadata(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _paths_to_strings(path_map: Mapping[str, Path]) -> dict[str, str]:
    return {key: str(value) for key, value in path_map.items()}


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
    *,
    column: str,
    estimator_ref: str = "overlay",
    comparator: str = "baseline",
) -> pd.DataFrame:
    mask = metrics["portfolio"].eq(portfolio)
    if regime != "full":
        mask &= metrics["regime"].eq(regime)
    if column not in metrics.columns:
        return pd.DataFrame(columns=[estimator_ref, comparator])
    subset = metrics.loc[mask, ["window_id", "estimator", column]]
    if subset.empty:
        return pd.DataFrame(columns=[estimator_ref, comparator])
    pivot = subset.pivot_table(
        index="window_id",
        columns="estimator",
        values=column,
        aggfunc="first",
    )
    if estimator_ref not in pivot.columns or comparator not in pivot.columns:
        return pd.DataFrame(columns=[estimator_ref, comparator])
    return pivot[[estimator_ref, comparator]].dropna()


def _aligned_dm_stat(
    metrics: pd.DataFrame,
    regime: str,
    portfolio: str,
    *,
    column: str = "sq_error",
    comparator: str = "baseline",
    valid_window_ids: set[int] | None = None,
) -> tuple[float, float, int]:
    aligned = _aligned_error_table(
        metrics,
        regime,
        portfolio,
        column=column,
        estimator_ref="overlay",
        comparator=comparator,
    )
    if valid_window_ids is not None and not aligned.empty:
        aligned = aligned.loc[aligned.index.isin(valid_window_ids)]
    n_eff = int(aligned.shape[0])
    if n_eff < 2:
        return float("nan"), float("nan"), n_eff
    dm_stat, p_value = dm_test(
        aligned["overlay"].to_numpy(),
        aligned[comparator].to_numpy(),
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
        "--factor-csv",
        dest="factors_csv",
        type=Path,
        default=None,
        help="Optional FF5+MOM factor CSV (falls back to MKT proxy when absent).",
    )
    parser.add_argument(
        "--use-factor-prewhiten",
        type=int,
        choices=[0, 1],
        default=None,
        help="Require registered FF5+MOM factors when prewhitening (default: 1).",
    )
    parser.add_argument("--window", type=int, default=None, help="Estimation window (days).")
    parser.add_argument("--horizon", type=int, default=None, help="Holdout horizon (days).")
    parser.add_argument("--start", type=str, default=None, help="Optional start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="Optional end date (YYYY-MM-DD).")
    parser.add_argument("--out", type=Path, default=None, help="Output directory.")
    parser.add_argument(
        "--assets-top",
        dest="assets_top",
        type=int,
        default=None,
        help="Optional cap on the number of assets (keeps first N tickers alphabetically).",
    )
    parser.add_argument(
        "--shrinker",
        type=str,
        default=None,
        choices=["rie", "lw", "oas", "cc", "sample", "quest", "ewma", "factor", "poet"],
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
        "--exec-mode",
        type=str,
        choices=["deterministic", "throughput"],
        default="deterministic",
        help="Execution profile controlling BLAS threads (default: deterministic).",
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
        "--overlay-delta-frac",
        dest="overlay_delta_frac",
        type=float,
        default=None,
        help="Relative delta_frac multiplier applied to MP edge thresholds (overrides defaults).",
    )
    parser.add_argument(
        "--overlay-seed",
        dest="overlay_seed",
        type=int,
        default=None,
        help="Optional seed override specifically for overlay search.",
    )
    parser.add_argument(
        "--mv-gamma",
        dest="mv_gamma",
        type=float,
        default=None,
        help="Ridge penalty applied to the min-variance solver (overrides defaults).",
    )
    parser.add_argument(
        "--mv-tau",
        dest="mv_tau",
        type=float,
        default=None,
        help="Turnover smoothing penalty for the legacy MV solver (default mirrors config).",
    )
    parser.add_argument(
        "--mv-box",
        dest="mv_box",
        type=str,
        default=None,
        help="Comma-separated lower/upper bounds for MV weights (e.g., 0.0,0.1).",
    )
    parser.add_argument(
        "--mv-box-lo",
        dest="mv_box_lo",
        type=float,
        default=None,
        help="Lower bound for MV weights (overrides defaults).",
    )
    parser.add_argument(
        "--mv-box-hi",
        dest="mv_box_hi",
        type=float,
        default=None,
        help="Upper bound for MV weights (overrides defaults).",
    )
    parser.add_argument(
        "--mv-turnover-bps",
        dest="mv_turnover_bps",
        type=float,
        default=None,
        help="One-way turnover cost applied to MV forecasts (in basis points).",
    )
    parser.add_argument(
        "--mv-condition-cap",
        dest="mv_condition_cap",
        type=float,
        default=None,
        help="Drop windows whose covariance condition number exceeds this cap.",
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
        "--coarse-candidate",
        dest="coarse_candidate",
        type=int,
        choices=[0, 1],
        default=None,
        help="Enable coarse eigenpair fallback before gating (1 to enable).",
    )
    parser.add_argument(
        "--gate-mode",
        dest="gate_mode",
        type=str,
        choices=["strict", "soft"],
        default=None,
        help="Overlay gating mode (strict enforces calibrated thresholds; soft ranks by score).",
    )
    parser.add_argument(
        "--gate-soft-max",
        dest="gate_soft_max",
        type=int,
        default=None,
        help="Maximum detections retained under soft gate (defaults to q_max).",
    )
    parser.add_argument(
        "--gate-delta-calibration",
        dest="gate_delta_calibration",
        type=Path,
        default=None,
        help="Optional path to calibrated delta_frac lookup JSON.",
    )
    parser.add_argument(
        "--gate-delta-frac-min",
        dest="gate_delta_frac_min",
        type=float,
        default=None,
        help="Minimum delta_frac required after calibration (strict gate).",
    )
    parser.add_argument(
        "--gate-delta-frac-max",
        dest="gate_delta_frac_max",
        type=float,
        default=None,
        help="Maximum delta_frac allowed after calibration (strict gate).",
    )
    parser.add_argument(
        "--gate-stability-min",
        dest="gate_stability_min",
        type=float,
        default=None,
        help="Minimum stability margin required for strict gate (defaults to stability_eta_deg).",
    )
    parser.add_argument(
        "--gate-alignment-min",
        dest="gate_alignment_min",
        type=float,
        default=None,
        help="Minimum alignment cosine required for strict gate.",
    )
    parser.add_argument(
        "--gate-accept-nonisolated",
        dest="gate_accept_nonisolated",
        action="store_true",
        help="Allow non-isolated detections to pass strict gate.",
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
        choices=["week", "dow", "vol", "dowxvol"],
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
    parser.add_argument(
        "--min-reps-dow",
        dest="min_reps_dow",
        type=int,
        default=None,
        help="Minimum replicates retained for Day-of-Week windows after balancing.",
    )
    parser.add_argument(
        "--min-reps-vol",
        dest="min_reps_vol",
        type=int,
        default=None,
        help="Minimum replicates retained for volatility-state windows after balancing.",
    )
    parser.add_argument(
        "--max-missing-asset",
        dest="max_missing_asset",
        type=float,
        default=None,
        help="Maximum missing fraction allowed per asset inside a detection window.",
    )
    parser.add_argument(
        "--max-missing-group-row",
        dest="max_missing_group_row",
        type=float,
        default=None,
        help="Maximum missing fraction allowed per replicate row before dropping.",
    )
    args = parser.parse_args(argv)
    if getattr(args, "mv_box", None) and (args.mv_box_lo is None or args.mv_box_hi is None):
        parts = [part.strip() for part in args.mv_box.split(",")]
        if len(parts) != 2:
            parser.error("--mv-box must be formatted as 'lo,hi'.")
        try:
            args.mv_box_lo = float(parts[0])
            args.mv_box_hi = float(parts[1])
        except ValueError as exc:  # pragma: no cover - argparse-level validation
            parser.error(f"--mv-box bounds must be numeric: {exc}")
    args.mv_box = None
    resolved = resolve_eval_config(vars(args))
    config = resolved.config
    resolved.resolved["exec_mode"] = args.exec_mode
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
            ("substitution_fraction", "Substitution"),
            ("gating_delta_frac", "Δ_frac"),
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


def _plot_histogram(series: pd.Series, path: Path, *, xlabel: str, title: str, bins: int = 20) -> Path | None:
    if plt is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if values.size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=min(bins, max(int(values.size / 5), 10)), color="C0", edgecolor="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _plot_acceptance_edge_histograms(diagnostics_df: pd.DataFrame, design: str, out_dir: Path) -> dict[str, Path]:
    design_key = (design or "unknown").lower()
    outputs: dict[str, Path] = {}
    if diagnostics_df.empty:
        return outputs
    acceptance_path = out_dir / f"acceptance_hist_{design_key}.png"
    edge_path = out_dir / f"edge_margin_hist_{design_key}.png"
    acc_series = diagnostics_df.get("acceptance_rate", pd.Series(dtype=float))
    edge_series = diagnostics_df.get("edge_margin_mean", pd.Series(dtype=float))
    acc_plot = _plot_histogram(
        acc_series,
        acceptance_path,
        xlabel="Acceptance rate",
        title=f"Acceptance distribution ({design_key})",
    )
    if acc_plot is not None:
        outputs["acceptance"] = acc_plot
    edge_plot = _plot_histogram(
        edge_series,
        edge_path,
        xlabel="Edge margin",
        title=f"Edge margin distribution ({design_key})",
    )
    if edge_plot is not None:
        outputs["edge_margin"] = edge_plot
    return outputs


def _detail_defaults() -> dict[str, object]:
    return {
        "design_ok": 0,
        "reps_by_label": "",
        "mp_edge_margin": float("nan"),
        "alignment_cos": float("nan"),
        "alignment_cos_p50": float("nan"),
        "leakage_offcomp": float("nan"),
        "stability_eta_pass": float("nan"),
        "bracket_status": "",
        "raw_outliers_found": 0,
        "pre_mp_edge_margin": float("nan"),
        "pre_alignment_cos": float("nan"),
        "pre_leakage_offcomp": float("nan"),
        "pre_stability_eta_pass": float("nan"),
        "pre_bracket_status": "",
    }


def _safe_nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def _safe_nanmedian(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _top_mean(values: Sequence[float], count: int) -> float:
    if count <= 0:
        return float("nan")
    filtered = [float(val) for val in values if np.isfinite(val)]
    if not filtered:
        return float("nan")
    filtered.sort(reverse=True)
    limit = min(len(filtered), max(1, int(count)))
    return float(np.mean(filtered[:limit]))


def _safe_share(successes: int, total: int) -> float:
    if total <= 0:
        return float("nan")
    return float(successes) / float(total)


def _required_replicates(design: str, config: EvalConfig) -> int:
    design_key = (design or "").lower()
    if design_key == "dow":
        return max(0, int(config.min_reps_dow))
    if design_key == "vol":
        return max(0, int(config.min_reps_vol))
    if design_key in {"dowxvol", "dowxmonth"}:
        return max(0, int(config.group_min_replicates))
    return max(0, int(config.group_min_replicates))


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
    if design == "dowxvol":
        return group_by_dow_vol(
            frame,
            vol_proxy=vol_proxy,
            calm_threshold=float(calm_threshold),
            crisis_threshold=float(crisis_threshold),
            min_replicates=min_replicates,
        )
    if design == "dowxmonth":
        return group_by_dow_month(frame, min_replicates=min_replicates)
    if design == "week":
        return group_by_week(frame, replicates=5)
    return group_by_week(frame, replicates=5)


def _min_variance_weights(
    covariance: np.ndarray,
    *,
    ridge: float,
    box: tuple[float, float],
    cache: MinVarMemo | None,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    weights, info = minvar_ridge_box(
        covariance,
        ridge=max(float(ridge), 0.0),
        box=box,
        cache=cache,
    )
    return weights, info


def _expected_shortfall(sigma: float, alpha: float = 0.05) -> float:
    z = stats.norm.ppf(alpha)
    return float(-(sigma * stats.norm.pdf(z) / alpha))


def _realised_tail_mean(returns: np.ndarray, var_threshold: float) -> float:
    tail = returns[returns < var_threshold]
    if tail.size == 0:
        return float("nan")
    return float(np.mean(tail))


def _safe_condition_number(matrix: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(matrix))
    except Exception:
        return float("inf")


def _qlike_loss(forecast_var: float, realised_var: float) -> float:
    if not np.isfinite(forecast_var):
        return float("nan")
    eps = 1e-12
    fvar = max(float(forecast_var), eps)
    rval = max(float(realised_var), eps)
    ratio = rval / fvar
    if ratio <= 0.0 or not np.isfinite(ratio):
        return float("nan")
    return float(ratio - math.log(ratio) - 1.0)


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


def _prepare_returns(
    config: EvalConfig,
) -> tuple[DailyPanel, pd.DataFrame, PrewhitenResult, PrewhitenTelemetry, FactorRegistryEntry | None]:
    loader_kwargs: dict[str, Any] = {"min_history": config.window + config.horizon + 10}
    if config.assets_top is not None:
        loader_kwargs["assets_top"] = int(config.assets_top)
    try:
        loader_cfg = DailyLoaderConfig(**loader_kwargs)
    except TypeError:
        loader_kwargs.pop("assets_top", None)
        loader_cfg = DailyLoaderConfig(**loader_kwargs)
    panel = load_daily_panel(config.returns_csv, config=loader_cfg)
    returns = panel.returns
    if config.assets_top is not None and returns.shape[1] > int(config.assets_top):
        ordered_columns = sorted(returns.columns)
        capped = ordered_columns[: int(config.assets_top)]
        returns = returns.loc[:, capped]
    if config.start:
        returns = returns.loc[returns.index >= pd.to_datetime(config.start)]
    if config.end:
        returns = returns.loc[returns.index <= pd.to_datetime(config.end)]
    if returns.shape[0] < config.window + config.horizon + 5:
        raise ValueError("Not enough observations for requested window and horizon.")
    returns = returns.sort_index()
    returns_start = returns.index.min()
    returns_end = returns.index.max()

    requested_mode = str(config.prewhiten or "off").lower()
    if requested_mode not in {"off", "ff5", "ff5mom"}:
        requested_mode = "off"

    factors_source: pd.DataFrame | None = None
    factor_entry: FactorRegistryEntry | None = None
    if requested_mode != "off":
        if config.use_factor_prewhiten:
            registry_keys = FACTOR_FALLBACKS.get(requested_mode, (requested_mode,))
            factor_error: FactorRegistryError | None = None
            for candidate in registry_keys:
                try:
                    required_cols = FACTOR_SETS.get(candidate, FACTOR_SETS.get(requested_mode, ()))
                    loaded, entry = load_registered_factors(
                        candidate,
                        start=returns_start,
                        end=returns_end,
                        required=required_cols,
                    )
                    factors_source = loaded
                    factor_entry = entry
                    break
                except FactorRegistryError as exc:
                    factor_error = exc
                    continue
            if factors_source is None:
                message = (
                    f"Registered factor dataset for mode '{requested_mode}' "
                    "is unavailable or lacks required coverage."
                )
                if factor_error is not None:
                    message = str(factor_error)
                raise RuntimeError(message)
        else:
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

    whitening, telemetry = apply_prewhitening(
        returns,
        factors=factors_source,
        requested_mode=requested_mode,
    )
    return panel, returns, whitening, telemetry, factor_entry


def run_evaluation(
    config: EvalConfig,
    *,
    resolved_config: Mapping[str, Any] | None = None,
) -> EvalOutputs:
    panel, raw_returns, whitening, prewhiten_meta, factor_entry = _prepare_returns(config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    residuals = whitening.residuals.sort_index()
    raw_returns = raw_returns.sort_index()
    residual_index_set = set(residuals.index)
    factor_tracking_required = bool(config.use_factor_prewhiten and prewhiten_meta.mode_effective != "off")
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
    resolved_payload["use_factor_prewhiten"] = bool(config.use_factor_prewhiten)
    resolved_payload["overlay_delta_frac"] = config.overlay_delta_frac
    if factor_entry is not None:
        resolved_payload["factors_dataset"] = {
            "key": factor_entry.key,
            "path": str(factor_entry.path),
            "sha256": factor_entry.sha256,
            "start_date": factor_entry.start_date,
            "end_date": factor_entry.end_date,
            "source": factor_entry.source,
            "note": factor_entry.note,
        }
    resolved_path.write_text(json.dumps(resolved_payload, indent=2, sort_keys=True))
    resolved_path_str = str(resolved_path)

    overlay_cfg = OverlayConfig(
        shrinker=config.shrinker,
        q_max=int(config.q_max) if config.q_max is not None else None,
        max_detections=int(config.q_max) if config.q_max is not None else None,
        edge_mode=str(config.edge_mode),
        seed=config.overlay_seed if config.overlay_seed is not None else config.seed,
        a_grid=int(config.overlay_a_grid),
        delta_frac=config.overlay_delta_frac,
        require_isolated=bool(config.require_isolated),
        cs_drop_top_frac=config.cs_drop_top_frac,
        ewma_halflife=float(config.ewma_halflife),
        gate_mode=str(config.gate_mode) if config.gate_mode else "strict",
        gate_soft_max=config.gate_soft_max,
        gate_delta_calibration=str(config.gate_delta_calibration)
        if config.gate_delta_calibration
        else None,
        gate_delta_frac_min=config.gate_delta_frac_min,
        gate_delta_frac_max=config.gate_delta_frac_max,
        gate_stability_min=config.gate_stability_min,
        gate_alignment_min=config.gate_alignment_min,
        gate_accept_nonisolated=bool(config.gate_accept_nonisolated),
        coarse_candidate=bool(config.coarse_candidate),
    )

    mv_gamma = float(config.mv_gamma)
    mv_tau = float(config.mv_tau)
    mv_box = (float(config.mv_box_lo), float(config.mv_box_hi))
    mv_turnover_bps = float(config.mv_turnover_bps)
    mv_condition_cap = float(config.mv_condition_cap)
    mv_cache = MinVarMemo()
    prev_mv_weights: dict[tuple[str, ...], np.ndarray] = {}

    def _evaluate_window(start: int) -> tuple[list[dict[str, object]], dict[str, object] | None]:
        fit_end = start + config.window
        hold_end = fit_end + config.horizon
        if hold_end > raw_returns.shape[0]:
            return [], None
        fit_idx = raw_returns.index[start:fit_end]
        hold_idx = raw_returns.index[fit_end:hold_end]
        if len(fit_idx) < config.window or len(hold_idx) < config.horizon:
            return [], None
        fit_labels = list(fit_idx)
        hold_labels = list(hold_idx)
        fit_base = raw_returns.loc[fit_labels]
        hold_base = raw_returns.loc[hold_labels]
        overlay_allowed = True
        if factor_tracking_required:
            needed_labels = fit_labels + hold_labels
            overlay_allowed = all(label in residual_index_set for label in needed_labels)
        if overlay_allowed:
            try:
                fit = residuals.loc[fit_labels]
                hold = residuals.loc[hold_labels]
            except KeyError:
                overlay_allowed = False
                fit = fit_base
                hold = hold_base
        else:
            fit = fit_base
            hold = hold_base
        factor_present = bool(overlay_allowed) if factor_tracking_required else True
        design = (config.group_design or "week").lower()
        required_groups = max(1, int(config.group_min_count))
        required_reps = max(0, _required_replicates(design, config))
        hold_start = pd.to_datetime(hold_labels[0])
        train_end = pd.to_datetime(fit_labels[-1])
        calm_cut, crisis_cut = _vol_thresholds(vol_proxy_past, train_end, config)
        train_vol_slice = vol_proxy_past.loc[:train_end].dropna()
        fallback_vol = float(train_vol_slice.iloc[-1]) if not train_vol_slice.empty else float("nan")
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
            fit_grouped, group_labels = _build_grouped_window(
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
                "factor_present": bool(factor_present),
                "changed_flag": 0,
            }
            diag_record["group_count_required"] = required_groups
            diag_record["group_replicates_required"] = required_reps
            diag_record.update(_detail_defaults())
            diag_record.update(
                {
                    "balance_reason": "grouping_error",
                    "balance_reps_per_group": "",
                    "balance_assets_dropped_pct": float("nan"),
                    "balance_days_dropped_pct": float("nan"),
                    "balance_target_reps": 0,
                }
            )
            return [], diag_record

        fit_grouped = fit_grouped.replace([np.inf, -np.inf], np.nan)
        grouped_rows_original = int(fit_grouped.shape[0])
        grouped_assets_original = int(fit_grouped.shape[1])

        nan_result = apply_nan_policy(
            fit_grouped,
            group_labels,
            max_missing_asset=float(config.max_missing_asset),
            max_missing_group_row=float(config.max_missing_group_row),
        )
        fit_clean = nan_result.frame.replace([np.inf, -np.inf], np.nan)
        group_labels_clean = nan_result.labels

        balance_result = build_balanced_window(
            fit_clean,
            group_labels_clean,
            min_replicates=required_reps,
        )
        fit_balanced = balance_result.frame.replace([np.inf, -np.inf], np.nan)
        group_labels = balance_result.labels
        balance_reason = balance_result.reason
        balance_target_reps = int(balance_result.telemetry.target_replicates)

        if fit_balanced.isna().any().any():
            valid_mask = fit_balanced.notna().all(axis=1)
            if not bool(valid_mask.all()):
                keep_positions = np.where(valid_mask.to_numpy(dtype=bool))[0]
                fit_balanced = fit_balanced.iloc[keep_positions]
                group_labels = group_labels[keep_positions]

        final_rows = int(fit_balanced.shape[0])
        final_assets = int(fit_balanced.shape[1])

        assets_drop_pct = (
            100.0 * max(0, grouped_assets_original - final_assets) / grouped_assets_original
            if grouped_assets_original > 0
            else 0.0
        )
        days_drop_pct = (
            100.0 * max(0, grouped_rows_original - final_rows) / grouped_rows_original
            if grouped_rows_original > 0
            else 0.0
        )

        if group_labels.size > 0:
            balance_counts_str, counts_map = _format_group_label_counts(group_labels, design)
        else:
            balance_counts_str, counts_map = ("", {})
        if counts_map:
            balance_target_reps = int(min(counts_map.values()))
        else:
            balance_target_reps = 0

        balance_diag_fields = {
            "balance_reason": balance_reason,
            "balance_reps_per_group": balance_counts_str,
            "balance_assets_dropped_pct": assets_drop_pct,
            "balance_days_dropped_pct": days_drop_pct,
            "balance_target_reps": balance_target_reps,
        }

        if final_rows == 0 or final_assets == 0 or group_labels.size == 0:
            reason_value = DiagnosticReason.BALANCE_FAILURE.value if config.reason_codes else ""
            diag_record = {
                "regime": regime,
                "detections": 0,
                "detection_rate": 0.0,
                "edge_margin_mean": 0.0,
                "stability_margin_mean": 0.0,
                "isolation_share": 0.0,
                "alignment_cos_mean": 0.0,
                "alignment_angle_mean": 0.0,
                "raw_detection_count": 0,
                "substitution_fraction": 0.0,
                "gating_mode": str(config.gate_mode or "strict"),
                "gating_initial": 0,
                "gating_accepted": 0,
                "gating_rejected": 0,
                "gating_soft_cap": overlay_cfg.gate_soft_max,
                "gating_delta_frac": overlay_cfg.gate_delta_frac_min,
                "reason_code": reason_value,
                "resolved_config_path": resolved_path_str,
                "calm_threshold": float(calm_cut),
                "crisis_threshold": float(crisis_cut),
                "vol_signal": float(vol_signal_value),
                "group_design": design,
                "group_count": 0,
                "group_replicates": 0,
                "prewhiten_r2_mean": prewhiten_meta.r2_mean,
                "prewhiten_r2_median": prewhiten_meta.r2_median,
                "prewhiten_mode_requested": prewhiten_meta.mode_requested,
                "prewhiten_mode_effective": prewhiten_meta.mode_effective,
                "prewhiten_factor_count": len(prewhiten_meta.factor_columns),
                "prewhiten_beta_abs_mean": prewhiten_meta.beta_abs_mean,
                "prewhiten_beta_abs_std": prewhiten_meta.beta_abs_std,
                "prewhiten_beta_abs_median": prewhiten_meta.beta_abs_median,
                "prewhiten_factors": ",".join(prewhiten_meta.factor_columns),
                "residual_energy_mean": 0.0,
                "acceptance_delta": 0.0,
                "group_label_counts": balance_counts_str,
                "group_observations": 0,
                "vol_state_label": hold_vol_state,
            }
            diag_record["group_count_required"] = required_groups
            diag_record["group_replicates_required"] = required_reps
            diag_record.update(_detail_defaults())
            diag_record["reps_by_label"] = balance_counts_str
            diag_record.update(balance_diag_fields)
            return [], diag_record

        if balance_reason in {"insufficient_reps", "empty_after_balance"}:
            reason_value = DiagnosticReason.BALANCE_FAILURE.value if config.reason_codes else ""
            diag_record = {
                "regime": regime,
                "detections": 0,
                "detection_rate": 0.0,
                "edge_margin_mean": 0.0,
                "stability_margin_mean": 0.0,
                "isolation_share": 0.0,
                "alignment_cos_mean": 0.0,
                "alignment_angle_mean": 0.0,
                "raw_detection_count": 0,
                "substitution_fraction": 0.0,
                "gating_mode": str(config.gate_mode or "strict"),
                "gating_initial": 0,
                "gating_accepted": 0,
                "gating_rejected": 0,
                "gating_soft_cap": overlay_cfg.gate_soft_max,
                "gating_delta_frac": overlay_cfg.gate_delta_frac_min,
                "reason_code": reason_value,
                "resolved_config_path": resolved_path_str,
                "calm_threshold": float(calm_cut),
                "crisis_threshold": float(crisis_cut),
                "vol_signal": float(vol_signal_value),
                "group_design": design,
                "group_count": len(counts_map),
                "group_replicates": balance_target_reps,
                "prewhiten_r2_mean": prewhiten_meta.r2_mean,
                "prewhiten_r2_median": prewhiten_meta.r2_median,
                "prewhiten_mode_requested": prewhiten_meta.mode_requested,
                "prewhiten_mode_effective": prewhiten_meta.mode_effective,
                "prewhiten_factor_count": len(prewhiten_meta.factor_columns),
                "prewhiten_beta_abs_mean": prewhiten_meta.beta_abs_mean,
                "prewhiten_beta_abs_std": prewhiten_meta.beta_abs_std,
                "prewhiten_beta_abs_median": prewhiten_meta.beta_abs_median,
                "prewhiten_factors": ",".join(prewhiten_meta.factor_columns),
                "residual_energy_mean": 0.0,
                "acceptance_delta": 0.0,
                "group_label_counts": balance_counts_str,
                "group_observations": int(sum(counts_map.values())),
                "vol_state_label": hold_vol_state,
                "factor_present": bool(factor_present),
                "changed_flag": 0,
            }
            diag_record["group_count_required"] = required_groups
            diag_record["group_replicates_required"] = required_reps
            diag_record.update(_detail_defaults())
            diag_record["reps_by_label"] = balance_counts_str
            diag_record.update(balance_diag_fields)
            return [], diag_record

        hold = hold.loc[:, fit_balanced.columns]
        hold = hold.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if hold.empty:
            reason_value = DiagnosticReason.HOLDOUT_EMPTY.value if config.reason_codes else ""
            diag_record = {
                "regime": regime,
                "detections": 0,
                "detection_rate": 0.0,
                "edge_margin_mean": 0.0,
                "stability_margin_mean": 0.0,
                "isolation_share": 0.0,
                "alignment_cos_mean": 0.0,
                "alignment_angle_mean": 0.0,
                "raw_detection_count": 0,
                "substitution_fraction": 0.0,
                "gating_mode": str(config.gate_mode or "strict"),
                "gating_initial": 0,
                "gating_accepted": 0,
                "gating_rejected": 0,
                "gating_soft_cap": overlay_cfg.gate_soft_max,
                "gating_delta_frac": overlay_cfg.gate_delta_frac_min,
                "reason_code": reason_value,
                "resolved_config_path": resolved_path_str,
                "calm_threshold": float(calm_cut),
                "crisis_threshold": float(crisis_cut),
                "vol_signal": float(vol_signal_value),
                "group_design": design,
                "group_count": len(counts_map),
                "group_replicates": balance_target_reps,
                "prewhiten_r2_mean": prewhiten_meta.r2_mean,
                "prewhiten_r2_median": prewhiten_meta.r2_median,
                "prewhiten_mode_requested": prewhiten_meta.mode_requested,
                "prewhiten_mode_effective": prewhiten_meta.mode_effective,
                "prewhiten_factor_count": len(prewhiten_meta.factor_columns),
                "prewhiten_beta_abs_mean": prewhiten_meta.beta_abs_mean,
                "prewhiten_beta_abs_std": prewhiten_meta.beta_abs_std,
                "prewhiten_beta_abs_median": prewhiten_meta.beta_abs_median,
                "prewhiten_factors": ",".join(prewhiten_meta.factor_columns),
                "residual_energy_mean": 0.0,
                "acceptance_delta": 0.0,
                "group_label_counts": balance_counts_str,
                "group_observations": int(sum(counts_map.values())),
                "vol_state_label": hold_vol_state,
                "factor_present": bool(factor_present),
                "changed_flag": 0,
            }
            diag_record["group_count_required"] = required_groups
            diag_record["group_replicates_required"] = required_reps
            diag_record.update(_detail_defaults())
            diag_record["reps_by_label"] = balance_counts_str
            diag_record.update(balance_diag_fields)
            return [], diag_record

        asset_key = tuple(str(col) for col in fit_balanced.columns)
        fit_matrix = fit_balanced.to_numpy(dtype=np.float64)
        p_assets = fit_matrix.shape[1]
        sample_cov = np.cov(fit_matrix, rowvar=False, ddof=1)
        group_ids, group_counts = np.unique(group_labels, return_counts=True)
        group_count = int(group_ids.size)
        replicates_per_group = int(group_counts.min()) if group_counts.size else 0

        group_label_counts = balance_counts_str
        group_observations = int(sum(counts_map.values()))

        detections: list[dict[str, object]] = []
        gating_info: dict[str, object] = {}
        detect_stats: dict[str, object] = {}
        reason = DiagnosticReason.NO_DETECTIONS
        if not overlay_allowed and factor_tracking_required:
            reason = DiagnosticReason.FACTOR_MISSING
        elif group_count < int(config.group_min_count):
            reason = DiagnosticReason.INSUFFICIENT_GROUPS
        else:
            try:
                detections = detect_spikes(
                    fit_matrix,
                    group_labels,
                    config=overlay_cfg,
                    stats=detect_stats,
                )
                gating_info = detect_stats.get("gating", {}) if isinstance(detect_stats, dict) else {}
                reason = DiagnosticReason.ACCEPTED if detections else DiagnosticReason.NO_DETECTIONS
            except np.linalg.LinAlgError:
                jitter_scale = max(float(np.nanstd(fit_matrix)) * 1e-6, 1e-8)
                rng_jitter = np.random.default_rng(config.seed + start + 101)
                perturbed = fit_matrix + rng_jitter.normal(scale=jitter_scale, size=fit_matrix.shape)
                detect_stats = {}
                try:
                    detections = detect_spikes(
                        perturbed,
                        group_labels,
                        config=overlay_cfg,
                        stats=detect_stats,
                    )
                    gating_info = detect_stats.get("gating", {}) if isinstance(detect_stats, dict) else {}
                    reason = DiagnosticReason.ACCEPTED if detections else DiagnosticReason.NO_DETECTIONS
                except Exception:
                    detections = []
                    reason = DiagnosticReason.DETECTION_ERROR
                    gating_info = {}
            except Exception:
                detections = []
                reason = DiagnosticReason.DETECTION_ERROR
                gating_info = {}

        alignment_cos_values: list[float] = []
        alignment_angle_values: list[float] = []
        alignment_cos_all: list[float] = []
        top_p = max(1, int(config.alignment_top_p))
        raw_detection_count = int(gating_info.get("initial", len(detections)))
        if detections:
            filtered: list[dict[str, object]] = []
            threshold = float(config.angle_min_cos) if config.angle_min_cos is not None else None
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
                if np.isfinite(cos_val):
                    alignment_cos_all.append(cos_val)
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
            alignment_cos_all = []
        pre_alignment_cos = _top_mean(alignment_cos_all, top_p)

        energy_values = [
            float(det.get("target_energy", det.get("lambda_hat", 0.0)) or 0.0)
            for det in detections
        ]
        energy_values = [val for val in energy_values if np.isfinite(val)]
        residual_energy_mean = float(np.mean(energy_values)) if energy_values else 0.0
        acceptance_delta = float(max(0, raw_detection_count - len(detections)))

        overlay_cov = apply_overlay(sample_cov, detections, observations=fit_matrix, config=overlay_cfg)
        baseline_cov = apply_overlay(sample_cov, [], observations=fit_matrix, config=overlay_cfg)
        baseline_cond = _safe_condition_number(baseline_cov)
        overlay_cond = _safe_condition_number(overlay_cov)
        diff_norm = 0.0
        if overlay_cov.size and baseline_cov.size:
            diff_norm = float(np.max(np.abs(overlay_cov - baseline_cov)))
        changed_flag = bool(diff_norm > 1e-10)
        condition_flag = bool(
            mv_condition_cap > 0.0
            and (not np.isfinite(baseline_cond) or baseline_cond > mv_condition_cap)
        )
        if condition_flag:
            reason = DiagnosticReason.CONDITION_CAP

        covariances: dict[str, np.ndarray] = {}
        baseline_errors: dict[str, str] = {}
        cond_map: dict[str, float] = {}
        metrics_block: list[dict[str, object]] = []
        mv_turnover_value = float("nan")
        mv_turnover_cost_bps = float("nan")
        mv_condition_penalized = float("nan")

        if not condition_flag:
            covariances = {
                "overlay": np.asarray(overlay_cov, dtype=np.float64),
                "baseline": np.asarray(baseline_cov, dtype=np.float64),
                "sample": np.asarray(sample_cov, dtype=np.float64),
                "scm": np.asarray(sample_cov, dtype=np.float64),
            }

            sample_count = int(fit_matrix.shape[0])

            def _add_covariance(name: str, builder: Callable[[], np.ndarray]) -> None:
                try:
                    matrix = np.asarray(builder(), dtype=np.float64)
                    if matrix.shape != sample_cov.shape:
                        raise ValueError(
                            f"unexpected covariance shape {matrix.shape} (expected {sample_cov.shape})"
                        )
                    covariances[name] = 0.5 * (matrix + matrix.T)
                except Exception as exc:  # pragma: no cover - propagated to diagnostics
                    baseline_errors[name] = str(exc)

            _add_covariance("rie", lambda: rie_covariance(sample_cov, sample_count=sample_count))
            _add_covariance("lw", lambda: baseline_lw_covariance(fit_matrix))
            _add_covariance("oas", lambda: baseline_oas_covariance(fit_matrix))
            _add_covariance("cc", lambda: baseline_cc_covariance(fit_matrix))
            _add_covariance("quest", lambda: baseline_quest_covariance(sample_cov, sample_count=sample_count))
            _add_covariance(
                "ewma",
                lambda: baseline_ewma_covariance(
                    fit_matrix,
                    halflife=float(config.ewma_halflife),
                ),
            )

            def _factor_cov_builder() -> np.ndarray:
                factors_df = getattr(whitening, "factors", pd.DataFrame())
                if not isinstance(factors_df, pd.DataFrame) or factors_df.empty:
                    raise ValueError("missing factor returns")
                aligned_index = factors_df.index.intersection(fit.index)
                factor_window = factors_df.loc[aligned_index].dropna(axis=0, how="any")
                if factor_window.shape[0] <= 1:
                    raise ValueError("insufficient factor observations for window")
                returns_window = fit.loc[factor_window.index].astype(np.float64)
                factor_window = factor_window.astype(np.float64)
                return observed_factor_covariance(returns_window, factor_window, add_intercept=True)

            _add_covariance("factor", _factor_cov_builder)

            def _poet_cov_builder() -> np.ndarray:
                window = fit.dropna(axis=0, how="any")
                if window.shape[0] <= 1:
                    raise ValueError("insufficient observations for POET")
                max_factors = int(min(10, max(window.shape[1] - 1, 1)))
                poet_result = poet_lite_covariance(window, max_factors=max_factors)
                return poet_result.covariance

            _add_covariance("poet", _poet_cov_builder)

            hold_matrix = hold.to_numpy(dtype=np.float64)
            eq_weights = np.full(p_assets, 1.0 / p_assets, dtype=np.float64)
            prev_weights = prev_mv_weights.get(asset_key)
            mv_weights, mv_solver_info = _min_variance_weights(
                baseline_cov,
                ridge=mv_gamma,
                box=mv_box,
                cache=mv_cache,
            )
            mv_condition_penalized = float(mv_solver_info.get("cond_penalized", float("nan")))
            if prev_weights is not None and prev_weights.shape == mv_weights.shape:
                mv_turnover_value = float(turnover(prev_weights, mv_weights))
            else:
                mv_turnover_value = 0.0
            mv_turnover_cost_bps = mv_turnover_value * mv_turnover_bps
            prev_mv_weights[asset_key] = mv_weights.copy()
            weights_map = {"ew": eq_weights, "mv": mv_weights}
            cond_map = {name: _safe_condition_number(matrix) for name, matrix in covariances.items()}

            for estimator, cov in covariances.items():
                for portfolio, weights in weights_map.items():
                    forecast_var = float(weights.T @ cov @ weights)
                    sigma = float(np.sqrt(max(forecast_var, 1e-12)))
                    realised_returns = hold_matrix @ weights
                    realised_var = (
                        float(np.var(realised_returns, ddof=1)) if realised_returns.size > 1 else float("nan")
                    )
                    var95 = float(stats.norm.ppf(0.05) * sigma)
                    es95 = _expected_shortfall(sigma)
                    violations = realised_returns < var95
                    violation_rate = float(np.mean(violations)) if violations.size else float("nan")
                    realised_es = _realised_tail_mean(realised_returns, var95)
                    mse = (forecast_var - realised_var) ** 2 if np.isfinite(realised_var) else float("nan")
                    es_error = (es95 - realised_es) ** 2 if np.isfinite(realised_es) else float("nan")
                    qlike_error = _qlike_loss(forecast_var, realised_var)
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
                            "qlike": qlike_error,
                            "cov_condition": cond_map.get(estimator, float("nan")),
                            "mv_turnover": mv_turnover_value if portfolio == "mv" else 0.0,
                            "mv_turnover_cost_bps": mv_turnover_cost_bps if portfolio == "mv" else 0.0,
                        }
                    )
        if detections:
            edge_margin_detail = [
                float(det.get("edge_margin", float("nan"))) for det in detections
            ]
            edge_margins = [
                float(det.get("edge_margin", 0.0) or 0.0) for det in detections
            ]
            stability_detail = [
                float(det.get("stability_margin", float("nan"))) for det in detections
            ]
            stability = [
                val if np.isfinite(val) else 0.0 for val in stability_detail
            ]
            off_component_ratios = [
                float(det.get("off_component_ratio", float("nan"))) for det in detections
            ]
            isolation = []
            for ratio in off_component_ratios:
                ratio_val = ratio if np.isfinite(ratio) else 1.0
                ratio_clip = min(1.0, max(0.0, ratio_val))
                isolation.append(1.0 - ratio_clip)
        else:
            edge_margin_detail = []
            edge_margins = []
            stability_detail = []
            stability = []
            off_component_ratios = []
            isolation = []
        alignment_cos_mean = float(np.mean(alignment_cos_values)) if alignment_cos_values else float("nan")
        alignment_angle_mean = float(np.mean(alignment_angle_values)) if alignment_angle_values else float("nan")

        mp_edge_margin_value = _safe_nanmean(edge_margin_detail)
        alignment_cos_p50 = _safe_nanmedian(alignment_cos_values)
        leakage_offcomp_value = _safe_nanmean(off_component_ratios)
        stability_threshold = (
            float(overlay_cfg.gate_stability_min)
            if overlay_cfg.gate_stability_min is not None
            else float(overlay_cfg.stability_eta_deg)
        )
        stability_finite = [val for val in stability_detail if np.isfinite(val)]
        stability_pass = sum(1 for val in stability_finite if val >= stability_threshold)
        stability_eta_share = _safe_share(stability_pass, len(stability_finite))
        solver_tokens = sorted(
            {
                str(det.get("solver_used", "")).strip().lower()
                for det in detections
                if det.get("solver_used")
            }
        )
        bracket_status = "none"
        if solver_tokens:
            if solver_tokens == ["grid"]:
                bracket_status = "grid"
            elif solver_tokens == ["rootfind"]:
                bracket_status = "rootfind"
            elif solver_tokens == ["auto"]:
                bracket_status = "auto"
            elif "rootfind" in solver_tokens and "grid" in solver_tokens:
                bracket_status = "mixed"
            else:
                bracket_status = "+".join(solver_tokens[:3])
        required_groups = max(1, int(config.group_min_count))
        required_reps = max(0, _required_replicates(design, config))
        design_ok_flag = int(
            group_count >= required_groups and replicates_per_group >= required_reps
        )

        reason_value = reason.value if config.reason_codes else ""
        gating_mode = str(gating_info.get("mode", overlay_cfg.gate_mode or "strict"))
        gating_initial = int(gating_info.get("initial", raw_detection_count))
        gating_accepted = int(gating_info.get("accepted", len(detections)))
        gating_rejected = int(gating_info.get("rejected", gating_initial - gating_accepted))
        gating_soft_cap = gating_info.get("soft_cap")
        gating_delta_frac = gating_info.get("delta_frac_used")
        substitution_fraction = (
            len(detections) / float(p_assets) if p_assets else 0.0
        )
        baseline_error_str = "" if not baseline_errors else ";".join(
            f"{key}:{value}" for key, value in sorted(baseline_errors.items())
        )
        pre_gate_stats = detect_stats.get("pre_gate", {}) if isinstance(detect_stats, dict) else {}
        raw_outliers_found = int(
            pre_gate_stats.get("raw_outliers_found", raw_detection_count)
        )
        pre_mp_edge_margin = float(pre_gate_stats.get("mp_edge_margin", float("nan")))
        pre_leakage_offcomp = float(pre_gate_stats.get("leakage_offcomp", float("nan")))
        pre_stability_eta_pass = float(pre_gate_stats.get("stability_eta_pass", float("nan")))
        pre_bracket_status = str(pre_gate_stats.get("bracket_status", ""))
        diag_record = {
            "window_id": start,
            "window_start": hold_start.isoformat(),
            "regime": regime,
            "detections": len(detections),
            "detection_rate": len(detections) / float(p_assets) if p_assets else 0.0,
            "acceptance_rate": len(detections) / float(p_assets) if p_assets else 0.0,
            "edge_margin_mean": float(np.mean(edge_margins)) if edge_margins else 0.0,
            "stability_margin_mean": float(np.mean(stability)) if stability else 0.0,
            "stability_eta": float(np.mean(stability)) if stability else 0.0,
            "isolation_share": float(np.mean(isolation)) if isolation else 0.0,
            "raw_detection_count": raw_detection_count,
            "substitution_fraction": substitution_fraction,
            "gating_mode": gating_mode,
            "gating_initial": gating_initial,
            "gating_accepted": gating_accepted,
            "gating_rejected": gating_rejected,
            "gating_soft_cap": int(gating_soft_cap) if gating_soft_cap is not None else np.nan,
            "gating_delta_frac": float(gating_delta_frac) if gating_delta_frac is not None else np.nan,
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
            "group_count_required": required_groups,
            "group_replicates_required": required_reps,
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
            "cov_condition_baseline": baseline_cond,
            "cov_condition_overlay": overlay_cond,
            "cov_condition_penalized": mv_condition_penalized,
            "mv_condition_cap": mv_condition_cap,
            "mv_condition_flag": bool(condition_flag),
            "mv_turnover": mv_turnover_value,
            "mv_turnover_cost_bps": mv_turnover_cost_bps,
            "baseline_errors": baseline_error_str,
            "factor_present": bool(factor_present),
            "changed_flag": int(changed_flag),
        }
        detail_fields = _detail_defaults()
        detail_fields.update(
            {
                "design_ok": design_ok_flag,
                "reps_by_label": group_label_counts,
                "mp_edge_margin": mp_edge_margin_value,
                "alignment_cos": alignment_cos_p50,
                "alignment_cos_p50": alignment_cos_p50,
                "leakage_offcomp": leakage_offcomp_value,
                "stability_eta_pass": stability_eta_share,
                "bracket_status": bracket_status,
                "raw_outliers_found": raw_outliers_found,
                "pre_mp_edge_margin": pre_mp_edge_margin,
                "pre_alignment_cos": pre_alignment_cos,
                "pre_leakage_offcomp": pre_leakage_offcomp,
                "pre_stability_eta_pass": pre_stability_eta_pass,
                "pre_bracket_status": pre_bracket_status,
            }
        )
        diag_record.update(detail_fields)
        diag_record.update(balance_diag_fields)
        for name, message in baseline_errors.items():
            diag_record[f"baseline_error_{name}"] = message
        return metrics_block, diag_record

    window_records: list[dict[str, object]] = []
    diagnostics_records: list[dict[str, object]] = []

    total_days = raw_returns.shape[0]
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
    metrics_detail_path = out_dir / "metrics_detail.csv"
    metrics_df.to_csv(metrics_detail_path, index=False)
    diagnostics_df = pd.DataFrame(diagnostics_records)
    if "group_label_counts" not in diagnostics_df.columns:
        diagnostics_df["group_label_counts"] = ""
    if "group_observations" not in diagnostics_df.columns:
        diagnostics_df["group_observations"] = np.nan
    if "vol_state_label" not in diagnostics_df.columns:
        diagnostics_df["vol_state_label"] = ""
    if "window_start" not in diagnostics_df.columns:
        diagnostics_df["window_start"] = ""
    if "raw_detection_count" not in diagnostics_df.columns:
        diagnostics_df["raw_detection_count"] = 0
    if "substitution_fraction" not in diagnostics_df.columns:
        diagnostics_df["substitution_fraction"] = 0.0
    if "gating_mode" not in diagnostics_df.columns:
        diagnostics_df["gating_mode"] = ""
    if "gating_initial" not in diagnostics_df.columns:
        diagnostics_df["gating_initial"] = 0
    if "gating_accepted" not in diagnostics_df.columns:
        diagnostics_df["gating_accepted"] = 0
    if "gating_rejected" not in diagnostics_df.columns:
        diagnostics_df["gating_rejected"] = 0
    if "gating_soft_cap" not in diagnostics_df.columns:
        diagnostics_df["gating_soft_cap"] = np.nan
    if "gating_delta_frac" not in diagnostics_df.columns:
        diagnostics_df["gating_delta_frac"] = np.nan
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
        "window_start",
        "regime",
        "detections",
        "detection_rate",
        "acceptance_rate",
        "edge_margin_mean",
        "stability_margin_mean",
        "stability_eta",
        "isolation_share",
        "alignment_cos_mean",
        "alignment_cos",
        "alignment_angle_mean",
        "raw_detection_count",
        "substitution_fraction",
        "gating_mode",
        "gating_initial",
        "gating_accepted",
        "gating_rejected",
        "gating_soft_cap",
        "gating_delta_frac",
        "reason_code",
        "resolved_config_path",
        "calm_threshold",
        "crisis_threshold",
        "vol_signal",
        "group_design",
        "group_count",
        "group_replicates",
        "group_count_required",
        "group_replicates_required",
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
        "reps_by_label",
        "group_observations",
        "vol_state_label",
        "cov_condition_baseline",
        "cov_condition_overlay",
        "cov_condition_penalized",
        "mv_condition_cap",
        "mv_condition_flag",
        "mv_turnover",
        "mv_turnover_cost_bps",
        "baseline_errors",
        "factor_present",
        "changed_flag",
        "design_ok",
        "mp_edge_margin",
        "alignment_cos_p50",
        "leakage_offcomp",
        "stability_eta_pass",
        "bracket_status",
        "raw_outliers_found",
        "pre_mp_edge_margin",
        "pre_alignment_cos",
        "pre_leakage_offcomp",
        "pre_stability_eta_pass",
        "pre_bracket_status",
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
                    "reps_by_label",
                    "bracket_status",
                }:
                    diagnostics_df[column] = ""
                elif column in {"factor_present", "changed_flag", "design_ok"}:
                    diagnostics_df[column] = 0
                elif column in {
                    "mp_edge_margin",
                    "alignment_cos",
                    "alignment_cos_p50",
                    "leakage_offcomp",
                    "stability_eta_pass",
                }:
                    diagnostics_df[column] = np.nan
                else:
                    diagnostics_df[column] = np.nan
        diagnostics_df = diagnostics_df[expected_diag_columns]

    regime_columns = [
        "window_id",
        "window_start",
        "regime",
        "vol_signal",
        "calm_threshold",
        "crisis_threshold",
        "vol_state_label",
    ]
    regime_df = diagnostics_df[regime_columns].copy() if not diagnostics_df.empty else pd.DataFrame(columns=regime_columns)
    regime_path = out_dir / "regime.csv"
    regime_df.to_csv(regime_path, index=False)

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
                qlike_mean=("qlike", "mean"),
                mv_turnover_mean=("mv_turnover", "mean"),
                mv_turnover_cost_bps=("mv_turnover_cost_bps", "mean"),
                cov_condition_median=("cov_condition", lambda s: float(np.nanmedian(pd.to_numeric(s, errors="coerce")))
                ),
                cov_condition_p90=("cov_condition", lambda s: float(np.nanquantile(pd.to_numeric(s, errors="coerce"), 0.9))
                ),
            )
            .reset_index()
        )
        summary.insert(0, "regime", regime)
        summaries.append(summary)
    summary_df = pd.concat(summaries, ignore_index=True)

    baseline_mask = summary_df["estimator"] == "baseline"
    baseline_df = summary_df[baseline_mask].rename(
        columns={
            "mse_mean": "baseline_mse",
            "es_mse_mean": "baseline_es_mse",
            "qlike_mean": "baseline_qlike_mean",
        }
    )[["regime", "portfolio", "baseline_mse", "baseline_es_mse", "baseline_qlike_mean"]]
    summary_df = summary_df.merge(baseline_df, on=["regime", "portfolio"], how="left")
    summary_df["delta_mse_vs_baseline"] = summary_df["mse_mean"] - summary_df["baseline_mse"]
    summary_df["delta_es_vs_baseline"] = summary_df["es_mse_mean"] - summary_df["baseline_es_mse"]
    summary_df["delta_qlike_vs_baseline"] = summary_df["qlike_mean"] - summary_df["baseline_qlike_mean"]
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
        "acceptance_rate": ("acceptance_rate", "mean"),
        "edge_margin_mean": ("edge_margin_mean", "mean"),
        "stability_margin_mean": ("stability_margin_mean", "mean"),
        "stability_eta": ("stability_eta", "mean"),
        "isolation_share": ("isolation_share", "mean"),
        "alignment_cos_mean": ("alignment_cos_mean", "mean"),
        "alignment_cos": ("alignment_cos", "mean"),
        "alignment_angle_mean": ("alignment_angle_mean", "mean"),
        "alignment_cos_p50": ("alignment_cos_p50", "mean"),
        "raw_detection_count": ("raw_detection_count", "mean"),
        "substitution_fraction": ("substitution_fraction", "mean"),
        "mp_edge_margin": ("mp_edge_margin", "mean"),
        "leakage_offcomp": ("leakage_offcomp", "mean"),
        "stability_eta_pass": ("stability_eta_pass", "mean"),
        "gating_initial": ("gating_initial", "mean"),
        "gating_accepted": ("gating_accepted", "mean"),
        "gating_rejected": ("gating_rejected", "mean"),
        "gating_soft_cap": ("gating_soft_cap", "mean"),
        "gating_delta_frac": ("gating_delta_frac", "mean"),
        "calm_threshold": ("calm_threshold", "mean"),
        "crisis_threshold": ("crisis_threshold", "mean"),
        "vol_signal": ("vol_signal", "mean"),
        "group_count": ("group_count", "mean"),
        "group_replicates": ("group_replicates", "mean"),
        "group_count_required": ("group_count_required", "mean"),
        "group_replicates_required": ("group_replicates_required", "mean"),
        "prewhiten_r2_mean": ("prewhiten_r2_mean", "mean"),
        "prewhiten_r2_median": ("prewhiten_r2_median", "mean"),
        "prewhiten_factor_count": ("prewhiten_factor_count", "mean"),
        "prewhiten_beta_abs_mean": ("prewhiten_beta_abs_mean", "mean"),
        "prewhiten_beta_abs_std": ("prewhiten_beta_abs_std", "mean"),
        "prewhiten_beta_abs_median": ("prewhiten_beta_abs_median", "mean"),
        "residual_energy_mean": ("residual_energy_mean", "mean"),
        "acceptance_delta": ("acceptance_delta", "mean"),
        "group_observations": ("group_observations", "mean"),
        "cov_condition_baseline": ("cov_condition_baseline", "mean"),
        "cov_condition_overlay": ("cov_condition_overlay", "mean"),
        "cov_condition_penalized": ("cov_condition_penalized", "mean"),
        "mv_condition_flag": ("mv_condition_flag", "mean"),
        "mv_turnover": ("mv_turnover", "mean"),
        "mv_turnover_cost_bps": ("mv_turnover_cost_bps", "mean"),
        "percent_changed": ("changed_flag", "mean"),
        "factor_present_share": ("factor_present", "mean"),
        "design_ok": ("design_ok", "mean"),
        "raw_outliers_found": ("raw_outliers_found", "mean"),
        "pre_mp_edge_margin": ("pre_mp_edge_margin", "mean"),
        "pre_alignment_cos": ("pre_alignment_cos", "mean"),
        "pre_leakage_offcomp": ("pre_leakage_offcomp", "mean"),
        "pre_stability_eta_pass": ("pre_stability_eta_pass", "mean"),
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
        diagnostics_summary["alignment_cos"] = np.nan
        diagnostics_summary["alignment_angle_mean"] = np.nan
        diagnostics_summary["raw_detection_count"] = np.nan
        diagnostics_summary["substitution_fraction"] = np.nan
        diagnostics_summary["percent_changed"] = np.nan
        diagnostics_summary["gating_initial"] = np.nan
        diagnostics_summary["gating_accepted"] = np.nan
        diagnostics_summary["gating_rejected"] = np.nan
        diagnostics_summary["gating_soft_cap"] = np.nan
        diagnostics_summary["gating_delta_frac"] = np.nan
        diagnostics_summary["group_design"] = ""
        diagnostics_summary["group_count"] = np.nan
        diagnostics_summary["group_replicates"] = np.nan
        diagnostics_summary["group_count_required"] = np.nan
        diagnostics_summary["group_replicates_required"] = np.nan
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
        diagnostics_summary["reps_by_label"] = ""
        diagnostics_summary["vol_state_label"] = ""
        diagnostics_summary["prewhiten_mode_requested"] = ""
        diagnostics_summary["prewhiten_mode_effective"] = ""
        diagnostics_summary["prewhiten_factors"] = ""
        diagnostics_summary["gating_mode"] = ""
        diagnostics_summary["factor_present_share"] = np.nan
        diagnostics_summary["mp_edge_margin"] = np.nan
        diagnostics_summary["alignment_cos_p50"] = np.nan
        diagnostics_summary["leakage_offcomp"] = np.nan
        diagnostics_summary["stability_eta_pass"] = np.nan
        diagnostics_summary["design_ok"] = np.nan
        diagnostics_summary["bracket_status"] = ""
        diagnostics_summary["raw_outliers_found"] = np.nan
        diagnostics_summary["pre_mp_edge_margin"] = np.nan
        diagnostics_summary["pre_alignment_cos"] = np.nan
        diagnostics_summary["pre_leakage_offcomp"] = np.nan
        diagnostics_summary["pre_stability_eta_pass"] = np.nan
        diagnostics_summary["pre_bracket_status"] = ""
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
        gating_mode_summary = (
            diagnostics_df.groupby("regime")["gating_mode"].agg(_mode_string).reset_index()
        )
        diagnostics_summary = diagnostics_summary.merge(gating_mode_summary, on="regime", how="left")
        if "group_label_counts" in diagnostics_df.columns:
            label_counts_summary = (
                diagnostics_df.groupby("regime")["group_label_counts"]
                .agg(_mode_string)
                .reset_index()
            )
            diagnostics_summary = diagnostics_summary.merge(label_counts_summary, on="regime", how="left")
        else:
            diagnostics_summary["group_label_counts"] = ""
        if "reps_by_label" in diagnostics_df.columns:
            reps_summary = (
                diagnostics_df.groupby("regime")["reps_by_label"]
                .agg(_mode_string)
                .reset_index()
            )
            diagnostics_summary = diagnostics_summary.merge(reps_summary, on="regime", how="left")
        else:
            diagnostics_summary["reps_by_label"] = ""
        if "vol_state_label" in diagnostics_df.columns:
            vol_state_summary = (
                diagnostics_df.groupby("regime")["vol_state_label"]
                .agg(_mode_string)
                .reset_index()
            )
            diagnostics_summary = diagnostics_summary.merge(vol_state_summary, on="regime", how="left")
        else:
            diagnostics_summary["vol_state_label"] = ""
        if "bracket_status" in diagnostics_df.columns:
            bracket_summary = (
                diagnostics_df.groupby("regime")["bracket_status"]
                .agg(_mode_string)
                .reset_index()
            )
            diagnostics_summary = diagnostics_summary.merge(bracket_summary, on="regime", how="left")
        else:
            diagnostics_summary["bracket_status"] = ""
        if "pre_bracket_status" in diagnostics_df.columns:
            pre_bracket_summary = (
                diagnostics_df.groupby("regime")["pre_bracket_status"]
                .agg(_mode_string)
                .reset_index()
            )
            diagnostics_summary = diagnostics_summary.merge(pre_bracket_summary, on="regime", how="left")
        else:
            diagnostics_summary["pre_bracket_status"] = ""
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

    def _fmt_scalar(value: object) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "nan"
        if not math.isfinite(numeric):
            return "nan"
        return f"{numeric:.3f}"

    full_row = diagnostics_summary[diagnostics_summary["regime"] == "full"]
    if not full_row.empty:
        row = full_row.iloc[0]
        residual_energy = row.get("residual_energy_mean", float("nan"))
        acceptance_delta = row.get("acceptance_delta", float("nan"))
        detection_rate = row.get("detection_rate", float("nan"))
        edge_margin = row.get("edge_margin_mean", float("nan"))
        alignment_cos = row.get("alignment_cos_mean", float("nan"))
        substitution_frac = row.get("substitution_fraction", float("nan"))
    else:
        residual_energy = float("nan")
        acceptance_delta = float("nan")
        detection_rate = float("nan")
        edge_margin = float("nan")
        alignment_cos = float("nan")
        substitution_frac = float("nan")

    print(
        "[diagnostics] prewhiten=%s r2_mean=%s residual_energy=%s acceptance_delta=%s detection_rate=%s edge_margin=%s alignment=%s substitution=%s"
        % (
            prewhiten_meta.mode_effective,
            _fmt_scalar(prewhiten_r2_mean),
            _fmt_scalar(residual_energy),
            _fmt_scalar(acceptance_delta),
            _fmt_scalar(detection_rate),
            _fmt_scalar(edge_margin),
            _fmt_scalar(alignment_cos),
            _fmt_scalar(substitution_frac),
        )
    )

    overlay_toggle_path = out_dir / "overlay_toggle.md"
    _write_overlay_toggle(overlay_toggle_path, diagnostics_summary)
    extra_plots = _plot_acceptance_edge_histograms(diagnostics_df, config.group_design or "", out_dir)
    regime_hist_config = [
        ("mp_edge_margin", "MP edge margin", "MP edge margin"),
        ("alignment_cos", "Alignment cos (p50)", "Alignment cos"),
        ("leakage_offcomp", "Leakage (off-comp ratio)", "Leakage / off-comp"),
        ("stability_eta_pass", "Stability η pass share", "Stability η pass share"),
        ("design_ok", "Design OK flag", "Design OK"),
        ("pre_mp_edge_margin", "Pre-gate MP edge margin", "Pre-gate MP edge"),
        ("pre_alignment_cos", "Pre-gate alignment cos (top q)", "Pre-gate alignment"),
        ("pre_leakage_offcomp", "Pre-gate leakage", "Pre-gate leakage"),
        ("pre_stability_eta_pass", "Pre-gate stability η pass", "Pre-gate stability η"),
        ("raw_outliers_found", "Raw outliers found", "Raw outliers"),
    ]
    for column, xlabel, title_prefix in regime_hist_config:
        extra_plots.update(
            _plot_regime_histograms(
                diagnostics_df,
                column,
                out_dir=out_dir,
                xlabel=xlabel,
                title_prefix=title_prefix,
            )
        )

    changed_windows_by_regime: dict[str, set[int]] = {regime: set() for regime in _REGIMES}
    changed_windows_by_regime["full"] = set()
    if {"changed_flag", "window_id"}.issubset(diagnostics_df.columns):
        diag_ids = diagnostics_df.dropna(subset=["window_id"]).copy()
        if not diag_ids.empty:
            diag_ids["window_id"] = diag_ids["window_id"].astype(int)
            changed_mask = diag_ids["changed_flag"].fillna(0).astype(int) == 1
            changed_windows_by_regime["full"] = set(diag_ids.loc[changed_mask, "window_id"])
            for regime_name in _REGIMES:
                regime_mask = diag_ids["regime"] == regime_name
                changed_windows_by_regime[regime_name] = set(
                    diag_ids.loc[regime_mask & changed_mask, "window_id"]
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
            "realised_var",
            "realised_es",
            "qlike",
            "mv_turnover",
            "mv_turnover_cost_bps",
            "cov_condition",
        ]
        available_cols = [col for col in risk_columns if col in risk_subset.columns]
        risk_subset[available_cols].to_csv(risk_path, index=False)
        outputs_risk[regime] = risk_path

        dm_rows = []
        comparators = ("baseline", "lw", "oas")
        for portfolio in ("ew", "mv"):
            for comparator in comparators:
                if regime == "full":
                    valid_ids = changed_windows_by_regime.get("full", set())
                else:
                    valid_ids = changed_windows_by_regime.get(regime, set())
                dm_stat, p_value, n_eff = _aligned_dm_stat(
                    metrics_df,
                    regime,
                    portfolio,
                    column="sq_error",
                    comparator=comparator,
                    valid_window_ids=valid_ids,
                )
                dm_stat_qlike, p_value_qlike, n_eff_qlike = _aligned_dm_stat(
                    metrics_df,
                    regime,
                    portfolio,
                    column="qlike",
                    comparator=comparator,
                    valid_window_ids=valid_ids,
                )
                dm_rows.append(
                    {
                        "portfolio": portfolio,
                        "baseline": comparator,
                        "dm_stat": dm_stat,
                        "p_value": p_value,
                        "n_effective": n_eff,
                        "dm_stat_qlike": dm_stat_qlike,
                        "p_value_qlike": p_value_qlike,
                        "n_effective_qlike": n_eff_qlike,
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

    plot_paths = _paths_to_strings(outputs_plots)
    if extra_plots:
        plot_paths.update({f"hist_{name}": str(path) for name, path in extra_plots.items()})
    run_metadata = {
        "git_sha": _current_git_sha(),
        "out_dir": str(out_dir),
        "config": resolved_payload,
        "execution": {
            "mode": resolved_payload.get("exec_mode", "deterministic"),
            "thread_caps": runtime.thread_caps_snapshot(),
        },
        "use_factor_prewhiten": bool(config.use_factor_prewhiten),
        "factors": {
            "key": factor_entry.key,
            "path": str(factor_entry.path),
            "sha256": factor_entry.sha256,
            "start_date": factor_entry.start_date,
            "end_date": factor_entry.end_date,
            "source": factor_entry.source,
            "note": factor_entry.note,
        }
        if factor_entry is not None
        else None,
        "outputs": {
            "metrics": _paths_to_strings(outputs_metrics),
            "risk": _paths_to_strings(outputs_risk),
            "dm": _paths_to_strings(outputs_dm),
            "diagnostics": _paths_to_strings(outputs_diag),
            "diagnostics_detail": _paths_to_strings(outputs_diag_detail),
            "plots": plot_paths,
            "overlay_toggle": str(overlay_toggle_path),
        },
    }
    _write_run_metadata(out_dir / "run.json", run_metadata)

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
    exec_mode = resolved.get("exec_mode", "deterministic")
    exec_settings = runtime.configure_exec_mode(exec_mode)
    resolved["exec_mode"] = exec_settings.mode
    run_evaluation(config, resolved_config=resolved)


if __name__ == "__main__":  # pragma: no cover
    main()
