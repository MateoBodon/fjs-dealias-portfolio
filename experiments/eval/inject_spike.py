#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from eval.balance import build_balanced_window
from eval.clean import apply_nan_policy
from experiments.daily.grouping import GroupingError
from experiments.eval.config import resolve_eval_config
import experiments.eval.run as eval_run
from fjs.overlay import OverlayConfig, detect_spikes


@dataclass(frozen=True)
class WindowSample:
    window_id: int
    matrix: np.ndarray
    group_labels: np.ndarray


def _parse_float_list(raw: str, name: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        tok = token.strip()
        if not tok:
            continue
        try:
            values.append(float(tok))
        except ValueError as exc:
            raise ValueError(f"invalid {name} value '{token}'") from exc
    if not values:
        raise ValueError(f"{name} grid must contain at least one value")
    return values


def _make_overlay_config(config: eval_run.EvalConfig) -> OverlayConfig:
    return OverlayConfig(
        shrinker=config.shrinker,
        q_max=int(config.q_max) if config.q_max is not None else None,
        max_detections=int(config.q_max) if config.q_max is not None else None,
        edge_mode=str(config.edge_mode),
        seed=config.overlay_seed if config.overlay_seed is not None else config.seed,
        a_grid=int(config.overlay_a_grid),
        delta_frac=getattr(config, "overlay_delta_frac", None),
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
        coarse_candidate=bool(getattr(config, "coarse_candidate", False)),
    )


def _collect_windows(
    config: eval_run.EvalConfig,
    raw_returns: pd.DataFrame,
    residuals: pd.DataFrame,
    vol_proxy_full: pd.Series,
    *,
    factor_tracking_required: bool,
    residual_index_set: set[pd.Timestamp],
) -> list[WindowSample]:
    windows: list[WindowSample] = []
    vol_proxy_past = vol_proxy_full.shift(1)
    total_days = raw_returns.shape[0]
    start_indices = range(0, total_days - config.window - config.horizon + 1)

    for start in start_indices:
        fit_end = start + config.window
        hold_end = fit_end + config.horizon
        fit_labels = list(raw_returns.index[start:fit_end])
        hold_labels = list(raw_returns.index[fit_end:hold_end])
        if len(fit_labels) < config.window or len(hold_labels) < config.horizon:
            continue
        fit_base = raw_returns.loc[fit_labels]
        overlay_allowed = True
        if factor_tracking_required:
            needed = fit_labels + hold_labels
            overlay_allowed = all(label in residual_index_set for label in needed)
        fit = residuals.loc[fit_labels] if overlay_allowed else fit_base

        train_end = pd.to_datetime(fit_labels[-1])
        hold_start = pd.to_datetime(hold_labels[0])
        calm_cut, crisis_cut = eval_run._vol_thresholds(vol_proxy_past, train_end, config)
        try:
            fit_grouped, group_labels = eval_run._build_grouped_window(
                fit,
                config=config,
                calm_threshold=calm_cut,
                crisis_threshold=crisis_cut,
                vol_proxy=vol_proxy_past,
            )
        except GroupingError:
            continue

        fit_grouped = fit_grouped.replace([np.inf, -np.inf], np.nan)
        nan_result = apply_nan_policy(
            fit_grouped,
            group_labels,
            max_missing_asset=float(config.max_missing_asset),
            max_missing_group_row=float(config.max_missing_group_row),
        )
        fit_clean = nan_result.frame.replace([np.inf, -np.inf], np.nan)
        balance_result = build_balanced_window(
            fit_clean,
            nan_result.labels,
            min_replicates=eval_run._required_replicates(config.group_design, config),
        )
        if balance_result.reason in {"empty_after_balance", "insufficient_reps"}:
            continue
        fit_balanced = balance_result.frame.replace([np.inf, -np.inf], np.nan)
        if fit_balanced.shape[0] == 0 or fit_balanced.shape[1] == 0:
            continue
        if fit_balanced.isna().any().any():
            valid_mask = fit_balanced.notna().all(axis=1)
            if not bool(valid_mask.all()):
                keep_positions = np.where(valid_mask.to_numpy(dtype=bool))[0]
                fit_balanced = fit_balanced.iloc[keep_positions]
                balance_labels = balance_result.labels[keep_positions]
            else:
                balance_labels = balance_result.labels
        else:
            balance_labels = balance_result.labels
        if fit_balanced.shape[0] == 0 or fit_balanced.shape[1] == 0:
            continue
        matrix = fit_balanced.to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(matrix).all():
            continue
        windows.append(
            WindowSample(
                window_id=start,
                matrix=matrix,
                group_labels=np.asarray(balance_labels, dtype=np.intp),
            )
        )
    return windows


def _baseline_fp(samples: list[WindowSample], overlay_cfg: OverlayConfig) -> tuple[list[bool], float]:
    flags: list[bool] = []
    for sample in samples:
        detections = detect_spikes(sample.matrix, sample.group_labels, config=overlay_cfg)
        flags.append(len(detections) > 0)
    rate = float(sum(flags)) / float(len(flags)) if samples else float("nan")
    return flags, rate


def _inject_spike(matrix: np.ndarray, rng: np.random.Generator, mu: float) -> np.ndarray:
    n_obs, n_assets = matrix.shape
    vec = rng.normal(size=n_assets)
    norm = np.linalg.norm(vec)
    if norm <= 0.0:
        vec = np.ones(n_assets, dtype=np.float64)
        norm = np.linalg.norm(vec)
    vec /= norm
    coeffs = rng.normal(size=n_obs)
    injection = np.outer(coeffs, vec) * math.sqrt(mu)
    return matrix + injection


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weak-spike injection evaluation on real residuals")
    parser.add_argument("--returns-csv", type=Path, required=True)
    parser.add_argument("--factors-csv", type=Path, default=None)
    parser.add_argument("--window", type=int, default=126)
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--assets-top", type=int, default=150)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument("--group-design", type=str, default="week")
    parser.add_argument("--use-factor-prewhiten", type=int, choices=[0, 1], default=1)
    parser.add_argument("--coarse-candidate", type=int, choices=[0, 1], default=0)
    parser.add_argument("--mu-grid", type=str, default="3,4,5")
    parser.add_argument("--inject-frac-min", type=float, default=0.05)
    parser.add_argument("--inject-frac-max", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("reports/figures"))
    args = parser.parse_args(argv)
    args.mu_values = _parse_float_list(args.mu_grid, "mu")
    if args.inject_frac_min <= 0 or args.inject_frac_max <= 0:
        raise ValueError("Injection fractions must be positive.")
    if args.inject_frac_max < args.inject_frac_min:
        raise ValueError("inject-frac-max must be >= inject-frac-min")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if plt is None:
        raise RuntimeError("matplotlib is required for injection plots")

    config_args = {
        "returns_csv": args.returns_csv,
        "factors_csv": args.factors_csv,
        "window": args.window,
        "horizon": args.horizon,
        "start": args.start,
        "end": args.end,
        "assets_top": args.assets_top,
        "config": args.config,
        "thresholds": args.thresholds,
        "group_design": args.group_design,
        "use_factor_prewhiten": args.use_factor_prewhiten,
        "coarse_candidate": args.coarse_candidate,
    }
    resolved = resolve_eval_config(config_args)
    config = resolved.config
    panel, raw_returns, whitening, telemetry, factor_entry = eval_run._prepare_returns(config)
    residuals = whitening.residuals.sort_index()
    raw_returns = raw_returns.sort_index()
    residual_index_set = set(residuals.index)
    factor_tracking_required = bool(config.use_factor_prewhiten and telemetry.mode_effective != "off")
    vol_proxy_full = eval_run._compute_vol_proxy(residuals, span=config.vol_ewma_span)

    samples = _collect_windows(
        config,
        raw_returns,
        residuals,
        vol_proxy_full,
        factor_tracking_required=factor_tracking_required,
        residual_index_set=residual_index_set,
    )
    if not samples:
        raise RuntimeError("No valid windows available for injection analysis.")

    overlay_cfg = _make_overlay_config(config)
    base_flags, fp_rate = _baseline_fp(samples, overlay_cfg)

    rng = np.random.default_rng(args.seed)
    summary_rows: list[dict[str, float]] = []
    recall_points: list[tuple[float, float]] = []

    for mu in args.mu_values:
        frac = float(rng.uniform(args.inject_frac_min, args.inject_frac_max))
        n_injected = max(1, int(round(frac * len(samples))))
        indices = rng.choice(len(samples), size=n_injected, replace=False)
        hits = 0
        for idx in indices:
            injected_matrix = _inject_spike(samples[idx].matrix, rng, mu)
            detections = detect_spikes(injected_matrix, samples[idx].group_labels, config=overlay_cfg)
            if detections:
                hits += 1
        recall = hits / n_injected if n_injected > 0 else float("nan")
        summary_rows.append(
            {
                "mu": mu,
                "fraction": frac,
                "n_injected": n_injected,
                "recall": recall,
                "false_positive_rate": fp_rate,
            }
        )
        recall_points.append((mu, recall))

    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_root / "inject_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    recall_fig = out_root / "inject_recall.png"
    fp_fig = out_root / "inject_fp.png"

    fig, ax = plt.subplots(figsize=(5, 3.5))
    recall_df = summary_df.sort_values("mu")
    ax.plot(recall_df["mu"], recall_df["recall"], marker="o")
    ax.set_xlabel("Injected spike μ")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(recall_fig, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["clean"], [fp_rate])
    ax.set_ylabel("False positive rate")
    ax.set_ylim(0, max(0.05, fp_rate * 1.2))
    fig.tight_layout()
    fig.savefig(fp_fig, dpi=200)
    plt.close(fig)

    manifest = {
        "config": eval_run._serialise_config(config),
        "summary_csv": summary_csv.as_posix(),
        "recall_figure": recall_fig.as_posix(),
        "fp_figure": fp_fig.as_posix(),
        "windows": len(samples),
        "false_positive_rate": fp_rate,
        "mu_grid": args.mu_values,
    }
    manifest_path = out_root / "inject_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[inject] windows={len(samples)} fp_rate={fp_rate:.3f} "
        + ", ".join(f"μ={row['mu']}: recall={row['recall']:.2f}" for row in summary_rows)
    )


if __name__ == "__main__":
    main()
