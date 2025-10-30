#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.equity_panel.run import (  # noqa: E402
    load_config,
    _prepare_data,
    _apply_preprocessing,
    _run_single_period,
)


GRID_PRESETS: dict[str, dict[str, Iterable[float]]] = {
    "default": {
        "delta_frac": [0.01, 0.02],
        "eps": [0.02, 0.03],
        "eta": [0.4, 0.6],
        "a_grid": [90, 120],
    }
}


@dataclass
class SweepParams:
    delta_frac: float | None
    eps: float
    eta: float
    a_grid: int

    def slug(self) -> str:
        pieces = [
            f"df{self.delta_frac:.3f}" if self.delta_frac is not None else "dfNone",
            f"eps{self.eps:.3f}",
            f"eta{self.eta:.2f}",
            f"a{self.a_grid}",
        ]
        return "_".join(pieces)


def _load_grid(arg: str) -> dict[str, list[float]]:
    if arg in GRID_PRESETS:
        return {key: list(values) for key, values in GRID_PRESETS[arg].items()}
    path = Path(arg)
    if not path.exists():
        raise FileNotFoundError(f"Grid specification '{arg}' not found.")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Grid specification must be a mapping.")
    result: dict[str, list[float]] = {}
    for key in ("delta_frac", "eps", "eta", "a_grid"):
        values = data.get(key)
        if values is None:
            raise ValueError(f"Grid file missing '{key}' values.")
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"Grid entry '{key}' must be a sequence.")
        result[key] = [float(v) for v in values]
    result["a_grid"] = [int(round(v)) for v in result["a_grid"]]
    return result


def _build_parameter_grid(grid_cfg: dict[str, list[float]]) -> list[SweepParams]:
    combos: list[SweepParams] = []
    for delta_frac, eps, eta, a_grid in product(
        grid_cfg.get("delta_frac", [None]),
        grid_cfg.get("eps", [0.03]),
        grid_cfg.get("eta", [0.4]),
        grid_cfg.get("a_grid", [180]),
    ):
        combos.append(
            SweepParams(
                delta_frac=float(delta_frac) if delta_frac is not None else None,
                eps=float(eps),
                eta=float(eta),
                a_grid=int(a_grid),
            )
        )
    return combos


def _load_factor_returns(config: Mapping[str, Any]) -> pd.DataFrame | None:
    factor_csv = config.get("factor_csv")
    if not factor_csv:
        return None
    path = Path(str(factor_csv)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Factor CSV not found at '{path}'.")
    frame = pd.read_csv(path)
    if frame.empty or frame.shape[1] < 2:
        raise ValueError("Factor CSV must include a date column plus factors.")
    date_candidates = [
        col
        for col in frame.columns
        if str(col).lower() in {"date", "timestamp", "time", "week", "period"}
    ]
    date_col = date_candidates[0] if date_candidates else frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.set_index(date_col).sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    if frame.empty:
        raise ValueError("Factor CSV contains no usable numeric data.")
    return frame


def _extract_metrics(
    run_dir: Path,
    estimators: Iterable[str],
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics_summary.csv"
    summary = json.loads(summary_path.read_text())
    metrics_df = pd.read_csv(metrics_path)

    detection_rate = float(summary.get("detection_rate", 0.0))
    edge_stats = summary.get("edge_margin_stats", {}) or {}
    edge_median = edge_stats.get("median")
    windows_eval = int(summary.get("rolling_windows_evaluated", 0) or 0)

    gating = summary.get("gating", {}) or {}
    substituted = int(gating.get("windows_substituted", 0) or 0)
    subs_frac = float(substituted / windows_eval) if windows_eval else float("nan")

    record: dict[str, Any] = {
        "detection_rate": detection_rate,
        "edge_margin_median": float(edge_median) if edge_median is not None else float("nan"),
        "substitution_fraction": subs_frac,
        "windows_evaluated": windows_eval,
    }

    target_estimators = {"dealias", "de-aliased", "de_aliased"}
    estimators_lower = {name.lower() for name in estimators}

    baseline = metrics_df[
        (metrics_df["strategy"] == "Equal Weight") & (metrics_df["estimator"].str.lower().isin(target_estimators))
    ]
    if baseline.empty:
        return record
    de_row = baseline.iloc[0]

    def _copy_dm(prefix: str, suffix: str) -> None:
        stat_col = f"{prefix}_{suffix}"
        p_col = f"{prefix}_{suffix.replace('stat', 'p')}" if "stat" in suffix else None
        if stat_col in de_row:
            record[f"{stat_col}"] = float(de_row[stat_col])
        if p_col and p_col in de_row:
            record[f"{p_col}"] = float(de_row[p_col])

    # Legacy MSE DM columns
    for estimator in ("lw", "oas"):
        if estimator in estimators_lower:
            _copy_dm("dm_stat_de_vs", estimator)
            _copy_dm("dm_p_de_vs", estimator)

    # QLIKE DM columns (named *_qlike in metrics summary)
    for estimator in ("lw", "oas"):
        stat_col = f"dm_stat_de_vs_{estimator}_qlike"
        p_col = f"dm_p_de_vs_{estimator}_qlike"
        if stat_col in de_row:
            record[stat_col] = float(de_row[stat_col])
        if p_col in de_row:
            record[p_col] = float(de_row[p_col])

    return record


def run_sweep(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).expanduser()
    config = load_config(config_path)

    # Allow CLI overrides while preserving defaults.
    if args.start_date:
        config["start_date"] = args.start_date
    if args.end_date:
        config["end_date"] = args.end_date
    if args.design:
        config["design"] = args.design

    grid_cfg = _load_grid(args.grid)
    sweep_params = _build_parameter_grid(grid_cfg)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        df = pd.DataFrame(
            {
                "delta_frac": [param.delta_frac for param in sweep_params],
                "eps": [param.eps for param in sweep_params],
                "eta": [param.eta for param in sweep_params],
                "a_grid": [param.a_grid for param in sweep_params],
            }
        )
        summary_path = output_root / "sweep_summary.csv"
        df.to_csv(summary_path, index=False)
        return summary_path

    daily_returns = _prepare_data(config)
    winsorize_q_cfg = config.get("winsorize_q")
    huber_c_cfg = config.get("huber_c")
    if winsorize_q_cfg is not None and huber_c_cfg is not None:
        raise ValueError("winsorize_q and huber_c preprocessing are mutually exclusive.")
    winsorize_q_val = float(winsorize_q_cfg) if winsorize_q_cfg is not None else None
    huber_c_val = float(huber_c_cfg) if huber_c_cfg is not None else None
    processed_returns, preprocess_flags = _apply_preprocessing(
        daily_returns,
        winsorize_q=winsorize_q_val,
        huber_c=huber_c_val,
    )

    factor_returns = _load_factor_returns(config)
    delta_base = float(config.get("dealias_delta", 0.0))
    delta_frac_base_raw = config.get("dealias_delta_frac")
    delta_frac_base = float(delta_frac_base_raw) if delta_frac_base_raw is not None else None
    off_leak_val = config.get("off_component_leak_cap")
    if off_leak_val is not None:
        off_leak_val = float(off_leak_val)
    energy_min_abs_raw = config.get("energy_min_abs")
    energy_min_abs_val = float(energy_min_abs_raw) if energy_min_abs_raw is not None else None
    minvar_box_cfg = config.get("minvar_box", (0.0, 0.05))
    if isinstance(minvar_box_cfg, Mapping):
        minvar_box_vals = (
            float(minvar_box_cfg.get("lo", 0.0)),
            float(minvar_box_cfg.get("hi", 0.05)),
        )
    else:
        minvar_box_vals = tuple(float(x) for x in minvar_box_cfg)
    turnover_cost = float(config.get("turnover_cost_bps", 0.0))
    minvar_ridge_val = float(config.get("minvar_ridge", 1e-3))
    nested_reps = int(config.get("nested_replicates", 5))
    design_mode = str(config.get("design", "oneway"))
    target_component = int(config.get("target_component", 0))
    oneway_solver = str(config.get("oneway_a_solver", "auto"))
    estimator_value = str(config.get("estimator", "dealias"))
    window_weeks = int(config["window_weeks"])
    horizon_weeks = int(config["horizon_weeks"])
    partial_week_policy = str(config.get("partial_week_policy", "drop"))

    records: list[dict[str, Any]] = []
    for idx, params in enumerate(sweep_params):
        run_dir = output_root / f"run_{idx:03d}_{params.slug()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        label = f"sweep_{idx:03d}"
        _run_single_period(
            processed_returns,
            start=config["start_date"],
            end=config["end_date"],
            output_dir=run_dir,
            window_weeks=window_weeks,
            horizon_weeks=horizon_weeks,
            delta=delta_base,
            delta_frac=params.delta_frac if params.delta_frac is not None else delta_frac_base,
            eps=float(params.eps),
            stability_eta=float(params.eta),
            signed_a=bool(config.get("signed_a", True)),
            target_component=target_component,
            partial_week_policy=partial_week_policy,
            precompute_panel=False,
            cache_dir=None,
            resume_cache=False,
            cs_drop_top_frac=float(config.get("cs_drop_top_frac", 0.05)),
            cs_sensitivity_frac=float(config.get("cs_sensitivity_frac", 0.0)),
            off_component_leak_cap=off_leak_val,
            sigma_ablation=False,
            label=label,
            crisis_label=None,
            design_mode=design_mode,
            nested_replicates=nested_reps,
            oneway_a_solver=oneway_solver,
            estimator=estimator_value,
            progress=False,
            a_grid=int(params.a_grid),
            energy_min_abs=energy_min_abs_val,
            factor_returns=factor_returns,
            minvar_ridge=minvar_ridge_val,
            minvar_box=minvar_box_vals,
            turnover_cost_bps=turnover_cost,
            preprocess_flags=preprocess_flags,
            gating=config.get("gating"),
        )

        metrics = _extract_metrics(run_dir, args.estimators)
        record = {
            "delta_frac": params.delta_frac if params.delta_frac is not None else delta_frac_base,
            "eps": params.eps,
            "eta": params.eta,
            "a_grid": params.a_grid,
            "run_path": str(run_dir),
            "design": design_mode,
        }
        record.update(metrics)
        records.append(record)

    summary_df = pd.DataFrame.from_records(records)
    summary_path = output_root / "sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep acceptance parameters for de-alias calibration.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/equity_panel/config.smoke.yaml",
        help="Base configuration YAML (default: smoke slice).",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="default",
        help="Grid preset name or YAML file describing delta_frac/eps/eta/a_grid values.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="experiments/equity_panel/sweeps",
        help="Directory to store sweep runs and summary CSV.",
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=["dealias", "lw", "oas", "cc", "tyler"],
        help="Estimators to include when reporting DM statistics.",
    )
    parser.add_argument("--design", type=str, default=None, help="Override design mode for the sweep.")
    parser.add_argument("--start-date", type=str, default=None, help="Override start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=None, help="Override end date (YYYY-MM-DD).")
    parser.add_argument("--dry-run", action="store_true", help="Skip execution and emit an empty sweep summary.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    summary_path = run_sweep(args)
    print(f"Sweep summary written to {summary_path}")


if __name__ == "__main__":
    main()
