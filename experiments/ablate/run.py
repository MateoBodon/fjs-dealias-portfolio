#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.eval.run import EvalConfig, run_evaluation  # noqa: E402
from tools.make_summary import write_summaries  # noqa: E402


BOOL_KEYS = {
    "require_isolated",
    "prewhiten",
}
INT_KEYS = {
    "window",
    "horizon",
    "seed",
    "overlay_a_grid",
    "overlay_seed",
    "q_max",
    "alignment_top_p",
    "vol_ewma_span",
    "bootstrap_samples",
    "calm_window_sample",
    "crisis_window_top_k",
}
FLOAT_KEYS = {
    "angle_min_cos",
    "cs_drop_top_frac",
    "mv_gamma",
    "mv_tau",
    "calm_quantile",
    "crisis_quantile",
}


@dataclass(frozen=True)
class PanelSpec:
    name: str
    returns_csv: Path
    factors_csv: Path | None = None
    start: str | None = None
    end: str | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Ablation config at {path} must be a mapping.")
    return dict(data)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"Cannot interpret {value!r} as boolean.")


def _coerce_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key in BOOL_KEYS:
        return _coerce_bool(value)
    if key in INT_KEYS:
        return int(value)
    if key in FLOAT_KEYS:
        return float(value)
    return value


def _normalise_defaults(defaults: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in defaults.items():
        result[key] = _coerce_value(key, value)
    return result


def _normalise_combo(combo: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in combo.items():
        result[key] = _coerce_value(key, value)
    return result


def _combo_identifier(params: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(params.keys()):
        value = params[key]
        if isinstance(value, float):
            formatted = f"{value:.6g}"
        else:
            formatted = str(value)
        formatted = formatted.replace("/", "-")
        parts.append(f"{key}-{formatted}")
    slug = "_".join(parts)
    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:8]
    safe_slug = slug.replace(".", "p").replace(" ", "")
    return f"{safe_slug}_{digest}"


def _is_default_combo(
    combo: Mapping[str, Any],
    defaults: Mapping[str, Any],
    keys: Sequence[str],
) -> bool:
    for key in keys:
        default_val = defaults.get(key)
        combo_val = combo.get(key)
        if key in FLOAT_KEYS:
            if default_val is None and combo_val is None:
                continue
            if default_val is None or combo_val is None:
                return False
            if not math.isclose(float(combo_val), float(default_val), rel_tol=1e-9, abs_tol=1e-9):
                return False
        else:
            if combo_val != default_val:
                return False
    return True


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_panels(section: Mapping[str, Any]) -> dict[str, PanelSpec]:
    panels: dict[str, PanelSpec] = {}
    for name, payload in section.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"Panel entry '{name}' must be a mapping.")
        returns_csv = Path(payload.get("returns_csv", "")).expanduser()
        if not returns_csv:
            raise ValueError(f"Panel '{name}' missing returns_csv path.")
        factors_csv = payload.get("factors_csv")
        factors_path = Path(factors_csv).expanduser() if factors_csv else None
        panels[name] = PanelSpec(
            name=name,
            returns_csv=returns_csv,
            factors_csv=factors_path,
            start=payload.get("start"),
            end=payload.get("end"),
        )
    return panels


def _extract_perf(perf_df: pd.DataFrame, regime: str, portfolio: str) -> pd.Series:
    mask = perf_df["regime"].astype(str).str.lower().eq(regime.lower())
    mask &= perf_df["portfolio"].astype(str).str.lower().eq(portfolio.lower())
    subset = perf_df.loc[mask]
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.iloc[0]


def _extract_detection(det_df: pd.DataFrame, regime: str) -> pd.Series:
    mask = det_df["regime"].astype(str).str.lower().eq(regime.lower())
    subset = det_df.loc[mask]
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.iloc[0]


def _safe_get(series: pd.Series, key: str) -> float:
    if key not in series:
        return float("nan")
    try:
        return float(series[key])
    except (TypeError, ValueError):
        return float("nan")


def run_ablation(
    config_path: Path,
    *,
    force: bool = False,
    limit: int | None = None,
    calm_window_sample: int | None = None,
    crisis_window_top_k: int | None = None,
) -> Path:
    cfg = _load_yaml(config_path)
    defaults = _normalise_defaults(cfg.get("defaults", {}))
    if calm_window_sample is not None:
        defaults["calm_window_sample"] = int(calm_window_sample)
    if crisis_window_top_k is not None:
        defaults["crisis_window_top_k"] = int(crisis_window_top_k)
    grid = cfg.get("grid", {})
    if not grid:
        raise ValueError("Ablation config requires a non-empty grid section.")
    panels_section = cfg.get("panels", {})
    if not panels_section:
        raise ValueError("Ablation config requires at least one panel definition.")
    panels = _load_panels(panels_section)

    io_cfg = cfg.get("io", {})
    cache_dir = Path(io_cfg.get("cache_dir", "ablations/cache")).resolve()
    out_csv = Path(io_cfg.get("out_csv", "ablations/ablation_matrix.csv")).resolve()
    summary_cfg = cfg.get("summary", {})
    summary_regime = str(summary_cfg.get("regime", "full"))
    summary_portfolios = summary_cfg.get("portfolios", ["ew", "mv"])
    if not summary_portfolios:
        summary_portfolios = ["ew", "mv"]

    grid_keys = list(grid.keys())
    if "panel" not in grid_keys:
        raise ValueError("Grid must include a 'panel' dimension.")

    grid_values = []
    for key in grid_keys:
        values = grid[key]
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            raise ValueError(f"Grid entry '{key}' must be an iterable of values.")
        grid_values.append(list(values))

    combinations: list[dict[str, Any]] = []
    for raw_values in product(*grid_values):
        combo_raw = dict(zip(grid_keys, raw_values))
        combo = _normalise_combo(combo_raw)
        panel_name = combo.get("panel")
        if panel_name not in panels:
            raise ValueError(f"Unknown panel '{panel_name}'. Available: {sorted(panels)}")
        combinations.append(combo)

    if limit is not None:
        combinations = combinations[: int(limit)]

    compare_keys = [key for key in grid_keys if key != "panel"]

    base_metrics: dict[str, dict[str, float]] = {}
    records: list[dict[str, Any]] = []

    _ensure_dir(cache_dir)
    _ensure_dir(out_csv.parent)

    for combo in combinations:
        panel_name = combo["panel"]
        panel_spec = panels[panel_name]
        combo_id = _combo_identifier({k: combo[k] for k in combo if k != "panel"} | {"panel": panel_name})
        run_dir = cache_dir / combo_id
        summary_perf_path = run_dir / "summary" / "summary_perf.csv"
        should_run = force or not summary_perf_path.exists()

        if should_run:
            calm_window_limit = combo.get("calm_window_sample", defaults.get("calm_window_sample"))
            if calm_window_limit is not None:
                calm_window_limit = int(calm_window_limit)
            crisis_window_limit = combo.get("crisis_window_top_k", defaults.get("crisis_window_top_k"))
            if crisis_window_limit is not None:
                crisis_window_limit = int(crisis_window_limit)
            _ensure_dir(run_dir)
            config = EvalConfig(
                returns_csv=panel_spec.returns_csv,
                factors_csv=panel_spec.factors_csv,
                window=int(defaults.get("window", 126)),
                horizon=int(defaults.get("horizon", 21)),
                out_dir=run_dir,
                start=panel_spec.start,
                end=panel_spec.end,
                shrinker=str(combo.get("shrinker", defaults.get("shrinker", "rie"))),
                seed=int(defaults.get("seed", 0)),
                calm_quantile=float(defaults.get("calm_quantile", 0.2)),
                crisis_quantile=float(defaults.get("crisis_quantile", 0.8)),
                vol_ewma_span=int(defaults.get("vol_ewma_span", 21)),
                config_path=None,
                thresholds_path=None,
                echo_config=False,
                reason_codes=True,
                workers=None,
                overlay_a_grid=int(defaults.get("overlay_a_grid", 60)),
                overlay_seed=int(defaults.get("overlay_seed", defaults.get("seed", 0))),
                mv_gamma=float(defaults.get("mv_gamma", 5e-4)),
                mv_tau=float(defaults.get("mv_tau", 0.0)),
                bootstrap_samples=int(defaults.get("bootstrap_samples", 0)),
                require_isolated=bool(combo.get("require_isolated", defaults.get("require_isolated", True))),
                q_max=int(combo.get("q_max", defaults.get("q_max", 1))),
                edge_mode=str(combo.get("edge_mode", defaults.get("edge_mode", "tyler"))),
                angle_min_cos=(
                    float(combo.get("angle_min_cos"))
                    if combo.get("angle_min_cos") is not None
                    else defaults.get("angle_min_cos")
                ),
                alignment_top_p=int(defaults.get("alignment_top_p", 3)),
                cs_drop_top_frac=(
                    float(combo.get("cs_drop_top_frac"))
                    if combo.get("cs_drop_top_frac") is not None
                    else defaults.get("cs_drop_top_frac")
                ),
                prewhiten=bool(combo.get("prewhiten", defaults.get("prewhiten", True))),
                calm_window_sample=calm_window_limit,
                crisis_window_top_k=crisis_window_limit,
            )
            print(f"[ablate] Running {combo_id}")
            run_evaluation(config)
        else:
            print(f"[ablate] Reusing cached run {combo_id}")

        artifacts_map = write_summaries([run_dir])
        artifacts = artifacts_map[run_dir]

        perf_df = artifacts.performance
        det_df = artifacts.detection

        perf_rows = {
            portfolio: _extract_perf(perf_df, summary_regime, portfolio)
            for portfolio in summary_portfolios
        }
        det_row = _extract_detection(det_df, summary_regime)

        record: dict[str, Any] = {
            "panel": panel_name,
            "run_path": str(run_dir),
        }
        record.update(combo)

        for portfolio in summary_portfolios:
            row = perf_rows.get(portfolio)
            prefix = f"{portfolio}_"
            if row is None or row.empty:
                record[f"{prefix}delta_mse_vs_baseline"] = float("nan")
                record[f"{prefix}delta_es_vs_baseline"] = float("nan")
                record[f"{prefix}var95_overlay"] = float("nan")
                record[f"{prefix}var95_baseline"] = float("nan")
                record[f"{prefix}dm_stat"] = float("nan")
                record[f"{prefix}dm_p_value"] = float("nan")
                record[f"{prefix}n_effective"] = float("nan")
            else:
                record[f"{prefix}delta_mse_vs_baseline"] = _safe_get(row, "delta_mse_vs_baseline")
                record[f"{prefix}delta_es_vs_baseline"] = _safe_get(row, "delta_es_vs_baseline")
                record[f"{prefix}var95_overlay"] = _safe_get(row, "var95_overlay")
                record[f"{prefix}var95_baseline"] = _safe_get(row, "var95_baseline")
                record[f"{prefix}dm_stat"] = _safe_get(row, "dm_stat")
                record[f"{prefix}dm_p_value"] = _safe_get(row, "dm_p_value")
                record[f"{prefix}n_effective"] = _safe_get(row, "n_effective")

        record["detections_mean"] = _safe_get(det_row, "detections_mean")
        record["detection_rate_mean"] = _safe_get(det_row, "detection_rate_mean")
        record["edge_margin_mean"] = _safe_get(det_row, "edge_margin_mean")
        record["stability_margin_mean"] = _safe_get(det_row, "stability_margin_mean")
        record["isolation_share_mean"] = _safe_get(det_row, "isolation_share_mean")
        record["alignment_cos_mean"] = _safe_get(det_row, "alignment_cos_mean")
        record["alignment_angle_mean"] = _safe_get(det_row, "alignment_angle_mean")
        record["detection_rate_median"] = _safe_get(det_row, "detection_rate_median")
        record["alignment_cos_median"] = _safe_get(det_row, "alignment_cos_median")
        record["reason_code_mode"] = det_row.get("reason_code", "") if isinstance(det_row, pd.Series) else ""

        metrics_snapshot = {
            "detections_mean": record["detections_mean"],
            "detection_rate_mean": record["detection_rate_mean"],
            "edge_margin_mean": record["edge_margin_mean"],
            "stability_margin_mean": record["stability_margin_mean"],
            "isolation_share_mean": record["isolation_share_mean"],
            "alignment_cos_mean": record["alignment_cos_mean"],
        }
        for portfolio in summary_portfolios:
            prefix = f"{portfolio}_"
            metrics_snapshot[f"{prefix}delta_mse_vs_baseline"] = record[f"{prefix}delta_mse_vs_baseline"]

        if _is_default_combo(combo, defaults, compare_keys):
            base_metrics[panel_name] = metrics_snapshot

        record_path = run_dir / "ablation_params.json"
        if should_run or not record_path.exists():
            payload = {"panel": panel_name, **combo}
            record_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        records.append(record)

    missing_base = [panel for panel in panels if panel not in base_metrics]
    if missing_base:
        raise RuntimeError(
            "Missing default combination for panels: " + ", ".join(sorted(missing_base))
        )

    df = pd.DataFrame(records)
    diff_keys = [
        "detections_mean",
        "detection_rate_mean",
        "edge_margin_mean",
        "stability_margin_mean",
        "isolation_share_mean",
        "alignment_cos_mean",
    ]
    for portfolio in summary_portfolios:
        diff_keys.append(f"{portfolio}_delta_mse_vs_baseline")

    for key in diff_keys:
        diff_col = f"{key}_vs_default"
        diffs: list[float] = []
        for _, row in df.iterrows():
            panel_name = row["panel"]
            base_val = base_metrics[panel_name].get(key, float("nan"))
            value = row.get(key, float("nan"))
            if np.isnan(value) or np.isnan(base_val):
                diffs.append(float("nan"))
            else:
                diffs.append(float(value) - float(base_val))
        df[diff_col] = diffs

    sort_columns = ["panel"] + [key for key in compare_keys if key in df.columns]
    df = df.sort_values(sort_columns).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[ablate] Wrote {out_csv}")
    return out_csv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run overlay ablation grid and summarise results.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/ablate/ablation_matrix.yaml"),
        help="Path to ablation matrix YAML configuration.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all combinations even when cached summaries exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N combinations (testing/debug).",
    )
    parser.add_argument(
        "--calm-window-sample",
        type=int,
        default=None,
        help="Uniform calm-window sample size applied to every evaluation.",
    )
    parser.add_argument(
        "--crisis-window-topk",
        type=int,
        default=None,
        help="Top-K crisis windows by edge margin retained for each evaluation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_ablation(
        args.config.resolve(),
        force=args.force,
        limit=args.limit,
        calm_window_sample=args.calm_window_sample,
        crisis_window_top_k=args.crisis_window_topk,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
