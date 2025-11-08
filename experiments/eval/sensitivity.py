#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - plotting dependency optional in CI
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from evaluation.dm import dm_test
from tools.verify_dataset import verify_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
EVAL_SCRIPT = REPO_ROOT / "experiments" / "eval" / "run.py"


@dataclass(frozen=True)
class Combo:
    require_isolated: bool
    alignment_min_cos: float | None
    delta_frac: float
    stability_eta: float

    @property
    def alignment_label(self) -> str:
        return "none" if self.alignment_min_cos is None else f"{self.alignment_min_cos:.2f}"

    @property
    def slug(self) -> str:
        iso = "iso1" if self.require_isolated else "iso0"
        align = f"align{self.alignment_label.replace('.', 'p')}"
        delta = f"delta{self.delta_frac:.2f}".replace(".", "p")
        eta = f"eta{self.stability_eta:.1f}".replace(".", "p")
        return f"{iso}_{align}_{delta}_{eta}"


def _parse_bool_grid(raw: str) -> list[bool]:
    mapping = {"true": True, "1": True, "false": False, "0": False}
    values: list[bool] = []
    for token in raw.split(","):
        tok = token.strip().lower()
        if not tok:
            continue
        if tok not in mapping:
            raise ValueError(f"invalid boolean grid token '{token}'")
        values.append(mapping[tok])
    if not values:
        raise ValueError("require_isolated grid must contain at least one value")
    return values


def _parse_alignment_grid(raw: str) -> list[float | None]:
    values: list[float | None] = []
    for token in raw.split(","):
        tok = token.strip().lower()
        if not tok:
            continue
        if tok in {"none", "nan"}:
            values.append(None)
            continue
        try:
            values.append(float(tok))
        except ValueError as exc:  # pragma: no cover - invalid input guard
            raise ValueError(f"invalid alignment cosine '{token}'") from exc
    if not values:
        raise ValueError("alignment grid must contain at least one entry")
    return values


def _parse_float_grid(raw: str, name: str) -> list[float]:
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
        raise ValueError(f"{name} grid must contain at least one entry")
    return values


def _ensure_matplotlib() -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for sensitivity heatmaps; install it or run within plotting-enabled env"
        )


def _thread_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    thread_env_vars = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]
    for var in thread_env_vars:
        env[var] = "1"
    current_path = env.get("PYTHONPATH")
    src_path = str(SRC_ROOT)
    if current_path:
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{current_path}"
    else:
        env["PYTHONPATH"] = src_path
    return env


def _run_evaluation(command: Sequence[str], env: dict[str, str]) -> None:
    subprocess.run(command, check=True, env=env)


def _build_command(base_args: argparse.Namespace, combo: Combo, run_dir: Path) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--returns-csv",
        str(base_args.returns_csv),
        "--window",
        str(base_args.window),
        "--horizon",
        str(base_args.horizon),
        "--start",
        base_args.slice_start,
        "--end",
        base_args.slice_end,
        "--out",
        str(run_dir),
        "--assets-top",
        str(base_args.assets_top),
        "--exec-mode",
        "deterministic",
        "--mv-turnover-bps",
        str(base_args.mv_turnover_bps),
        "--mv-condition-cap",
        str(base_args.mv_condition_cap),
        "--use-factor-prewhiten",
        "1" if base_args.use_factor_prewhiten else "0",
        "--gate-stability-min",
        f"{combo.stability_eta:.3f}",
        "--overlay-delta-frac",
        f"{combo.delta_frac:.4f}",
        "--group-design",
        base_args.group_design,
    ]
    if base_args.workers is not None:
        cmd.extend(["--workers", str(base_args.workers)])
    if base_args.config:
        cmd.extend(["--config", str(base_args.config)])
    if base_args.thresholds:
        cmd.extend(["--thresholds", str(base_args.thresholds)])
    if base_args.factors_csv:
        cmd.extend(["--factors-csv", str(base_args.factors_csv)])
    if combo.require_isolated:
        cmd.append("--require-isolated")
    else:
        cmd.append("--allow-non-isolated")
    if combo.alignment_min_cos is not None:
        cmd.extend(["--angle-min-cos", f"{combo.alignment_min_cos:.3f}"])
    if base_args.q_max is not None:
        cmd.extend(["--q-max", str(base_args.q_max)])
    return cmd


def _load_first_row(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"diagnostics summary at '{path}' is empty")
    return df.iloc[0]


def _changed_window_ids(detail_source: Path | pd.DataFrame) -> list[int]:
    if isinstance(detail_source, Path):
        detail_df = pd.read_csv(detail_source)
    else:
        detail_df = detail_source
    if detail_df.empty:
        return []
    regime_series = detail_df["regime"].astype(str).str.lower()
    full_rows = detail_df[regime_series == "full"]
    if full_rows.empty:
        return []
    mask = full_rows.get("changed_flag", pd.Series(dtype=float)).fillna(0).astype(int) == 1
    if not mask.any():
        return []
    ids = pd.to_numeric(full_rows.loc[mask, "window_id"], errors="coerce").dropna().astype(int)
    return ids.tolist()


def _mean_delta_sq_error(metrics_path: Path, changed_ids: set[int], portfolio: str) -> float:
    if not metrics_path.exists():
        return float("nan")
    df = pd.read_csv(metrics_path)
    if df.empty:
        return float("nan")
    subset = df[(df["regime"] == "full") & (df["portfolio"] == portfolio)]
    if subset.empty:
        return float("nan")
    pivot = subset.pivot_table(index="window_id", columns="estimator", values="sq_error", aggfunc="first")
    if "overlay" not in pivot.columns or "baseline" not in pivot.columns:
        return float("nan")
    valid_ids = [idx for idx in pivot.index if idx in changed_ids]
    if not valid_ids:
        return float("nan")
    diffs = pivot.loc[valid_ids, "overlay"] - pivot.loc[valid_ids, "baseline"]
    diffs = diffs.replace([np.inf, -np.inf], np.nan).dropna()
    if diffs.empty:
        return float("nan")
    return float(diffs.mean())


def _dm_stats_from_metrics(
    metrics_path: Path,
    changed_ids: set[int],
    portfolio: str,
) -> tuple[float, float, int]:
    if not metrics_path.exists() or not changed_ids:
        return float("nan"), float("nan"), 0
    df = pd.read_csv(metrics_path)
    if df.empty:
        return float("nan"), float("nan"), 0
    subset = df[(df["regime"] == "full") & (df["portfolio"] == portfolio)]
    if subset.empty:
        return float("nan"), float("nan"), 0
    pivot = subset.pivot_table(index="window_id", columns="estimator", values="sq_error", aggfunc="first")
    if pivot.empty or "overlay" not in pivot.columns or "baseline" not in pivot.columns:
        return float("nan"), float("nan"), 0
    valid_ids = pivot.index.isin(list(changed_ids))
    filtered = pivot[valid_ids]
    if filtered.empty:
        return float("nan"), float("nan"), 0
    overlay_errors = pd.to_numeric(filtered["overlay"], errors="coerce").to_numpy(dtype=np.float64)
    baseline_errors = pd.to_numeric(filtered["baseline"], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(overlay_errors) & np.isfinite(baseline_errors)
    n_eff = int(mask.sum())
    if n_eff <= 1:
        return float("nan"), float("nan"), 0
    dm_stat, p_value = dm_test(overlay_errors[mask], baseline_errors[mask])
    if not np.isfinite(dm_stat):
        return float("nan"), float("nan"), 0
    return dm_stat, p_value, n_eff


def _plot_heatmap(
    subset: pd.DataFrame,
    delta_values: Sequence[float],
    stability_values: Sequence[float],
    metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> Path | None:
    if subset.empty:
        return None
    pivot = subset.pivot_table(index="stability_eta", columns="delta_frac", values=metric, aggfunc="first")
    pivot = pivot.reindex(index=stability_values, columns=delta_values)
    data = pivot.to_numpy(dtype=np.float64)
    if np.all(~np.isfinite(data)):
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(delta_values)))
    ax.set_xticklabels([f"{val:.2f}" for val in delta_values])
    ax.set_yticks(np.arange(len(stability_values)))
    ax.set_yticklabels([f"{val:.2f}" for val in stability_values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i, eta in enumerate(stability_values):
        for j, delta in enumerate(delta_values):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(metric, rotation=270, labelpad=15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _full_regime(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty or "regime" not in detail_df.columns:
        return detail_df
    mask = detail_df["regime"].astype(str).str.lower() == "full"
    subset = detail_df[mask]
    if subset.empty:
        return detail_df
    return subset


def _metric_series(
    detail_df: pd.DataFrame,
    column: str,
    changed_ids: set[int] | None,
) -> pd.Series:
    frame = _full_regime(detail_df)
    if frame.empty or column not in frame.columns:
        return pd.Series(dtype=float)
    filtered = frame
    if changed_ids and "window_id" in frame.columns:
        window_ids = pd.to_numeric(frame["window_id"], errors="coerce")
        mask = window_ids.isin(list(changed_ids))
        filtered = frame[mask]
        if filtered.empty:
            filtered = frame
    series = pd.to_numeric(filtered[column], errors="coerce").dropna()
    return series


def _median_metric(detail_df: pd.DataFrame, column: str, changed_ids: set[int]) -> float:
    series = _metric_series(detail_df, column, changed_ids if changed_ids else None)
    if series.empty:
        return float("nan")
    return float(series.median())


def _plot_metric_histograms(
    metric_map: dict[tuple[bool, str], list[float]],
    *,
    figures_dir: Path,
    slug: str,
    xlabel: str,
    title_prefix: str,
) -> list[str]:
    if plt is None or not metric_map:
        return []
    paths: list[str] = []
    for (ri, align_label), values in sorted(
        metric_map.items(), key=lambda item: (int(not item[0][0]), item[0][1])
    ):
        if not values:
            continue
        data = np.asarray(values, dtype=np.float64)
        data = data[np.isfinite(data)]
        if data.size == 0:
            continue
        bin_count = int(max(10, min(40, round(np.sqrt(data.size)))))
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.hist(data, bins=bin_count, color="#1f77b4", alpha=0.85)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        label = align_label.replace(".", "p")
        iso_label = "on" if ri else "off"
        ax.set_title(f"{title_prefix} (iso={iso_label}, align={align_label})")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = figures_dir / f"{slug}_ri{'1' if ri else '0'}_align{label}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path.as_posix())
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-data sensitivity sweep for overlay detector")
    parser.add_argument("--returns-csv", type=Path, required=True)
    parser.add_argument("--factors-csv", type=Path, default=None)
    parser.add_argument("--slice-start", type=str, required=True)
    parser.add_argument("--slice-end", type=str, required=True)
    parser.add_argument("--assets-top", type=int, default=150)
    parser.add_argument("--window", type=int, default=126)
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=Path("data/registry.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/rc-sensitivity"))
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--group-design", type=str, default="week")
    parser.add_argument("--mv-turnover-bps", type=float, default=5.0)
    parser.add_argument("--mv-condition-cap", type=float, default=1e6)
    parser.add_argument("--q-max", type=int, default=1)
    parser.add_argument("--use-factor-prewhiten", type=int, choices=[0, 1], default=1)
    parser.add_argument("--require-isolated-grid", type=str, default="true,false")
    parser.add_argument("--alignment-cos-grid", type=str, default="none,0.7,0.8,0.9")
    parser.add_argument("--delta-frac-grid", type=str, default="0.0,0.01,0.02")
    parser.add_argument("--stability-eta-grid", type=str, default="0.3,0.4,0.5")
    args = parser.parse_args(argv)
    args.require_isolated_values = _parse_bool_grid(args.require_isolated_grid)
    args.alignment_values = _parse_alignment_grid(args.alignment_cos_grid)
    args.delta_frac_values = _parse_float_grid(args.delta_frac_grid, "delta_frac")
    args.stability_eta_values = _parse_float_grid(args.stability_eta_grid, "stability_eta")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if plt is None:
        _ensure_matplotlib()
    returns_csv = args.returns_csv.resolve()
    if not returns_csv.exists():
        raise FileNotFoundError(f"returns CSV '{returns_csv}' not found")
    verify_dataset(returns_csv, args.registry.resolve())
    if args.factors_csv is not None and not args.factors_csv.exists():
        raise FileNotFoundError(f"factors CSV '{args.factors_csv}' not found")

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    label = args.label or f"rc-sensitivity-{timestamp}"
    out_dir = args.out.resolve() / label
    runs_dir = out_dir / "runs"
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    runs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    combos: list[Combo] = [
        Combo(ri, align, delta, eta)
        for ri in args.require_isolated_values
        for align in args.alignment_values
        for delta in args.delta_frac_values
        for eta in args.stability_eta_values
    ]

    env = _thread_env()
    results: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []
    alignment_hist_data: dict[tuple[bool, str], list[float]] = defaultdict(list)
    leakage_hist_data: dict[tuple[bool, str], list[float]] = defaultdict(list)

    for combo in combos:
        run_dir = runs_dir / combo.slug
        run_dir.mkdir(parents=True, exist_ok=True)
        run_json = run_dir / "run.json"
        should_run = True
        if args.skip_existing and run_json.exists():
            should_run = False
        if should_run:
            cmd = _build_command(args, combo, run_dir)
            print(f"[sensitivity] Running {' '.join(cmd)}", flush=True)
            _run_evaluation(cmd, env)
        if not run_json.exists():
            raise RuntimeError(f"expected '{run_json}' to exist after evaluation")

        diag_path = run_dir / "full" / "diagnostics.csv"
        if not diag_path.exists():
            raise RuntimeError(f"missing diagnostics at '{diag_path}'")
        diag_row = _load_first_row(diag_path)
        acceptance_rate = float(diag_row.get("acceptance_rate", float("nan")))
        leakage = float(diag_row.get("leakage_offcomp", float("nan")))
        percent_changed = float(diag_row.get("percent_changed", 0.0))

        detail_root = run_dir / "diagnostics_detail.csv"
        if not detail_root.exists():
            raise RuntimeError(f"missing diagnostics detail at '{detail_root}'")
        detail_df = pd.read_csv(detail_root)
        changed_ids = set(_changed_window_ids(detail_df))
        n_changed = len(changed_ids)
        edge_margin_median = _median_metric(detail_df, "mp_edge_margin", changed_ids)

        alignment_series = _metric_series(detail_df, "alignment_cos", changed_ids)
        if not alignment_series.empty:
            alignment_hist_data[(combo.require_isolated, combo.alignment_label)].extend(
                alignment_series.tolist()
            )
        leakage_series = _metric_series(detail_df, "leakage_offcomp", changed_ids)
        if not leakage_series.empty:
            leakage_hist_data[(combo.require_isolated, combo.alignment_label)].extend(
                leakage_series.tolist()
            )

        edge_margin_mean = float(diag_row.get("mp_edge_margin", diag_row.get("edge_margin_mean", float("nan"))))
        if not np.isfinite(edge_margin_median):
            edge_margin_median = edge_margin_mean

        metrics_detail_path = run_dir / "metrics_detail.csv"
        delta_mse_ew = _mean_delta_sq_error(metrics_detail_path, changed_ids, "ew")
        delta_mse_mv = _mean_delta_sq_error(metrics_detail_path, changed_ids, "mv")

        dm_stat_ew, dm_p_ew, dm_n_ew = _dm_stats_from_metrics(metrics_detail_path, changed_ids, "ew")
        dm_stat_mv, dm_p_mv, dm_n_mv = _dm_stats_from_metrics(metrics_detail_path, changed_ids, "mv")

        result_record = {
            "require_isolated": combo.require_isolated,
            "alignment_min_cos": combo.alignment_min_cos,
            "alignment_label": combo.alignment_label,
            "delta_frac": combo.delta_frac,
            "stability_eta": combo.stability_eta,
            "acceptance_rate": acceptance_rate,
            "mp_edge_margin_mean": edge_margin_mean,
            "mp_edge_margin_median": edge_margin_median,
            "leakage_offcomp": leakage,
            "percent_changed": percent_changed,
            "n_changed_windows": n_changed,
            "delta_mse_ew": delta_mse_ew,
            "delta_mse_mv": delta_mse_mv,
            "dm_stat_ew": dm_stat_ew,
            "dm_p_ew": dm_p_ew,
            "dm_n_ew": dm_n_ew,
            "dm_stat_mv": dm_stat_mv,
            "dm_p_mv": dm_p_mv,
            "dm_n_mv": dm_n_mv,
            "run_dir": run_dir.as_posix(),
        }
        results.append(result_record)
        run_manifest.append(
            {
                "combo": {
                    "require_isolated": combo.require_isolated,
                    "alignment_min_cos": combo.alignment_min_cos,
                    "delta_frac": combo.delta_frac,
                    "stability_eta": combo.stability_eta,
                },
                "paths": {
                    "run_dir": run_dir.as_posix(),
                    "diagnostics": diag_path.as_posix(),
                    "diagnostics_detail": detail_root.as_posix(),
                    "metrics_detail": metrics_detail_path.as_posix(),
                    "run_json": run_json.as_posix(),
                },
            }
        )

    results_df = pd.DataFrame(results)
    summary_path = tables_dir / "sensitivity_summary.csv"
    results_df.to_csv(summary_path, index=False)

    changed_table = results_df[
        [
            "require_isolated",
            "alignment_label",
            "delta_frac",
            "stability_eta",
            "n_changed_windows",
            "percent_changed",
        ]
    ].copy()
    changed_table.to_csv(tables_dir / "changed_windows.csv", index=False)

    heatmap_metrics = {
        "acceptance_rate": "Acceptance Rate",
        "mp_edge_margin_median": "Median MP Edge Margin",
    }
    heatmap_paths: list[str] = []
    _ensure_matplotlib()
    for metric, title in heatmap_metrics.items():
        for ri in args.require_isolated_values:
            for align in args.alignment_values:
                subset = results_df[
                    (results_df["require_isolated"] == ri)
                    & (results_df["alignment_min_cos"] == align)
                ]
                label = "none" if align is None else f"{float(align):.2f}"
                fig_path = (
                    figures_dir
                    / f"{metric}_ri{'1' if ri else '0'}_align{label.replace('.', 'p')}.png"
                )
                plotted = _plot_heatmap(
                    subset,
                    args.delta_frac_values,
                    args.stability_eta_values,
                    metric,
                    f"{title} (iso={'on' if ri else 'off'}, align={label})",
                    "delta_frac",
                    "stability_eta",
                    fig_path,
                )
                if plotted is not None:
                    heatmap_paths.append(plotted.as_posix())

    alignment_hist_paths = _plot_metric_histograms(
        alignment_hist_data,
        figures_dir=figures_dir,
        slug="alignment_hist",
        xlabel="Alignment cos",
        title_prefix="Alignment cos distribution",
    )
    leakage_hist_paths = _plot_metric_histograms(
        leakage_hist_data,
        figures_dir=figures_dir,
        slug="leakage_hist",
        xlabel="Leakage (off-comp)",
        title_prefix="Leakage distribution",
    )

    manifest_path = out_dir / "run_manifest.json"
    manifest_payload = {
        "label": label,
        "created_at": timestamp,
        "returns_csv": returns_csv.as_posix(),
        "factors_csv": args.factors_csv.as_posix() if args.factors_csv else None,
        "slice": {
            "start": args.slice_start,
            "end": args.slice_end,
            "assets_top": args.assets_top,
            "window": args.window,
            "horizon": args.horizon,
        },
        "grid": {
            "require_isolated": args.require_isolated_values,
            "alignment_min_cos": args.alignment_values,
            "delta_frac": args.delta_frac_values,
            "stability_eta": args.stability_eta_values,
        },
        "runs": run_manifest,
        "summary_csv": summary_path.as_posix(),
        "heatmaps": heatmap_paths,
        "alignment_histograms": alignment_hist_paths,
        "leakage_histograms": leakage_hist_paths,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    print(f"[sensitivity] Completed sweep with {len(results_df)} combinations. Summary: {summary_path}")


if __name__ == "__main__":
    main()
