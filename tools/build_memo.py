#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from report.gather import collect_estimator_panel, find_runs, load_run

EW_STRATEGIES = ("Equal Weight",)
MV_STRATEGIES = ("Min-Variance (box)", "Min-Variance (long-only)", "Min-Variance")
REJECTION_REASONS = ["edge_buffer", "off_component_ratio", "stability_fail", "energy_floor", "neg_mu"]


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _discover_run_paths(entries: Iterable[dict]) -> list[Path]:
    run_paths: list[Path] = []
    for entry in entries:
        if not entry:
            continue
        if "path" in entry:
            candidate = Path(entry["path"]).resolve()
            if candidate.is_dir():
                run_paths.append(candidate)
            continue
        root = entry.get("root")
        if not root:
            continue
        pattern = entry.get("pattern")
        try:
            run_paths.extend(find_runs(root, pattern=pattern))
        except FileNotFoundError:
            print(f"[build_memo] Skipping missing root '{root}'", file=sys.stderr)
            continue
    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in run_paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _latest_summary_dir(root: Path = Path("reports")) -> Optional[Path]:
    candidates = sorted(
        (p for p in root.glob("rc-*") if (p / "summary").is_dir()),
        key=lambda path: path.name,
    )
    if not candidates:
        return None
    return candidates[-1] / "summary"


def _load_summary_artifacts(config: dict) -> dict[str, Any]:
    summary_cfg = config.get("summary", {}) or {}

    summary_dir_cfg = summary_cfg.get("path")
    summary_dir = Path(summary_dir_cfg).resolve() if summary_dir_cfg else _latest_summary_dir()
    if summary_dir is not None and not summary_dir.is_dir():
        summary_dir = None

    def _resolve_path(key: str, default_name: str | None) -> Optional[Path]:
        candidate = summary_cfg.get(key)
        if candidate:
            path = Path(candidate).resolve()
            return path if path.exists() else None
        if default_name is None or summary_dir is None:
            return None
        path = summary_dir / default_name
        return path if path.exists() else None

    perf_csv = _resolve_path("perf_csv", "summary_perf.csv")
    detection_csv = _resolve_path("detection_csv", "summary_detection.csv")
    kill_json = _resolve_path("kill_criteria", "kill_criteria.json")
    limitations_md = _resolve_path("limitations", "limitations.md")

    ablation_candidate = summary_cfg.get("ablation_csv")
    if ablation_candidate:
        ablation_path = Path(ablation_candidate).resolve()
    else:
        ablation_path = Path("ablations/ablation_matrix.csv").resolve()
    if not ablation_path.exists():
        ablation_path = None

    def _read_csv(path: Optional[Path]) -> pd.DataFrame:
        if path is None:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    perf_df = _read_csv(perf_csv)
    det_df = _read_csv(detection_csv)
    ablation_df = _read_csv(ablation_path)

    if kill_json is not None:
        try:
            kill_payload = json.loads(kill_json.read_text(encoding="utf-8"))
        except Exception:
            kill_payload = None
    else:
        kill_payload = None

    if limitations_md is not None:
        try:
            limitations_text = limitations_md.read_text(encoding="utf-8").strip()
        except Exception:
            limitations_text = ""
    else:
        limitations_text = ""

    return {
        "summary_dir": summary_dir,
        "perf_df": perf_df,
        "det_df": det_df,
        "kill_data": kill_payload,
        "limitations_text": limitations_text,
        "ablation_matrix": ablation_df,
    }


def _format_kill_criteria_payload(kill_data: Any) -> tuple[str, str]:
    if not isinstance(kill_data, dict):
        return "(kill criteria unavailable)", "UNKNOWN"
    criteria = kill_data.get("criteria", [])
    lines: list[str] = []
    status_flags: list[Optional[bool]] = []
    for crit in criteria:
        if not isinstance(crit, dict):
            continue
        status = crit.get("pass")
        status_flags.append(status)
        label = crit.get("label") or crit.get("key", "criterion")
        raw_val = crit.get("value")
        if isinstance(raw_val, (int, float)) and not np.isnan(raw_val):
            value_str = f"{raw_val:.3g}"
        elif raw_val is None:
            value_str = "n/a"
        else:
            value_str = str(raw_val)
        threshold = crit.get("threshold")
        if isinstance(threshold, dict):
            threshold_str = json.dumps(threshold, sort_keys=True)
        else:
            threshold_str = str(threshold)
        tag = "PASS" if status is True else "FAIL" if status is False else "N/A"
        lines.append(f"- [{tag}] {label} (value: {value_str}, threshold: {threshold_str})")
    if not lines:
        lines = ["- (no criteria evaluated)"]
    flagged = [flag for flag in status_flags if flag is not None]
    if flagged and all(flagged):
        status = "PASS"
    elif flagged and not all(flagged):
        status = "CHECK"
    else:
        status = "UNKNOWN"
    return "\n".join(lines), status


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no data)"
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            val = row[col]
            if isinstance(val, (float, int)) and not pd.isna(val):
                values.append(f"{val:.4g}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_percent(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _pick_strategy_row(group: pd.DataFrame, strategies: Iterable[str]) -> pd.Series | None:
    for strategy in strategies:
        exact = group[group["strategy"] == strategy]
        if not exact.empty:
            return exact.iloc[0]
        prefixed = group[group["strategy"].str.startswith(strategy, na=False)]
        if not prefixed.empty:
            return prefixed.iloc[0]
    return None


def _format_delta(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    abs_val = abs(value)
    if abs_val >= 1e-2:
        return f"{value:.3f}"
    if abs_val >= 1e-4:
        return f"{value:.4f}"
    return f"{value:.2e}"


def _format_ci(lo: float, hi: float) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return "n/a"
    return f"[{_format_delta(lo)}, {_format_delta(hi)}]"


def _format_edge_metric(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if abs(value) >= 1e-2:
        return f"{value:.3f}"
    return f"{value:.2e}"


def _format_pvalue(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if value >= 0.001:
        return f"{value:.3f}"
    return f"{value:.1e}"


def _format_detection(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_windows(value: float | int) -> str:
    if pd.isna(value):
        return "n/a"
    return str(int(round(float(value))))


def _prettify_reason(reason: str) -> str:
    return reason.replace("_", " ")


def _collect_rejection_records(summary_df: pd.DataFrame, run_tag: str) -> list[dict[str, float]]:
    if summary_df.empty:
        return []
    records: list[dict[str, float]] = []
    for reason in REJECTION_REASONS:
        column = f"rejection_stats.{reason}"
        value = float(summary_df[column].iloc[0]) if column in summary_df else 0.0
        records.append({"run": run_tag, "rejection_reason": reason, "count": value})
    return records


def _build_key_tables(
    panel_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if panel_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    records: list[dict[str, float | str]] = []
    for (run_tag, estimator), group in panel_df.groupby(["run", "estimator"], sort=False):
        if estimator == "De-aliased":
            continue
        ew_row = _pick_strategy_row(group, EW_STRATEGIES)
        mv_row = _pick_strategy_row(group, MV_STRATEGIES)
        if ew_row is None and mv_row is None:
            continue

        detection_series = group["detection_rate"].dropna() if "detection_rate" in group else pd.Series(dtype=float)
        crisis_series = group["crisis_label"].dropna() if "crisis_label" in group else pd.Series(dtype=object)
        edge_med_series = group["edge_margin_median"].dropna() if "edge_margin_median" in group else pd.Series(dtype=float)
        edge_iqr_series = group["edge_margin_iqr"].dropna() if "edge_margin_iqr" in group else pd.Series(dtype=float)
        design_series = group["design"].dropna() if "design" in group else pd.Series(dtype=object)
        windows_series = (
            group["rolling_windows_evaluated"].dropna()
            if "rolling_windows_evaluated" in group
            else pd.Series(dtype=float)
        )
        substitution_series = (
            group["substitution_fraction"].dropna()
            if "substitution_fraction" in group
            else pd.Series(dtype=float)
        )
        no_iso_series = (
            group["skip_no_isolated_share"].dropna()
            if "skip_no_isolated_share" in group
            else pd.Series(dtype=float)
        )
        edge_mode_series = group["edge_mode"].dropna() if "edge_mode" in group else pd.Series(dtype=object)
        edge_mode_value = str(edge_mode_series.iloc[0]) if not edge_mode_series.empty else ""
        gating_mode_series = group["gating_mode"].dropna() if "gating_mode" in group else pd.Series(dtype=object)
        gating_mode_value = str(gating_mode_series.iloc[0]) if not gating_mode_series.empty else ""
        delta_frac_med_series = (
            group["delta_frac_used_median"].dropna()
            if "delta_frac_used_median" in group
            else pd.Series(dtype=float)
        )
        delta_frac_used_median = float(delta_frac_med_series.iloc[0]) if not delta_frac_med_series.empty else float("nan")
        var_kupiec_series = group["var_kupiec_p"].dropna() if "var_kupiec_p" in group else pd.Series(dtype=float)
        var_kupiec_value = float(var_kupiec_series.iloc[0]) if not var_kupiec_series.empty else float("nan")
        var_indep_series = group["var_independence_p"].dropna() if "var_independence_p" in group else pd.Series(dtype=float)
        var_indep_value = float(var_indep_series.iloc[0]) if not var_indep_series.empty else float("nan")
        es_series = group["es_shortfall_p"].dropna() if "es_shortfall_p" in group else pd.Series(dtype=float)
        es_value = float(es_series.iloc[0]) if not es_series.empty else float("nan")

        record: dict[str, float | str] = {
            "run": run_tag,
            "estimator": estimator,
            "crisis_label": str(crisis_series.iloc[0]) if not crisis_series.empty else "",
            "detection_rate": float(detection_series.iloc[0]) if not detection_series.empty else float("nan"),
            "design": str(design_series.iloc[0]) if not design_series.empty else "",
            "delta_mse_ew": float("nan"),
            "ci_lo_ew": float("nan"),
            "ci_hi_ew": float("nan"),
            "dm_p_ew": float("nan"),
            "dm_p_ew_qlike": float("nan"),
            "delta_mse_mv": float("nan"),
            "ci_lo_mv": float("nan"),
            "ci_hi_mv": float("nan"),
            "dm_p_mv": float("nan"),
            "dm_p_mv_qlike": float("nan"),
            "edge_margin_median": float(edge_med_series.iloc[0]) if not edge_med_series.empty else float("nan"),
            "edge_margin_iqr": float(edge_iqr_series.iloc[0]) if not edge_iqr_series.empty else float("nan"),
            "n_windows": float("nan"),
            "rolling_windows_evaluated": float(windows_series.iloc[0]) if not windows_series.empty else float("nan"),
            "mean_qlike": float("nan"),
            "substitution_fraction": float(substitution_series.iloc[0]) if not substitution_series.empty else float("nan"),
            "skip_no_isolated_share": float(no_iso_series.iloc[0]) if not no_iso_series.empty else float("nan"),
            "edge_mode": edge_mode_value,
            "gating_mode": gating_mode_value,
            "delta_frac_used_median": delta_frac_used_median,
            "var_kupiec_p": var_kupiec_value,
            "var_independence_p": var_indep_value,
            "es_shortfall_p": es_value,
        }

        if ew_row is not None:
            record["delta_mse_ew"] = float(ew_row.get("delta_mse_vs_de", float("nan")))
            record["ci_lo_ew"] = float(ew_row.get("ci_lo", float("nan")))
            record["ci_hi_ew"] = float(ew_row.get("ci_hi", float("nan")))
            record["dm_p_ew"] = float(ew_row.get("dm_p", float("nan")))
            record["dm_p_ew_qlike"] = float(ew_row.get("dm_p_qlike", float("nan")))
            record["n_windows"] = float(ew_row.get("n_windows", float("nan")))
            record["mean_qlike"] = float(ew_row.get("mean_qlike", float("nan")))
        if mv_row is not None:
            record["delta_mse_mv"] = float(mv_row.get("delta_mse_vs_de", float("nan")))
            record["ci_lo_mv"] = float(mv_row.get("ci_lo", float("nan")))
            record["ci_hi_mv"] = float(mv_row.get("ci_hi", float("nan")))
            record["dm_p_mv"] = float(mv_row.get("dm_p", float("nan")))
            record["dm_p_mv_qlike"] = float(mv_row.get("dm_p_qlike", float("nan")))
            if pd.isna(record["n_windows"]):
                record["n_windows"] = float(mv_row.get("n_windows", float("nan")))
            if pd.isna(record["mean_qlike"]):
                record["mean_qlike"] = float(mv_row.get("mean_qlike", float("nan")))

        records.append(record)

    if not records:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    numeric_df = pd.DataFrame(records).sort_values(["run", "estimator"]).reset_index(drop=True)

    display_rows: list[dict[str, str]] = []
    for _, row in numeric_df.iterrows():
        detection_display = _format_detection(row["detection_rate"])
        design_value = str(row.get("design", "")).lower()
        windows_val = row.get("rolling_windows_evaluated", float("nan"))
        skip_share_val = float(row.get("skip_no_isolated_share", float("nan")))
        if (
            design_value == "nested"
            and not pd.isna(row.get("detection_rate"))
            and float(row.get("detection_rate", 0.0)) == 0.0
            and not pd.isna(windows_val)
            and float(windows_val) > 0.0
            and detection_display != "n/a"
        ):
            detection_display += " (no accepted detections; check guardrails)"
        if not pd.isna(skip_share_val) and skip_share_val >= 0.2:
            detection_display += f" ⚠ gate-no_iso {skip_share_val * 100:.0f}%"
        display_rows.append(
            {
                "run": row["run"],
                "crisis_label": row["crisis_label"] or "n/a",
                "estimator": row["estimator"],
                "edge|gate": f"{row.get('edge_mode', '') or 'n/a'} | {row.get('gating_mode', '') or 'n/a'}",
                "detection_rate": detection_display,
                "delta_mse_ew": _format_delta(row["delta_mse_ew"]),
                "CI_EW": _format_ci(row["ci_lo_ew"], row["ci_hi_ew"]),
                "DM_p_EW": _format_pvalue(row["dm_p_ew"]),
                "DM_p_EW_QLIKE": _format_pvalue(row.get("dm_p_ew_qlike", float("nan"))),
                "delta_mse_mv": _format_delta(row["delta_mse_mv"]),
                "CI_MV": _format_ci(row["ci_lo_mv"], row["ci_hi_mv"]),
                "DM_p_MV": _format_pvalue(row["dm_p_mv"]),
                "DM_p_MV_QLIKE": _format_pvalue(row.get("dm_p_mv_qlike", float("nan"))),
                "edge_margin_median": _format_edge_metric(row["edge_margin_median"]),
                "edge_margin_IQR": _format_edge_metric(row["edge_margin_iqr"]),
                "mean_qlike": _format_edge_metric(row.get("mean_qlike", float("nan"))),
                "substitution_fraction": _format_detection(row.get("substitution_fraction", float("nan"))),
                "no_iso_skip_share": _format_percent(skip_share_val),
                "delta_frac_used": _format_edge_metric(row.get("delta_frac_used_median", float("nan"))),
                "VaR_pof": _format_pvalue(row.get("var_kupiec_p", float("nan"))),
                "VaR_indep": _format_pvalue(row.get("var_independence_p", float("nan"))),
                "ES_p": _format_pvalue(row.get("es_shortfall_p", float("nan"))),
                "n_windows": _format_windows(row["n_windows"]),
            }
        )

    display_df = pd.DataFrame(
        display_rows,
        columns=[
            "run",
            "crisis_label",
            "estimator",
            "edge|gate",
            "detection_rate",
            "delta_mse_ew",
            "CI_EW",
            "DM_p_EW",
            "DM_p_EW_QLIKE",
            "delta_mse_mv",
            "CI_MV",
            "DM_p_MV",
            "DM_p_MV_QLIKE",
            "edge_margin_median",
            "edge_margin_IQR",
            "mean_qlike",
            "substitution_fraction",
            "no_iso_skip_share",
            "delta_frac_used",
            "VaR_pof",
            "VaR_indep",
            "ES_p",
            "n_windows",
        ],
    )

    qlike_display_rows: list[dict[str, str]] = []
    for _, row in numeric_df.iterrows():
        qlike_display_rows.append(
            {
                "run": row["run"],
                "crisis_label": row["crisis_label"] or "n/a",
                "estimator": row["estimator"],
                "DM_p_EW_QLIKE": _format_pvalue(row["dm_p_ew_qlike"]),
                "DM_p_MV_QLIKE": _format_pvalue(row["dm_p_mv_qlike"]),
                "mean_qlike": _format_edge_metric(row["mean_qlike"]),
                "substitution_fraction": _format_detection(row.get("substitution_fraction", float("nan"))),
                "no_iso_skip_share": _format_percent(row.get("skip_no_isolated_share", float("nan"))),
            }
        )

    qlike_display_df = pd.DataFrame(
        qlike_display_rows,
        columns=[
            "run",
            "crisis_label",
            "estimator",
            "DM_p_EW_QLIKE",
            "DM_p_MV_QLIKE",
            "mean_qlike",
            "substitution_fraction",
            "no_iso_skip_share",
        ],
    )

    qlike_numeric_df = numeric_df[
        [
            "run",
            "estimator",
            "dm_p_ew_qlike",
            "dm_p_mv_qlike",
            "mean_qlike",
            "substitution_fraction",
            "skip_no_isolated_share",
        ]
    ].copy()

    return numeric_df, display_df, qlike_numeric_df, qlike_display_df


def _build_rejection_tables(rejection_records: list[dict[str, float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not rejection_records:
        return pd.DataFrame(), pd.DataFrame()

    raw_df = pd.DataFrame(rejection_records)
    pivot = (
        raw_df.pivot_table(index="run", columns="rejection_reason", values="count", aggfunc="sum", fill_value=0)
        .reset_index()
    )

    for reason in REJECTION_REASONS:
        if reason not in pivot.columns:
            pivot[reason] = 0.0
    pivot = pivot[["run", *REJECTION_REASONS]]

    totals = pivot[REJECTION_REASONS].sum(axis=1)
    percent_df = pivot.copy()
    for reason in REJECTION_REASONS:
        shares = percent_df[reason].astype(float)
        percent_df[reason] = shares.divide(totals.where(totals > 0, 1.0)).where(totals > 0, 0.0) * 100.0

    markdown_df = percent_df.copy()
    for reason in REJECTION_REASONS:
        markdown_df[reason] = markdown_df[reason].apply(lambda val: f"{val:.1f}%")

    return percent_df, markdown_df


def build_memo(config_path: Path) -> Path:
    config = _load_config(config_path)
    gallery_cfg = config.get("gallery", {}) or {}
    gallery_name = gallery_cfg.get("name", "memo")
    run_entries = config.get("runs", []) or []
    run_paths = _discover_run_paths(run_entries)
    if not run_paths:
        raise ValueError("No runs discovered for memo generation.")

    gallery_root = Path("figures") / gallery_name
    gallery_root.mkdir(parents=True, exist_ok=True)
    summary_plots_dir = gallery_root / "summary"
    summary_plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    panel_frames: list[pd.DataFrame] = []
    run_lines: list[str] = []
    artifact_map: dict[str, list[str]] = defaultdict(list)
    rejection_records: list[dict[str, float]] = []
    ablation_frames: list[pd.DataFrame] = []
    ablation_figures: list[tuple[str, Path]] = []
    alignment_medians: dict[str, float] = {}
    no_iso_shares: dict[str, float] = {}
    nested_scope_notes: list[str] = []
    detail_frames: list[pd.DataFrame] = []
    reason_records: list[dict[str, object]] = []

    for run_path in run_paths:
        frames = load_run(run_path)
        panel_df = collect_estimator_panel([run_path])
        if not panel_df.empty:
            panel_frames.append(panel_df.assign(run=run_path.name))
        summary_df = frames["summary"]
        run_meta_path = run_path / "run_meta.json"
        run_meta = {}
        if run_meta_path.exists():
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        summary_json_path = run_path / "summary.json"
        summary_json: dict[str, object] = {}
        if summary_json_path.exists():
            summary_json = json.loads(summary_json_path.read_text(encoding="utf-8"))
        detection_df = frames.get("detections", pd.DataFrame())
        detail_df = frames.get("diagnostics_detail", pd.DataFrame())
        if not detail_df.empty:
            detail_with_run = detail_df.copy()
            detail_with_run["run"] = run_path.name
            if "reason_code" in detail_with_run:
                detail_with_run["reason_code"] = detail_with_run["reason_code"].fillna("unknown")
            else:
                detail_with_run["reason_code"] = "unknown"
            detail_frames.append(detail_with_run)
            reason_counts = (
                detail_with_run.groupby(["run", "regime", "reason_code"], dropna=False)
                .size()
                .reset_index(name="count")
            )
            reason_records.extend(reason_counts.to_dict("records"))

        design = run_meta.get("design") or summary_df.get("design").iloc[0] if not summary_df.empty and "design" in summary_df else "n/a"
        nested = run_meta.get("nested_replicates") or summary_df.get("nested_replicates").iloc[0] if not summary_df.empty and "nested_replicates" in summary_df else "n/a"
        start_date = summary_df.get("start_date").iloc[0] if not summary_df.empty and "start_date" in summary_df else "?"
        end_date = summary_df.get("end_date").iloc[0] if not summary_df.empty and "end_date" in summary_df else "?"
        crisis_label = ""
        if "crisis_label" in panel_df and not panel_df["crisis_label"].dropna().empty:
            crisis_label = str(panel_df["crisis_label"].dropna().iloc[0])
        elif not summary_df.empty and "label" in summary_df:
            crisis_label = str(summary_df["label"].iloc[0])
        estimators = sorted(panel_df["estimator"].unique()) if not panel_df.empty else []

        alignment_median = float("nan")
        if not detection_df.empty and "angle_min_deg" in detection_df:
            angle_series = detection_df["angle_min_deg"].dropna()
            if not angle_series.empty:
                alignment_median = float(angle_series.median())
                alignment_medians[run_path.name] = alignment_median

        no_iso_share = float("nan")
        gating_payload = summary_json.get("gating", {}) if isinstance(summary_json, dict) else {}
        skip_entries = gating_payload.get("skip_reasons", []) if isinstance(gating_payload, dict) else []
        if isinstance(skip_entries, list):
            for entry in skip_entries:
                if isinstance(entry, dict) and entry.get("reason") == "no_isolated_spike":
                    try:
                        count_val = float(entry.get("count", 0.0))
                    except (TypeError, ValueError):
                        count_val = 0.0
                    windows_total = summary_json.get("rolling_windows_evaluated") if isinstance(summary_json, dict) else None
                    try:
                        windows_val = float(windows_total) if windows_total is not None else float("nan")
                    except (TypeError, ValueError):
                        windows_val = float("nan")
                    if not pd.isna(windows_val) and windows_val > 0:
                        no_iso_share = count_val / windows_val
                        no_iso_shares[run_path.name] = no_iso_share
                    break

        run_label = crisis_label or run_path.name
        alignment_fragment = (
            f", median angle={alignment_median:.1f}°"
            if not pd.isna(alignment_median)
            else ""
        )
        no_iso_fragment = (
            f" [no_isolated {no_iso_share * 100:.0f}%]"
            if not pd.isna(no_iso_share) and no_iso_share >= 0.2
            else ""
        )
        nested_scope_payload = summary_json.get("nested_scope") if isinstance(summary_json, dict) else {}
        nested_scope_fragment = ""
        if isinstance(nested_scope_payload, dict) and nested_scope_payload.get("de_scoped_equity"):
            nested_scope_fragment = " — nested scope de-scoped (no isolated spikes)"
            nested_scope_notes.append(f"{run_label}: nested equity de-scoped (gating blocked all windows)")
        edge_mode_value = ""
        if isinstance(summary_json, dict) and summary_json.get("edge_mode"):
            edge_mode_value = str(summary_json.get("edge_mode"))
        gating_mode_value = ""
        if isinstance(gating_payload, dict) and gating_payload.get("mode"):
            gating_mode_value = str(gating_payload.get("mode"))
        delta_frac_median = float("nan")
        if not detection_df.empty and "delta_frac_used" in detection_df:
            try:
                delta_series = pd.to_numeric(detection_df["delta_frac_used"], errors="coerce").dropna()
            except Exception:
                delta_series = pd.Series(dtype=float)
            if not delta_series.empty:
                delta_frac_median = float(delta_series.median())
        badge_bits: list[str] = []
        if edge_mode_value:
            badge_bits.append(f"edge={edge_mode_value}")
        if gating_mode_value:
            badge_bits.append(f"gate={gating_mode_value}")
        if not pd.isna(delta_frac_median):
            badge_bits.append(f"df~{delta_frac_median:.3f}")
        edge_mode_fragment = f" [{' | '.join(badge_bits)}]" if badge_bits else ""
        run_lines.append(
            f"- **{run_label}** (design={design}, J={nested}, period={start_date} → {end_date}{alignment_fragment}{no_iso_fragment}){nested_scope_fragment}{edge_mode_fragment} — estimators: {', '.join(estimators) if estimators else 'n/a'}"
        )

        rejection_records.extend(_collect_rejection_records(summary_df, run_path.name))

        ablation_path = run_path / "ablation_summary.csv"
        if ablation_path.exists():
            try:
                ablation_df = pd.read_csv(ablation_path)
            except pd.errors.EmptyDataError:
                ablation_df = pd.DataFrame()
            if not ablation_df.empty:
                ablation_df.insert(0, "run", run_path.name)
                ablation_frames.append(ablation_df)

        artifact_base = gallery_root / run_path.name
        tables_dir = artifact_base / "tables"
        plots_dir = artifact_base / "plots"
        if tables_dir.exists():
            artifact_map[run_path.name].append(f"tables: {tables_dir}")
        if plots_dir.exists():
            artifact_map[run_path.name].append(f"plots: {plots_dir}")

        ablation_table_path = tables_dir / "ablation.csv"
        if ablation_table_path.exists():
            artifact_map[run_path.name].append(f"ablation_table: {ablation_table_path}")
        ablation_heatmap_path = plots_dir / "ablation_heatmap.png"
        if ablation_heatmap_path.exists():
            artifact_map[run_path.name].append(f"ablation_heatmap: {ablation_heatmap_path}")
            ablation_figures.append((run_path.name, ablation_heatmap_path))

        if run_meta_path.exists():
            artifact_map[run_path.name].append(f"run_meta: {run_meta_path}")
        if run_path.name not in artifact_map:
            artifact_map[run_path.name].append("(no gallery artifacts located)")

    if panel_frames:
        combined_table = pd.concat(panel_frames, ignore_index=True, sort=False)
    else:
        combined_table = pd.DataFrame(columns=["run", "crisis_label", "estimator"])

    if detail_frames:
        detail_combined = pd.concat(detail_frames, ignore_index=True, sort=False)
    else:
        detail_combined = pd.DataFrame()

    (
        key_numeric_df,
        key_display_df,
        key_qlike_numeric_df,
        key_qlike_display_df,
    ) = _build_key_tables(combined_table)
    key_table_md = _markdown_table(key_display_df) if not key_display_df.empty else "(no data)"
    key_qlike_table_md = _markdown_table(key_qlike_display_df) if not key_qlike_display_df.empty else "(no data)"

    if key_numeric_df.empty:
        detection_mean = float("nan")
        delta_share = float("nan")
        delta_median = float("nan")
        dm_sig = 0
        dm_total = 0
        dm_sig_qlike = 0
        dm_total_qlike = 0
        substitution_mean = float("nan")
    else:
        per_run_detection = key_numeric_df.groupby("run")["detection_rate"].max()
        detection_mean = float(per_run_detection.dropna().mean()) if not per_run_detection.dropna().empty else float("nan")

        delta_series = key_numeric_df["delta_mse_ew"].dropna()
        delta_share = float((delta_series < 0).mean()) if not delta_series.empty else float("nan")
        delta_median = float(delta_series.median()) if not delta_series.empty else float("nan")

        dm_columns = []
        if "dm_p_ew" in key_numeric_df.columns:
            dm_columns.append(key_numeric_df["dm_p_ew"])
        if "dm_p_mv" in key_numeric_df.columns:
            dm_columns.append(key_numeric_df["dm_p_mv"])
        if dm_columns:
            dm_concat = pd.concat(dm_columns).dropna()
            dm_sig = int((dm_concat < 0.05).sum())
            dm_total = int(dm_concat.shape[0])
        else:
            dm_sig = 0
            dm_total = 0

        dm_columns_qlike = []
        if "dm_p_ew_qlike" in key_qlike_numeric_df.columns:
            dm_columns_qlike.append(key_qlike_numeric_df["dm_p_ew_qlike"])
        if "dm_p_mv_qlike" in key_qlike_numeric_df.columns:
            dm_columns_qlike.append(key_qlike_numeric_df["dm_p_mv_qlike"])
        if dm_columns_qlike:
            dm_concat_qlike = pd.concat(dm_columns_qlike).dropna()
            dm_sig_qlike = int((dm_concat_qlike < 0.05).sum())
            dm_total_qlike = int(dm_concat_qlike.shape[0])
        else:
            dm_sig_qlike = 0
            dm_total_qlike = 0

        if "substitution_fraction" in key_qlike_numeric_df.columns:
            sub_series = key_qlike_numeric_df["substitution_fraction"].dropna()
            substitution_mean = float(sub_series.mean()) if not sub_series.empty else float("nan")
        else:
            substitution_mean = float("nan")

    bullet_detection = (
        f"Average detection coverage across RC runs: {_format_percent(detection_mean)}."
        if not pd.isna(detection_mean)
        else "Detection coverage metrics unavailable."
    )
    bullet_mse = (
        f"Delta MSE (EW) < 0 in {_format_percent(delta_share)} of comparisons (median Delta MSE (EW) = {delta_median:.4g})."
        if not pd.isna(delta_share)
        else "Delta MSE directionality not available."
    )
    bullet_dm = (
        f"{dm_sig} of {dm_total} Diebold–Mariano tests show p < 0.05."
        if dm_total
        else "No Diebold–Mariano statistics reported."
    )
    bullet_dm_qlike = (
        f"{dm_sig_qlike} of {dm_total_qlike} QLIKE DM tests show p < 0.05."
        if dm_total_qlike
        else "No QLIKE Diebold–Mariano statistics reported."
    )
    bullet_substitution = (
        f"Accepted detections substitute in {_format_percent(substitution_mean)} of evaluated windows."
        if not pd.isna(substitution_mean)
        else "Substitution share not available."
    )
    if alignment_medians:
        alignment_series = pd.Series(alignment_medians)
        bullet_alignment = (
            f"Median alignment angle across runs: {alignment_series.median():.1f}° (mean {alignment_series.mean():.1f}°)."
        )
    else:
        bullet_alignment = "Alignment diagnostics unavailable."

    if no_iso_shares:
        no_iso_series = pd.Series(no_iso_shares)
        bullet_no_iso = (
            f"'no_isolated_spike' gate skipped {_format_percent(no_iso_series.mean())} of windows on average (max {_format_percent(no_iso_series.max())})."
        )
    else:
        bullet_no_iso = "No 'no_isolated_spike' gate activations recorded."

    summary_artifacts = _load_summary_artifacts(config)
    summary_perf_df = summary_artifacts["perf_df"]
    summary_det_df = summary_artifacts["det_df"]
    kill_data = summary_artifacts["kill_data"]
    limitations_text = summary_artifacts["limitations_text"]
    ablation_matrix_df = summary_artifacts["ablation_matrix"]

    summary_perf_table_md = "(summary performance unavailable)"
    summary_detection_table_md = "(summary detection unavailable)"
    kill_criteria_md = "(kill criteria unavailable)"
    kill_overall_status = "UNKNOWN"
    limitations_fragment = limitations_text if limitations_text else "No critical limitations detected under current criteria."
    ablation_matrix_table_md = "(global ablation matrix unavailable)"

    if not summary_perf_df.empty:
        full_perf = summary_perf_df[summary_perf_df["regime"].astype(str).str.lower() == "full"].copy()
        if not full_perf.empty:
            display_cols = [
                "portfolio",
                "delta_mse_vs_baseline",
                "var95_overlay",
                "var95_baseline",
                "dm_p_value",
                "n_effective",
            ]
            existing_cols = [col for col in display_cols if col in full_perf.columns]
            perf_display = full_perf[existing_cols].copy()
            perf_display.rename(
                columns={
                    "delta_mse_vs_baseline": "ΔMSE vs baseline",
                    "var95_overlay": "VaR95 (overlay)",
                    "var95_baseline": "VaR95 (baseline)",
                    "dm_p_value": "DM p-value",
                    "n_effective": "DM n_effective",
                },
                inplace=True,
            )
            summary_perf_table_md = _markdown_table(perf_display)

            ew_row = _row_for(full_perf, "full", "ew")
            mv_row = _row_for(full_perf, "full", "mv")
            ew_delta = _numeric(ew_row, "delta_mse_vs_baseline")
            mv_delta = _numeric(mv_row, "delta_mse_vs_baseline")
            if not np.isnan(ew_delta) and not np.isnan(mv_delta):
                bullet_mse = (
                    f"Full regime ΔMSE vs baseline: EW {ew_delta:.3g}, MV {mv_delta:.3g}."
                )
            ew_dm = _numeric(ew_row, "dm_p_value")
            mv_dm = _numeric(mv_row, "dm_p_value")
            if not np.isnan(ew_dm) or not np.isnan(mv_dm):
                parts = []
                if not np.isnan(ew_dm):
                    parts.append(f"EW p={_format_pvalue(ew_dm)}")
                if not np.isnan(mv_dm):
                    parts.append(f"MV p={_format_pvalue(mv_dm)}")
                bullet_dm = "Full regime DM tests: " + ", ".join(parts) + "."

    if not summary_det_df.empty:
        full_det = summary_det_df[summary_det_df["regime"].astype(str).str.lower() == "full"].copy()
        if not full_det.empty:
            detection_cols = [
                "detection_rate_mean",
                "detection_rate_median",
                "edge_margin_mean",
                "stability_margin_mean",
                "isolation_share_mean",
                "alignment_cos_mean",
                "reason_code_mode",
            ]
            existing_det = [col for col in detection_cols if col in full_det.columns]
            det_display = full_det[existing_det].copy()
            percent_cols = {
                "detection_rate_mean",
                "detection_rate_median",
                "isolation_share_mean",
            }
            for col in det_display.columns:
                if det_display[col].dtype.kind in {"f", "i"}:
                    if col in percent_cols:
                        det_display[col] = det_display[col].apply(lambda v: _format_percent(v) if pd.notna(v) else "n/a")
                    elif col == "alignment_cos_mean":
                        det_display[col] = det_display[col].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "n/a")
                    else:
                        det_display[col] = det_display[col].apply(lambda v: _format_edge_metric(v) if pd.notna(v) else "n/a")
            det_display.rename(
                columns={
                    "detection_rate_mean": "Detection rate (mean)",
                    "detection_rate_median": "Detection rate (median)",
                    "edge_margin_mean": "Edge margin",
                    "stability_margin_mean": "Stability margin",
                    "isolation_share_mean": "Isolation share",
                    "alignment_cos_mean": "Alignment cos",
                    "reason_code_mode": "Reason code",
                },
                inplace=True,
            )
            summary_detection_table_md = _markdown_table(det_display)

            det_row = full_det.iloc[0]
            det_rate_mean = _numeric(det_row, "detection_rate_mean")
            reason_code_mode = det_row.get("reason_code_mode", "")
            edge_margin_mean = _numeric(det_row, "edge_margin_mean")
            stability_mean = _numeric(det_row, "stability_margin_mean")
            if not np.isnan(det_rate_mean):
                bullet_detection = (
                    f"Full regime detection { _format_percent(det_rate_mean) } (reason: {reason_code_mode or 'n/a'}); "
                    f"edge margin {edge_margin_mean:.3f}, stability {stability_mean:.3f}."
                )

    kill_criteria_md, kill_overall_status = _format_kill_criteria_payload(kill_data)

    if ablation_matrix_df is not None and not ablation_matrix_df.empty:
        display_cols = [
            "panel",
            "edge_mode",
            "require_isolated",
            "q_max",
            "shrinker",
            "prewhiten",
            "ew_delta_mse_vs_baseline_vs_default",
            "detections_mean_vs_default",
        ]
        existing_cols = [col for col in display_cols if col in ablation_matrix_df.columns]
        matrix_display = ablation_matrix_df.sort_values(
            by="ew_delta_mse_vs_baseline_vs_default",
            ascending=True,
            na_position="last",
        ).head(5)[existing_cols].copy()
        if not matrix_display.empty:
            if "require_isolated" in matrix_display.columns:
                matrix_display["require_isolated"] = matrix_display["require_isolated"].map({True: "yes", False: "no"})
            if "prewhiten" in matrix_display.columns:
                matrix_display["prewhiten"] = matrix_display["prewhiten"].map({True: "on", False: "off"})
            matrix_display.rename(
                columns={
                    "ew_delta_mse_vs_baseline_vs_default": "ΔMSE(EW) vs default",
                    "detections_mean_vs_default": "ΔDetections vs default",
                },
                inplace=True,
            )
            ablation_matrix_table_md = _markdown_table(matrix_display)


    rejection_percent_df, rejection_markdown_df = _build_rejection_tables(rejection_records)
    if rejection_markdown_df.empty:
        rejection_table_md = "(no rejection diagnostics)"
        coverage_paragraph = "What limits coverage: rejection diagnostics unavailable."
    else:
        rejection_table_md = _markdown_table(
            rejection_markdown_df[
                ["run", *REJECTION_REASONS]
            ]
        )
        reason_means = rejection_percent_df[REJECTION_REASONS].mean().sort_values(ascending=False)
        summary_bits = [
            f"{_prettify_reason(reason)} {share:.1f}%"
            for reason, share in reason_means.head(2).items()
            if share > 0
        ]
        if not summary_bits:
            summary_bits = ["no dominant rejection signals"]
        coverage_paragraph = "What limits coverage: " + "; ".join(summary_bits) + "."

    ablation_table_md = ""
    ablation_caption = ""
    ablation_figure_md = ""
    if ablation_frames:
        ablation_df = pd.concat(ablation_frames, ignore_index=True, sort=False)
        if "mse_gain" not in ablation_df.columns and {"mse_alias", "mse_de"}.issubset(ablation_df.columns):
            ablation_df["mse_gain"] = ablation_df["mse_alias"] - ablation_df["mse_de"]
        display_columns = ["run", "delta_frac", "eps", "eta", "a_grid", "mse_gain", "detection_rate"]
        for col in display_columns:
            if col not in ablation_df.columns:
                ablation_df[col] = float("nan")
        top_df = (
            ablation_df.dropna(subset=["mse_gain"], how="all")
            .sort_values("mse_gain", ascending=False)
            .head(6)
            .copy()
        )
        if not top_df.empty:
            display_rows: list[dict[str, str]] = []
            for _, row in top_df.iterrows():
                display_rows.append(
                    {
                        "run": row["run"],
                        "delta_frac": f"{row['delta_frac']:.3f}" if pd.notna(row["delta_frac"]) else "n/a",
                        "eps": f"{row['eps']:.3f}" if pd.notna(row["eps"]) else "n/a",
                        "eta": f"{row['eta']:.2f}" if pd.notna(row["eta"]) else "n/a",
                        "a_grid": _format_windows(row["a_grid"]),
                        "delta_mse_ew": _format_delta(row["mse_gain"]),
                        "detection_rate": _format_detection(row["detection_rate"]),
                    }
                )
            ablation_display_df = pd.DataFrame(
                display_rows,
                columns=[
                    "run",
                    "delta_frac",
                    "eps",
                    "eta",
                    "a_grid",
                    "delta_mse_ew",
                    "detection_rate",
                ],
            )
            ablation_table_md = _markdown_table(ablation_display_df)

            run_count = int(ablation_df["run"].nunique())
            combo_count = int(ablation_df.shape[0])
            caption_parts = [
                f"Ablation sweep covers {combo_count} parameter sets across {run_count} run(s)."
            ]
            best_row = top_df.iloc[0]
            if pd.notna(best_row["mse_gain"]):
                delta_str = f"{best_row['delta_frac']:.3f}" if pd.notna(best_row["delta_frac"]) else "n/a"
                eps_str = f"{best_row['eps']:.3f}" if pd.notna(best_row["eps"]) else "n/a"
                eta_val = best_row["eta"] if "eta" in best_row else float("nan")
                eta_str = f"{eta_val:.2f}" if pd.notna(eta_val) else "n/a"
                a_str = _format_windows(best_row["a_grid"])
                caption_parts.append(
                    f"Best Delta MSE (EW) of {_format_delta(best_row['mse_gain'])} occurs at delta={delta_str}, eps={eps_str}, eta={eta_str}, a_grid={a_str}."
                )
            coverage_series = ablation_df["detection_rate"].dropna()
            if not coverage_series.empty:
                caption_parts.append(
                    f"Coverage spans {coverage_series.min() * 100:.0f}% - {coverage_series.max() * 100:.0f}% (median {coverage_series.median() * 100:.0f}%)."
                )
            else:
                caption_parts.append("Coverage metrics were not reported for this sweep.")

            ablation_caption = " ".join(caption_parts)

            for run_name, figure_path in ablation_figures:
                if figure_path.exists():
                    ablation_figure_md = f"![Ablation heatmap ({run_name})]({figure_path.as_posix()})"
                    break

    if reason_records:
        reason_df = pd.DataFrame(reason_records)
        totals_df = (
            reason_df.groupby(["run", "regime"], as_index=False)["count"].sum().rename(columns={"count": "total"})
        )
        pivot_reason = (
            reason_df.pivot_table(
                index=["run", "regime"],
                columns="reason_code",
                values="count",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
            .merge(totals_df, on=["run", "regime"], how="left")
        )
        for col in pivot_reason.columns:
            if col not in {"run", "regime", "total"}:
                pivot_reason[col] = pivot_reason[col] / pivot_reason["total"].replace(0, np.nan)
        pivot_reason = pivot_reason.drop(columns=["total"], errors="ignore")
        display_reason = pivot_reason.copy()
        display_reason = display_reason.rename(columns=lambda col: _prettify_reason(col) if col not in {"run", "regime"} else col)
        for col in display_reason.columns:
            if col not in {"run", "regime"}:
                display_reason[col] = display_reason[col].apply(_format_percent)
        reason_table_md = _markdown_table(display_reason)
    else:
        reason_table_md = "(no reason-code diagnostics)"

    summary_plot_paths: list[Path] = []
    if not detail_combined.empty:
        edge_series = pd.to_numeric(detail_combined.get("edge_margin_mean"), errors="coerce")
        if edge_series is not None:
            edge_values = edge_series.dropna()
        else:
            edge_values = pd.Series(dtype=float)
        if not edge_values.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(edge_values, bins=20, color="#1f77b4", alpha=0.85)
            ax.set_title("Edge Margin Mean Distribution")
            ax.set_xlabel("Edge margin mean")
            ax.set_ylabel("Frequency")
            fig.tight_layout()
            edge_hist_path = summary_plots_dir / "edge_margin_hist.png"
            fig.savefig(edge_hist_path, dpi=200)
            plt.close(fig)
            edge_margin_hist_md = f"![Edge margin distribution]({edge_hist_path.as_posix()})"
            summary_plot_paths.append(edge_hist_path)
        else:
            edge_margin_hist_md = "(edge margin histogram unavailable)"

        iso_series = pd.to_numeric(detail_combined.get("isolation_share"), errors="coerce")
        if iso_series is not None:
            iso_df = (
                detail_combined.assign(isolation_share=iso_series)
                .dropna(subset=["isolation_share"])
                .groupby("run")
                ["isolation_share"].mean()
                .sort_values(ascending=False)
                .reset_index()
            )
        else:
            iso_df = pd.DataFrame()
        if not iso_df.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(iso_df["run"], iso_df["isolation_share"] * 100.0, color="#ff7f0e")
            ax.set_xlabel("Mean isolation share (%)")
            ax.set_title("Isolation Share by Run")
            fig.tight_layout()
            iso_bar_path = summary_plots_dir / "isolation_share_bar.png"
            fig.savefig(iso_bar_path, dpi=200)
            plt.close(fig)
            isolation_share_bar_md = f"![Isolation share by run]({iso_bar_path.as_posix()})"
            summary_plot_paths.append(iso_bar_path)
        else:
            isolation_share_bar_md = "(isolation share chart unavailable)"

        stability_cols = {col for col in detail_combined.columns if col in {"edge_margin_mean", "stability_margin_mean"}}
        if {"edge_margin_mean", "stability_margin_mean"}.issubset(stability_cols):
            scatter_df = detail_combined.copy()
            scatter_df["edge_margin_mean"] = pd.to_numeric(scatter_df["edge_margin_mean"], errors="coerce")
            scatter_df["stability_margin_mean"] = pd.to_numeric(scatter_df["stability_margin_mean"], errors="coerce")
            scatter_summary = (
                scatter_df.dropna(subset=["edge_margin_mean", "stability_margin_mean"])
                .groupby("run")[["edge_margin_mean", "stability_margin_mean"]]
                .mean()
                .reset_index()
            )
        else:
            scatter_summary = pd.DataFrame()
        if not scatter_summary.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(
                scatter_summary["edge_margin_mean"],
                scatter_summary["stability_margin_mean"],
                c="#2ca02c",
                alpha=0.85,
            )
            for _, row in scatter_summary.iterrows():
                ax.annotate(row["run"], (row["edge_margin_mean"], row["stability_margin_mean"]), fontsize=8)
            ax.set_xlabel("Edge margin mean")
            ax.set_ylabel("Stability margin mean")
            ax.set_title("Direction vs Stability")
            fig.tight_layout()
            scatter_path = summary_plots_dir / "direction_stability_scatter.png"
            fig.savefig(scatter_path, dpi=200)
            plt.close(fig)
            direction_stability_plot_md = f"![Direction vs stability]({scatter_path.as_posix()})"
            summary_plot_paths.append(scatter_path)
        else:
            direction_stability_plot_md = "(direction vs stability plot unavailable)"
    else:
        edge_margin_hist_md = "(edge margin histogram unavailable)"
        isolation_share_bar_md = "(isolation share chart unavailable)"
        direction_stability_plot_md = "(direction vs stability plot unavailable)"

    if summary_plot_paths:
        artifact_map["summary"].append(f"plots: {summary_plots_dir.as_posix()}")

    artifact_lines = [
        f"- {run}: " + "; ".join(paths)
        for run, paths in artifact_map.items()
    ]
    artifact_list = "\n".join(artifact_lines) if artifact_lines else "- (no artifacts located)"

    problem_statement = (
        "This memo synthesizes smoke and crisis release-candidate runs, "
        "highlighting estimator deltas, coverage bottlenecks, and ablation sensitivities."
    )

    key_results_intro = (
        "Values are measured versus the de-aliased baseline; negative Delta MSE indicates an improvement. "
        "The first table reports MSE deltas and DM p-values, while the second summarises QLIKE DM statistics and substitution share."
    )

    if not ablation_table_md:
        ablation_table_md = "(no ablation sweep detected)"
    if not ablation_caption:
        ablation_caption = "No ablation sweep artifacts were detected for this gallery."

    if nested_scope_notes:
        nested_scope_section = "\n## Nested Scope\n" + "\n".join(
            f"- {note}" for note in nested_scope_notes
        ) + "\n"
    else:
        nested_scope_section = ""

    template_path = Path("reports/templates/memo.md.j2")
    template = template_path.read_text(encoding="utf-8")
    context = {
        "gallery_name": gallery_name,
        "problem_statement": problem_statement,
        "run_summary": "\n".join(run_lines) if run_lines else "(no runs)",
        "nested_scope_section": nested_scope_section,
        "key_results_intro": key_results_intro,
        "key_results_table": key_table_md,
        "key_results_table_qlike": key_qlike_table_md,
        "coverage_paragraph": coverage_paragraph,
        "rejection_table_md": rejection_table_md,
        "ablation_caption": ablation_caption,
        "ablation_table_md": ablation_table_md,
        "ablation_figure_md": ablation_figure_md or "",
        "bullet_detection": bullet_detection,
        "bullet_mse": bullet_mse,
        "bullet_dm": bullet_dm,
        "bullet_dm_qlike": bullet_dm_qlike,
        "bullet_substitution": bullet_substitution,
        "bullet_alignment": bullet_alignment,
        "bullet_no_iso": bullet_no_iso,
        "reason_table_md": reason_table_md,
        "edge_margin_hist_md": edge_margin_hist_md,
        "isolation_share_bar_md": isolation_share_bar_md,
        "direction_stability_plot_md": direction_stability_plot_md,
        "artifact_list": artifact_list,
        "summary_perf_table_md": summary_perf_table_md,
        "summary_detection_table_md": summary_detection_table_md,
        "kill_criteria_md": kill_criteria_md,
        "kill_overall_status": kill_overall_status,
        "limitations_fragment": limitations_fragment,
        "ablation_matrix_table_md": ablation_matrix_table_md,
    }
    memo_text = template.format(**context)

    memo_path = reports_dir / "memo.md"
    memo_path.write_text(memo_text, encoding="utf-8")
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    memo_timestamp_path = reports_dir / f"memo_{timestamp}.md"
    memo_timestamp_path.write_text(memo_text, encoding="utf-8")

    print(f"Memo written to {memo_path} and {memo_timestamp_path}")
    return memo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Markdown memo summarizing RC runs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/equity_panel/config.rc.yaml"),
        help="YAML configuration describing runs to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_memo(args.config.resolve())


if __name__ == "__main__":
    main()
