#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

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


def _build_key_tables(panel_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if panel_df.empty:
        return pd.DataFrame(), pd.DataFrame()

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
            "delta_mse_mv": float("nan"),
            "ci_lo_mv": float("nan"),
            "ci_hi_mv": float("nan"),
            "dm_p_mv": float("nan"),
            "edge_margin_median": float(edge_med_series.iloc[0]) if not edge_med_series.empty else float("nan"),
            "edge_margin_iqr": float(edge_iqr_series.iloc[0]) if not edge_iqr_series.empty else float("nan"),
            "n_windows": float("nan"),
            "rolling_windows_evaluated": float(windows_series.iloc[0]) if not windows_series.empty else float("nan"),
        }

        if ew_row is not None:
            record["delta_mse_ew"] = float(ew_row.get("delta_mse_vs_de", float("nan")))
            record["ci_lo_ew"] = float(ew_row.get("ci_lo", float("nan")))
            record["ci_hi_ew"] = float(ew_row.get("ci_hi", float("nan")))
            record["dm_p_ew"] = float(ew_row.get("dm_p", float("nan")))
            record["n_windows"] = float(ew_row.get("n_windows", float("nan")))
        if mv_row is not None:
            record["delta_mse_mv"] = float(mv_row.get("delta_mse_vs_de", float("nan")))
            record["ci_lo_mv"] = float(mv_row.get("ci_lo", float("nan")))
            record["ci_hi_mv"] = float(mv_row.get("ci_hi", float("nan")))
            record["dm_p_mv"] = float(mv_row.get("dm_p", float("nan")))
            if pd.isna(record["n_windows"]):
                record["n_windows"] = float(mv_row.get("n_windows", float("nan")))

        records.append(record)

    if not records:
        return pd.DataFrame(), pd.DataFrame()

    numeric_df = pd.DataFrame(records).sort_values(["run", "estimator"]).reset_index(drop=True)

    display_rows: list[dict[str, str]] = []
    for _, row in numeric_df.iterrows():
        detection_display = _format_detection(row["detection_rate"])
        design_value = str(row.get("design", "")).lower()
        windows_val = row.get("rolling_windows_evaluated", float("nan"))
        if (
            design_value == "nested"
            and not pd.isna(row.get("detection_rate"))
            and float(row.get("detection_rate", 0.0)) == 0.0
            and not pd.isna(windows_val)
            and float(windows_val) > 0.0
            and detection_display != "n/a"
        ):
            detection_display += " (no accepted detections; check guardrails)"
        display_rows.append(
            {
                "run": row["run"],
                "crisis_label": row["crisis_label"] or "n/a",
                "estimator": row["estimator"],
                "detection_rate": detection_display,
                "delta_mse_ew": _format_delta(row["delta_mse_ew"]),
                "CI_EW": _format_ci(row["ci_lo_ew"], row["ci_hi_ew"]),
                "DM_p_EW": _format_pvalue(row["dm_p_ew"]),
                "delta_mse_mv": _format_delta(row["delta_mse_mv"]),
                "CI_MV": _format_ci(row["ci_lo_mv"], row["ci_hi_mv"]),
                "DM_p_MV": _format_pvalue(row["dm_p_mv"]),
                "edge_margin_median": _format_edge_metric(row["edge_margin_median"]),
                "edge_margin_IQR": _format_edge_metric(row["edge_margin_iqr"]),
                "n_windows": _format_windows(row["n_windows"]),
            }
        )

    display_df = pd.DataFrame(
        display_rows,
        columns=[
            "run",
            "crisis_label",
            "estimator",
            "detection_rate",
            "delta_mse_ew",
            "CI_EW",
            "DM_p_EW",
            "delta_mse_mv",
            "CI_MV",
            "DM_p_MV",
            "edge_margin_median",
            "edge_margin_IQR",
            "n_windows",
        ],
    )

    return numeric_df, display_df


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
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    panel_frames: list[pd.DataFrame] = []
    run_lines: list[str] = []
    artifact_map: dict[str, list[str]] = defaultdict(list)
    rejection_records: list[dict[str, float]] = []
    ablation_frames: list[pd.DataFrame] = []
    ablation_figures: list[tuple[str, Path]] = []

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

        run_label = crisis_label or run_path.name
        run_lines.append(
            f"- **{run_label}** (design={design}, J={nested}, period={start_date} → {end_date}) — estimators: {', '.join(estimators) if estimators else 'n/a'}"
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

    key_numeric_df, key_display_df = _build_key_tables(combined_table)
    key_table_md = _markdown_table(key_display_df) if not key_display_df.empty else "(no data)"

    if key_numeric_df.empty:
        detection_mean = float("nan")
        delta_share = float("nan")
        delta_median = float("nan")
        dm_sig = 0
        dm_total = 0
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
        "Values are measured versus the de-aliased baseline; negative Delta MSE indicates an improvement."
    )

    if not ablation_table_md:
        ablation_table_md = "(no ablation sweep detected)"
    if not ablation_caption:
        ablation_caption = "No ablation sweep artifacts were detected for this gallery."

    template_path = Path("reports/templates/memo.md.j2")
    template = template_path.read_text(encoding="utf-8")
    context = {
        "gallery_name": gallery_name,
        "problem_statement": problem_statement,
        "run_summary": "\n".join(run_lines) if run_lines else "(no runs)",
        "key_results_intro": key_results_intro,
        "key_results_table": key_table_md,
        "coverage_paragraph": coverage_paragraph,
        "rejection_table_md": rejection_table_md,
        "ablation_caption": ablation_caption,
        "ablation_table_md": ablation_table_md,
        "ablation_figure_md": ablation_figure_md or "",
        "bullet_detection": bullet_detection,
        "bullet_mse": bullet_mse,
        "bullet_dm": bullet_dm,
        "artifact_list": artifact_list,
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
