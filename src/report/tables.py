from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["table_estimators_panel", "table_rejections", "table_ablation"]

DEFAULT_FIG_ROOT = Path("figures")
EW_LABEL = "Equal Weight"
EW_STRATEGIES: Sequence[str] = ("Equal Weight",)
MV_LABEL_PREFIX = "Min-Variance"
MV_STRATEGIES: Sequence[str] = ("Min-Variance (box)", "Min-Variance (long-only)", "Min-Variance")
CI_COLUMNS = ("ci_lo_ew", "ci_hi_ew", "ci_lo_mv", "ci_hi_mv")


def _single_run_tag(df: pd.DataFrame) -> str:
    if "run" not in df or df["run"].nunique() != 1:
        raise ValueError("Expected a single run in the provided DataFrame.")
    return df["run"].iloc[0]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _format_float(value: float) -> str:
    if pd.isna(value):
        return "nan"
    if abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.4f}"


def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for _, row in df.iterrows():
        values = [
            _format_float(row[col]) if isinstance(row[col], (float, np.floating)) else str(row[col])
            for col in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_strategy_row(group: pd.DataFrame, candidates: Sequence[str]) -> pd.Series | None:
    for name in candidates:
        exact = group[group["strategy"] == name]
        if not exact.empty:
            return exact.iloc[0]
        prefix = group[group["strategy"].str.startswith(name, na=False)]
        if not prefix.empty:
            return prefix.iloc[0]
    return None


def table_estimators_panel(df: pd.DataFrame, *, root: Path = DEFAULT_FIG_ROOT) -> tuple[Path, Path, Path]:
    """Create estimator panel comparison tables and return paths to CSV, Markdown, and LaTeX outputs."""

    if df.empty:
        raise ValueError("Estimator DataFrame is empty.")

    run_tag = _single_run_tag(df)
    output_dir = root / run_tag / "tables"
    _ensure_dir(output_dir)

    rows: list[dict[str, object]] = []
    for estimator, group in df.groupby("estimator"):
        ew_row = _find_strategy_row(group, EW_STRATEGIES)
        mv_row = _find_strategy_row(group, MV_STRATEGIES)

        detection_series = group["detection_rate"].dropna() if "detection_rate" in group else pd.Series(dtype=float)
        detection_rate = float(detection_series.iloc[0]) if not detection_series.empty else np.nan

        crisis_series = group["crisis_label"].dropna() if "crisis_label" in group else pd.Series(dtype=object)
        crisis_label = str(crisis_series.iloc[0]) if not crisis_series.empty else ""

        edge_med = group["edge_margin_median"].dropna() if "edge_margin_median" in group else pd.Series(dtype=float)
        edge_iqr = group["edge_margin_iqr"].dropna() if "edge_margin_iqr" in group else pd.Series(dtype=float)

        substitution_series = (
            group["substitution_fraction"].dropna()
            if "substitution_fraction" in group
            else pd.Series(dtype=float)
        )
        record: dict[str, object] = {
            "estimator": estimator,
            "detection_rate": detection_rate,
            "crisis_label": crisis_label,
            "edge_margin_median": float(edge_med.iloc[0]) if not edge_med.empty else np.nan,
            "edge_margin_iqr": float(edge_iqr.iloc[0]) if not edge_iqr.empty else np.nan,
            "dm_p_ew_qlike": np.nan,
            "dm_p_mv_qlike": np.nan,
            "mean_qlike": np.nan,
            "substitution_fraction": float(substitution_series.iloc[0]) if not substitution_series.empty else np.nan,
        }

        if ew_row is not None:
            record.update(
                {
                    "delta_mse_ew": float(ew_row.get("delta_mse_vs_de", np.nan)),
                    "ci_lo_ew": float(ew_row.get("ci_lo", np.nan)),
                    "ci_hi_ew": float(ew_row.get("ci_hi", np.nan)),
                    "dm_p_ew": float(ew_row.get("dm_p", np.nan)),
                    "n_windows": int(ew_row.get("n_windows", 0)),
                    "dm_p_ew_qlike": float(ew_row.get("dm_p_qlike", np.nan)),
                    "mean_qlike": float(ew_row.get("mean_qlike", np.nan)),
                }
            )
        if mv_row is not None:
            record.update(
                {
                    "delta_mse_mv": float(mv_row.get("delta_mse_vs_de", np.nan)),
                    "ci_lo_mv": float(mv_row.get("ci_lo", np.nan)),
                    "ci_hi_mv": float(mv_row.get("ci_hi", np.nan)),
                    "dm_p_mv": float(mv_row.get("dm_p", np.nan)),
                    "n_windows": int(mv_row.get("n_windows", record.get("n_windows", 0))),
                    "dm_p_mv_qlike": float(mv_row.get("dm_p_qlike", np.nan)),
                }
            )

        rows.append(record)

    table_df = pd.DataFrame(rows).sort_values("estimator").reset_index(drop=True)

    csv_path = output_dir / "estimators.csv"
    md_path = output_dir / "estimators.md"
    tex_path = output_dir / "estimators.tex"

    table_df.to_csv(csv_path, index=False)

    md_df = table_df.copy()
    for lo_col, hi_col in (("ci_lo_ew", "ci_hi_ew"), ("ci_lo_mv", "ci_hi_mv")):
        if lo_col in md_df.columns and hi_col in md_df.columns:
            suffix = lo_col.rsplit("_", 1)[-1].upper()
            display_col = f"CI_{suffix}"
            ci_values = []
            for lo, hi in zip(md_df[lo_col], md_df[hi_col]):
                if pd.isna(lo) or pd.isna(hi):
                    ci_values.append("n/a")
                else:
                    ci_values.append(f"[{_format_float(float(lo))}, {_format_float(float(hi))}]")
            md_df[display_col] = ci_values
    md_df = md_df.drop(columns=[col for col in CI_COLUMNS if col in md_df.columns], errors="ignore")
    _write_markdown(md_df, md_path)

    table_df.to_latex(tex_path, index=False, float_format=lambda x: _format_float(float(x)))

    return csv_path, md_path, tex_path


def table_rejections(df: pd.DataFrame, *, root: Path = DEFAULT_FIG_ROOT) -> tuple[Path, Path, Path]:
    """Generate a rejection reason summary table."""

    if df.empty:
        raise ValueError("Rejection DataFrame is empty.")

    run_tag = _single_run_tag(df)
    output_dir = root / run_tag / "tables"
    _ensure_dir(output_dir)

    reasons = ["other", "edge_buffer", "off_component_ratio", "stability_fail", "energy_floor", "neg_mu"]
    pivot = (
        df.pivot_table(index="run", columns="rejection_reason", values="count", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    for reason in reasons:
        if reason not in pivot.columns:
            pivot[reason] = 0.0
    pivot = pivot[["run", *reasons]]

    totals = pivot[reasons].sum(axis=1)
    percent_df = pivot.copy()
    for reason in reasons:
        percent_df[reason] = np.where(
            totals > 0,
            (pivot[reason] / totals) * 100.0,
            0.0,
        )

    csv_path = output_dir / "rejections.csv"
    md_path = output_dir / "rejections.md"
    tex_path = output_dir / "rejections.tex"

    percent_df.to_csv(csv_path, index=False)

    md_df = percent_df.copy()
    for reason in reasons:
        md_df[reason] = md_df[reason].apply(lambda value: f"{value:.1f}%")
    _write_markdown(md_df, md_path)

    percent_df.to_latex(tex_path, index=False, float_format=lambda x: _format_float(float(x)))

    return csv_path, md_path, tex_path


def table_ablation(df: pd.DataFrame, *, root: Path = DEFAULT_FIG_ROOT) -> tuple[Path, Path, Path]:
    """Summarise ablation grids when available."""

    if df.empty:
        raise ValueError("Ablation DataFrame is empty.")

    run_tag = _single_run_tag(df)
    output_dir = root / run_tag / "tables"
    _ensure_dir(output_dir)

    table = df.copy()
    if "mse_gain" not in table.columns and {"mse_alias", "mse_de"}.issubset(table.columns):
        table["mse_gain"] = table["mse_alias"] - table["mse_de"]

    keep_cols = [col for col in ["delta_frac", "eps", "a_grid", "eta", "detection_rate", "mse_gain"] if col in table.columns]
    keep_cols = ["run"] + keep_cols
    table = table[keep_cols]

    csv_path = output_dir / "ablation.csv"
    md_path = output_dir / "ablation.md"
    tex_path = output_dir / "ablation.tex"

    table.to_csv(csv_path, index=False)
    _write_markdown(table, md_path)
    table.to_latex(tex_path, index=False)

    return csv_path, md_path, tex_path
