from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["table_estimators_panel", "table_rejections", "table_ablation"]

DEFAULT_FIG_ROOT = Path("figures")
EW_LABEL = "Equal Weight"
MV_LABEL_PREFIX = "Min-Variance"


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


def table_estimators_panel(df: pd.DataFrame, *, root: Path = DEFAULT_FIG_ROOT) -> tuple[Path, Path, Path]:
    """Create estimator panel comparison tables and return paths to CSV, Markdown, and LaTeX outputs."""

    if df.empty:
        raise ValueError("Estimator DataFrame is empty.")

    run_tag = _single_run_tag(df)
    output_dir = root / run_tag / "tables"
    _ensure_dir(output_dir)

    crisis_label = df["crisis_label"].dropna().iloc[0] if "crisis_label" in df and not df["crisis_label"].dropna().empty else ""

    rows = []
    for estimator, group in df.groupby("estimator"):
        detection_rate = (
            float(group["detection_rate"].dropna().iloc[0])
            if not group["detection_rate"].dropna().empty
            else np.nan
        )

        def _strategy_value(label: str, column: str) -> float:
            selection = group[group["strategy"] == label][column]
            if not selection.empty:
                return float(selection.iloc[0])
            selection = group[group["strategy"].str.startswith(label, na=False)][column]
            return float(selection.iloc[0]) if not selection.empty else np.nan

        delta_ew = _strategy_value(EW_LABEL, "delta_mse_vs_de")
        dm_ew = _strategy_value(EW_LABEL, "dm_p")
        n_windows = _strategy_value(EW_LABEL, "n_windows")

        mv_values = group[group["strategy"].str.startswith(MV_LABEL_PREFIX, na=False)]
        delta_mv = float(mv_values["delta_mse_vs_de"].iloc[0]) if not mv_values.empty else np.nan
        dm_mv = float(mv_values["dm_p"].iloc[0]) if not mv_values.empty else np.nan
        n_windows_mv = float(mv_values["n_windows"].iloc[0]) if not mv_values.empty else np.nan

        rows.append(
            {
                "estimator": estimator,
                "detection_rate": detection_rate,
                "delta_mse_ew": delta_ew,
                "delta_mse_mv": delta_mv,
                "dm_p_ew": dm_ew,
                "dm_p_mv": dm_mv,
                "n_windows": n_windows if not np.isnan(n_windows) else n_windows_mv,
                "crisis_label": crisis_label,
            }
        )

    table_df = pd.DataFrame(rows).sort_values("estimator").reset_index(drop=True)

    csv_path = output_dir / "estimators.csv"
    md_path = output_dir / "estimators.md"
    tex_path = output_dir / "estimators.tex"

    table_df.to_csv(csv_path, index=False)
    _write_markdown(table_df, md_path)
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
    pivot = df.pivot_table(index="run", columns="rejection_reason", values="count", fill_value=0).reset_index()
    for reason in reasons:
        if reason not in pivot.columns:
            pivot[reason] = 0
    pivot = pivot[[col for col in ["run", *reasons] if col in pivot.columns]]
    csv_path = output_dir / "rejections.csv"
    md_path = output_dir / "rejections.md"
    tex_path = output_dir / "rejections.tex"

    pivot.to_csv(csv_path, index=False)
    _write_markdown(pivot, md_path)
    pivot.to_latex(tex_path, index=False)

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
