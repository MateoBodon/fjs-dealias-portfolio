#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import csv


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []
            return [dict(row) for row in reader]
    except OSError:
        return []


def _scalar(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mode_from_resolved(run_dir: Path) -> str:
    resolved_path = run_dir / "resolved_config.json"
    if resolved_path.exists():
        try:
            data = json.loads(resolved_path.read_text(encoding="utf-8"))
            mode = data.get("prewhiten_mode_effective") or data.get("prewhiten_mode_requested")
            if isinstance(mode, str) and mode:
                return mode.lower()
        except (json.JSONDecodeError, OSError):
            pass
    diag_path = run_dir / "full" / "diagnostics.csv"
    diag_rows = _read_csv_rows(diag_path)
    if diag_rows:
        mode = diag_rows[0].get("prewhiten_mode_effective")
        if isinstance(mode, str) and mode:
            return mode.lower()
    return "unknown"


@dataclass(slots=True)
class RunSummary:
    path: Path
    mode: str
    detection_rate: float
    delta_mse: dict[str, float]
    es_error: dict[str, float]
    sign_p: dict[str, float]


def _portfolio_value(rows: list[dict[str, str]], portfolio: str, column: str) -> float:
    for row in rows:
        if (
            row.get("regime", "").strip().lower() == "full"
            and row.get("estimator", "").strip().lower() == "overlay"
            and row.get("portfolio", "").strip().lower() == portfolio.lower()
        ):
            return _scalar(row.get(column))
    return float("nan")


def _sign_p_value(path: Path, portfolio: str) -> float:
    rows = _read_csv_rows(path)
    for row in rows:
        if (
            row.get("portfolio", "").strip().lower() == portfolio.lower()
            and row.get("baseline", "").strip().lower() == "baseline"
            and row.get("test", "").strip().lower() == "sign"
        ):
            return _scalar(row.get("p_value"))
    return float("nan")


def _load_run_summary(run_dir: Path) -> RunSummary:
    full_dir = run_dir / "full"
    diag_rows = _read_csv_rows(full_dir / "diagnostics.csv")
    detection_rate = float("nan")
    if diag_rows:
        detection_rate = _scalar(diag_rows[0].get("detection_rate"))
    metrics_rows = _read_csv_rows(full_dir / "metrics.csv")
    delta_mse = {
        "ew": _portfolio_value(metrics_rows, "ew", "delta_mse_vs_baseline"),
        "mv": _portfolio_value(metrics_rows, "mv", "delta_mse_vs_baseline"),
    }
    es_error = {
        portfolio: (
            _portfolio_value(metrics_rows, portfolio, "es95")
            - _portfolio_value(metrics_rows, portfolio, "realised_es")
        )
        for portfolio in ("ew", "mv")
    }
    sign_p = {
        portfolio: _sign_p_value(run_dir / "dm_flip_only.csv", portfolio)
        for portfolio in ("ew", "mv")
    }
    return RunSummary(
        path=run_dir,
        mode=_mode_from_resolved(run_dir),
        detection_rate=detection_rate,
        delta_mse=delta_mse,
        es_error=es_error,
        sign_p=sign_p,
    )


def _build_effect_rows(off: RunSummary, on: RunSummary, label_off: str, label_on: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = [
        (off, label_off or off.mode or "off"),
        (on, label_on or on.mode or "ff5mom"),
    ]
    for summary, label in ordered:
        rows.append(
            {
                "prewhiten_mode": label,
                "detection_rate": summary.detection_rate,
                "delta_mse_ew": summary.delta_mse.get("ew"),
                "delta_mse_mv": summary.delta_mse.get("mv"),
                "es95_error_ew": summary.es_error.get("ew"),
                "es95_error_mv": summary.es_error.get("mv"),
                "sign_p_ew": summary.sign_p.get("ew"),
                "sign_p_mv": summary.sign_p.get("mv"),
            }
        )
    delta_row = {
        "prewhiten_mode": f"{ordered[1][1]}_minus_{ordered[0][1]}",
        "detection_rate": ordered[1][0].detection_rate - ordered[0][0].detection_rate,
        "delta_mse_ew": ordered[1][0].delta_mse.get("ew") - ordered[0][0].delta_mse.get("ew"),
        "delta_mse_mv": ordered[1][0].delta_mse.get("mv") - ordered[0][0].delta_mse.get("mv"),
        "es95_error_ew": ordered[1][0].es_error.get("ew") - ordered[0][0].es_error.get("ew"),
        "es95_error_mv": ordered[1][0].es_error.get("mv") - ordered[0][0].es_error.get("mv"),
        "sign_p_ew": ordered[1][0].sign_p.get("ew"),
        "sign_p_mv": ordered[1][0].sign_p.get("mv"),
    }
    rows.append(delta_row)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise paired prewhiten runs (off vs ff5mom) into a comparison CSV."
    )
    parser.add_argument("--off", type=Path, required=True, help="Directory containing the prewhiten=off run.")
    parser.add_argument("--on", type=Path, required=True, help="Directory containing the prewhiten=ff5mom run.")
    parser.add_argument("--label-off", type=str, default="off", help="Label for the OFF run (default: off).")
    parser.add_argument("--label-on", type=str, default="ff5mom", help="Label for the ON run (default: ff5mom).")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (defaults to <on>/prewhiten_effect.csv).",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Also copy the output CSV into the OFF run directory for convenience.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    off_dir = args.off.resolve()
    on_dir = args.on.resolve()
    if not off_dir.exists():
        raise FileNotFoundError(f"OFF run directory '{off_dir}' does not exist.")
    if not on_dir.exists():
        raise FileNotFoundError(f"ON run directory '{on_dir}' does not exist.")

    off_summary = _load_run_summary(off_dir)
    on_summary = _load_run_summary(on_dir)
    rows = _build_effect_rows(off_summary, on_summary, args.label_off, args.label_on)

    headers = [
        "prewhiten_mode",
        "detection_rate",
        "delta_mse_ew",
        "delta_mse_mv",
        "es95_error_ew",
        "es95_error_mv",
        "sign_p_ew",
        "sign_p_mv",
    ]
    out_path = args.out.resolve() if args.out else (on_dir / "prewhiten_effect.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[prewhiten_effect] Wrote {out_path}")
    if args.mirror:
        mirror_path = off_dir / out_path.name
        with mirror_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"[prewhiten_effect] Mirrored to {mirror_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
