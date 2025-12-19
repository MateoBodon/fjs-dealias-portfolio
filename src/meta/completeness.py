from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def _load_json(path: Path) -> dict[str, Any]:
    """Read JSON from ``path`` if it exists; otherwise return an empty mapping."""

    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


@dataclass(frozen=True, slots=True)
class CompletenessResult:
    label: str
    path: Path
    run_type: str
    present: bool
    is_complete: bool
    missing_files: list[str]
    incomplete_reason: str | None
    cap_active: bool
    cap_sources: list[str]
    window_coverage: float | None
    windows_evaluated: int | None
    windows_total: int | None

    @property
    def status(self) -> str:
        if not self.present:
            return "missing"
        if self.is_complete:
            return "complete"
        return "incomplete"

    @property
    def excluded_from_aggregate(self) -> bool:
        coverage_block = self.window_coverage is not None and self.window_coverage < 1.0
        return (not self.is_complete) or self.cap_active or coverage_block

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["excluded_from_aggregate"] = self.excluded_from_aggregate
        payload["path"] = str(self.path)
        return payload


def _window_stats_from_manifest(manifest: Mapping[str, Any]) -> tuple[int | None, int | None, float | None, bool, list[str], str | None]:
    windows_block = manifest.get("windows") if isinstance(manifest.get("windows"), Mapping) else {}
    evaluated = _coerce_int(_first(windows_block or manifest, ["windows_evaluated", "windows_completed", "windows_after_regime"]))
    total = _coerce_int(_first(windows_block or manifest, ["windows_total", "windows_requested"]))
    coverage = _first(windows_block or manifest, ["window_coverage", "coverage"])
    cap_active = bool(_first(windows_block or manifest, ["cap_active"]) or False)
    cap_sources = _first(windows_block or manifest, ["cap_sources", "cap_configured_sources"]) or []
    if not isinstance(cap_sources, list):
        cap_sources = [str(cap_sources)]
    if coverage is None and evaluated is not None and total:
        try:
            coverage = float(evaluated) / float(total) if total else None
        except ZeroDivisionError:
            coverage = None
    try:
        coverage = None if coverage is None else float(coverage)
    except (TypeError, ValueError):
        coverage = None
    reason = _first(windows_block or manifest, ["incomplete_reason"])
    if isinstance(reason, str) and not reason.strip():
        reason = None
    return evaluated, total, coverage, cap_active, list(cap_sources), reason


def evaluate_eval_run(
    run_dir: Path,
    *,
    label: str | None = None,
    require_manifest: bool = True,
    allow_unknown_coverage: bool = False,
    run_type: str = "daily",
) -> CompletenessResult:
    """Assess completeness for a daily overlay evaluation (rc-lite/rc) run directory."""

    label = label or run_dir.name
    present = run_dir.exists()
    missing_files: list[str] = []

    manifest = _load_json(run_dir / "run_manifest.json")
    manifest_source = "run_manifest.json"
    if not manifest:
        manifest = _load_json(run_dir / "run.json")
        manifest_source = "run.json"
    if not manifest and require_manifest:
        missing_files.append(manifest_source)

    metrics_present = any(run_dir.rglob("metrics.csv"))
    diagnostics_present = any(run_dir.rglob("diagnostics.csv"))
    if not metrics_present:
        missing_files.append("metrics.csv")
    if not diagnostics_present:
        missing_files.append("diagnostics.csv")

    evaluated, total, coverage, cap_active, cap_sources, manifest_reason = _window_stats_from_manifest(manifest)
    incomplete_reason = manifest_reason
    if not present:
        incomplete_reason = "run directory missing"
    elif missing_files and not incomplete_reason:
        incomplete_reason = f"missing files: {', '.join(sorted(set(missing_files)))}"
    elif coverage is None and not allow_unknown_coverage and not incomplete_reason and (manifest or metrics_present or diagnostics_present):
        incomplete_reason = "window_coverage_unknown"
    elif coverage is not None and coverage < 1.0 and not incomplete_reason:
        incomplete_reason = f"window_coverage<{coverage:.3g}"

    is_complete = not missing_files and not incomplete_reason

    return CompletenessResult(
        label=label,
        path=run_dir,
        run_type=run_type,
        present=present,
        is_complete=is_complete,
        missing_files=missing_files,
        incomplete_reason=incomplete_reason,
        cap_active=cap_active,
        cap_sources=cap_sources,
        window_coverage=coverage,
        windows_evaluated=evaluated,
        windows_total=total,
    )


def _locate_payload_dir(base: Path) -> Path:
    """Return the most likely payload directory (handles tagged weekly outputs)."""

    direct_hits = ("detection_summary.csv", "summary.json", "metrics_summary.csv", "run_meta.json")
    if any((base / name).exists() for name in direct_hits):
        return base
    candidates = [child for child in base.iterdir() if child.is_dir()]
    for child in candidates:
        if any((child / name).exists() for name in direct_hits):
            return child
    return base


def evaluate_weekly_run(run_dir: Path, *, label: str | None = None) -> CompletenessResult:
    """Assess completeness for a weekly equity_panel run directory."""

    label = label or run_dir.name
    present = run_dir.exists()
    payload_dir = _locate_payload_dir(run_dir) if present else run_dir

    missing_files: list[str] = []
    summary_path = payload_dir / "summary.json"
    det_path = payload_dir / "detection_summary.csv"
    metrics_path = payload_dir / "metrics_summary.csv"

    if not summary_path.exists():
        missing_files.append("summary.json")
    if not det_path.exists():
        missing_files.append("detection_summary.csv")
    if not metrics_path.exists():
        missing_files.append("metrics_summary.csv")

    summary = _load_json(summary_path)
    balanced_weeks = _coerce_int(summary.get("balanced_weeks"))
    window_weeks = _coerce_int(summary.get("window_weeks"))
    horizon_weeks = _coerce_int(summary.get("horizon_weeks"))
    windows_evaluated = _coerce_int(summary.get("rolling_windows_evaluated"))
    cap_active = bool(summary.get("cap_active", False))
    cap_sources = summary.get("cap_sources") or []
    if not isinstance(cap_sources, list):
        cap_sources = [str(cap_sources)]

    expected_total = None
    if balanced_weeks is not None and window_weeks is not None and horizon_weeks is not None:
        expected_total = max(balanced_weeks - window_weeks - horizon_weeks + 1, 0)
    coverage = None
    if expected_total and windows_evaluated is not None:
        try:
            coverage = float(windows_evaluated) / float(expected_total) if expected_total else None
        except ZeroDivisionError:
            coverage = None

    incomplete_reason = None
    if not present:
        incomplete_reason = "run directory missing"
    elif missing_files:
        incomplete_reason = f"missing files: {', '.join(sorted(set(missing_files)))}"
    elif expected_total is None or windows_evaluated is None:
        incomplete_reason = "window_coverage_unknown"
    elif coverage is not None and coverage < 1.0:
        incomplete_reason = f"window_coverage<{coverage:.3g}"

    is_complete = not missing_files and not incomplete_reason

    return CompletenessResult(
        label=label,
        path=run_dir,
        run_type="weekly",
        present=present,
        is_complete=is_complete,
        missing_files=missing_files,
        incomplete_reason=incomplete_reason,
        cap_active=cap_active,
        cap_sources=list(cap_sources),
        window_coverage=coverage,
        windows_evaluated=windows_evaluated,
        windows_total=expected_total,
    )
