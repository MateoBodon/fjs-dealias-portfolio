"""
Configuration helpers for de-aliasing detection and overlay defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Sequence

import yaml

__all__ = [
    "CalibratedThresholds",
    "DetectionSettings",
    "get_detection_settings",
    "override_detection_settings",
    "load_detection_settings",
]


@dataclass(frozen=True)
class CalibratedThresholds:
    """Threshold summary parsed from calibration outputs."""

    min_margin: float
    fpr: float | None = None
    tpr: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionSettings:
    """Resolved detection defaults used throughout the pipeline."""

    delta: float = 0.5
    delta_frac: float = 0.0
    a_grid_size: int = 120
    min_margin: float = 0.05
    min_isolation: float = 0.05
    require_isolated: bool = True
    angle_min_cos: float = 0.05
    t_eps: float = 0.06
    q_max: int = 1
    cs_drop_top_frac: float = 0.05
    edge_mode: str = "tyler"
    huber_c: float = 2.5
    rng_seed: int | None = None
    off_component_cap: float | None = 0.3

    def with_overrides(self, **kwargs: object) -> "DetectionSettings":
        data = self.__dict__ | kwargs
        return DetectionSettings(**data)


_DETECTION_CACHE: DetectionSettings | None = None


def get_detection_settings(force_reload: bool = False) -> DetectionSettings:
    """Return cached detection settings, reloading from disk when requested."""

    global _DETECTION_CACHE
    if force_reload or _DETECTION_CACHE is None:
        _DETECTION_CACHE = load_detection_settings()
    return _DETECTION_CACHE


def override_detection_settings(settings: DetectionSettings | None) -> None:
    """Override the cached detection settings (primarily for tests)."""

    global _DETECTION_CACHE
    _DETECTION_CACHE = settings


def load_detection_settings(
    *,
    config_path: Path | None = None,
    thresholds_paths: Sequence[Path] | None = None,
) -> DetectionSettings:
    """
    Load detection defaults from YAML plus calibration thresholds when present.
    """

    defaults = DetectionSettings()
    yaml_path = config_path or Path("configs/detection.yaml")
    config_data = _read_yaml_dict(yaml_path)

    threshold_candidates: list[Path] = []
    if thresholds_paths:
        threshold_candidates.extend(thresholds_paths)

    raw_thresholds = config_data.get("thresholds_path")
    if isinstance(raw_thresholds, str):
        threshold_candidates.append(Path(raw_thresholds))
    elif isinstance(raw_thresholds, Iterable):
        threshold_candidates.extend(Path(item) for item in raw_thresholds)

    threshold_candidates.extend(_default_threshold_candidates())
    thresholds = _load_thresholds(threshold_candidates)

    merged = defaults.__dict__ | {
        key: config_data.get(key, getattr(defaults, key))
        for key in defaults.__dict__.keys()
    }

    if thresholds is not None:
        merged["min_margin"] = thresholds.min_margin

    # Normalise types
    merged["delta"] = float(merged["delta"])
    merged["delta_frac"] = float(merged["delta_frac"])
    merged["a_grid_size"] = int(merged["a_grid_size"])
    merged["min_margin"] = float(merged["min_margin"])
    merged["min_isolation"] = float(merged["min_isolation"])
    merged["require_isolated"] = bool(merged["require_isolated"])
    merged["angle_min_cos"] = float(merged["angle_min_cos"])
    merged["t_eps"] = float(merged["t_eps"])
    merged["q_max"] = int(merged["q_max"])
    merged["cs_drop_top_frac"] = float(merged["cs_drop_top_frac"])
    merged["edge_mode"] = str(merged["edge_mode"]).strip().lower()
    merged["huber_c"] = float(merged["huber_c"])
    rng_seed = merged.get("rng_seed")
    if rng_seed is not None:
        merged["rng_seed"] = int(rng_seed)
    if merged.get("off_component_cap") is not None:
        merged["off_component_cap"] = float(merged["off_component_cap"])

    return DetectionSettings(**merged)


def _read_yaml_dict(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Detection config at {path} must be a mapping.")
    return loaded


def _default_threshold_candidates() -> list[Path]:
    candidates = [Path("reports/calibration/thresholds.json")]
    reports_root = Path("reports")
    if reports_root.exists():
        rc_paths = sorted(
            reports_root.glob("rc-*/calibration/thresholds.json"),
            reverse=True,
        )
        candidates.extend(rc_paths)
    return candidates


def _load_thresholds(candidates: Sequence[Path]) -> CalibratedThresholds | None:
    for path in candidates:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            recommended = payload.get("recommended", {})
            margin = recommended.get("min_margin")
            if margin is None:
                margin_grid = payload.get("margin_grid") or []
                if margin_grid:
                    margin = float(margin_grid[0])
            if margin is None:
                continue
            fpr = recommended.get("fpr")
            tpr = recommended.get("tpr", {})
            return CalibratedThresholds(
                min_margin=float(margin),
                fpr=(None if fpr is None else float(fpr)),
                tpr={str(k): float(v) for k, v in tpr.items() if _is_number(v)},
            )
    return None


def _is_number(value: object) -> bool:
    try:
        float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return True
