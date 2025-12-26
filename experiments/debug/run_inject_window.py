#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from fjs.overlay import OverlayConfig, detect_spikes


def _load_metadata(raw: np.ndarray) -> dict:
    try:
        text = str(raw.item())
    except Exception:
        text = str(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _overlay_from_metadata(payload: dict) -> OverlayConfig:
    config_raw = payload.get("overlay_config", {}) if isinstance(payload, dict) else {}
    allowed = {field.name for field in fields(OverlayConfig)}
    filtered = {key: value for key, value in config_raw.items() if key in allowed}
    return OverlayConfig(**filtered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detect_spikes on a saved debug window")
    parser.add_argument("--input", type=Path, required=True, help="Path to debug_window.npz")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Missing debug window: {args.input}")

    data = np.load(args.input, allow_pickle=True)
    matrix = data["matrix"]
    labels = data["group_labels"]
    payload = _load_metadata(data["metadata"])
    cfg = _overlay_from_metadata(payload)

    stats: dict = {}
    detections = detect_spikes(matrix, labels, config=cfg, stats=stats)

    print("detections", len(detections))
    print("gating", json.dumps(stats.get("gating", {}), indent=2, sort_keys=True, default=str))
    print("diagnostics", json.dumps(stats.get("diagnostics", {}), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
