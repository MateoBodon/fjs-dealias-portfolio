from __future__ import annotations

import json
from pathlib import Path

from experiments.synthetic.calibrate_thresholds import _build_defaults_payload


def test_build_defaults_payload_filters_and_formats(tmp_path: Path) -> None:
    thresholds = {
        "scm": {
            "G36": {
                "r12-16": {
                    "p64-96": {
                        "delta": 0.35,
                        "delta_frac": 0.02,
                        "stability_eta_deg": 0.4,
                        "fpr": 0.015,
                        "replicates": 14,
                        "replicates_bin_bounds": (12.0, 16.0),
                        "p_bin_bounds": (64.0, 96.0),
                    }
                }
            }
        },
        "tyler": {
            "G36": {
                "r12-16": {
                    "p64-96": {
                        "delta": 0.45,
                        "delta_frac": 0.03,
                        "stability_eta_deg": 0.5,
                        "fpr": 0.03,
                        "replicates": 14,
                    }
                }
            }
        },
    }

    payload = _build_defaults_payload(thresholds, alpha=0.02, thresholds_path=Path("calibration/test.json"))
    assert payload["alpha"] == 0.02
    defaults = payload["defaults"]
    assert "scm" in defaults
    assert "tyler" not in defaults  # FPR filtered out
    entry = defaults["scm"]["G36"]["r12-16"]["p64-96"]
    assert entry["delta"] == 0.35
    assert entry["delta_frac"] == 0.02
    assert entry["stability_eta_deg"] == 0.4
    assert entry["fpr"] == 0.015
