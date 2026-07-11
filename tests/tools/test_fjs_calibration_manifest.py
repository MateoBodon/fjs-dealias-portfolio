from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from experiments.synthetic.harness_utils import simulate_panel
from tools.fjs_m4_contract import (
    code_input_fingerprint,
    exact_binomial_interval_95,
    stable_sha256,
)

from tools import freeze_fjs_m4_manifest, run_fjs_calibration_manifest


def _write_smoke_manifest(path: Path) -> dict:
    freeze_fjs_m4_manifest.main(
        [
            "--profile",
            "smoke",
            "--out",
            str(path),
            "--seed-base",
            "9000",
            "--limit-cells",
            "2",
        ]
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _fake_checkpoint(
    task: dict, *, exec_mode: str = "throughput", workers: int = 1
) -> dict:
    manifest = task["manifest"]
    cell = task["cell"]
    stable_payload = {
        "manifest_id": manifest["manifest_id"],
        "cell_id": str(cell["cell_id"]),
        "cell_spec_digest": run_fjs_calibration_manifest._cell_spec_digest(
            manifest, cell
        ),
        "threshold_entry": {
            "delta_frac": 0.01,
            "stability_eta_deg": 0.3,
            "fpr": 0.05,
            "power": None,
        },
        "curve": [
            {
                "mu": 0.0,
                "inject_mode": "between",
                "detection_rate": 0.0,
                "acceptance_rate": 0.0,
                "n_windows": int(task["trials_null"]),
                "n_detected": 0,
                "n_accepted": 0,
                "detection_ci_low": 0.0,
                "detection_ci_high": 0.5,
            },
            {
                "mu": float(cell["power_mu"]),
                "inject_mode": "between",
                "detection_rate": 1.0,
                "acceptance_rate": 1.0,
                "n_windows": int(task["trials_null"]),
                "n_detected": int(task["trials_null"]),
                "n_accepted": int(task["trials_null"]),
                "detection_ci_low": 0.5,
                "detection_ci_high": 1.0,
            },
        ],
        "trial_rows": [],
        "gate_metrics": {
            "null_detection_rate": 0.0,
            "null_detection_ci_low": 0.0,
            "null_detection_ci_high": 0.5,
            "strong_detection_rate": 1.0,
            "strong_acceptance_rate": 1.0,
            "detection_gain": 1.0,
            "monotone_detection": True,
            "monotone_acceptance": True,
            "planted_component_accept_share": 1.0,
            "nuisance_component_accept_share": 0.0,
            "acceptance_exceeds_detection": False,
        },
    }
    return {
        "checkpoint_meta": {
            "manifest_id": manifest["manifest_id"],
            "manifest_digest": manifest["manifest_digest"],
            "expected_cell_set_digest": manifest["expected_cell_set_digest"],
            "cell_id": str(cell["cell_id"]),
            "cell_spec_digest": run_fjs_calibration_manifest._cell_spec_digest(
                manifest, cell
            ),
            "trials_null": int(task["trials_null"]),
            "trials_alt": int(task["trials_alt"]),
            "delta_frac_grid": list(manifest["sweep"]["delta_frac_grid"]),
            "stability_grid": list(manifest["sweep"]["stability_grid"]),
            "code_tree_sha": run_fjs_calibration_manifest.git_tree_sha(),
            "code_input_fingerprint_sha256": code_input_fingerprint()["sha256"],
            "environment_fingerprint_sha256": (
                run_fjs_calibration_manifest.environment_fingerprint(
                    exec_mode, workers
                )["sha256"]
            ),
            "stable_payload_sha256": stable_sha256(stable_payload),
        },
        "stable_payload": stable_payload,
    }


def test_freeze_manifest_profiles_have_distinct_ids(tmp_path: Path) -> None:
    full_path = tmp_path / "full.json"
    smoke_path = tmp_path / "smoke.json"
    freeze_fjs_m4_manifest.main(["--profile", "full", "--out", str(full_path)])
    freeze_fjs_m4_manifest.main(["--profile", "smoke", "--out", str(smoke_path)])
    full = json.loads(full_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    assert full["manifest_id"] == "fjs-m4-full-target-between-v2"
    assert smoke["manifest_id"] == "fjs-m4-smoke-target-between-v2"
    assert full["purpose"] != smoke["purpose"]
    assert full["predeclaration_contract"]["nominal_size"] == 0.05
    assert smoke["predeclaration_contract"]["power_boundary_multiplier"] == 1.5
    assert full["execution_readiness"]["full_execution_ready"] is False
    assert full["execution_readiness"]["aws_execution_authorized"] is False


def test_exact_binomial_interval_has_correct_two_sided_upper_bound() -> None:
    zero_low, zero_high = exact_binomial_interval_95(0, 6)
    one_low, one_high = exact_binomial_interval_95(1, 6)
    all_low, all_high = exact_binomial_interval_95(6, 6)
    assert zero_low == 0.0
    assert zero_high == pytest.approx(1.0 - 0.025 ** (1.0 / 6.0), abs=1e-12)
    assert one_low == pytest.approx(0.0042107445, abs=1e-9)
    assert one_high == pytest.approx(0.6412345789, abs=1e-9)
    assert all_low == pytest.approx(0.025 ** (1.0 / 6.0), abs=1e-12)
    assert all_high == 1.0


def test_simulate_panel_returns_exact_planted_directions() -> None:
    kwargs = {
        "n_assets": 5,
        "n_groups": 8,
        "replicates": 2,
        "spike_strength": 1.5,
        "noise_variance": 1.0,
        "signal_to_noise": 0.35,
    }
    observations, groups = simulate_panel(np.random.default_rng(123), **kwargs)
    with_truth = simulate_panel(np.random.default_rng(123), **kwargs, return_dirs=True)
    truth_observations, truth_groups, planted, nuisance = with_truth
    assert np.array_equal(observations, truth_observations)
    assert np.array_equal(groups, truth_groups)
    assert np.linalg.norm(planted) == pytest.approx(1.0)
    assert np.linalg.norm(nuisance) == pytest.approx(1.0)


def test_full_manifest_execution_fails_closed_before_run_root(tmp_path: Path) -> None:
    manifest_path = tmp_path / "full.json"
    freeze_fjs_m4_manifest.main(["--profile", "full", "--out", str(manifest_path)])
    with pytest.raises(RuntimeError, match="Full M4 execution is fail-closed"):
        run_fjs_calibration_manifest.main(
            [
                "--manifest",
                str(manifest_path),
                "--run-id",
                "blocked_full",
                "--run-root-base",
                str(tmp_path / "runs"),
            ]
        )
    assert not (tmp_path / "runs" / "blocked_full").exists()


def test_smoke_artifacts_cannot_be_written_inside_repository(tmp_path: Path) -> None:
    manifest_path = tmp_path / "smoke.json"
    _write_smoke_manifest(manifest_path)
    with pytest.raises(ValueError, match="Smoke artifacts must remain outside"):
        run_fjs_calibration_manifest.main(
            [
                "--manifest",
                str(manifest_path),
                "--run-id",
                "blocked_smoke",
                "--run-root-base",
                str(run_fjs_calibration_manifest.ROOT / "reports" / "synthetic"),
            ]
        )


def test_runner_resume_matches_fresh_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    run_base = tmp_path / "runs"
    payload = _write_smoke_manifest(manifest_path)

    def fake_run_single_cell(task: dict) -> dict:
        return _fake_checkpoint(task)

    monkeypatch.setattr(
        run_fjs_calibration_manifest, "_run_single_cell", fake_run_single_cell
    )

    with pytest.raises(RuntimeError, match="Intentional interruption"):
        run_fjs_calibration_manifest.main(
            [
                "--manifest",
                str(manifest_path),
                "--run-id",
                "resume_case",
                "--run-root-base",
                str(run_base),
                "--workers",
                "1",
                "--interrupt-after-completions",
                "1",
            ]
        )

    resumed_thresholds = tmp_path / "resumed_thresholds.json"
    resumed_defaults = tmp_path / "resumed_defaults.json"
    run_fjs_calibration_manifest.main(
        [
            "--manifest",
            str(manifest_path),
            "--run-id",
            "resume_case",
            "--run-root-base",
            str(run_base),
            "--workers",
            "1",
            "--resume",
            "--out",
            str(resumed_thresholds),
            "--defaults-out",
            str(resumed_defaults),
        ]
    )

    fresh_thresholds = tmp_path / "fresh_thresholds.json"
    fresh_defaults = tmp_path / "fresh_defaults.json"
    run_fjs_calibration_manifest.main(
        [
            "--manifest",
            str(manifest_path),
            "--run-id",
            "fresh_case",
            "--run-root-base",
            str(run_base),
            "--workers",
            "1",
            "--out",
            str(fresh_thresholds),
            "--defaults-out",
            str(fresh_defaults),
        ]
    )

    resumed_payload = json.loads(resumed_thresholds.read_text(encoding="utf-8"))
    fresh_payload = json.loads(fresh_thresholds.read_text(encoding="utf-8"))
    assert resumed_payload["stable"] == fresh_payload["stable"]
    assert (
        resumed_payload["stable"]["stable_reducer_sha256"]
        == fresh_payload["stable"]["stable_reducer_sha256"]
    )

    summary = json.loads(
        (run_base / "resume_case" / "run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["completed_cells"] == len(payload["cells"])


def test_runner_rejects_manifest_mismatch_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    run_base = tmp_path / "runs"
    payload = _write_smoke_manifest(manifest_path)
    monkeypatch.setattr(
        run_fjs_calibration_manifest,
        "_run_single_cell",
        lambda task: _fake_checkpoint(task),
    )

    run_fjs_calibration_manifest.main(
        [
            "--manifest",
            str(manifest_path),
            "--run-id",
            "mismatch_case",
            "--run-root-base",
            str(run_base),
            "--workers",
            "1",
        ]
    )

    payload["cells"][0]["power_mu"] = float(payload["cells"][0]["power_mu"]) + 0.25
    payload["cell_digests"][payload["cells"][0]["cell_id"]] = stable_sha256(
        payload["cells"][0]
    )
    payload["expected_cell_set_digest"] = stable_sha256(
        [
            {"cell_id": key, "sha256": payload["cell_digests"][key]}
            for key in sorted(payload["cell_digests"])
        ]
    )
    payload["manifest_digest"] = stable_sha256(
        {k: v for k, v in payload.items() if k != "manifest_digest"}
    )
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Checkpoint mismatch"):
        run_fjs_calibration_manifest.main(
            [
                "--manifest",
                str(manifest_path),
                "--run-id",
                "mismatch_case",
                "--run-root-base",
                str(run_base),
                "--workers",
                "1",
                "--resume",
            ]
        )


def test_runner_rejects_extra_stale_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    run_base = tmp_path / "runs"
    _write_smoke_manifest(manifest_path)
    monkeypatch.setattr(
        run_fjs_calibration_manifest,
        "_run_single_cell",
        lambda task: _fake_checkpoint(task),
    )
    run_fjs_calibration_manifest.main(
        [
            "--manifest",
            str(manifest_path),
            "--run-id",
            "contam_case",
            "--run-root-base",
            str(run_base),
            "--workers",
            "1",
        ]
    )
    stale = run_base / "contam_case" / "cells" / "stale_extra.json"
    stale.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Out-of-scope stale cell checkpoint"):
        run_fjs_calibration_manifest.main(
            [
                "--manifest",
                str(manifest_path),
                "--run-id",
                "contam_case",
                "--run-root-base",
                str(run_base),
                "--workers",
                "1",
                "--resume",
            ]
        )


def test_zero_overrides_remain_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    run_base = tmp_path / "runs"
    _write_smoke_manifest(manifest_path)
    seen: dict[str, int] = {}

    def fake(task: dict) -> dict:
        seen["trials_alt"] = int(task["trials_alt"])
        return _fake_checkpoint(task)

    monkeypatch.setattr(run_fjs_calibration_manifest, "_run_single_cell", fake)
    run_fjs_calibration_manifest.main(
        [
            "--manifest",
            str(manifest_path),
            "--run-id",
            "zero_case",
            "--run-root-base",
            str(run_base),
            "--workers",
            "1",
            "--trials-alt-override",
            "0",
        ]
    )
    assert seen["trials_alt"] == 0
