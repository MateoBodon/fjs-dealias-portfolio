#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.synthetic.harness_utils import (  # noqa: E402
    simulate_panel,
    write_run_metadata,
)
from tools.fjs_m4_contract import (  # noqa: E402
    code_input_fingerprint,
    environment_fingerprint,
    exact_binomial_interval_95,
    git_tree_sha,
    stable_json_dumps,
    stable_sha256,
)

from fjs.detector_contract import candidate_source_counts  # noqa: E402
from fjs.invariance_contract import (  # noqa: E402
    InvarianceTolerances,
    assess_invariance,
    build_detector_signature,
    deterministic_asset_permutation,
    deterministic_group_label_permutation,
    deterministic_rescaling,
    deterministic_row_permutation,
    standardize_columns,
)
from fjs.overlay import OverlayConfig, detect_spikes  # noqa: E402
from meta import runtime  # noqa: E402
from synthetic.calibration import (  # noqa: E402
    CalibrationConfig,
    calibrate_thresholds,
)


@dataclass(frozen=True)
class RunPaths:
    run_root: Path
    cells_dir: Path
    logs_dir: Path
    progress_log: Path
    reducer_path: Path
    summary_path: Path
    metadata_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Run the frozen FJS calibration manifest with fail-closed checkpoint/restart."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument(
        "--run-root-base",
        type=Path,
        default=ROOT / "reports" / "synthetic" / "calib",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--exec-mode", choices=["deterministic", "throughput"], default="throughput"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--trials-null-override", type=int, default=None)
    parser.add_argument("--trials-alt-override", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--defaults-out", type=Path, default=None)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--scratch-root", type=Path, default=None)
    parser.add_argument("--cell-timeout-seconds", type=int, default=1800)
    parser.add_argument("--interrupt-after-completions", type=int, default=None)
    parser.add_argument(
        "--instance-hourly-usd",
        type=float,
        default=None,
        help=(
            "Optional fresh authoritative on-demand hourly rate used only for "
            "a completed full-run cost receipt; never inferred from smoke timing."
        ),
    )
    parser.add_argument(
        "--worker-cell-id", type=str, default=None, help=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "manifest_id",
        "manifest_digest",
        "cells",
        "cell_digests",
        "expected_cell_set_digest",
        "sweep",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Manifest missing required keys: {missing}")
    actual_digest = stable_sha256(
        {k: v for k, v in payload.items() if k != "manifest_digest"}
    )
    if actual_digest != payload["manifest_digest"]:
        raise ValueError("Manifest digest mismatch.")
    schema_version = int(payload["schema_version"])
    if schema_version not in {2, 3}:
        raise ValueError(f"Unsupported FJS calibration schema {schema_version}.")
    if schema_version == 3:
        from tools.fjs_m4_contract_v3 import validate_manifest_v3

        validate_manifest_v3(payload)
    return payload


def _scoped_cells(
    manifest: Mapping[str, Any], max_cells: int | None
) -> list[dict[str, Any]]:
    cells = list(manifest["cells"])
    if max_cells is not None:
        cells = cells[: max(0, int(max_cells))]
    return cells


def _paths(args: argparse.Namespace) -> RunPaths:
    run_root = args.run_root_base.expanduser().resolve() / str(args.run_id)
    return RunPaths(
        run_root=run_root,
        cells_dir=run_root / "cells",
        logs_dir=run_root / "logs",
        progress_log=run_root / "progress.jsonl",
        reducer_path=run_root / "reducer_payload.json",
        summary_path=run_root / "run_summary.json",
        metadata_path=run_root / "run.json",
    )


def _output_paths(args: argparse.Namespace, paths: RunPaths) -> tuple[Path, Path]:
    thresholds = (
        args.out.expanduser().resolve()
        if args.out is not None
        else (paths.run_root / "thresholds.json")
    )
    defaults = (
        args.defaults_out.expanduser().resolve()
        if args.defaults_out is not None
        else (paths.run_root / "defaults.json")
    )
    return thresholds, defaults


def _append_progress(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def _safe_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(stable_json_dumps(payload) + "\n", encoding="utf-8")
    tmp.replace(path)


def _cell_spec_digest(manifest: Mapping[str, Any], cell: Mapping[str, Any]) -> str:
    return str(manifest["cell_digests"][str(cell["cell_id"])])


def _scope_cell_set_digest(
    manifest: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]
) -> str:
    return stable_sha256(
        [
            {
                "cell_id": str(cell["cell_id"]),
                "sha256": _cell_spec_digest(manifest, cell),
            }
            for cell in sorted(cells, key=lambda item: str(item["cell_id"]))
        ]
    )


def _cell_checkpoint_valid(
    checkpoint_path: Path,
    *,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    trials_null: int,
    trials_alt: int,
    exec_mode: str,
    workers: int,
) -> bool:
    if not checkpoint_path.exists():
        return False
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    meta = payload.get("checkpoint_meta", {})
    required = {
        "manifest_id": manifest["manifest_id"],
        "manifest_digest": manifest["manifest_digest"],
        "expected_cell_set_digest": manifest["expected_cell_set_digest"],
        "cell_id": str(cell["cell_id"]),
        "cell_spec_digest": _cell_spec_digest(manifest, cell),
        "trials_null": int(trials_null),
        "trials_alt": int(trials_alt),
        "delta_frac_grid": list(manifest["sweep"]["delta_frac_grid"]),
        "stability_grid": list(manifest["sweep"]["stability_grid"]),
        "code_tree_sha": git_tree_sha(),
        "code_input_fingerprint_sha256": code_input_fingerprint()["sha256"],
        "environment_fingerprint_sha256": environment_fingerprint(exec_mode, workers)[
            "sha256"
        ],
    }
    for key, value in required.items():
        if meta.get(key) != value:
            raise ValueError(f"Checkpoint mismatch for {checkpoint_path.name}: {key}")
    stable_payload = payload.get("stable_payload")
    if stable_payload is None:
        raise ValueError(f"Checkpoint missing stable_payload: {checkpoint_path}")
    if stable_sha256(stable_payload) != meta.get("stable_payload_sha256"):
        raise ValueError(f"Checkpoint payload hash mismatch: {checkpoint_path}")
    return True


def _build_overlay_config(
    cell: Mapping[str, Any], threshold: Mapping[str, Any]
) -> OverlayConfig:
    return OverlayConfig(
        q_max=2,
        delta=float(cell["delta_abs"]),
        delta_frac=float(threshold["delta_frac"]),
        eps=0.02,
        stability_eta_deg=float(threshold["stability_eta_deg"]),
        a_grid=120,
        require_isolated=True,
        off_component_cap=0.3,
        edge_mode=str(cell["edge_mode"]),
    )


def _detector_signature_for_panel(
    observations: np.ndarray,
    groups: np.ndarray,
    *,
    config: OverlayConfig,
    direction_indexer: np.ndarray | None = None,
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    accepted = detect_spikes(observations, groups, config=config, stats=stats)
    return build_detector_signature(
        stats.get("pre_gate", {}),
        accepted,
        direction_indexer=direction_indexer,
    )


def _evaluate_invariance_for_panel(
    observations: np.ndarray,
    groups: np.ndarray,
    *,
    config: OverlayConfig,
    invariance_seed: int,
    tolerances: InvarianceTolerances,
) -> dict[str, Any]:
    reference = _detector_signature_for_panel(
        observations,
        groups,
        config=config,
    )
    row_order = deterministic_row_permutation(observations.shape[0])
    asset_order = deterministic_asset_permutation(
        observations.shape[1], invariance_seed
    )
    inverse_asset_order = np.argsort(asset_order)
    standardized = standardize_columns(observations)
    standardized_rescaled = standardize_columns(
        observations * deterministic_rescaling(observations.shape[1])
    )
    standardized_error = float(np.max(np.abs(standardized - standardized_rescaled)))
    comparisons = {
        "standardized_rescaling": (
            _detector_signature_for_panel(standardized, groups, config=config),
            _detector_signature_for_panel(
                standardized_rescaled,
                groups,
                config=config,
            ),
        ),
        "deterministic_row_order": (
            reference,
            _detector_signature_for_panel(
                observations[row_order],
                groups[row_order],
                config=config,
            ),
        ),
        "asset_permutation": (
            reference,
            _detector_signature_for_panel(
                observations[:, asset_order],
                groups,
                config=config,
                direction_indexer=inverse_asset_order,
            ),
        ),
        "group_label_permutation": (
            reference,
            _detector_signature_for_panel(
                observations,
                deterministic_group_label_permutation(groups),
                config=config,
            ),
        ),
    }
    assessment = assess_invariance(comparisons, tolerances=tolerances)
    assessment["standardized_matrix_max_abs_error"] = standardized_error
    if standardized_error > tolerances.standardized_matrix_atol:
        check = assessment["checks"]["standardized_rescaling"]
        check["passed"] = False
        check["reasons"].append("standardized_matrix_outside_tolerance")
        assessment["passed"] = False
        if "standardized_rescaling" not in assessment["failed_checks"]:
            assessment["failed_checks"].insert(0, "standardized_rescaling")
    return assessment


def _evaluate_invariance_checks(
    *,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    threshold: Mapping[str, Any],
) -> dict[str, Any]:
    if int(manifest["schema_version"]) != 3:
        raise ValueError("Invariance execution is available only for schema v3.")
    contract = manifest["invariance_contract"]
    tolerances = InvarianceTolerances.from_mapping(contract["tolerances"])
    config = _build_overlay_config(cell, threshold)
    seed = int(cell["invariance_seed"])
    evaluations = []
    for role, mu in (
        ("null", 0.0),
        ("power", float(cell["power_mu"])),
    ):
        observations, groups, _, _ = simulate_panel(
            np.random.default_rng(seed),
            n_assets=int(cell["p_assets"]),
            n_groups=int(cell["n_groups"]),
            replicates=int(cell["replicates"]),
            spike_strength=float(mu),
            noise_variance=1.0,
            signal_to_noise=0.35,
            return_dirs=True,
        )
        evaluations.append(
            {
                "role": role,
                "mu": float(mu),
                "seed": seed,
                "assessment": _evaluate_invariance_for_panel(
                    observations,
                    groups,
                    config=config,
                    invariance_seed=seed,
                    tolerances=tolerances,
                ),
            }
        )
    failed_roles = [
        item["role"] for item in evaluations if not item["assessment"]["passed"]
    ]
    return {
        "contract_id": str(contract["contract_id"]),
        "contract_sha256": str(contract["sha256"]),
        "passed": not failed_roles,
        "failed_roles": failed_roles,
        "evaluations": evaluations,
    }


def _evaluate_trials(
    *,
    cell: Mapping[str, Any],
    threshold: Mapping[str, Any],
    mu_values: Sequence[float],
    trial_counts: Mapping[float, int],
    seed_base: int,
) -> list[dict[str, Any]]:
    cfg = _build_overlay_config(cell, threshold)
    rows: list[dict[str, Any]] = []
    for mu in mu_values:
        trial_count = int(trial_counts.get(float(mu), 0))
        for trial_index in range(trial_count):
            # Paired seeds keep the noise, planted direction, and nuisance
            # direction fixed across mu for a given trial index.
            rng = np.random.default_rng(seed_base + trial_index)
            observations, groups, planted_direction, nuisance_direction = (
                simulate_panel(
                    rng,
                    n_assets=int(cell["p_assets"]),
                    n_groups=int(cell["n_groups"]),
                    replicates=int(cell["replicates"]),
                    spike_strength=float(mu),
                    noise_variance=1.0,
                    signal_to_noise=0.35,
                    return_dirs=True,
                )
            )
            stats: dict[str, Any] = {}
            accepted = detect_spikes(observations, groups, config=cfg, stats=stats)
            pre_gate = stats.get("pre_gate", {})
            detected = int(pre_gate.get("raw_outliers_found", 0) > 0)
            accepted_flag = int(len(accepted) > 0)
            sq_cos_values = []
            nuisance_sq_cos_values = []
            nuisance_count = 0
            for det in accepted:
                eigvec = np.asarray(det["eigvec"], dtype=np.float64)
                sq_cos = float(np.dot(eigvec, planted_direction) ** 2)
                nuisance_sq_cos = float(np.dot(eigvec, nuisance_direction) ** 2)
                sq_cos_values.append(sq_cos)
                nuisance_sq_cos_values.append(nuisance_sq_cos)
                if nuisance_sq_cos >= 0.80:
                    nuisance_count += 1
            accepted_sources = candidate_source_counts(accepted)
            non_fjs_count = sum(
                int(count)
                for source, count in accepted_sources.items()
                if source != "fjs"
            )
            rows.append(
                {
                    "mu": float(mu),
                    "trial_index": int(trial_index),
                    "detected": detected,
                    "accepted": accepted_flag,
                    "candidate_source_counts_pre_gate": pre_gate.get(
                        "candidate_sources", ""
                    ),
                    "candidate_source_counts_accepted": accepted_sources,
                    "accepted_count": int(len(accepted)),
                    "direction_squared_cosine_max": max(sq_cos_values)
                    if sq_cos_values
                    else 0.0,
                    "direction_squared_cosine_mean": (
                        float(np.mean(sq_cos_values)) if sq_cos_values else 0.0
                    ),
                    "direction_squared_cosine_sum": float(sum(sq_cos_values)),
                    "nuisance_squared_cosine_max": (
                        max(nuisance_sq_cos_values) if nuisance_sq_cos_values else 0.0
                    ),
                    "planted_component_hit": int(
                        any(value >= 0.80 for value in sq_cos_values)
                    ),
                    "nuisance_component_accept_count": int(nuisance_count),
                    "non_fjs_accept_count": non_fjs_count,
                }
            )
    return rows


def _aggregate_curve(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    mu_values = sorted({float(row["mu"]) for row in rows})
    curve = []
    for mu in mu_values:
        subset = [row for row in rows if float(row["mu"]) == mu]
        n = len(subset)
        detected = sum(int(row["detected"]) for row in subset)
        accepted = sum(int(row["accepted"]) for row in subset)
        lo, hi = exact_binomial_interval_95(detected, n)
        curve.append(
            {
                "mu": float(mu),
                "inject_mode": "between",
                "detection_rate": detected / n,
                "acceptance_rate": accepted / n,
                "n_windows": n,
                "n_detected": detected,
                "n_accepted": accepted,
                "detection_ci_low": lo,
                "detection_ci_high": hi,
            }
        )
    return curve


def _monotone(values: Sequence[float]) -> bool:
    return all(
        right + 1e-12 >= left for left, right in zip(values, values[1:], strict=False)
    )


def _assess_statistical_gates(
    cell: Mapping[str, Any], gate_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    nominal_size = float(cell["nominal_size"])
    checks = {
        "null_interval_contains_nominal_size": (
            float(gate_metrics["null_detection_ci_low"])
            <= nominal_size
            <= float(gate_metrics["null_detection_ci_high"])
        ),
        "null_interval_upper_within_limit": (
            float(gate_metrics["null_detection_ci_high"])
            <= float(cell["null_upper_bound_max"])
        ),
        "strong_detection_at_or_above_minimum": (
            float(gate_metrics["strong_detection_rate"])
            >= float(cell["power_detection_min"])
        ),
        "strong_acceptance_at_or_above_minimum": (
            float(gate_metrics["strong_acceptance_rate"])
            >= float(cell["power_acceptance_min"])
        ),
        "detection_gain_at_or_above_minimum": (
            float(gate_metrics["detection_gain"]) >= float(cell["power_gain_min"])
        ),
        "detection_curve_monotone": bool(gate_metrics["monotone_detection"]),
        "acceptance_curve_monotone": bool(gate_metrics["monotone_acceptance"]),
        "acceptance_never_exceeds_detection": not bool(
            gate_metrics["acceptance_exceeds_detection"]
        ),
        "direction_squared_cosine_at_or_above_minimum": (
            float(gate_metrics["direction_squared_cosine_mean"])
            >= float(cell["direction_squared_cosine_min"])
        ),
        "planted_component_share_at_or_above_minimum": (
            float(gate_metrics["planted_component_accept_share"])
            >= float(cell["planted_component_accept_share_min"])
        ),
        "nuisance_component_share_within_limit": (
            float(gate_metrics["nuisance_component_accept_share"])
            <= float(cell["nuisance_component_accept_share_max"])
        ),
        "accepted_candidates_are_fjs_only": (
            int(gate_metrics["non_fjs_accept_count"]) == 0
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "statistical_gate_pass": not failed,
        "checks": checks,
        "failed_checks": failed,
        "full_detector_gate_pass": False,
        "full_detector_gate_blockers": [
            "invariance_reducer_not_yet_hash_bound",
            "cell_specific_independent_detection_boundary_unbound",
            "real_design_cell_manifest_not_yet_bound",
        ],
    }


def _assess_cell_gates(
    *,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    gate_metrics: Mapping[str, Any],
    invariance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    statistical = _assess_statistical_gates(cell, gate_metrics)
    if int(manifest["schema_version"]) != 3:
        return statistical
    if invariance is None:
        raise ValueError("Schema v3 cells require an invariance assessment.")

    role_results = {
        str(item["role"]): bool(item["assessment"]["passed"])
        for item in invariance["evaluations"]
    }
    if set(role_results) != {"null", "power"}:
        raise ValueError("Schema v3 invariance must cover null and power roles.")
    checks = dict(statistical["checks"])
    checks.update(
        {
            "null_invariance_pass": role_results["null"],
            "power_invariance_pass": role_results["power"],
        }
    )
    failed_checks = [name for name, passed in checks.items() if not passed]
    local_scientific_pass = not failed_checks
    blockers = []
    if not statistical["statistical_gate_pass"]:
        blockers.append("statistical_detector_gate_failed")
    if not bool(invariance["passed"]):
        blockers.append("invariance_gate_failed")
    blockers.extend(
        str(item)
        for item in manifest.get("execution_readiness", {}).get("blockers", [])
    )
    return {
        "statistical_gate_pass": bool(statistical["statistical_gate_pass"]),
        "invariance_gate_pass": bool(invariance["passed"]),
        "local_scientific_gate_pass": local_scientific_pass,
        "checks": checks,
        "failed_checks": failed_checks,
        "full_detector_gate_pass": local_scientific_pass and not blockers,
        "full_detector_gate_blockers": blockers,
    }


def _stable_cell_payload(
    *,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    trials_null: int,
    trials_alt: int,
) -> dict[str, Any]:
    config = CalibrationConfig(
        p_assets=int(cell["p_assets"]),
        n_groups=int(cell["n_groups"]),
        replicates=int(cell["replicates"]),
        alpha=float(cell["nominal_size"]),
        trials_null=int(trials_null),
        trials_alt=0,
        delta_abs=float(cell["delta_abs"]),
        eps=float(manifest["sweep"]["eps"]),
        delta_frac_grid=tuple(float(v) for v in manifest["sweep"]["delta_frac_grid"]),
        stability_grid=tuple(float(v) for v in manifest["sweep"]["stability_grid"]),
        spike_strength=float(cell["power_mu"]),
        edge_modes=(str(cell["edge_mode"]),),
        q_max=int(manifest["sweep"]["q_max"]),
        seed=int(cell["seed"]),
        workers=1,
        batch_size=50,
    )
    threshold_result = calibrate_thresholds(config)
    threshold_entry = threshold_result.thresholds[str(cell["edge_mode"])].to_dict()
    mu_values = [
        0.0,
        float(independently_power_boundary(cell)),
        float(cell["power_mu"]),
    ]
    trial_rows = _evaluate_trials(
        cell=cell,
        threshold=threshold_entry,
        mu_values=mu_values,
        trial_counts={
            0.0: int(trials_null),
            float(independently_power_boundary(cell)): int(trials_alt),
            float(cell["power_mu"]): int(trials_alt),
        },
        seed_base=int(cell["seed"]) * 10,
    )
    curve = _aggregate_curve(trial_rows)
    if not curve or float(curve[0]["mu"]) != 0.0 or len(curve) < 2:
        raise ValueError(
            "Calibration cell requires a null row and at least one positive-mu row."
        )
    detection_rates = [float(row["detection_rate"]) for row in curve]
    acceptance_rates = [float(row["acceptance_rate"]) for row in curve]
    strong = curve[-1]
    null = curve[0]
    accepted_trials = [
        row
        for row in trial_rows
        if int(row["accepted"]) > 0 and float(row["mu"]) == float(cell["power_mu"])
    ]
    accepted_count = len(accepted_trials)
    accepted_candidate_count = sum(
        int(row["accepted_count"]) for row in accepted_trials
    )
    direction_squared_cosine_mean = (
        sum(float(row["direction_squared_cosine_sum"]) for row in accepted_trials)
        / accepted_candidate_count
        if accepted_candidate_count
        else 0.0
    )
    planted_share = (
        sum(int(row["planted_component_hit"]) for row in accepted_trials)
        / accepted_count
        if accepted_count
        else 0.0
    )
    nuisance_share = (
        sum(int(row["nuisance_component_accept_count"] > 0) for row in accepted_trials)
        / accepted_count
        if accepted_count
        else 0.0
    )
    non_fjs_accept_count = sum(
        int(row["non_fjs_accept_count"]) for row in accepted_trials
    )
    gate_metrics = {
        "null_detection_rate": float(null["detection_rate"]),
        "null_detection_ci_low": float(null["detection_ci_low"]),
        "null_detection_ci_high": float(null["detection_ci_high"]),
        "strong_detection_rate": float(strong["detection_rate"]),
        "strong_acceptance_rate": float(strong["acceptance_rate"]),
        "detection_gain": float(strong["detection_rate"])
        - float(null["detection_rate"]),
        "monotone_detection": _monotone(detection_rates),
        "monotone_acceptance": _monotone(acceptance_rates),
        "direction_squared_cosine_mean": direction_squared_cosine_mean,
        "planted_component_accept_share": planted_share,
        "nuisance_component_accept_share": nuisance_share,
        "non_fjs_accept_count": non_fjs_accept_count,
        "acceptance_exceeds_detection": any(
            a > d + 1e-12
            for a, d in zip(acceptance_rates, detection_rates, strict=False)
        ),
    }
    invariance_payload = (
        _evaluate_invariance_checks(
            manifest=manifest,
            cell=cell,
            threshold=threshold_entry,
        )
        if int(manifest["schema_version"]) == 3
        else None
    )
    stable_payload = {
        "manifest_id": manifest["manifest_id"],
        "cell_id": str(cell["cell_id"]),
        "cell_spec_digest": _cell_spec_digest(manifest, cell),
        "threshold_entry": threshold_entry,
        "curve": curve,
        "trial_rows": trial_rows,
        "gate_metrics": gate_metrics,
        "gate_assessment": _assess_cell_gates(
            manifest=manifest,
            cell=cell,
            gate_metrics=gate_metrics,
            invariance=invariance_payload,
        ),
    }
    if invariance_payload is not None:
        stable_payload["invariance"] = invariance_payload
    return stable_payload


def independently_power_boundary(cell: Mapping[str, Any]) -> float:
    boundary = cell.get("detection_boundary")
    if isinstance(boundary, Mapping):
        return float(boundary["derived"]["population_eigenvalue_boundary"])
    return float(cell["power_mu"]) / 1.5


def _run_single_cell(task: dict[str, Any]) -> dict[str, Any]:
    manifest = task["manifest"]
    cell = task["cell"]
    trials_null = int(task["trials_null"])
    trials_alt = int(task["trials_alt"])
    exec_mode = str(task["exec_mode"])
    workers = int(task["workers"])
    stable_payload = _stable_cell_payload(
        manifest=manifest,
        cell=cell,
        trials_null=trials_null,
        trials_alt=trials_alt,
    )
    checkpoint_meta = {
        "manifest_id": manifest["manifest_id"],
        "manifest_digest": manifest["manifest_digest"],
        "expected_cell_set_digest": manifest["expected_cell_set_digest"],
        "cell_id": str(cell["cell_id"]),
        "cell_spec_digest": _cell_spec_digest(manifest, cell),
        "trials_null": trials_null,
        "trials_alt": trials_alt,
        "delta_frac_grid": list(manifest["sweep"]["delta_frac_grid"]),
        "stability_grid": list(manifest["sweep"]["stability_grid"]),
        "code_tree_sha": git_tree_sha(),
        "code_input_fingerprint_sha256": code_input_fingerprint()["sha256"],
        "environment_fingerprint_sha256": environment_fingerprint(exec_mode, workers)[
            "sha256"
        ],
        "stable_payload_sha256": stable_sha256(stable_payload),
    }
    return {
        "checkpoint_meta": checkpoint_meta,
        "stable_payload": stable_payload,
    }


def _ensure_clean_start(args: argparse.Namespace, paths: RunPaths) -> None:
    if args.resume:
        if not paths.cells_dir.exists():
            raise FileNotFoundError(
                f"Resume requested but no checkpoint dir exists: {paths.cells_dir}"
            )
        return
    if paths.run_root.exists():
        raise FileExistsError(f"Run root already exists: {paths.run_root}")
    paths.cells_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)


def _prepare_scratch(
    args: argparse.Namespace, manifest: Mapping[str, Any], paths: RunPaths
) -> Path:
    if args.scratch_root is not None:
        scratch = args.scratch_root.expanduser().resolve()
    else:
        scratch = paths.run_root / (
            "scratch"
            if not manifest["artifacts"]["smoke_outputs_must_be_temp"]
            else "tmp_scratch"
        )
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_execution_scope(
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    paths: RunPaths,
    thresholds_out: Path,
    defaults_out: Path,
) -> None:
    profile = str(manifest.get("profile", "")).strip().lower()
    readiness = manifest.get("execution_readiness", {})
    if profile == "full" and not bool(readiness.get("full_execution_ready", False)):
        blockers = readiness.get("blockers", ["full_execution_not_ready"])
        raise RuntimeError(
            "Full M4 execution is fail-closed: " + ", ".join(map(str, blockers))
        )
    if profile == "full" and (
        args.trials_null_override is not None
        or args.trials_alt_override is not None
        or args.max_cells is not None
    ):
        raise ValueError(
            "Full M4 execution cannot use cell or trial overrides; use the "
            "separate smoke profile for bounded diagnostics."
        )
    if profile == "smoke":
        repo_paths = [
            path
            for path in (paths.run_root, thresholds_out, defaults_out)
            if _is_within(path, ROOT)
        ]
        if repo_paths:
            rendered = ", ".join(str(path) for path in repo_paths)
            raise ValueError(
                "Smoke artifacts must remain outside the repository: " + rendered
            )
    if args.instance_hourly_usd is not None and args.instance_hourly_usd < 0.0:
        raise ValueError("--instance-hourly-usd must be non-negative.")


def _launch_worker(
    *,
    args: argparse.Namespace,
    script_path: Path,
    manifest_path: Path,
    cell_id: str,
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.Popen[Any]:
    cmd = [
        sys.executable,
        str(script_path),
        "--manifest",
        str(manifest_path),
        "--run-id",
        str(args.run_id),
        "--run-root-base",
        str(args.run_root_base),
        "--exec-mode",
        str(args.exec_mode),
        "--workers",
        str(args.workers),
        "--worker-cell-id",
        cell_id,
        "--cell-timeout-seconds",
        str(args.cell_timeout_seconds),
    ]
    if args.trials_null_override is not None:
        cmd.extend(["--trials-null-override", str(args.trials_null_override)])
    if args.trials_alt_override is not None:
        cmd.extend(["--trials-alt-override", str(args.trials_alt_override)])
    with (
        stdout_path.open("w", encoding="utf-8") as out_handle,
        stderr_path.open("w", encoding="utf-8") as err_handle,
    ):
        return subprocess.Popen(
            cmd,
            stdout=out_handle,
            stderr=err_handle,
            cwd=ROOT,
            start_new_session=True,
        )


def _terminate_worker(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait(timeout=5)


def _validate_existing_checkpoints(
    *,
    manifest: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    paths: RunPaths,
    trials_null: int,
    trials_alt: int,
    exec_mode: str,
    workers: int,
) -> set[str]:
    expected_ids = {str(cell["cell_id"]) for cell in cells}
    seen_ids: set[str] = set()
    for checkpoint_path in sorted(paths.cells_dir.glob("*.json")):
        cell_id = checkpoint_path.stem
        if cell_id not in expected_ids:
            raise ValueError(
                f"Out-of-scope stale cell checkpoint detected: {checkpoint_path.name}"
            )
        cell = next(cell for cell in cells if str(cell["cell_id"]) == cell_id)
        _cell_checkpoint_valid(
            checkpoint_path,
            manifest=manifest,
            cell=cell,
            trials_null=trials_null,
            trials_alt=trials_alt,
            exec_mode=exec_mode,
            workers=workers,
        )
        seen_ids.add(cell_id)
    return seen_ids


def _reduce_expected_cells(
    *,
    manifest: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    paths: RunPaths,
    trials_null: int,
    trials_alt: int,
    exec_mode: str,
    workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_ids = [str(cell["cell_id"]) for cell in cells]
    observed_ids = sorted(path.stem for path in paths.cells_dir.glob("*.json"))
    if sorted(expected_ids) != observed_ids:
        raise ValueError(
            "Checkpoint set mismatch. "
            f"expected={sorted(expected_ids)} observed={observed_ids}"
        )
    cell_payloads = []
    for cell in cells:
        checkpoint_path = paths.cells_dir / f"{cell['cell_id']}.json"
        _cell_checkpoint_valid(
            checkpoint_path,
            manifest=manifest,
            cell=cell,
            trials_null=trials_null,
            trials_alt=trials_alt,
            exec_mode=exec_mode,
            workers=workers,
        )
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        cell_payloads.append(checkpoint["stable_payload"])
    stable_reducer = {
        "manifest_id": manifest["manifest_id"],
        "manifest_digest": manifest["manifest_digest"],
        "expected_cell_set_digest": manifest["expected_cell_set_digest"],
        "scope_cell_set_digest": _scope_cell_set_digest(manifest, cells),
        "scope_is_full_manifest": len(cells) == int(manifest["sweep"]["cells_total"]),
        "execution_readiness": manifest.get("execution_readiness", {}),
        "statistical_gate_pass_count": sum(
            int(
                bool(
                    payload.get("gate_assessment", {}).get(
                        "statistical_gate_pass", False
                    )
                )
            )
            for payload in cell_payloads
        ),
        "full_detector_gate_pass": False,
        "cells": cell_payloads,
    }
    if int(manifest["schema_version"]) == 3:
        stable_reducer.update(
            {
                "invariance_gate_pass_count": sum(
                    int(
                        bool(
                            payload.get("gate_assessment", {}).get(
                                "invariance_gate_pass", False
                            )
                        )
                    )
                    for payload in cell_payloads
                ),
                "local_scientific_gate_pass_count": sum(
                    int(
                        bool(
                            payload.get("gate_assessment", {}).get(
                                "local_scientific_gate_pass", False
                            )
                        )
                    )
                    for payload in cell_payloads
                ),
            }
        )
    stable_reducer["stable_reducer_sha256"] = stable_sha256(
        {k: v for k, v in stable_reducer.items() if k != "stable_reducer_sha256"}
    )
    meta_reducer = {
        "generated_at": _now(),
        "code_tree_sha": git_tree_sha(),
        "code_input_fingerprint": code_input_fingerprint(),
        "environment_fingerprint": environment_fingerprint(exec_mode, workers),
        "trials_null": trials_null,
        "trials_alt": trials_alt,
    }
    return stable_reducer, meta_reducer


def _write_outputs(
    *,
    stable_reducer: Mapping[str, Any],
    meta_reducer: Mapping[str, Any],
    thresholds_out: Path,
    defaults_out: Path,
) -> None:
    _safe_write_json(thresholds_out, {"stable": stable_reducer, "meta": meta_reducer})
    _safe_write_json(
        defaults_out,
        {
            "stable_reducer_sha256": stable_reducer["stable_reducer_sha256"],
            "manifest_id": stable_reducer["manifest_id"],
            "cells": [cell["cell_id"] for cell in stable_reducer["cells"]],
        },
    )


def _runtime_cost_summary(
    *,
    manifest: Mapping[str, Any],
    cells_completed: int,
    elapsed: float,
    instance_hourly_usd: float | None,
) -> dict[str, Any]:
    full_cells = int(manifest["sweep"]["cells_total"])
    trials_null = int(manifest["sweep"]["trials_null"])
    trials_alt = int(manifest["sweep"]["trials_alt"])
    # One null calibration pass plus null/boundary/strong curve evaluation.
    trial_work_units_per_cell = (2 * trials_null) + (2 * trials_alt)
    trial_work_units = full_cells * trial_work_units_per_cell
    complete_full_run = (
        str(manifest.get("profile", "")) == "full" and cells_completed == full_cells
    )
    full_runtime_seconds = elapsed if complete_full_run else None
    cost_usd = (
        (full_runtime_seconds / 3600.0) * float(instance_hourly_usd)
        if full_runtime_seconds is not None and instance_hourly_usd is not None
        else None
    )
    return {
        "full_cells": full_cells,
        "trial_work_units_per_cell": trial_work_units_per_cell,
        "full_work_units": trial_work_units,
        "cost_status": (
            "actual_completed_full_run"
            if complete_full_run
            else "not_estimable_from_smoke_or_partial_timing"
        ),
        "full_runtime_seconds": full_runtime_seconds,
        "full_runtime_hours": (
            full_runtime_seconds / 3600.0 if full_runtime_seconds is not None else None
        ),
        "instance_hourly_usd": instance_hourly_usd,
        "actual_compute_cost_usd": (
            round(cost_usd, 4) if cost_usd is not None else None
        ),
        "estimate_requirement": (
            None
            if complete_full_run
            else "fresh authoritative price plus stratified full-shape benchmark"
        ),
    }


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    cells = _scoped_cells(manifest, args.max_cells)
    paths = _paths(args)
    thresholds_out, defaults_out = _output_paths(args, paths)
    _validate_execution_scope(
        args=args,
        manifest=manifest,
        paths=paths,
        thresholds_out=thresholds_out,
        defaults_out=defaults_out,
    )
    trials_null = int(
        manifest["sweep"]["trials_null"]
        if args.trials_null_override is None
        else args.trials_null_override
    )
    trials_alt = int(
        manifest["sweep"]["trials_alt"]
        if args.trials_alt_override is None
        else args.trials_alt_override
    )

    if args.worker_cell_id is not None:
        cell = next(
            (item for item in cells if str(item["cell_id"]) == args.worker_cell_id),
            None,
        )
        if cell is None:
            raise ValueError(f"Worker cell not found: {args.worker_cell_id}")
        exec_settings = runtime.configure_exec_mode(args.exec_mode)
        paths.cells_dir.mkdir(parents=True, exist_ok=True)
        payload = _run_single_cell(
            {
                "manifest": manifest,
                "cell": cell,
                "trials_null": trials_null,
                "trials_alt": trials_alt,
                "exec_mode": exec_settings.mode,
                "workers": max(1, int(args.workers)),
            }
        )
        _safe_write_json(paths.cells_dir / f"{cell['cell_id']}.json", payload)
        return paths.cells_dir / f"{cell['cell_id']}.json"

    _ensure_clean_start(args, paths)
    _prepare_scratch(args, manifest, paths)
    exec_settings = runtime.configure_exec_mode(args.exec_mode)
    start = time.perf_counter()
    completed_ids = _validate_existing_checkpoints(
        manifest=manifest,
        cells=cells,
        paths=paths,
        trials_null=trials_null,
        trials_alt=trials_alt,
        exec_mode=exec_settings.mode,
        workers=max(1, int(args.workers)),
    )
    pending = [cell for cell in cells if str(cell["cell_id"]) not in completed_ids]
    _append_progress(
        paths.progress_log,
        {
            "event": "run_start",
            "ts": _now(),
            "manifest_id": manifest["manifest_id"],
            "manifest_digest": manifest["manifest_digest"],
            "queued_cells": len(pending),
            "completed_cells": len(completed_ids),
            "expected_cell_set_digest": manifest["expected_cell_set_digest"],
            "scope_cell_set_digest": _scope_cell_set_digest(manifest, cells),
        },
    )

    finished_now = 0
    if pending:
        if max(1, int(args.workers)) == 1:
            for cell in pending:
                payload = _run_single_cell(
                    {
                        "manifest": manifest,
                        "cell": cell,
                        "trials_null": trials_null,
                        "trials_alt": trials_alt,
                        "exec_mode": exec_settings.mode,
                        "workers": 1,
                    }
                )
                _safe_write_json(paths.cells_dir / f"{cell['cell_id']}.json", payload)
                finished_now += 1
                _append_progress(
                    paths.progress_log,
                    {
                        "event": "cell_complete",
                        "ts": _now(),
                        "cell_id": cell["cell_id"],
                        "finished_now": finished_now,
                    },
                )
                if args.interrupt_after_completions is not None and finished_now >= int(
                    args.interrupt_after_completions
                ):
                    raise RuntimeError(
                        "Intentional interruption after requested number of "
                        "completions."
                    )
        else:
            active: list[
                tuple[Mapping[str, Any], subprocess.Popen[Any], Path, Path, float]
            ] = []
            queue = list(pending)
            try:
                while queue or active:
                    while queue and len(active) < max(1, int(args.workers)):
                        cell = queue.pop(0)
                        stdout_path = paths.logs_dir / f"{cell['cell_id']}.stdout.log"
                        stderr_path = paths.logs_dir / f"{cell['cell_id']}.stderr.log"
                        proc = _launch_worker(
                            args=args,
                            script_path=Path(__file__).resolve(),
                            manifest_path=manifest_path,
                            cell_id=str(cell["cell_id"]),
                            stdout_path=stdout_path,
                            stderr_path=stderr_path,
                        )
                        active.append(
                            (
                                cell,
                                proc,
                                stdout_path,
                                stderr_path,
                                time.perf_counter(),
                            )
                        )
                    time.sleep(0.05)
                    next_active = []
                    for cell, proc, stdout_path, stderr_path, launched_at in active:
                        rc = proc.poll()
                        if rc is None:
                            if time.perf_counter() - launched_at > float(
                                args.cell_timeout_seconds
                            ):
                                _terminate_worker(proc)
                                raise TimeoutError(f"Cell timed out: {cell['cell_id']}")
                            next_active.append(
                                (cell, proc, stdout_path, stderr_path, launched_at)
                            )
                            continue
                        if rc != 0:
                            stdout_text = (
                                stdout_path.read_text(encoding="utf-8")
                                if stdout_path.exists()
                                else ""
                            )
                            stderr_text = (
                                stderr_path.read_text(encoding="utf-8")
                                if stderr_path.exists()
                                else ""
                            )
                            raise RuntimeError(
                                f"Worker failed for {cell['cell_id']} rc={rc}"
                                f"\nstdout:\n{stdout_text}\nstderr:\n{stderr_text}"
                            )
                        finished_now += 1
                        _append_progress(
                            paths.progress_log,
                            {
                                "event": "cell_complete",
                                "ts": _now(),
                                "cell_id": cell["cell_id"],
                                "finished_now": finished_now,
                            },
                        )
                        if (
                            args.interrupt_after_completions is not None
                            and finished_now >= int(args.interrupt_after_completions)
                        ):
                            raise RuntimeError(
                                "Intentional interruption after requested "
                                "number of completions."
                            )
                    active = next_active
            finally:
                for _, running_proc, _, _, _ in active:
                    _terminate_worker(running_proc)

    stable_reducer, meta_reducer = _reduce_expected_cells(
        manifest=manifest,
        cells=cells,
        paths=paths,
        trials_null=trials_null,
        trials_alt=trials_alt,
        exec_mode=exec_settings.mode,
        workers=max(1, int(args.workers)),
    )
    _safe_write_json(
        paths.reducer_path, {"stable": stable_reducer, "meta": meta_reducer}
    )
    _write_outputs(
        stable_reducer=stable_reducer,
        meta_reducer=meta_reducer,
        thresholds_out=thresholds_out,
        defaults_out=defaults_out,
    )
    elapsed = time.perf_counter() - start
    summary = {
        "manifest_id": manifest["manifest_id"],
        "manifest_digest": manifest["manifest_digest"],
        "expected_cell_set_digest": manifest["expected_cell_set_digest"],
        "scope_cell_set_digest": _scope_cell_set_digest(manifest, cells),
        "execution_readiness": manifest.get("execution_readiness", {}),
        "full_detector_gate_pass": False,
        "completed_cells": len(cells),
        "elapsed_seconds": elapsed,
        "runtime_cost_estimate": _runtime_cost_summary(
            manifest=manifest,
            cells_completed=len(cells),
            elapsed=elapsed,
            instance_hourly_usd=args.instance_hourly_usd,
        ),
    }
    _safe_write_json(paths.summary_path, summary)
    write_run_metadata(
        paths.metadata_path,
        config={
            "run_id": args.run_id,
            "manifest_id": manifest["manifest_id"],
            "manifest_digest": manifest["manifest_digest"],
            "exec_mode": exec_settings.mode,
            "workers": int(args.workers),
            "trials_null": trials_null,
            "trials_alt": trials_alt,
            "elapsed_seconds": elapsed,
        },
        extra={"summary": summary},
    )
    return paths.summary_path


if __name__ == "__main__":  # pragma: no cover
    main()
