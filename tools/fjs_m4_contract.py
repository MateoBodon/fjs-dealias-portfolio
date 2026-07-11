from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

FULL_MANIFEST_ID = "fjs-m4-full-target-between-v2"
SMOKE_MANIFEST_ID = "fjs-m4-smoke-target-between-v2"
SCHEMA_VERSION = 2

FULL_P_ASSETS = (64, 80, 96, 128, 160, 188, 200)
FULL_N_GROUPS = (36, 60, 80)
FULL_REPLICATES = (12, 16, 20)
FULL_DELTA_ABS = (0.35, 0.50)
FULL_EDGE_MODES = ("scm", "tyler")

SMOKE_P_ASSETS = (4,)
SMOKE_N_GROUPS = (8,)
SMOKE_REPLICATES = (2,)
SMOKE_DELTA_ABS = (0.30,)
SMOKE_EDGE_MODES = ("scm", "tyler")

CODE_INPUT_PATHS = (
    "experiments/synthetic/harness_utils.py",
    "src/fjs/balanced.py",
    "src/fjs/dealias.py",
    "src/fjs/detector_contract.py",
    "src/fjs/gating.py",
    "src/fjs/mp.py",
    "src/fjs/overlay.py",
    "src/meta/runtime.py",
    "src/synthetic/calibration.py",
    "src/synthetic/threshold_eval.py",
    "tools/fjs_m4_contract.py",
    "tools/freeze_fjs_m4_manifest.py",
    "tools/run_fjs_calibration_manifest.py",
)


def stable_json_dumps(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def stable_sha256(payload: Mapping[str, Any] | Sequence[Any] | str | bytes) -> str:
    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = stable_json_dumps(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_tree_sha() -> str:
    return git_value("rev-parse", "HEAD^{tree}")


def code_input_fingerprint() -> dict[str, Any]:
    """Hash the exact executable inputs, including uncommitted content.

    A Git tree alone is insufficient while a runner is being prepared in a
    worktree: it identifies ``HEAD`` but not the bytes Python will execute.
    Checkpoints bind both values so a resume cannot silently cross a code edit.
    """

    files = []
    for relative in CODE_INPUT_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required M4 code input is missing: {relative}")
        files.append(
            {
                "path": relative,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload: dict[str, Any] = {"files": files}
    payload["sha256"] = stable_sha256(payload)
    return payload


def environment_fingerprint(exec_mode: str, workers: int) -> dict[str, Any]:
    payload = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "exec_mode": exec_mode,
        "workers": workers,
        "cwd": str(ROOT),
    }
    payload["sha256"] = stable_sha256(payload)
    return payload


def exact_binomial_interval_95(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    from math import comb

    def pmf(k: int, p: float) -> float:
        return comb(total, k) * (p**k) * ((1.0 - p) ** (total - k))

    def cdf(k: int, p: float) -> float:
        return sum(pmf(i, p) for i in range(0, k + 1))

    alpha = 0.05
    if successes == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, successes / total
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            upper_tail = 1.0 - cdf(successes - 1, mid)
            if upper_tail > alpha / 2.0:
                hi = mid
            else:
                lo = mid
        lower = 0.5 * (lo + hi)
    if successes == total:
        upper = 1.0
    else:
        lo, hi = successes / total, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lower_tail = cdf(successes, mid)
            # P(X <= successes) decreases as p increases.  Keep the root
            # bracket ordered as cdf(lo) >= alpha/2 >= cdf(hi).
            if lower_tail > alpha / 2.0:
                lo = mid
            else:
                hi = mid
        upper = 0.5 * (lo + hi)
    return float(lower), float(upper)


@dataclass(frozen=True)
class ManifestProfile:
    manifest_id: str
    purpose: str
    p_assets: tuple[int, ...]
    n_groups: tuple[int, ...]
    replicates: tuple[int, ...]
    delta_abs: tuple[float, ...]
    edge_modes: tuple[str, ...]
    trials_null: int
    trials_alt: int
    delta_frac_grid: tuple[float, ...]
    stability_grid: tuple[float, ...]


def manifest_profile(profile: str) -> ManifestProfile:
    normalized = profile.strip().lower()
    if normalized == "full":
        return ManifestProfile(
            manifest_id=FULL_MANIFEST_ID,
            purpose=(
                "Frozen full M4 calibration contract for the detector stop-line "
                "and AWS sizing; not to be launched in this ticket."
            ),
            p_assets=FULL_P_ASSETS,
            n_groups=FULL_N_GROUPS,
            replicates=FULL_REPLICATES,
            delta_abs=FULL_DELTA_ABS,
            edge_modes=FULL_EDGE_MODES,
            trials_null=200,
            trials_alt=200,
            delta_frac_grid=(0.01, 0.015, 0.02, 0.025, 0.03),
            stability_grid=(0.30, 0.40, 0.50, 0.60),
        )
    if normalized == "smoke":
        return ManifestProfile(
            manifest_id=SMOKE_MANIFEST_ID,
            purpose=(
                "Bounded two-cell deterministic real-kernel smoke for M4 "
                "checkpoint/restart, mismatch rejection, and reducer-equality "
                "evidence only."
            ),
            p_assets=SMOKE_P_ASSETS,
            n_groups=SMOKE_N_GROUPS,
            replicates=SMOKE_REPLICATES,
            delta_abs=SMOKE_DELTA_ABS,
            edge_modes=SMOKE_EDGE_MODES,
            trials_null=6,
            trials_alt=6,
            delta_frac_grid=(0.01,),
            stability_grid=(0.30,),
        )
    raise ValueError(f"Unsupported manifest profile {profile!r}")


def build_cells(
    profile: ManifestProfile,
    *,
    seed_base: int,
    limit_cells: int | None = None,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    seed = int(seed_base)
    index = 0
    for p_assets in profile.p_assets:
        for n_groups in profile.n_groups:
            for replicates in profile.replicates:
                for delta_abs in profile.delta_abs:
                    for edge_mode in profile.edge_modes:
                        delta_token = int(round(float(delta_abs) * 1000))
                        cell = {
                            "cell_id": (
                                f"p{p_assets}_g{n_groups}_r{replicates}_"
                                f"d{delta_token}_{edge_mode}"
                            ),
                            "index": index,
                            "seed": seed,
                            "p_assets": p_assets,
                            "n_groups": n_groups,
                            "replicates": replicates,
                            "delta_abs": float(delta_abs),
                            "edge_mode": edge_mode,
                            "inject_mode": "between",
                            "invariance_checks": [
                                "standardized_rescaling",
                                "deterministic_row_order",
                                "asset_permutation",
                                "group_label_permutation",
                            ],
                            "attribution_fields": [
                                "fjs_detection_count",
                                "fjs_acceptance_count",
                                "candidate_source_counts_pre_gate",
                                "candidate_source_counts_accepted",
                                "direction_squared_cosine_mean",
                                "planted_component_accept_share",
                                "nuisance_component_accept_share",
                            ],
                        }
                        cells.append(cell)
                        seed += 1
                        index += 1
                        if limit_cells is not None and len(cells) >= limit_cells:
                            return cells
    return cells


def independently_computed_boundary(cell: Mapping[str, Any]) -> float:
    # The bounded smoke binds to the scalar reference mechanism boundary only.
    # It is deliberately not presented as the cell-specific asymptotic boundary
    # required for the full calibration.  Full execution remains fail-closed
    # below until that independent calculation is hash-bound.
    return 1.0


def build_manifest(
    *,
    profile_name: str,
    seed_base: int,
    limit_cells: int | None = None,
) -> dict[str, Any]:
    profile = manifest_profile(profile_name)
    cells = build_cells(profile, seed_base=seed_base, limit_cells=limit_cells)
    cell_specs = []
    for cell in cells:
        boundary = independently_computed_boundary(cell)
        power_mu = 1.5 * boundary
        cell_specs.append(
            {
                **cell,
                "nominal_size": 0.05,
                "null_upper_bound_max": 0.075,
                "power_mu": power_mu,
                "power_detection_min": 0.80,
                "power_acceptance_min": 0.80,
                "power_gain_min": 0.50,
                "direction_squared_cosine_min": 0.80,
                "planted_component_accept_share_min": 0.90,
                "nuisance_component_accept_share_max": 0.10,
            }
        )
    cell_digest_map = {spec["cell_id"]: stable_sha256(spec) for spec in cell_specs}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": profile.manifest_id,
        "profile": profile_name,
        "purpose": profile.purpose,
        "promotion_allowed": False,
        "execution_readiness": {
            "full_execution_ready": False,
            "aws_execution_authorized": False,
            "blockers": [
                "cell_specific_independent_detection_boundary_unbound",
                "invariance_reducer_not_yet_hash_bound",
                "real_design_cell_manifest_not_yet_bound",
                "fresh_authoritative_aws_admission_required",
            ],
            "smoke_scope": (
                "checkpoint_restart_and_reducer_integrity_only"
                if profile_name == "smoke"
                else None
            ),
        },
        "claim_boundary": {
            "mechanism_calibration_only": True,
            "empirical_claims_forbidden": True,
            "notes": [
                "Synthetic and semi-synthetic detector evidence are mechanism "
                "calibration only.",
                "Empirical claims remain separately gated by development, "
                "confirmation, and holdout rules.",
            ],
        },
        "predeclaration_contract": {
            "nominal_size": 0.05,
            "exact_binomial_interval": "95%",
            "power_boundary_multiplier": 1.5,
            "detection_and_acceptance_separate": True,
            "monotonic_power_required": True,
            "real_design_adequacy_required": True,
            "inject_mode": "between",
            "paired_trial_seeds_across_mu": True,
            "boundary_status": "reference_scalar_placeholder_not_full_cell_boundary",
        },
        "sweep": {
            "trials_null": profile.trials_null,
            "trials_alt": profile.trials_alt,
            "delta_frac_grid": list(profile.delta_frac_grid),
            "stability_grid": list(profile.stability_grid),
            "q_max": 2,
            "eps": 0.02,
            "p_assets": list(profile.p_assets),
            "n_groups": list(profile.n_groups),
            "replicates": list(profile.replicates),
            "delta_abs_grid": list(profile.delta_abs),
            "edge_modes": list(profile.edge_modes),
            "cells_total": len(cell_specs),
            "expected_cell_ids": [spec["cell_id"] for spec in cell_specs],
        },
        "artifacts": {
            "run_root_template": "reports/synthetic/calib/{run_id}",
            "smoke_outputs_must_be_temp": profile_name == "smoke",
            "canonical_promotion_requires_explicit_action": True,
        },
        "cells": cell_specs,
        "cell_digests": cell_digest_map,
    }
    manifest["expected_cell_set_digest"] = stable_sha256(
        [{"cell_id": k, "sha256": cell_digest_map[k]} for k in sorted(cell_digest_map)]
    )
    manifest["manifest_digest"] = stable_sha256(
        {k: v for k, v in manifest.items() if k not in {"manifest_digest"}}
    )
    return manifest
