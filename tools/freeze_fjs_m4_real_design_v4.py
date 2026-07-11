#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from tools.fjs_m4_contract_v3 import validate_manifest_v3  # noqa: E402

from fjs.real_design_contract import (  # noqa: E402
    FactorBinding,
    RealDesignCellSpec,
    SourcePartitionBinding,
    bind_factor_source,
    bind_source_partition,
    covariance_contract,
    derive_real_design_cell,
    file_sha256,
    load_bound_factors,
    load_filtered_sources,
    residualization_contract,
    source_contract,
    stable_json_dumps,
    stable_sha256,
    universe_contract,
    validate_real_design_cell,
    write_real_design_cell,
)

SCHEMA_VERSION = 4
MANIFEST_ID = "fjs-m4-real-design-bounded-proof-v4"
EXPECTED_V2_HASHES = {
    "calibration/manifests/fjs_m4_full_target_between_v2.json": (
        "aa444e283fa99048e77353d7912c00baf5552c33cc777fc0b8137fe074448b22"
    ),
    "calibration/manifests/fjs_m4_smoke_target_between_v2.json": (
        "ccca12d54fd73a0ea88e885297176c2de07153528af81083e8076960bd8cb5ef"
    ),
}
EXPECTED_V3_HASHES = {
    "calibration/manifests/fjs_m4_full_target_between_v3.json": (
        "0be2557e3cef75d871f3209f145ec0cd4bd9a5e0ca50b4ea632a234b72e00849"
    ),
    "calibration/manifests/fjs_m4_smoke_target_between_v3.json": (
        "8a55a1fd26bc0e010a897fc81f5d36ab2a551e1469749fb31ed0d6a684c45478"
    ),
}
CONTRACT_BINDING_PATHS = (
    "src/fjs/real_design_contract.py",
    "tools/freeze_fjs_m4_real_design_v4.py",
)
EXTERNAL_BLOCKERS = (
    "real_design_full_generation_not_run",
    "trusted_route_admission_required",
    "fresh_authoritative_aws_admission_required",
)


def required_development_partitions() -> list[str]:
    return [
        f"month={year:04d}-{month:02d}"
        for year in range(2013, 2019)
        for month in range(1, 13)
    ]


def _repo_binding(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required v4 contract input is missing: {relative}")
    return {
        "path": relative,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _contract_bindings() -> dict[str, dict[str, Any]]:
    return {relative: _repo_binding(relative) for relative in CONTRACT_BINDING_PATHS}


def frozen_predecessor_hashes() -> dict[str, str]:
    observed = {
        relative: file_sha256(ROOT / relative)
        for relative in (*EXPECTED_V2_HASHES, *EXPECTED_V3_HASHES)
    }
    expected = {**EXPECTED_V2_HASHES, **EXPECTED_V3_HASHES}
    if observed != expected:
        raise ValueError(
            "Frozen v2/v3 manifest bytes changed; refusing to create a v4 generation."
        )
    return observed


def bind_base_v3(path: Path) -> dict[str, Any]:
    base = path.expanduser().resolve()
    expected_path = (
        ROOT / "calibration/manifests/fjs_m4_full_target_between_v3.json"
    ).resolve()
    if base != expected_path:
        raise ValueError("V4 must extend the exact frozen full v3 manifest.")
    observed_hash = file_sha256(base)
    expected_hash = EXPECTED_V3_HASHES[
        "calibration/manifests/fjs_m4_full_target_between_v3.json"
    ]
    if observed_hash != expected_hash:
        raise ValueError("Frozen full v3 manifest hash mismatch.")
    payload = json.loads(base.read_text(encoding="utf-8"))
    validate_manifest_v3(payload)
    return {
        "path": "calibration/manifests/fjs_m4_full_target_between_v3.json",
        "sha256": observed_hash,
        "size_bytes": base.stat().st_size,
        "manifest_id": str(payload["manifest_id"]),
        "manifest_digest": str(payload["manifest_digest"]),
        "tree_generation": "frozen-v3-input",
    }


def build_manifest_v4(
    *,
    base_v3: Mapping[str, Any],
    source_bindings: Sequence[SourcePartitionBinding],
    factor_binding: FactorBinding,
    source_scan_receipt: Mapping[str, Any],
    cell_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    if not source_bindings:
        raise ValueError("V4 requires at least one exact CRSP source binding.")
    source_months = sorted({binding.partition for binding in source_bindings})
    if any("2025" in value for value in source_months):
        raise ValueError("The 2025 holdout cannot appear in a v4 source binding.")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": MANIFEST_ID,
        "profile": "bounded-proof",
        "purpose": (
            "Additive v4 proof of the exact realistic-design input freezer on a "
            "bounded local CRSP partition/sample; no full derivation or outcome."
        ),
        "promotion_allowed": False,
        "execution_readiness": {
            "real_design_contract_ready": True,
            "bounded_source_proof_ready": True,
            "real_design_full_generation_complete": False,
            "full_execution_ready": False,
            "aws_execution_authorized": False,
            "blockers": list(EXTERNAL_BLOCKERS),
        },
        "claim_boundary": {
            "development_only": True,
            "mechanism_calibration_only": True,
            "empirical_claims_forbidden": True,
            "full_outcomes_unobserved": True,
            "legacy_ticker_csv_used": False,
            "legacy_ticker_csv_provenance_inferred": False,
            "holdout_2025_opened": False,
            "source_proof_is_not_full_generation": True,
        },
        "predecessor": dict(base_v3),
        "frozen_predecessor_hashes": frozen_predecessor_hashes(),
        "contract_bindings": _contract_bindings(),
        "contracts": {
            "source": source_contract(),
            "universe": universe_contract(),
            "residualization": residualization_contract(),
            "covariance": covariance_contract(),
        },
        "full_generation_contract": {
            "required_partition_months": required_development_partitions(),
            "required_partition_count": 72,
            "partition_content_hash_required": True,
            "receipt_status_required": "ok",
            "truncated_scan_allowed": False,
            "all_cells_must_be_serialized_and_hash_bound": True,
            "successor_generation_required": True,
        },
        "bounded_proof": {
            "source_partition_months": source_months,
            "source_partitions": [binding.to_dict() for binding in source_bindings],
            "factor_source": factor_binding.to_dict(),
            "source_scan_receipt": dict(source_scan_receipt),
            "cell_artifacts": [dict(cell_artifact)],
        },
    }
    manifest["manifest_digest"] = stable_sha256(manifest)
    return manifest


def _verify_binding_digest(payload: Mapping[str, Any], digest_key: str) -> None:
    expected = stable_sha256(
        {key: value for key, value in payload.items() if key != digest_key}
    )
    if payload.get(digest_key) != expected:
        raise ValueError(f"Binding digest mismatch for {payload.get('path')!r}.")


def validate_manifest_v4(manifest: Mapping[str, Any]) -> None:
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("The v4 validator requires schema_version=4.")
    if manifest.get("manifest_id") != MANIFEST_ID:
        raise ValueError("Unknown FJS M4 v4 manifest identity.")
    if manifest.get("profile") != "bounded-proof":
        raise ValueError("This generation is deliberately bounded-proof only.")
    if manifest.get("execution_readiness", {}).get("blockers") != list(
        EXTERNAL_BLOCKERS
    ):
        raise ValueError("V4 external stop-lines changed.")
    if manifest.get("execution_readiness", {}).get("full_execution_ready") is not False:
        raise ValueError("Bounded v4 proof cannot be full-execution ready.")
    if (
        manifest.get("execution_readiness", {}).get("aws_execution_authorized")
        is not False
    ):
        raise ValueError("Bounded v4 proof cannot authorize AWS.")
    claim_boundary = manifest.get("claim_boundary")
    if not isinstance(claim_boundary, Mapping):
        raise ValueError("V4 claim boundary is missing.")
    if claim_boundary.get("legacy_ticker_csv_used") is not False:
        raise ValueError("V4 cannot use the legacy ticker CSV.")
    if claim_boundary.get("holdout_2025_opened") is not False:
        raise ValueError("V4 cannot open the 2025 holdout.")

    frozen = manifest.get("frozen_predecessor_hashes")
    if frozen != frozen_predecessor_hashes():
        raise ValueError("V4 frozen predecessor hash set mismatch.")
    base = manifest.get("predecessor")
    if not isinstance(base, Mapping):
        raise ValueError("V4 predecessor binding is missing.")
    expected_base = bind_base_v3(ROOT / str(base["path"]))
    if dict(base) != expected_base:
        raise ValueError("V4 predecessor binding mismatch.")
    if manifest.get("contract_bindings") != _contract_bindings():
        raise ValueError("V4 executable contract binding mismatch.")
    if manifest.get("contracts") != {
        "source": source_contract(),
        "universe": universe_contract(),
        "residualization": residualization_contract(),
        "covariance": covariance_contract(),
    }:
        raise ValueError("V4 scientific contract mismatch.")

    full_contract = manifest.get("full_generation_contract")
    if not isinstance(full_contract, Mapping):
        raise ValueError("V4 full-generation contract is missing.")
    if (
        full_contract.get("required_partition_months")
        != required_development_partitions()
    ):
        raise ValueError("V4 required development partition set mismatch.")
    proof = manifest.get("bounded_proof")
    if not isinstance(proof, Mapping):
        raise ValueError("V4 bounded proof is missing.")
    partitions = proof.get("source_partitions")
    if not isinstance(partitions, list) or not partitions:
        raise ValueError("V4 bounded proof has no source partitions.")
    for partition in partitions:
        if not isinstance(partition, Mapping):
            raise ValueError("V4 source partition binding must be a mapping.")
        _verify_binding_digest(partition, "binding_sha256")
        source = Path(str(partition["path"]))
        if not source.is_file() or file_sha256(source) != partition["sha256"]:
            raise ValueError(f"V4 bound source changed: {source}")
        if "2025" in str(partition.get("partition", "")):
            raise ValueError("V4 source partition reaches the 2025 holdout.")
    factor = proof.get("factor_source")
    if not isinstance(factor, Mapping):
        raise ValueError("V4 factor binding is missing.")
    _verify_binding_digest(factor, "binding_sha256")
    factor_path = Path(str(factor["path"]))
    if not factor_path.is_file() or file_sha256(factor_path) != factor["sha256"]:
        raise ValueError("V4 bound factor file changed.")

    artifacts = proof.get("cell_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError("V4 bounded proof must contain exactly one cell artifact.")
    artifact = artifacts[0]
    cell_path = Path(str(artifact["path"]))
    if not cell_path.is_file() or file_sha256(cell_path) != artifact["sha256"]:
        raise ValueError("V4 cell artifact hash mismatch.")
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    validate_real_design_cell(cell)
    if cell.get("cell_digest") != artifact.get("cell_digest"):
        raise ValueError("V4 cell artifact digest mismatch.")
    if cell.get("claim_boundary", {}).get("proof_only") is not True:
        raise ValueError("V4 bounded cell must remain proof-only.")

    expected_manifest_digest = stable_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )
    if manifest.get("manifest_digest") != expected_manifest_digest:
        raise ValueError("V4 manifest digest mismatch.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Freeze a bounded FJS M4 v4 realistic-design input proof."
    )
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--factors-csv", type=Path, required=True)
    parser.add_argument("--factor-registry", type=Path, required=True)
    parser.add_argument(
        "--base-v3-manifest",
        type=Path,
        default=ROOT / "calibration/manifests/fjs_m4_full_target_between_v3.json",
    )
    parser.add_argument("--cell-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--factor-fit-start", required=True)
    parser.add_argument("--factor-fit-end", required=True)
    parser.add_argument("--formation-date", required=True)
    parser.add_argument("--window-start", required=True)
    parser.add_argument("--window-end", required=True)
    parser.add_argument("--universe-size", type=int, default=60)
    parser.add_argument("--min-factor-observations", type=int, default=252)
    parser.add_argument("--min-window-observations", type=int, default=100)
    parser.add_argument("--min-pairwise-observations", type=int, default=60)
    parser.add_argument("--max-cap-staleness-days", type=int, default=10)
    parser.add_argument("--chunksize", type=int, default=25_000)
    parser.add_argument("--max-source-rows-per-partition", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    if len(args.source) != len(args.receipt):
        raise ValueError("Every --source requires one position-matched --receipt.")
    frozen_predecessor_hashes()
    base_v3 = bind_base_v3(args.base_v3_manifest)
    source_bindings = [
        bind_source_partition(source, receipt)
        for source, receipt in zip(args.source, args.receipt, strict=True)
    ]
    factor_binding = bind_factor_source(args.factors_csv, args.factor_registry)
    spec = RealDesignCellSpec(
        cell_id=str(args.cell_id),
        factor_fit_start=str(args.factor_fit_start),
        factor_fit_end=str(args.factor_fit_end),
        formation_date=str(args.formation_date),
        window_start=str(args.window_start),
        window_end=str(args.window_end),
        universe_size=int(args.universe_size),
        min_factor_observations=int(args.min_factor_observations),
        min_window_observations=int(args.min_window_observations),
        min_pairwise_observations=int(args.min_pairwise_observations),
        max_cap_staleness_days=int(args.max_cap_staleness_days),
        proof_only=True,
    )
    spec.validate()
    source_frame, scan_receipt = load_filtered_sources(
        source_bindings,
        start=spec.factor_fit_start,
        end=spec.window_end,
        chunksize=int(args.chunksize),
        max_source_rows_per_partition=args.max_source_rows_per_partition,
    )
    factors = load_bound_factors(
        factor_binding,
        start=spec.factor_fit_start,
        end=spec.window_end,
    )
    cell = derive_real_design_cell(
        source_frame,
        factors,
        spec=spec,
        source_bindings=source_bindings,
        factor_binding=factor_binding,
        scan_receipt=scan_receipt,
    )
    cell_artifact = write_real_design_cell(cell, args.cell_out)
    manifest = build_manifest_v4(
        base_v3=base_v3,
        source_bindings=source_bindings,
        factor_binding=factor_binding,
        source_scan_receipt=scan_receipt,
        cell_artifact=cell_artifact,
    )
    validate_manifest_v4(manifest)
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    validate_manifest_v4(json.loads(out.read_text(encoding="utf-8")))
    return out


if __name__ == "__main__":  # pragma: no cover
    main()
