from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fjs.real_design_contract import (
    bind_factor_source,
    bind_source_partition,
    file_sha256,
    stable_json_dumps,
    stable_sha256,
    validate_real_design_cell,
)

ROOT = Path(__file__).resolve().parents[2]
CELL_RECEIPT_SCHEMA = "fjs-real-design-cell-receipt/v1"
CHECKPOINT_SCHEMA = "fjs-real-design-finalizer-checkpoint/v1"
FINAL_MANIFEST_SCHEMA = "fjs-real-design-full-manifest/v1"
READBACK_SCHEMA = "fjs-real-design-full-readback/v1"
FINALIZER_CONTRACT_ID = "fjs-m4-v4-72-month-finalizer-v1"
GENERATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{5,127}$")

PUBLISHED_V4_LOCAL_HEAD = "90746fdb7c74dd333c14f78d1bfe7cec0016c952"
PUBLISHED_V4_TREE = "040ffb6c8407d9bf7b8b887dc611e37948fb437d"
PUBLISHED_V4_REMOTE_COMMIT = "967fcf5b2c0db171972a85d575149859ddb2ad05"
PUBLISHED_V4_PROOF_CELL_SHA256 = (
    "855d3a57673d2bee0ae40c06eb96268ee64dffd6039d6207da52c99d8896d208"
)
PUBLISHED_V4_PROOF_MANIFEST_SHA256 = (
    "e2d2d9880dc9c5e4533085ce2b396ea6aa152043bed8aa82172c46d4865a3f39"
)

CONTRACT_BINDING_PATHS = (
    "src/fjs/real_design_contract.py",
    "tools/freeze_fjs_m4_real_design_v4.py",
    "src/fjs/real_design_finalizer.py",
    "tools/finalize_fjs_m4_real_design_v4.py",
)
FINAL_EXECUTION_BLOCKERS = (
    "trusted_route_admission_required",
    "fresh_authoritative_aws_admission_required",
    "full_detector_calibration_outcomes_not_run",
)
FORBIDDEN_CELL_KEYS = {
    "aws",
    "aws_job",
    "aws_job_id",
    "calibration_results",
    "detector_results",
    "metrics",
    "outcome",
    "outcomes",
    "performance",
    "results",
    "submission",
}

CELL_RECEIPT_KEYS = {
    "schema",
    "generation_id",
    "month",
    "cell_id",
    "cell_artifact",
    "primary_source_binding",
    "source_partition_months",
    "source_binding_sha256",
    "factor_binding_sha256",
    "boundaries",
    "identity_sha256",
    "receipt_digest",
}
CHECKPOINT_KEYS = {
    "schema",
    "generation_id",
    "contract",
    "boundaries",
    "predecessor",
    "contract_bindings",
    "completed_cells",
    "completion_count",
    "complete",
    "checkpoint_digest",
}
FINAL_MANIFEST_KEYS = {
    "schema",
    "generation_id",
    "contract",
    "boundaries",
    "predecessor",
    "contract_bindings",
    "execution_readiness",
    "source_catalog",
    "factor_source",
    "cells",
    "source_set_digest",
    "cell_set_digest",
    "manifest_digest",
}


def required_months() -> list[str]:
    return [
        f"{year:04d}-{month:02d}"
        for year in range(2013, 2019)
        for month in range(1, 13)
    ]


def expected_cell_id(month: str) -> str:
    if month not in required_months():
        raise ValueError(f"Month is outside the frozen 2013-2018 set: {month!r}")
    return f"fjs-real-design-{month}-v4"


def _validate_generation_id(generation_id: str) -> str:
    value = str(generation_id)
    if GENERATION_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Invalid generation_id: {value!r}")
    if "2025" in value:
        raise ValueError("The generation identity may not reference the 2025 holdout.")
    return value


def boundary_contract() -> dict[str, Any]:
    return {
        "development_months_only": required_months(),
        "legacy_ticker_csv_used": False,
        "legacy_ticker_csv_provenance_inferred": False,
        "outcomes_present": False,
        "detector_run_present": False,
        "empirical_claims_allowed": False,
        "promotion_allowed": False,
        "holdout_2025_opened": False,
        "aws_execution_authorized": False,
        "readback_required_before_any_execution": True,
    }


def generation_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contract_id": FINALIZER_CONTRACT_ID,
        "required_months": required_months(),
        "required_month_count": 72,
        "expected_cell_ids": [expected_cell_id(month) for month in required_months()],
        "one_primary_source_per_month": True,
        "one_cell_per_month": True,
        "cell_identity_bound_to_generation": True,
        "exact_source_receipt_and_content_hash_required": True,
        "missing_month_policy": "fail",
        "duplicate_month_policy": "fail_unless_exact_idempotent_restart",
        "conflicting_source_policy": "fail",
        "finalization_policy": "all_72_then_independent_readback",
        "outcome_fields_forbidden": True,
    }
    payload["sha256"] = stable_sha256(payload)
    return payload


def _file_binding(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"Finalizer contract input is missing: {relative}")
    return {
        "path": relative,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def contract_bindings() -> dict[str, dict[str, Any]]:
    return {relative: _file_binding(relative) for relative in CONTRACT_BINDING_PATHS}


def predecessor_binding() -> dict[str, Any]:
    return {
        "local_head": PUBLISHED_V4_LOCAL_HEAD,
        "tree": PUBLISHED_V4_TREE,
        "remote_commit": PUBLISHED_V4_REMOTE_COMMIT,
        "branch": "portfolio/fjs-recenter-m1-20260710",
        "bounded_proof_cell_file_sha256": PUBLISHED_V4_PROOF_CELL_SHA256,
        "bounded_proof_manifest_file_sha256": PUBLISHED_V4_PROOF_MANIFEST_SHA256,
        "real_design_contract_sha256": file_sha256(
            ROOT / "src/fjs/real_design_contract.py"
        ),
        "bounded_freezer_sha256": file_sha256(
            ROOT / "tools/freeze_fjs_m4_real_design_v4.py"
        ),
    }


def _assert_exact_keys(
    payload: Mapping[str, Any], expected: set[str], label: str
) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"{label} keys mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _digest_without(payload: Mapping[str, Any], key: str) -> str:
    return stable_sha256(
        {name: value for name, value in payload.items() if name != key}
    )


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> Path:
    out = path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.tmp")
    temporary.write_text(stable_json_dumps(payload) + "\n", encoding="utf-8")
    os.replace(temporary, out)
    return out


def _rebind_source(observed: Mapping[str, Any]) -> dict[str, Any]:
    rebound = bind_source_partition(
        Path(str(observed["path"])),
        Path(str(observed["receipt_manifest_path"])),
    ).to_dict()
    if rebound != dict(observed):
        raise ValueError(
            f"Source binding changed for partition {observed.get('partition')!r}."
        )
    return rebound


def _rebind_factor(observed: Mapping[str, Any]) -> dict[str, Any]:
    rebound = bind_factor_source(
        Path(str(observed["path"])), Path(str(observed["registry_path"]))
    ).to_dict()
    if rebound != dict(observed):
        raise ValueError("Factor binding changed after cell creation.")
    return rebound


def _read_cell(path: Path) -> dict[str, Any]:
    cell_path = path.expanduser().resolve()
    if not cell_path.is_file():
        raise FileNotFoundError(f"Real-design cell artifact is missing: {cell_path}")
    payload = json.loads(cell_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Real-design cell artifact must be a JSON object.")
    validate_real_design_cell(payload)
    forbidden = _forbidden_cell_keys(payload)
    if forbidden:
        raise ValueError(
            "Real-design input cell contains forbidden outcome/external fields: "
            f"{forbidden}"
        )
    return payload


def _forbidden_cell_keys(payload: object, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            name = str(key)
            locator = f"{prefix}.{name}" if prefix else name
            if name.lower() in FORBIDDEN_CELL_KEYS:
                found.append(locator)
            found.extend(_forbidden_cell_keys(value, locator))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_forbidden_cell_keys(value, f"{prefix}[{index}]"))
    return sorted(found)


def build_cell_receipt(
    *, generation_id: str, month: str, cell_path: Path
) -> dict[str, Any]:
    generation = _validate_generation_id(generation_id)
    required = required_months()
    if month not in required:
        raise ValueError(f"Cell month is not required by v4: {month!r}")
    cell_file = cell_path.expanduser().resolve()
    cell = _read_cell(cell_file)
    expected_id = expected_cell_id(month)
    if (
        cell.get("cell_id") != expected_id
        or cell.get("spec", {}).get("cell_id") != expected_id
    ):
        raise ValueError(
            f"Cell identity mismatch for {month}: expected {expected_id!r}."
        )
    claim = cell.get("claim_boundary")
    if not isinstance(claim, Mapping):
        raise ValueError("Cell claim boundary is missing.")
    if claim.get("proof_only") is not False:
        raise ValueError("Full-generation cells may not be proof-only.")
    if claim.get("legacy_ticker_csv_used") is not False:
        raise ValueError("Full-generation cells may not use the legacy ticker CSV.")
    if claim.get("holdout_2025_opened") is not False:
        raise ValueError("The 2025 holdout must remain unopened.")

    raw_sources = cell.get("source_partitions")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Full-generation cell is missing source partitions.")
    rebound_sources = []
    for raw in raw_sources:
        if not isinstance(raw, Mapping):
            raise ValueError("Cell source binding must be a mapping.")
        rebound_sources.append(_rebind_source(raw))
    source_months = sorted(
        str(item["partition"]).removeprefix("month=") for item in rebound_sources
    )
    if len(set(source_months)) != len(source_months):
        raise ValueError("Cell contains duplicate source partition months.")
    if any(value not in required for value in source_months):
        raise ValueError("Cell source bindings leave the frozen 2013-2018 set.")
    primary = [
        item for item in rebound_sources if item["partition"] == f"month={month}"
    ]
    if len(primary) != 1:
        raise ValueError(
            f"Cell {expected_id} must bind exactly one primary month={month} source."
        )
    raw_factor = cell.get("factor_source")
    if not isinstance(raw_factor, Mapping):
        raise ValueError("Cell factor binding is missing.")
    factor = _rebind_factor(raw_factor)

    artifact = {
        "path": str(cell_file),
        "sha256": file_sha256(cell_file),
        "size_bytes": cell_file.stat().st_size,
        "cell_digest": str(cell["cell_digest"]),
    }
    identity_payload = {
        "generation_id": generation,
        "month": month,
        "cell_id": expected_id,
        "cell_artifact": artifact,
        "primary_source_binding_sha256": primary[0]["binding_sha256"],
        "source_binding_sha256": sorted(
            str(item["binding_sha256"]) for item in rebound_sources
        ),
        "factor_binding_sha256": factor["binding_sha256"],
    }
    receipt: dict[str, Any] = {
        "schema": CELL_RECEIPT_SCHEMA,
        "generation_id": generation,
        "month": month,
        "cell_id": expected_id,
        "cell_artifact": artifact,
        "primary_source_binding": primary[0],
        "source_partition_months": source_months,
        "source_binding_sha256": identity_payload["source_binding_sha256"],
        "factor_binding_sha256": factor["binding_sha256"],
        "boundaries": boundary_contract(),
        "identity_sha256": stable_sha256(identity_payload),
    }
    receipt["receipt_digest"] = stable_sha256(receipt)
    return receipt


def validate_cell_receipt(
    receipt: Mapping[str, Any], *, revalidate_artifact: bool
) -> None:
    _assert_exact_keys(receipt, CELL_RECEIPT_KEYS, "cell receipt")
    generation = _validate_generation_id(str(receipt["generation_id"]))
    month = str(receipt["month"])
    if month not in required_months():
        raise ValueError(f"Cell receipt month is not required: {month!r}")
    if receipt["cell_id"] != expected_cell_id(month):
        raise ValueError("Cell receipt identity does not match its month.")
    if receipt["boundaries"] != boundary_contract():
        raise ValueError("Cell receipt boundary mismatch.")
    primary = receipt["primary_source_binding"]
    if not isinstance(primary, Mapping) or primary.get("partition") != f"month={month}":
        raise ValueError("Cell receipt primary source does not match its month.")
    months = receipt["source_partition_months"]
    if not isinstance(months, list) or months != sorted(set(str(v) for v in months)):
        raise ValueError("Cell receipt source month list is not unique and sorted.")
    if month not in months or any(value not in required_months() for value in months):
        raise ValueError("Cell receipt source months violate frozen coverage.")
    source_digests = receipt["source_binding_sha256"]
    if not isinstance(source_digests, list) or source_digests != sorted(
        set(str(value) for value in source_digests)
    ):
        raise ValueError("Cell receipt source digests are not unique and sorted.")
    artifact = receipt["cell_artifact"]
    if not isinstance(artifact, Mapping):
        raise ValueError("Cell receipt artifact is missing.")
    identity_payload = {
        "generation_id": generation,
        "month": month,
        "cell_id": receipt["cell_id"],
        "cell_artifact": dict(artifact),
        "primary_source_binding_sha256": primary["binding_sha256"],
        "source_binding_sha256": source_digests,
        "factor_binding_sha256": receipt["factor_binding_sha256"],
    }
    if receipt["identity_sha256"] != stable_sha256(identity_payload):
        raise ValueError("Cell receipt restart identity digest mismatch.")
    if receipt["receipt_digest"] != _digest_without(receipt, "receipt_digest"):
        raise ValueError("Cell receipt digest mismatch.")
    if revalidate_artifact:
        rebuilt = build_cell_receipt(
            generation_id=generation,
            month=month,
            cell_path=Path(str(artifact["path"])),
        )
        if rebuilt != dict(receipt):
            raise ValueError(f"Cell receipt readback changed for {month}.")


def new_checkpoint(generation_id: str) -> dict[str, Any]:
    generation = _validate_generation_id(generation_id)
    checkpoint: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "generation_id": generation,
        "contract": generation_contract(),
        "boundaries": boundary_contract(),
        "predecessor": predecessor_binding(),
        "contract_bindings": contract_bindings(),
        "completed_cells": [],
        "completion_count": 0,
        "complete": False,
    }
    checkpoint["checkpoint_digest"] = stable_sha256(checkpoint)
    return checkpoint


def validate_checkpoint(
    checkpoint: Mapping[str, Any], *, revalidate_artifacts: bool
) -> None:
    _assert_exact_keys(checkpoint, CHECKPOINT_KEYS, "finalizer checkpoint")
    _validate_generation_id(str(checkpoint["generation_id"]))
    if checkpoint["contract"] != generation_contract():
        raise ValueError("Finalizer checkpoint contract mismatch.")
    if checkpoint["boundaries"] != boundary_contract():
        raise ValueError("Finalizer checkpoint boundary mismatch.")
    if checkpoint["predecessor"] != predecessor_binding():
        raise ValueError("Finalizer checkpoint predecessor mismatch.")
    if checkpoint["contract_bindings"] != contract_bindings():
        raise ValueError("Finalizer checkpoint executable binding mismatch.")
    cells = checkpoint["completed_cells"]
    if not isinstance(cells, list):
        raise ValueError("Finalizer checkpoint cells must be a list.")
    months: list[str] = []
    cell_ids: list[str] = []
    artifact_paths: list[str] = []
    for receipt in cells:
        if not isinstance(receipt, Mapping):
            raise ValueError("Finalizer checkpoint cell receipt must be a mapping.")
        validate_cell_receipt(receipt, revalidate_artifact=revalidate_artifacts)
        if receipt["generation_id"] != checkpoint["generation_id"]:
            raise ValueError("Checkpoint contains a cross-generation cell receipt.")
        months.append(str(receipt["month"]))
        cell_ids.append(str(receipt["cell_id"]))
        artifact_paths.append(str(receipt["cell_artifact"]["path"]))
    if months != sorted(months) or len(months) != len(set(months)):
        raise ValueError("Checkpoint months must be unique and sorted.")
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("Checkpoint cell identities must be unique.")
    if len(artifact_paths) != len(set(artifact_paths)):
        raise ValueError("Checkpoint artifact paths must be unique.")
    if int(checkpoint["completion_count"]) != len(cells):
        raise ValueError("Checkpoint completion count mismatch.")
    is_complete = months == required_months()
    if bool(checkpoint["complete"]) != is_complete:
        raise ValueError("Checkpoint completeness flag mismatch.")
    if checkpoint["checkpoint_digest"] != _digest_without(
        checkpoint, "checkpoint_digest"
    ):
        raise ValueError("Finalizer checkpoint digest mismatch.")


def register_cell(
    checkpoint: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    validate_checkpoint(checkpoint, revalidate_artifacts=False)
    validate_cell_receipt(receipt, revalidate_artifact=True)
    if receipt["generation_id"] != checkpoint["generation_id"]:
        raise ValueError("Cannot register a cell from another generation.")
    updated = copy.deepcopy(dict(checkpoint))
    cells = list(updated["completed_cells"])
    existing = [item for item in cells if item["month"] == receipt["month"]]
    if existing:
        if len(existing) == 1 and existing[0] == dict(receipt):
            return updated
        raise ValueError(
            f"Conflicting duplicate month registration: {receipt['month']}"
        )
    if any(item["cell_id"] == receipt["cell_id"] for item in cells):
        raise ValueError(f"Duplicate cell identity: {receipt['cell_id']}")
    if any(
        item["cell_artifact"]["path"] == receipt["cell_artifact"]["path"]
        for item in cells
    ):
        raise ValueError("A cell artifact path cannot satisfy two months.")
    cells.append(copy.deepcopy(dict(receipt)))
    cells.sort(key=lambda item: str(item["month"]))
    updated["completed_cells"] = cells
    updated["completion_count"] = len(cells)
    updated["complete"] = [str(item["month"]) for item in cells] == required_months()
    updated["checkpoint_digest"] = _digest_without(updated, "checkpoint_digest")
    validate_checkpoint(updated, revalidate_artifacts=False)
    return updated


def write_checkpoint(checkpoint: Mapping[str, Any], path: Path) -> Path:
    validate_checkpoint(checkpoint, revalidate_artifacts=False)
    return _atomic_write_json(checkpoint, path)


def load_checkpoint(
    path: Path, *, revalidate_artifacts: bool = False
) -> dict[str, Any]:
    checkpoint_path = path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Finalizer checkpoint is missing: {checkpoint_path}")
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Finalizer checkpoint must be a JSON object.")
    validate_checkpoint(payload, revalidate_artifacts=revalidate_artifacts)
    return payload


def checkpoint_status(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    validate_checkpoint(checkpoint, revalidate_artifacts=False)
    completed = [str(item["month"]) for item in checkpoint["completed_cells"]]
    missing = [month for month in required_months() if month not in set(completed)]
    return {
        "generation_id": checkpoint["generation_id"],
        "completion_count": len(completed),
        "required_count": 72,
        "complete": not missing,
        "completed_months": completed,
        "missing_months": missing,
        "checkpoint_digest": checkpoint["checkpoint_digest"],
        "full_execution_ready": False,
        "aws_execution_authorized": False,
        "outcomes_present": False,
        "holdout_2025_opened": False,
    }


def _execution_readiness() -> dict[str, Any]:
    return {
        "real_design_full_generation_complete": True,
        "independent_readback_required": True,
        "full_execution_ready": False,
        "aws_execution_authorized": False,
        "outcomes_present": False,
        "blockers": list(FINAL_EXECUTION_BLOCKERS),
    }


def build_final_manifest(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    validate_checkpoint(checkpoint, revalidate_artifacts=True)
    status = checkpoint_status(checkpoint)
    if not status["complete"]:
        raise ValueError(
            "Cannot finalize an incomplete v4 generation; missing months: "
            f"{status['missing_months']}"
        )
    cells = copy.deepcopy(list(checkpoint["completed_cells"]))
    source_catalog = [copy.deepcopy(item["primary_source_binding"]) for item in cells]
    source_catalog.sort(key=lambda item: str(item["partition"]))
    source_months = [
        str(item["partition"]).removeprefix("month=") for item in source_catalog
    ]
    if source_months != required_months():
        raise ValueError(
            "Final source catalog is incomplete, duplicated, or misordered."
        )

    factor_source: dict[str, Any] | None = None
    source_by_month = {
        str(item["partition"]).removeprefix("month="): item for item in source_catalog
    }
    for receipt in cells:
        cell = _read_cell(Path(str(receipt["cell_artifact"]["path"])))
        raw_factor = cell["factor_source"]
        if factor_source is None:
            factor_source = copy.deepcopy(raw_factor)
        elif factor_source != raw_factor:
            raise ValueError("Full generation contains inconsistent factor bindings.")
        for source in cell["source_partitions"]:
            source_month = str(source["partition"]).removeprefix("month=")
            if source_month not in source_by_month:
                raise ValueError(
                    f"Cell {receipt['cell_id']} references an unfinalized source month."
                )
            if source_by_month[source_month] != source:
                raise ValueError(
                    f"Cell {receipt['cell_id']} conflicts with the source catalog."
                )
    if factor_source is None:
        raise ValueError("Full generation contains no factor binding.")
    _rebind_factor(factor_source)

    manifest: dict[str, Any] = {
        "schema": FINAL_MANIFEST_SCHEMA,
        "generation_id": checkpoint["generation_id"],
        "contract": generation_contract(),
        "boundaries": boundary_contract(),
        "predecessor": predecessor_binding(),
        "contract_bindings": contract_bindings(),
        "execution_readiness": _execution_readiness(),
        "source_catalog": source_catalog,
        "factor_source": factor_source,
        "cells": cells,
        "source_set_digest": stable_sha256(source_catalog),
        "cell_set_digest": stable_sha256(cells),
    }
    manifest["manifest_digest"] = stable_sha256(manifest)
    validate_final_manifest(manifest, revalidate_artifacts=True)
    return manifest


def validate_final_manifest(
    manifest: Mapping[str, Any], *, revalidate_artifacts: bool
) -> None:
    _assert_exact_keys(manifest, FINAL_MANIFEST_KEYS, "final manifest")
    generation = _validate_generation_id(str(manifest["generation_id"]))
    if manifest["contract"] != generation_contract():
        raise ValueError("Final manifest contract mismatch.")
    if manifest["boundaries"] != boundary_contract():
        raise ValueError("Final manifest boundary mismatch.")
    if manifest["predecessor"] != predecessor_binding():
        raise ValueError("Final manifest predecessor mismatch.")
    if manifest["contract_bindings"] != contract_bindings():
        raise ValueError("Final manifest executable binding mismatch.")
    if manifest["execution_readiness"] != _execution_readiness():
        raise ValueError("Final manifest execution boundary mismatch.")

    sources = manifest["source_catalog"]
    cells = manifest["cells"]
    if not isinstance(sources, list) or not isinstance(cells, list):
        raise ValueError("Final manifest source and cell catalogs must be lists.")
    if len(sources) != 72 or len(cells) != 72:
        raise ValueError("Final manifest requires exactly 72 sources and 72 cells.")
    source_months = [str(item["partition"]).removeprefix("month=") for item in sources]
    if source_months != required_months():
        raise ValueError("Final manifest source months are incomplete or duplicated.")
    cell_months = [str(item["month"]) for item in cells]
    if cell_months != required_months():
        raise ValueError("Final manifest cell months are incomplete or duplicated.")
    if [str(item["cell_id"]) for item in cells] != [
        expected_cell_id(month) for month in required_months()
    ]:
        raise ValueError("Final manifest cell identities are incomplete or misordered.")
    artifact_paths = [str(item["cell_artifact"]["path"]) for item in cells]
    if len(set(artifact_paths)) != 72:
        raise ValueError("Final manifest cell artifact paths must be unique.")
    if manifest["source_set_digest"] != stable_sha256(sources):
        raise ValueError("Final manifest source-set digest mismatch.")
    if manifest["cell_set_digest"] != stable_sha256(cells):
        raise ValueError("Final manifest cell-set digest mismatch.")
    if manifest["manifest_digest"] != _digest_without(manifest, "manifest_digest"):
        raise ValueError("Final manifest aggregate digest mismatch.")

    source_by_month: dict[str, Mapping[str, Any]] = {}
    if revalidate_artifacts:
        for source in sources:
            if not isinstance(source, Mapping):
                raise ValueError("Final source binding must be a mapping.")
            rebound = _rebind_source(source)
            month = str(rebound["partition"]).removeprefix("month=")
            source_by_month[month] = rebound
        factor = manifest["factor_source"]
        if not isinstance(factor, Mapping):
            raise ValueError("Final factor source must be a mapping.")
        _rebind_factor(factor)

    for receipt in cells:
        if not isinstance(receipt, Mapping):
            raise ValueError("Final cell receipt must be a mapping.")
        validate_cell_receipt(receipt, revalidate_artifact=False)
        if receipt["generation_id"] != generation:
            raise ValueError("Final manifest contains a cross-generation receipt.")
        if revalidate_artifacts:
            cell_path = Path(str(receipt["cell_artifact"]["path"]))
            if file_sha256(cell_path) != receipt["cell_artifact"]["sha256"]:
                raise ValueError(f"Final cell artifact changed: {cell_path}")
            cell = _read_cell(cell_path)
            if cell["cell_digest"] != receipt["cell_artifact"]["cell_digest"]:
                raise ValueError("Final cell digest changed after checkpointing.")
            if cell["cell_id"] != receipt["cell_id"]:
                raise ValueError("Final cell identity changed after checkpointing.")
            if cell["factor_source"] != manifest["factor_source"]:
                raise ValueError("Final cell factor binding conflicts with manifest.")
            for source in cell["source_partitions"]:
                month = str(source["partition"]).removeprefix("month=")
                if month not in source_by_month or source_by_month[month] != source:
                    raise ValueError(
                        "Final cell source conflicts with manifest catalog."
                    )


def write_final_manifest(manifest: Mapping[str, Any], path: Path) -> Path:
    validate_final_manifest(manifest, revalidate_artifacts=True)
    return _atomic_write_json(manifest, path)


def load_final_manifest(
    path: Path, *, revalidate_artifacts: bool = True
) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Final v4 manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Final v4 manifest must be a JSON object.")
    validate_final_manifest(payload, revalidate_artifacts=revalidate_artifacts)
    return payload


def independent_readback(path: Path) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    manifest = load_final_manifest(manifest_path, revalidate_artifacts=True)
    receipt: dict[str, Any] = {
        "schema": READBACK_SCHEMA,
        "generation_id": manifest["generation_id"],
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "manifest_digest": manifest["manifest_digest"],
        "source_count": len(manifest["source_catalog"]),
        "cell_count": len(manifest["cells"]),
        "source_set_digest": manifest["source_set_digest"],
        "cell_set_digest": manifest["cell_set_digest"],
        "boundaries": boundary_contract(),
        "passed": True,
    }
    receipt["readback_digest"] = stable_sha256(receipt)
    return receipt


def write_readback(receipt: Mapping[str, Any], path: Path) -> Path:
    expected_keys = {
        "schema",
        "generation_id",
        "manifest_path",
        "manifest_file_sha256",
        "manifest_digest",
        "source_count",
        "cell_count",
        "source_set_digest",
        "cell_set_digest",
        "boundaries",
        "passed",
        "readback_digest",
    }
    _assert_exact_keys(receipt, expected_keys, "readback receipt")
    if receipt["schema"] != READBACK_SCHEMA or receipt["passed"] is not True:
        raise ValueError("Readback receipt is not a passing v4 receipt.")
    if receipt["boundaries"] != boundary_contract():
        raise ValueError("Readback receipt boundary mismatch.")
    if receipt["source_count"] != 72 or receipt["cell_count"] != 72:
        raise ValueError("Readback receipt count mismatch.")
    if receipt["readback_digest"] != _digest_without(receipt, "readback_digest"):
        raise ValueError("Readback receipt digest mismatch.")
    return _atomic_write_json(receipt, path)
