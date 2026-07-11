from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from fjs.real_design_contract import (
    FACTOR_COLUMNS,
    RealDesignCellSpec,
    _serialize_float_matrix,
    _serialize_int_matrix,
    _serialize_missingness,
    bind_factor_source,
    bind_source_partition,
    covariance_contract,
    file_sha256,
    residualization_contract,
    source_contract,
    stable_sha256,
    universe_contract,
    write_real_design_cell,
)
from fjs.real_design_finalizer import (
    build_cell_receipt,
    build_final_manifest,
    checkpoint_status,
    expected_cell_id,
    independent_readback,
    load_checkpoint,
    load_final_manifest,
    new_checkpoint,
    register_cell,
    required_months,
    validate_final_manifest,
    write_checkpoint,
    write_final_manifest,
    write_readback,
)
from tools import finalize_fjs_m4_real_design_v4


def _source_and_factor_bindings(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, Path]]:
    raw_root = tmp_path / "raw"
    items = []
    source_paths: dict[str, Path] = {}
    for month in required_months():
        source = raw_root / f"month={month}" / "data.csv.gz"
        source.parent.mkdir(parents=True)
        source.write_bytes(f"bounded-source-{month}\n".encode())
        source_paths[month] = source
        items.append(
            {
                "status": "ok",
                "path": str(source.resolve()),
                "partition": f"month={month}",
                "rows": 10,
                "size_bytes": source.stat().st_size,
                "schema": "crsp",
                "table": "wrds_dsfv2_query",
                "date_column": "dlycaldt",
            }
        )
    receipt = tmp_path / "receipts" / "manifest.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps({"status": "ok", "items": items}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    factors = tmp_path / "data" / "factors" / "ff5mom_daily.csv"
    factors.parent.mkdir(parents=True)
    factors.write_text(
        "date,MKT,SMB,HML,RMW,CMA,RF,MOM\n"
        "2013-01-02,0.01,0.001,-0.002,0.003,0.004,0.0001,0.005\n",
        encoding="utf-8",
    )
    registry = tmp_path / "data" / "factors" / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "datasets": {
                    "data/factors/ff5mom_daily.csv": {
                        "path": str(factors.resolve()),
                        "sha256": file_sha256(factors),
                        "columns": [*FACTOR_COLUMNS, "RF"],
                        "start_date": "2013-01-02",
                        "end_date": "2018-12-31",
                        "source": "fixture.ken_french.ff5_umd_daily",
                    }
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        {
            month: bind_source_partition(path, receipt)
            for month, path in source_paths.items()
        },
        {"receipt": receipt, "factors": factors, "registry": registry},
    )


def _cell_payload(
    *, month: str, source_binding: object, factor_binding: object
) -> dict[str, object]:
    cell_id = expected_cell_id(month)
    spec = RealDesignCellSpec(
        cell_id=cell_id,
        factor_fit_start="2013-01-02",
        factor_fit_end="2013-01-10",
        formation_date="2013-01-10",
        window_start="2013-01-11",
        window_end="2013-01-14",
        universe_size=2,
        min_factor_observations=7,
        min_window_observations=2,
        min_pairwise_observations=2,
        max_cap_staleness_days=3,
        proof_only=False,
    )
    members = [
        {
            "rank": 1,
            "permno": 10001,
            "lagged_market_cap": 2_000_000.0,
            "cap_observation_date": "2013-01-10",
        },
        {
            "rank": 2,
            "permno": 10002,
            "lagged_market_cap": 1_000_000.0,
            "cap_observation_date": "2013-01-10",
        },
    ]
    source = source_binding.to_dict()
    factor = factor_binding.to_dict()
    cell: dict[str, object] = {
        "schema": "fjs-real-design-cell/v1",
        "cell_id": cell_id,
        "purpose": "Synthetic finalizer contract fixture; no result.",
        "claim_boundary": {
            "development_only": True,
            "mechanism_calibration_only": True,
            "empirical_claims_forbidden": True,
            "promotion_allowed": False,
            "proof_only": False,
            "legacy_ticker_csv_used": False,
            "holdout_2025_opened": False,
        },
        "spec": spec.to_dict(),
        "source_contract": source_contract(),
        "source_partitions": [source],
        "source_scan_receipt": {
            "partitions": [
                {
                    "binding_sha256": source["binding_sha256"],
                    "rows_scanned": 10,
                    "rows_receipted": 10,
                    "rows_after_frozen_filters_and_date_bounds": 10,
                    "scan_truncated": False,
                }
            ],
            "sha256": stable_sha256({"month": month, "rows": 10}),
        },
        "factor_source": factor,
        "residualization_contract": residualization_contract(),
        "covariance_contract": covariance_contract(),
        "universe": {
            "contract": universe_contract(),
            "formation_date": "2013-01-10",
            "members": members,
            "member_set_sha256": stable_sha256(members),
        },
        "factor_fit": {
            "start": "2013-01-02",
            "end": "2013-01-10",
            "window_start": "2013-01-11",
            "observations_per_asset": [7, 7],
            "coefficient_order": ["intercept", *FACTOR_COLUMNS],
            "coefficients": _serialize_float_matrix(np.zeros((2, 7))),
        },
        "window_geometry": {
            "dates": ["2013-01-11", "2013-01-14"],
            "week_labels": ["2013-01-07", "2013-01-14"],
            "weekday_slots": [4, 0],
            "group_sizes": [1, 1],
            "p_assets": 2,
            "n_dates": 2,
            "n_groups": 2,
            "replicate_slots": 5,
            "complete_balanced_groups": 0,
            "between_degrees_of_freedom": 1,
            "within_degrees_of_freedom": 0,
            "between_aspect_ratio": 2.0,
            "within_aspect_ratio": None,
        },
        "missingness": {
            "observed_mask": _serialize_missingness(np.ones((2, 2), dtype=bool)),
            "observed_per_asset": [2, 2],
            "observed_per_date": [2, 2],
            "missing_fraction": 0.0,
        },
        "residual_covariance": {
            "matrix": _serialize_float_matrix(np.eye(2)),
            "pairwise_observation_counts": _serialize_int_matrix(
                np.full((2, 2), 2, dtype=np.int32)
            ),
            "diagnostics": {
                "raw_min_eigenvalue": 1.0,
                "raw_max_eigenvalue": 1.0,
                "projection_floor": 1e-10,
                "projected_min_eigenvalue": 1.0,
            },
        },
    }
    cell["cell_digest"] = stable_sha256(cell)
    return cell


def _generation(tmp_path: Path) -> tuple[list[Path], dict[str, Path]]:
    source_bindings, paths = _source_and_factor_bindings(tmp_path)
    factor_binding = bind_factor_source(paths["factors"], paths["registry"])
    cells = []
    for month in required_months():
        cell = _cell_payload(
            month=month,
            source_binding=source_bindings[month],
            factor_binding=factor_binding,
        )
        path = tmp_path / "cells" / f"{expected_cell_id(month)}.json"
        write_real_design_cell(cell, path)
        cells.append(path)
    return cells, paths


def _complete_checkpoint(tmp_path: Path) -> tuple[dict[str, object], list[Path]]:
    cells, _ = _generation(tmp_path)
    checkpoint = new_checkpoint("fjs-m4-v4-synthetic-finalizer")
    for month, cell in zip(required_months(), cells, strict=True):
        receipt = build_cell_receipt(
            generation_id=checkpoint["generation_id"],
            month=month,
            cell_path=cell,
        )
        checkpoint = register_cell(checkpoint, receipt)
    return checkpoint, cells


@pytest.mark.unit
def test_finalizer_restart_and_independent_readback_are_byte_stable(
    tmp_path: Path,
) -> None:
    cells, _ = _generation(tmp_path)
    generation = "fjs-m4-v4-synthetic-finalizer"
    checkpoint = new_checkpoint(generation)
    first = build_cell_receipt(
        generation_id=generation, month="2013-01", cell_path=cells[0]
    )
    checkpoint = register_cell(checkpoint, first)
    digest_after_first = checkpoint["checkpoint_digest"]
    assert register_cell(checkpoint, first) == checkpoint
    assert checkpoint["checkpoint_digest"] == digest_after_first

    checkpoint_path = tmp_path / "state" / "checkpoint.json"
    write_checkpoint(checkpoint, checkpoint_path)
    resumed = load_checkpoint(checkpoint_path, revalidate_artifacts=True)
    assert checkpoint_status(resumed)["completion_count"] == 1
    with pytest.raises(ValueError, match="incomplete"):
        build_final_manifest(resumed)

    for month, cell in zip(required_months()[1:], cells[1:], strict=True):
        receipt = build_cell_receipt(
            generation_id=generation, month=month, cell_path=cell
        )
        resumed = register_cell(resumed, receipt)
    assert checkpoint_status(resumed)["complete"] is True
    assert checkpoint_status(resumed)["missing_months"] == []
    write_checkpoint(resumed, checkpoint_path)

    manifest = build_final_manifest(load_checkpoint(checkpoint_path))
    first_manifest = tmp_path / "final" / "manifest-a.json"
    second_manifest = tmp_path / "final" / "manifest-b.json"
    write_final_manifest(manifest, first_manifest)
    write_final_manifest(manifest, second_manifest)
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    loaded = load_final_manifest(first_manifest)
    assert loaded == manifest
    assert loaded["execution_readiness"]["full_execution_ready"] is False
    assert loaded["execution_readiness"]["aws_execution_authorized"] is False
    assert loaded["execution_readiness"]["outcomes_present"] is False
    assert loaded["boundaries"]["holdout_2025_opened"] is False

    readback = independent_readback(first_manifest)
    assert readback["passed"] is True
    assert readback["source_count"] == 72
    assert readback["cell_count"] == 72
    receipt_path = tmp_path / "final" / "readback.json"
    write_readback(readback, receipt_path)
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == readback


@pytest.mark.unit
def test_finalizer_rejects_conflicts_missing_months_and_cross_generation(
    tmp_path: Path,
) -> None:
    cells, _ = _generation(tmp_path)
    checkpoint = new_checkpoint("fjs-m4-v4-conflict-test")
    receipt = build_cell_receipt(
        generation_id=checkpoint["generation_id"],
        month="2013-01",
        cell_path=cells[0],
    )
    checkpoint = register_cell(checkpoint, receipt)
    with pytest.raises(ValueError, match="incomplete"):
        build_final_manifest(checkpoint)

    alternate = tmp_path / "alternate" / cells[0].name
    alternate.parent.mkdir(parents=True)
    shutil.copyfile(cells[0], alternate)
    conflicting = build_cell_receipt(
        generation_id=checkpoint["generation_id"],
        month="2013-01",
        cell_path=alternate,
    )
    with pytest.raises(ValueError, match="Conflicting duplicate month"):
        register_cell(checkpoint, conflicting)

    foreign = build_cell_receipt(
        generation_id="fjs-m4-v4-foreign-generation",
        month="2013-02",
        cell_path=cells[1],
    )
    with pytest.raises(ValueError, match="another generation"):
        register_cell(checkpoint, foreign)
    with pytest.raises(ValueError, match="identity mismatch"):
        build_cell_receipt(
            generation_id=checkpoint["generation_id"],
            month="2013-02",
            cell_path=cells[0],
        )
    with pytest.raises(ValueError, match="2013-2018"):
        expected_cell_id("2025-01")
    with pytest.raises(ValueError, match="2025"):
        new_checkpoint("fjs-m4-v4-2025-forbidden")

    outcome_cell = json.loads(cells[0].read_text(encoding="utf-8"))
    outcome_cell["outcomes"] = {"detector_rate": 1.0}
    outcome_cell["cell_digest"] = stable_sha256(
        {key: value for key, value in outcome_cell.items() if key != "cell_digest"}
    )
    outcome_path = tmp_path / "forbidden-outcome-cell.json"
    outcome_path.write_text(
        json.dumps(outcome_cell, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden outcome"):
        build_cell_receipt(
            generation_id=checkpoint["generation_id"],
            month="2013-01",
            cell_path=outcome_path,
        )


@pytest.mark.unit
def test_final_manifest_rejects_missing_duplicate_and_tampered_artifacts(
    tmp_path: Path,
) -> None:
    checkpoint, cells = _complete_checkpoint(tmp_path)
    manifest = build_final_manifest(checkpoint)

    missing = copy.deepcopy(manifest)
    missing["source_catalog"].pop()
    with pytest.raises(ValueError, match="exactly 72"):
        validate_final_manifest(missing, revalidate_artifacts=False)

    duplicate = copy.deepcopy(manifest)
    duplicate["source_catalog"][1] = copy.deepcopy(duplicate["source_catalog"][0])
    duplicate["source_set_digest"] = stable_sha256(duplicate["source_catalog"])
    duplicate["manifest_digest"] = stable_sha256(
        {key: value for key, value in duplicate.items() if key != "manifest_digest"}
    )
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        validate_final_manifest(duplicate, revalidate_artifacts=False)

    aggregate_tamper = copy.deepcopy(manifest)
    aggregate_tamper["cell_set_digest"] = "0" * 64
    with pytest.raises(ValueError, match="cell-set digest"):
        validate_final_manifest(aggregate_tamper, revalidate_artifacts=False)

    manifest_path = tmp_path / "final.json"
    write_final_manifest(manifest, manifest_path)
    cells[0].write_bytes(cells[0].read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="cell artifact changed|digest mismatch"):
        independent_readback(manifest_path)


@pytest.mark.unit
def test_finalizer_cli_init_register_status_and_incomplete_stop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cells, _ = _generation(tmp_path)
    checkpoint = tmp_path / "cli" / "checkpoint.json"
    finalize_fjs_m4_real_design_v4.main(
        [
            "init",
            "--checkpoint",
            str(checkpoint),
            "--generation-id",
            "fjs-m4-v4-cli-finalizer",
        ]
    )
    finalize_fjs_m4_real_design_v4.main(
        [
            "register",
            "--checkpoint",
            str(checkpoint),
            "--month",
            "2013-01",
            "--cell",
            str(cells[0]),
        ]
    )
    finalize_fjs_m4_real_design_v4.main(["status", "--checkpoint", str(checkpoint)])
    output = capsys.readouterr().out
    assert '"completion_count": 1' in output
    assert '"full_execution_ready": false' in output
    with pytest.raises(ValueError, match="incomplete"):
        finalize_fjs_m4_real_design_v4.main(
            [
                "finalize",
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(tmp_path / "cli" / "manifest.json"),
            ]
        )
