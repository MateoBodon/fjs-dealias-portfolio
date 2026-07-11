from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from tools.fjs_m4_contract_v3 import independently_computed_oneway_boundary

from fjs.real_design_contract import (
    FactorBinding,
    SourcePartitionBinding,
    bind_source_partition,
    file_sha256,
    stable_sha256,
)
from fjs.rolling_geometry_contract import (
    BOUNDED_PROOF_ENDPOINT_MONTH,
    GEOMETRY_COLUMNS,
    WINDOW_WEEKS,
    RollingGeometrySpec,
    build_rolling_geometry_manifest,
    build_rolling_geometry_proof,
    geometry_logical_sha256,
    headline_calibration_claim,
    load_geometry_only_sources,
    resolve_spec,
    rolling_geometry_contract,
    rolling_window_start,
    source_months_for_window,
    validate_rolling_geometry_proof,
)
from tools import freeze_fjs_m5_rolling_geometry


def _calendar() -> pd.DatetimeIndex:
    formation = pd.Timestamp("2013-01-31")
    start = rolling_window_start(formation)
    return pd.bdate_range(start, formation)


def _factor_binding(tmp_path: Path, dates: pd.DatetimeIndex) -> FactorBinding:
    factor = tmp_path / "ff5mom_daily.csv"
    registry = tmp_path / "registry.json"
    factor.write_text(
        "date\n" + "\n".join(value.date().isoformat() for value in dates) + "\n",
        encoding="utf-8",
    )
    registry.write_text("{}\n", encoding="utf-8")
    return FactorBinding(
        path=str(factor.resolve()),
        sha256=file_sha256(factor),
        size_bytes=factor.stat().st_size,
        registry_path=str(registry.resolve()),
        registry_sha256=file_sha256(registry),
        registry_key="fixture",
        source="fixture",
        start_date="2010-01-01",
        end_date="2018-12-31",
        columns=("MKT", "SMB", "HML", "RMW", "CMA", "MOM", "RF"),
        binding_sha256="fixture-factor-binding",
    )


def _source_binding(tmp_path: Path, month: str) -> SourcePartitionBinding:
    source = tmp_path / "sources" / f"month={month}" / "data.csv.gz"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"fixture")
    return SourcePartitionBinding(
        path=str(source.resolve()),
        partition=f"month={month}",
        sha256=file_sha256(source),
        size_bytes=source.stat().st_size,
        receipt_manifest_path=str((tmp_path / f"receipt-{month}.json").resolve()),
        receipt_manifest_sha256="fixture-receipt",
        receipt_status="ok",
        receipt_rows=1,
        receipt_size_bytes=source.stat().st_size,
        receipt_schema="crsp",
        receipt_table="wrds_dsfv2_query",
        receipt_date_column="dlycaldt",
        binding_sha256=f"fixture-source-binding-{month}",
    )


def _geometry_frame(*, one_natural_missing: bool) -> pd.DataFrame:
    dates = _calendar()
    rows = []
    for asset_index, permno in enumerate(range(10001, 10063), start=1):
        for date in dates:
            if (
                one_natural_missing
                and permno == 10062
                and date == dates[len(dates) // 2]
            ):
                continue
            rows.append(
                {
                    "permno": permno,
                    "dlycaldt": date,
                    "dlycap": 1_000_000.0 + asset_index * 10_000.0,
                }
            )
    return pd.DataFrame(rows, columns=list(GEOMETRY_COLUMNS))


def _proof(tmp_path: Path, *, one_natural_missing: bool) -> dict[str, object]:
    dates = _calendar()
    spec = resolve_spec(BOUNDED_PROOF_ENDPOINT_MONTH, dates)
    source_frame = _geometry_frame(one_natural_missing=one_natural_missing)
    source_months = source_months_for_window(spec.window_start, spec.window_end)
    source_bindings = [_source_binding(tmp_path, month) for month in source_months]
    scan = {
        "requested_start": spec.window_start,
        "requested_end": spec.window_end,
        "chunksize": 7,
        "partitions": [
            {
                "binding_sha256": binding.binding_sha256,
                "rows_scanned": binding.receipt_rows,
                "rows_receipted": binding.receipt_rows,
                "rows_after_frozen_filters_and_date_bounds": 1,
                "scan_truncated": False,
            }
            for binding in source_bindings
        ],
        "expected_source_months": source_months,
        "source_binding_set_digest": stable_sha256(
            [binding.binding_sha256 for binding in source_bindings]
        ),
        "rows_after_all_filters": len(source_frame),
        "exact_duplicate_rows_collapsed": 0,
        "return_values_persisted": False,
        "return_presence_and_validity_read": True,
        "logical_geometry_sha256": geometry_logical_sha256(source_frame),
    }
    scan["sha256"] = stable_sha256(scan)
    return build_rolling_geometry_proof(
        source_frame,
        spec=spec,
        source_bindings=source_bindings,
        factor_binding=_factor_binding(tmp_path, dates),
        scan_receipt=scan,
    )


@pytest.mark.unit
def test_v5_contract_freezes_claim_geometry_and_endpoint() -> None:
    contract = rolling_geometry_contract()
    claim = headline_calibration_claim()
    assert contract["window"]["calendar_week_count"] == WINDOW_WEEKS
    assert contract["window"]["bounded_real_proof_endpoint_month"] == "2013-01"
    assert contract["headline_calibration_claim"] == claim
    assert claim["null_gate"]["exact_95pct_upper_bound_max"] == 0.075
    assert claim["planted_gate"]["detection_rate_min"] == 0.80
    assert claim["outcomes_observed_when_frozen"] is False


@pytest.mark.unit
def test_v5_geometry_only_proof_passes_and_matches_boundary(
    tmp_path: Path,
) -> None:
    proof = _proof(tmp_path, one_natural_missing=True)
    validate_rolling_geometry_proof(proof)
    assert proof["coverage_proof_passed"] is True
    assert proof["geometry_metrics"]["n_groups"] == 156
    assert proof["geometry_metrics"]["missing_cells"] == 1
    assert proof["geometry_metrics"]["observed_assets_per_date_min"] == 59
    assert set(proof["source_bindings"][0]).isdisjoint({"dlyret", "returns"})

    complete = proof["geometry_metrics"]["complete_balanced_groups"]
    independent = independently_computed_oneway_boundary(
        p_assets=60,
        n_groups=complete,
        replicates=5,
        target_rank=1,
    )
    assert proof["target_boundary_feasibility"][
        "population_eigenvalue_boundary"
    ] == pytest.approx(independent.population_eigenvalue_boundary)

    manifest = build_rolling_geometry_manifest([proof])
    assert manifest["coverage_proof_passed"] is True
    assert manifest["full_72_endpoint_derivation_run"] is False
    assert manifest["detector_outcomes_present"] is False


@pytest.mark.unit
def test_v5_natural_missingness_gate_fails_closed(tmp_path: Path) -> None:
    proof = _proof(tmp_path, one_natural_missing=False)
    assert proof["coverage_gates"]["natural_missing_cells_min"] is False
    assert proof["coverage_proof_passed"] is False
    tampered = copy.deepcopy(proof)
    tampered["geometry_metrics"]["missing_cells"] = 1
    with pytest.raises(ValueError, match="metrics|gate result|digest"):
        validate_rolling_geometry_proof(tampered)


@pytest.mark.unit
def test_v5_rejects_nonfrozen_proof_endpoint_and_holdout() -> None:
    formation = pd.Timestamp("2013-02-28")
    spec = RollingGeometrySpec(
        endpoint_month="2013-02",
        formation_date=formation.date().isoformat(),
        window_start=rolling_window_start(formation).date().isoformat(),
        window_end=formation.date().isoformat(),
        proof_only=True,
    )
    with pytest.raises(ValueError, match="frozen to 2013-01"):
        spec.validate()
    forbidden = RollingGeometrySpec(
        endpoint_month="2025-01",
        formation_date="2025-01-31",
        window_start="2022-02-07",
        window_end="2025-01-31",
        proof_only=False,
    )
    with pytest.raises(ValueError, match="2013-2018"):
        forbidden.validate()


@pytest.mark.unit
def test_v5_source_loader_drops_return_values_and_collapses_exact_duplicates(
    tmp_path: Path,
) -> None:
    rows = []
    for date in ("2010-02-08", "2010-02-09"):
        rows.append(
            {
                "permno": 10001,
                "dlycaldt": date,
                "securitytype": "EQTY",
                "securitysubtype": "COM",
                "sharetype": "NS",
                "usincflg": "Y",
                "primaryexch": "N",
                "conditionaltype": "RW",
                "tradingstatusflg": "A",
                "dlyprc": 20.0,
                "dlycap": 1_000_000.0,
                "dlyret": 0.01,
                "dlyretmissflg": None,
                "dlydelflg": None,
            }
        )
    rows.append(copy.deepcopy(rows[0]))
    source = tmp_path / "raw" / "month=2010-02" / "data.csv.gz"
    source.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(source, index=False, compression="gzip")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "ok",
                "items": [
                    {
                        "status": "ok",
                        "path": str(source.resolve()),
                        "partition": "month=2010-02",
                        "rows": len(rows),
                        "size_bytes": source.stat().st_size,
                        "schema": "crsp",
                        "table": "wrds_dsfv2_query",
                        "date_column": "dlycaldt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    binding = bind_source_partition(source, receipt)
    geometry, scan = load_geometry_only_sources(
        [binding], start="2010-02-08", end="2010-02-09", chunksize=2
    )
    assert tuple(geometry.columns) == GEOMETRY_COLUMNS
    assert len(geometry) == 2
    assert scan["exact_duplicate_rows_collapsed"] == 1
    assert scan["return_values_persisted"] is False
    assert scan["return_presence_and_validity_read"] is True


@pytest.mark.unit
def test_v5_proof_rejects_unbound_source_geometry_and_factor_calendar(
    tmp_path: Path,
) -> None:
    proof = _proof(tmp_path, one_natural_missing=True)
    source_frame = _geometry_frame(one_natural_missing=True)
    changed = source_frame.copy()
    changed.loc[0, "dlycap"] += 1.0
    with pytest.raises(ValueError, match="source frame changed"):
        validate_rolling_geometry_proof(proof, source_frame=changed)

    incomplete = copy.deepcopy(proof)
    incomplete["source_bindings"].pop()
    incomplete["proof_digest"] = stable_sha256(
        {key: value for key, value in incomplete.items() if key != "proof_digest"}
    )
    with pytest.raises(ValueError, match="exact unique ordered"):
        validate_rolling_geometry_proof(incomplete)

    factor_path = Path(proof["factor_binding"]["path"])
    factor_path.write_text("date\n2013-01-31\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Factor file changed"):
        validate_rolling_geometry_proof(proof)


@pytest.mark.unit
def test_v5_freezer_refuses_detailed_proof_inside_git_worktree(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the Git worktree"):
        freeze_fjs_m5_rolling_geometry.main(
            [
                "--proof-out",
                str(freeze_fjs_m5_rolling_geometry.ROOT / "forbidden-proof.json"),
                "--manifest-out",
                str(tmp_path / "manifest.json"),
                "--receipt-out",
                str(tmp_path / "receipt.json"),
                "--expected-git-head",
                "0" * 40,
                "--expected-git-tree",
                "0" * 40,
                "--published-remote-commit",
                "0" * 40,
            ]
        )
