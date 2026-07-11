from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fjs.real_design_contract import (
    FACTOR_COLUMNS,
    RealDesignCellSpec,
    bind_factor_source,
    bind_source_partition,
    derive_real_design_cell,
    file_sha256,
    load_bound_factors,
    load_filtered_sources,
    validate_real_design_cell,
    write_real_design_cell,
)
from tools import freeze_fjs_m4_real_design_v4


def _fixture_inputs(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(20260711)
    dates = pd.bdate_range("2013-01-02", "2013-01-31")
    factor_values = rng.normal(0.0, 0.01, size=(len(dates), len(FACTOR_COLUMNS)))
    factors = pd.DataFrame(factor_values, columns=list(FACTOR_COLUMNS))
    factors.insert(0, "date", dates.strftime("%Y-%m-%d"))
    factors["RF"] = 0.00005
    factor_path = tmp_path / "data" / "factors" / "ff5mom_daily.csv"
    factor_path.parent.mkdir(parents=True)
    factors.to_csv(factor_path, index=False)

    rows: list[dict[str, object]] = []
    for asset_index, permno in enumerate(range(10001, 10009)):
        beta = np.array([0.0001 * asset_index, 0.7, 0.1, -0.2, 0.15, 0.05, 0.25])
        for date_index, date in enumerate(dates):
            if permno == 10008 and date == pd.Timestamp("2013-01-23"):
                continue
            design = np.concatenate(([1.0], factor_values[date_index]))
            residual = 0.00001 * (asset_index + 1) * ((date_index % 5) - 2)
            rows.append(
                {
                    "permno": permno,
                    "dlycaldt": date.date().isoformat(),
                    "securitytype": "EQTY",
                    "securitysubtype": "COM",
                    "sharetype": "NS",
                    "usincflg": "Y",
                    "primaryexch": ("N", "A", "Q")[asset_index % 3],
                    "conditionaltype": "RW",
                    "tradingstatusflg": "A",
                    "dlyprc": 20.0 + asset_index,
                    "dlycap": 1_000_000.0 + 10_000.0 * asset_index + date_index,
                    "dlyret": 0.00005 + float(design @ beta) + residual,
                    "dlyretmissflg": None,
                    "dlydelflg": None,
                }
            )
    rows.append(
        {
            "permno": 99999,
            "dlycaldt": "2013-01-15",
            "securitytype": "FUND",
            "securitysubtype": "ETF",
            "sharetype": "NS",
            "usincflg": "Y",
            "primaryexch": "N",
            "conditionaltype": "RW",
            "tradingstatusflg": "A",
            "dlyprc": 100.0,
            "dlycap": 1_000_000_000.0,
            "dlyret": 0.01,
            "dlyretmissflg": None,
            "dlydelflg": None,
        }
    )
    source = tmp_path / "raw" / "month=2013-01" / "data.csv.gz"
    source.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(source, index=False, compression="gzip")

    receipt = tmp_path / "receipt" / "manifest.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps(
            {
                "status": "ok",
                "items": [
                    {
                        "status": "ok",
                        "path": str(source.resolve()),
                        "partition": "month=2013-01",
                        "rows": len(rows),
                        "size_bytes": source.stat().st_size,
                        "schema": "crsp",
                        "table": "wrds_dsfv2_query",
                        "date_column": "dlycaldt",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    registry = tmp_path / "data" / "factors" / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "datasets": {
                    "data/factors/ff5mom_daily.csv": {
                        "path": str(factor_path.resolve()),
                        "sha256": file_sha256(factor_path),
                        "columns": [*FACTOR_COLUMNS, "RF"],
                        "start_date": dates.min().date().isoformat(),
                        "end_date": dates.max().date().isoformat(),
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
    return {
        "source": source,
        "receipt": receipt,
        "factors": factor_path,
        "registry": registry,
    }


def _spec() -> RealDesignCellSpec:
    return RealDesignCellSpec(
        cell_id="proof_2013_01",
        factor_fit_start="2013-01-02",
        factor_fit_end="2013-01-15",
        formation_date="2013-01-15",
        window_start="2013-01-16",
        window_end="2013-01-31",
        universe_size=4,
        min_factor_observations=8,
        min_window_observations=10,
        min_pairwise_observations=9,
        max_cap_staleness_days=3,
        proof_only=True,
    )


def _refresh_receipt(inputs: dict[str, Path], rows: int) -> None:
    payload = json.loads(inputs["receipt"].read_text(encoding="utf-8"))
    payload["items"][0]["rows"] = rows
    payload["items"][0]["size_bytes"] = inputs["source"].stat().st_size
    inputs["receipt"].write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _derive_fixture_cell(tmp_path: Path) -> tuple[dict[str, object], dict[str, Path]]:
    inputs = _fixture_inputs(tmp_path)
    source_binding = bind_source_partition(inputs["source"], inputs["receipt"])
    factor_binding = bind_factor_source(inputs["factors"], inputs["registry"])
    source_frame, scan = load_filtered_sources(
        [source_binding],
        start="2013-01-02",
        end="2013-01-31",
        chunksize=7,
    )
    factors = load_bound_factors(
        factor_binding,
        start="2013-01-02",
        end="2013-01-31",
    )
    cell = derive_real_design_cell(
        source_frame,
        factors,
        spec=_spec(),
        source_bindings=[source_binding],
        factor_binding=factor_binding,
        scan_receipt=scan,
    )
    return cell, inputs


@pytest.mark.unit
def test_v4_real_design_cell_is_deterministic_and_hash_bound(tmp_path: Path) -> None:
    cell, inputs = _derive_fixture_cell(tmp_path)
    validate_real_design_cell(cell)
    members = cell["universe"]["members"]
    assert [entry["permno"] for entry in members] == [10008, 10007, 10006, 10005]
    assert 99999 not in [entry["permno"] for entry in members]
    assert cell["claim_boundary"] == {
        "development_only": True,
        "mechanism_calibration_only": True,
        "empirical_claims_forbidden": True,
        "promotion_allowed": False,
        "proof_only": True,
        "legacy_ticker_csv_used": False,
        "holdout_2025_opened": False,
    }
    assert cell["source_partitions"][0]["sha256"] == file_sha256(inputs["source"])
    assert cell["missingness"]["missing_fraction"] > 0.0
    assert cell["factor_fit"]["end"] < cell["factor_fit"]["window_start"]
    assert cell["window_geometry"]["p_assets"] == 4

    out = tmp_path / "proof-cell.json"
    artifact = write_real_design_cell(cell, out)
    assert artifact["sha256"] == file_sha256(out)
    second = json.loads(out.read_text(encoding="utf-8"))
    validate_real_design_cell(second)
    assert second == cell


@pytest.mark.unit
def test_v4_universe_prefilters_fit_and_window_history_before_cap_rank(
    tmp_path: Path,
) -> None:
    inputs = _fixture_inputs(tmp_path)
    frame = pd.read_csv(inputs["source"])
    fit_dates = sorted(
        frame.loc[
            frame["permno"].eq(10008) & frame["dlycaldt"].le("2013-01-15"),
            "dlycaldt",
        ].unique()
    )
    window_dates = sorted(
        frame.loc[
            frame["permno"].eq(10007) & frame["dlycaldt"].ge("2013-01-16"),
            "dlycaldt",
        ].unique()
    )
    insufficient_fit = frame["permno"].eq(10008) & frame["dlycaldt"].isin(fit_dates[:3])
    insufficient_window = frame["permno"].eq(10007) & frame["dlycaldt"].isin(
        window_dates[:3]
    )
    frame = frame.loc[~(insufficient_fit | insufficient_window)].copy()
    frame.to_csv(inputs["source"], index=False, compression="gzip")
    _refresh_receipt(inputs, len(frame))

    source_binding = bind_source_partition(inputs["source"], inputs["receipt"])
    factor_binding = bind_factor_source(inputs["factors"], inputs["registry"])
    source_frame, scan = load_filtered_sources(
        [source_binding], start="2013-01-02", end="2013-01-31", chunksize=7
    )
    factors = load_bound_factors(factor_binding, start="2013-01-02", end="2013-01-31")
    cell = derive_real_design_cell(
        source_frame,
        factors,
        spec=_spec(),
        source_bindings=[source_binding],
        factor_binding=factor_binding,
        scan_receipt=scan,
    )

    assert [entry["permno"] for entry in cell["universe"]["members"]] == [
        10006,
        10005,
        10004,
        10003,
    ]
    validate_real_design_cell(cell)


@pytest.mark.unit
def test_v4_source_binding_detects_post_binding_mutation(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path)
    binding = bind_source_partition(inputs["source"], inputs["receipt"])
    inputs["source"].write_bytes(inputs["source"].read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="changed after binding"):
        load_filtered_sources(
            [binding],
            start="2013-01-02",
            end="2013-01-31",
        )


@pytest.mark.unit
def test_v4_collapses_only_exact_analytical_duplicates(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path)
    frame = pd.read_csv(inputs["source"])
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_csv(inputs["source"], index=False, compression="gzip")
    _refresh_receipt(inputs, len(frame))
    binding = bind_source_partition(inputs["source"], inputs["receipt"])
    filtered, scan = load_filtered_sources(
        [binding], start="2013-01-02", end="2013-01-31", chunksize=7
    )
    assert scan["exact_duplicate_rows_collapsed"] == 1
    assert not filtered.duplicated(subset=["dlycaldt", "permno"]).any()

    frame = pd.read_csv(inputs["source"])
    conflict = frame.iloc[[0]].copy()
    conflict["dlyret"] = pd.to_numeric(conflict["dlyret"]) + 0.25
    frame = pd.concat([frame, conflict], ignore_index=True)
    frame.to_csv(inputs["source"], index=False, compression="gzip")
    _refresh_receipt(inputs, len(frame))
    binding = bind_source_partition(inputs["source"], inputs["receipt"])
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        load_filtered_sources(
            [binding], start="2013-01-02", end="2013-01-31", chunksize=7
        )


@pytest.mark.unit
def test_v4_contract_refuses_the_2025_holdout() -> None:
    spec = RealDesignCellSpec(
        cell_id="forbidden_holdout",
        factor_fit_start="2025-01-02",
        factor_fit_end="2025-01-15",
        formation_date="2025-01-15",
        window_start="2025-01-16",
        window_end="2025-01-31",
        proof_only=True,
    )
    with pytest.raises(ValueError, match="2013-2018|holdout"):
        spec.validate()


@pytest.mark.unit
def test_v4_freezer_preserves_v2_v3_and_stays_fail_closed(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path)
    frozen_paths = {
        **freeze_fjs_m4_real_design_v4.EXPECTED_V2_HASHES,
        **freeze_fjs_m4_real_design_v4.EXPECTED_V3_HASHES,
    }
    before = {
        relative: file_sha256(freeze_fjs_m4_real_design_v4.ROOT / relative)
        for relative in frozen_paths
    }
    cell_out = tmp_path / "proof" / "cell.json"
    manifest_out = tmp_path / "proof" / "manifest.json"
    args = [
        "--source",
        str(inputs["source"]),
        "--receipt",
        str(inputs["receipt"]),
        "--factors-csv",
        str(inputs["factors"]),
        "--factor-registry",
        str(inputs["registry"]),
        "--cell-out",
        str(cell_out),
        "--out",
        str(manifest_out),
        "--cell-id",
        "proof_2013_01",
        "--factor-fit-start",
        "2013-01-02",
        "--factor-fit-end",
        "2013-01-15",
        "--formation-date",
        "2013-01-15",
        "--window-start",
        "2013-01-16",
        "--window-end",
        "2013-01-31",
        "--universe-size",
        "4",
        "--min-factor-observations",
        "8",
        "--min-window-observations",
        "10",
        "--min-pairwise-observations",
        "9",
        "--max-cap-staleness-days",
        "3",
        "--chunksize",
        "7",
    ]
    freeze_fjs_m4_real_design_v4.main(args)
    first_manifest = manifest_out.read_bytes()
    first_cell = cell_out.read_bytes()
    freeze_fjs_m4_real_design_v4.main(args)
    assert manifest_out.read_bytes() == first_manifest
    assert cell_out.read_bytes() == first_cell

    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    freeze_fjs_m4_real_design_v4.validate_manifest_v4(manifest)
    assert manifest["execution_readiness"] == {
        "real_design_contract_ready": True,
        "bounded_source_proof_ready": True,
        "real_design_full_generation_complete": False,
        "full_execution_ready": False,
        "aws_execution_authorized": False,
        "blockers": [
            "real_design_full_generation_not_run",
            "trusted_route_admission_required",
            "fresh_authoritative_aws_admission_required",
        ],
    }
    assert manifest["full_generation_contract"]["required_partition_count"] == 72
    assert manifest["claim_boundary"]["legacy_ticker_csv_used"] is False
    assert manifest["claim_boundary"]["holdout_2025_opened"] is False
    assert {
        relative: file_sha256(freeze_fjs_m4_real_design_v4.ROOT / relative)
        for relative in frozen_paths
    } == before
