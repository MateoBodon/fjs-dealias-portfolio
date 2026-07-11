#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from tools.freeze_fjs_m5_rolling_geometry import (  # noqa: E402
    RECEIPT_2010_2017,
    SOURCE_ROOT,
    _atomic_json,
    _git_value,
    _validate_published_remote,
)

from fjs.real_design_contract import (  # noqa: E402
    bind_factor_source,
    bind_source_partition,
    file_sha256,
    stable_json_dumps,
    stable_sha256,
)
from fjs.rolling_geometry_contract import (  # noqa: E402
    BOUNDED_PROOF_ENDPOINT_MONTH,
    WARMUP_START,
    load_bound_factor_calendar,
    load_geometry_only_sources,
    resolve_spec,
    source_months_for_window,
    validate_rolling_geometry_proof,
)
from fjs.seasoned_geometry_contract import (  # noqa: E402
    build_seasoned_geometry_manifest,
    build_seasoned_geometry_proof,
    validate_seasoned_geometry_proof,
)

CONTRACT_BINDING_PATHS = (
    "src/fjs/seasoned_geometry_contract.py",
    "tools/freeze_fjs_m6_seasoned_geometry.py",
    "docs/strategy/FJS_M6_SEASONED_GEOMETRY_CONTRACT.md",
    "src/fjs/rolling_geometry_contract.py",
    "tools/freeze_fjs_m5_rolling_geometry.py",
)


def _implementation_bindings() -> dict[str, dict[str, Any]]:
    bindings = {}
    for relative in CONTRACT_BINDING_PATHS:
        path = ROOT / relative
        bindings[relative] = {
            "path": relative,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return bindings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Freeze the one-endpoint FJS M6 seasoned-universe geometry proof."
    )
    parser.add_argument("--proof-out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--receipt", type=Path, default=RECEIPT_2010_2017)
    parser.add_argument(
        "--factors-csv",
        type=Path,
        default=ROOT / "data/factors/ff5mom_daily.csv",
    )
    parser.add_argument(
        "--factor-registry",
        type=Path,
        default=ROOT / "data/factors/registry.json",
    )
    parser.add_argument("--chunksize", type=int, default=25_000)
    parser.add_argument("--expected-git-head", required=True)
    parser.add_argument("--expected-git-tree", required=True)
    parser.add_argument("--published-remote-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    proof_path = args.proof_out.expanduser().resolve()
    if proof_path == ROOT or ROOT in proof_path.parents:
        raise ValueError("The detailed M6 proof must remain outside Git.")
    head = _git_value("rev-parse", "HEAD")
    tree = _git_value("rev-parse", "HEAD^{tree}")
    branch = _git_value("branch", "--show-current")
    if head != args.expected_git_head or tree != args.expected_git_tree:
        raise ValueError("M6 execution does not match the frozen commit/tree.")
    if (
        subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", *CONTRACT_BINDING_PATHS],
            cwd=ROOT,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("M6 contract binding files differ from the freeze.")
    _validate_published_remote(
        commit=args.published_remote_commit,
        expected_tree=tree,
        branch=branch,
    )

    factor_binding = bind_factor_source(args.factors_csv, args.factor_registry)
    broad_calendar = load_bound_factor_calendar(
        factor_binding,
        start=WARMUP_START.date().isoformat(),
        end="2013-01-31",
    )
    spec = resolve_spec(BOUNDED_PROOF_ENDPOINT_MONTH, broad_calendar)
    months = source_months_for_window(spec.window_start, spec.window_end)
    source_bindings = [
        bind_source_partition(
            args.source_root / f"month={month}" / "data.csv.gz",
            args.receipt,
        )
        for month in months
    ]
    geometry, scan = load_geometry_only_sources(
        source_bindings,
        start=spec.window_start,
        end=spec.window_end,
        chunksize=int(args.chunksize),
    )
    proof, filtered = build_seasoned_geometry_proof(
        geometry,
        spec=spec,
        source_bindings=source_bindings,
        factor_binding=factor_binding,
        scan_receipt=scan,
    )
    bindings = _implementation_bindings()
    execution = {
        "git_head": head,
        "git_tree": tree,
        "published_remote_commit": args.published_remote_commit,
        "branch": branch,
        "bounded_endpoint_count": 1,
        "full_72_endpoint_derivation_run": False,
    }
    proof["implementation_bindings"] = bindings
    proof["execution_identity"] = execution
    proof["proof_digest"] = stable_sha256(
        {key: value for key, value in proof.items() if key != "proof_digest"}
    )
    validate_seasoned_geometry_proof(proof, filtered_frame=filtered)
    proof_path = _atomic_json(proof, proof_path)

    manifest = build_seasoned_geometry_manifest(proof)
    manifest["implementation_bindings"] = bindings
    manifest["execution_identity"] = execution
    manifest["manifest_digest"] = stable_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )
    manifest_path = _atomic_json(manifest, args.manifest_out)

    reread = json.loads(proof_path.read_text(encoding="utf-8"))
    validate_seasoned_geometry_proof(reread, filtered_frame=filtered)
    validate_rolling_geometry_proof(
        reread["base_v5_geometry_proof"],
        source_frame=filtered,
        revalidate_external=True,
    )
    reread_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if reread_manifest["manifest_digest"] != stable_sha256(
        {
            key: value
            for key, value in reread_manifest.items()
            if key != "manifest_digest"
        }
    ):
        raise ValueError("M6 manifest readback digest mismatch.")

    base = proof["base_v5_geometry_proof"]
    selection = proof["selection_receipt"]
    receipt: dict[str, Any] = {
        "schema": "fjs-seasoned-geometry-readback/v1",
        "endpoint_month": BOUNDED_PROOF_ENDPOINT_MONTH,
        "proof_path": str(proof_path),
        "proof_file_sha256": file_sha256(proof_path),
        "proof_digest": proof["proof_digest"],
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "manifest_digest": manifest["manifest_digest"],
        "source_partition_count": len(source_bindings),
        "rows_scanned": sum(int(item["rows_scanned"]) for item in scan["partitions"]),
        "rows_after_filters": scan["rows_after_all_filters"],
        "exact_duplicates_collapsed": scan["exact_duplicate_rows_collapsed"],
        "seasoned_eligibility": {
            "calendar_date_count": selection["calendar_date_count"],
            "minimum_asset_observations": selection["minimum_asset_observations"],
            "guaranteed_pairwise_lower_bound": selection[
                "guaranteed_pairwise_lower_bound"
            ],
            "anchor_week_count": selection["anchor_week_count"],
            "anchor_date_count": selection["anchor_date_count"],
            "eligible_candidate_count": selection["eligible_candidate_count"],
            "selection_digest": selection["selection_digest"],
        },
        "geometry_metrics": base["geometry_metrics"],
        "target_boundary_feasibility": base["target_boundary_feasibility"],
        "coverage_gates": base["coverage_gates"],
        "coverage_proof_passed": base["coverage_proof_passed"],
        "readback_passed": True,
        "return_values_persisted": False,
        "full_72_endpoint_derivation_run": False,
        "detector_outcomes_present": False,
        "aws_execution_authorized": False,
        "holdout_2025_opened": False,
    }
    receipt["readback_digest"] = stable_sha256(receipt)
    receipt_path = _atomic_json(receipt, args.receipt_out)
    print(stable_json_dumps(receipt), flush=True)
    if receipt["coverage_proof_passed"] is not True:
        raise ValueError("The frozen M6 coverage proof failed.")
    return receipt_path


if __name__ == "__main__":  # pragma: no cover
    main()
