#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

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
    build_rolling_geometry_manifest,
    build_rolling_geometry_proof,
    load_bound_factor_calendar,
    load_geometry_only_sources,
    resolve_spec,
    source_months_for_window,
    validate_rolling_geometry_proof,
)

SOURCE_ROOT = Path(
    "/Volumes/Storage/Data/WRDS/raw/crsp/wrds_dsfv2_query/"
    "snapshot=20260707_045553_global_project_priority"
)
RECEIPT_2010_2017 = Path(
    "/Volumes/Storage/Data/wrds/_manifests/"
    "20260707T214900Z_worker8_crsp_dsfv2_month_2017_2010_csvgz/manifest.json"
)
CONTRACT_BINDING_PATHS = (
    "src/fjs/rolling_geometry_contract.py",
    "tools/freeze_fjs_m5_rolling_geometry.py",
    "docs/strategy/FJS_M5_ROLLING_GEOMETRY_CONTRACT.md",
)
GITHUB_REPOSITORY = "MateoBodon/fjs-dealias-portfolio"


def _atomic_json(payload: Mapping[str, Any], path: Path) -> Path:
    out = path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.tmp")
    temporary.write_text(stable_json_dumps(payload) + "\n", encoding="utf-8")
    os.replace(temporary, out)
    return out


def _git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _implementation_bindings() -> dict[str, dict[str, Any]]:
    bindings = {}
    for relative in CONTRACT_BINDING_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"V5 contract binding is missing: {relative}")
        bindings[relative] = {
            "path": relative,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return bindings


def _validate_published_remote(*, commit: str, expected_tree: str, branch: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Published remote commit must be an exact 40-hex identity.")
    remote = subprocess.check_output(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
        cwd=ROOT,
        text=True,
    ).strip()
    fields = remote.split()
    if len(fields) != 2 or fields[0] != commit:
        raise ValueError("Authoritative remote branch HEAD does not match the receipt.")
    remote_tree = subprocess.check_output(
        [
            "gh",
            "api",
            f"repos/{GITHUB_REPOSITORY}/git/commits/{commit}",
            "--jq",
            ".tree.sha",
        ],
        cwd=ROOT,
        text=True,
    ).strip()
    if remote_tree != expected_tree:
        raise ValueError("Authoritative remote commit tree does not match the freeze.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Freeze the one-endpoint, geometry-only FJS M5 rolling proof."
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
        raise ValueError(
            "The PERMNO/member/mask proof must remain outside the Git worktree."
        )
    observed_head = _git_value("rev-parse", "HEAD")
    observed_tree = _git_value("rev-parse", "HEAD^{tree}")
    if observed_head != str(args.expected_git_head) or observed_tree != str(
        args.expected_git_tree
    ):
        raise ValueError(
            "V5 execution does not match the predeclared frozen commit/tree."
        )
    binding_diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *CONTRACT_BINDING_PATHS],
        cwd=ROOT,
        check=False,
    )
    if binding_diff.returncode != 0:
        raise ValueError("V5 contract binding files differ from the frozen commit.")
    observed_branch = _git_value("branch", "--show-current")
    _validate_published_remote(
        commit=str(args.published_remote_commit),
        expected_tree=observed_tree,
        branch=observed_branch,
    )
    factor_binding = bind_factor_source(args.factors_csv, args.factor_registry)
    broad_calendar = load_bound_factor_calendar(
        factor_binding,
        start=WARMUP_START.date().isoformat(),
        end="2013-01-31",
    )
    spec = resolve_spec(BOUNDED_PROOF_ENDPOINT_MONTH, broad_calendar)
    months = source_months_for_window(spec.window_start, spec.window_end)
    bindings = [
        bind_source_partition(
            args.source_root / f"month={month}" / "data.csv.gz",
            args.receipt,
        )
        for month in months
    ]
    geometry, scan = load_geometry_only_sources(
        bindings,
        start=spec.window_start,
        end=spec.window_end,
        chunksize=int(args.chunksize),
    )
    proof = build_rolling_geometry_proof(
        geometry,
        spec=spec,
        source_bindings=bindings,
        factor_binding=factor_binding,
        scan_receipt=scan,
    )
    implementation_bindings = _implementation_bindings()
    execution_identity = {
        "git_head": observed_head,
        "git_tree": observed_tree,
        "published_remote_commit": str(args.published_remote_commit),
        "branch": observed_branch,
        "bounded_endpoint_count": 1,
        "full_72_endpoint_derivation_run": False,
    }
    proof["implementation_bindings"] = implementation_bindings
    proof["execution_identity"] = execution_identity
    proof["proof_digest"] = stable_sha256(
        {key: value for key, value in proof.items() if key != "proof_digest"}
    )
    validate_rolling_geometry_proof(proof)
    proof_path = _atomic_json(proof, args.proof_out)

    manifest = build_rolling_geometry_manifest([proof])
    manifest["implementation_bindings"] = implementation_bindings
    manifest["execution_identity"] = execution_identity
    manifest["manifest_digest"] = stable_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )
    manifest_path = _atomic_json(manifest, args.manifest_out)

    reread_proof = json.loads(proof_path.read_text(encoding="utf-8"))
    validate_rolling_geometry_proof(
        reread_proof,
        source_frame=geometry,
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
        raise ValueError("V5 manifest readback digest mismatch.")
    if reread_manifest["proof_cells"][0]["proof_digest"] != proof["proof_digest"]:
        raise ValueError("V5 manifest/proof identity mismatch.")

    receipt: dict[str, Any] = {
        "schema": "fjs-rolling-geometry-readback/v1",
        "endpoint_month": BOUNDED_PROOF_ENDPOINT_MONTH,
        "proof_path": str(proof_path),
        "proof_file_sha256": file_sha256(proof_path),
        "proof_digest": proof["proof_digest"],
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "manifest_digest": manifest["manifest_digest"],
        "source_partition_count": len(bindings),
        "rows_scanned": sum(int(item["rows_scanned"]) for item in scan["partitions"]),
        "rows_after_filters": scan["rows_after_all_filters"],
        "exact_duplicates_collapsed": scan["exact_duplicate_rows_collapsed"],
        "geometry_metrics": copy_geometry_metrics(proof["geometry_metrics"]),
        "target_boundary_feasibility": dict(proof["target_boundary_feasibility"]),
        "coverage_gates": dict(proof["coverage_gates"]),
        "coverage_proof_passed": proof["coverage_proof_passed"],
        "full_72_endpoint_derivation_run": False,
        "return_values_persisted": False,
        "detector_outcomes_present": False,
        "aws_execution_authorized": False,
        "holdout_2025_opened": False,
        "readback_passed": True,
    }
    receipt["readback_digest"] = stable_sha256(receipt)
    receipt_path = _atomic_json(receipt, args.receipt_out)
    print(stable_json_dumps(receipt), flush=True)
    if receipt["coverage_proof_passed"] is not True:
        raise ValueError(
            "The frozen V5 bounded coverage proof failed; no full derivation "
            "is allowed."
        )
    return receipt_path


def copy_geometry_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(metrics), sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
