#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from experiments.synthetic.calibrate_thresholds import build_planned_jobs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Shard synthetic calibration grid across multiple instances.")
    parser.add_argument("--p-assets", type=int, nargs="+", required=True, help="Asset dimensions to sweep.")
    parser.add_argument("--n-groups", type=int, nargs="+", required=True, help="Replicate groups per window.")
    parser.add_argument("--replicates", type=int, nargs="+", required=True, help="Replicates per group.")
    parser.add_argument("--delta-abs-grid", type=float, nargs="+", required=True, help="Absolute δ grid.")
    parser.add_argument(
        "--edge-modes",
        type=str,
        nargs="+",
        default=["scm", "tyler"],
        help="Edge modes to evaluate (default: scm tyler).",
    )
    parser.add_argument("--shards", type=int, default=2, help="Number of shards to emit (default: 2).")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["round_robin", "contiguous"],
        default="round_robin",
        help="Assignment strategy for shards (default: round_robin).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/synthetic/calib/manifest.jsonl"),
        help="Output manifest path (default: reports/synthetic/calib/manifest.jsonl).",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id hint for downstream jobs.")
    parser.add_argument("--alpha", type=float, default=0.02, help="Target alpha for metadata (default: 0.02).")
    parser.add_argument(
        "--delta-frac-grid",
        type=float,
        nargs="*",
        default=[0.01, 0.015, 0.02, 0.025, 0.03],
        help="Delta_frac grid metadata (default: 0.01 ... 0.03).",
    )
    parser.add_argument(
        "--stability-grid",
        type=float,
        nargs="*",
        default=[0.30, 0.40, 0.50, 0.60],
        help="Stability eta grid metadata (default: 0.30 ... 0.60).",
    )
    parser.add_argument("--trials-null", type=int, default=300, help="Null trials metadata (default: 300).")
    parser.add_argument("--trials-alt", type=int, default=200, help="Alt trials metadata (default: 200).")
    return parser.parse_args(argv)


def _shard_jobs(jobs: list[dict[str, object]], shard_count: int, strategy: str) -> list[list[dict[str, object]]]:
    shards: list[list[dict[str, object]]] = [[] for _ in range(shard_count)]
    if shard_count <= 1:
        shards[0] = list(jobs)
        return shards
    if strategy == "contiguous":
        chunk = (len(jobs) + shard_count - 1) // shard_count
        for idx in range(shard_count):
            start = idx * chunk
            end = min(start + chunk, len(jobs))
            shards[idx] = jobs[start:end]
        return shards
    for idx, job in enumerate(jobs):
        shards[idx % shard_count].append(job)
    return shards


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    jobs = build_planned_jobs(args.p_assets, args.n_groups, args.replicates, args.delta_abs_grid)
    if not jobs:
        raise ValueError("No jobs were generated; check the provided grids.")
    shard_count = max(1, int(args.shards))
    shards = _shard_jobs(jobs, shard_count, args.strategy)
    timestamp = datetime.now(timezone.utc).isoformat()

    out_path = args.out.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for shard_idx, shard_jobs in enumerate(shards):
            payload = {
                "shard": shard_idx,
                "job_count": len(shard_jobs),
                "cell_count": len(shard_jobs) * len(args.edge_modes),
                "edge_modes": [str(mode) for mode in args.edge_modes],
                "jobs": shard_jobs,
                "run_id": args.run_id,
                "generated_at": timestamp,
                "strategy": args.strategy,
                "alpha": float(args.alpha),
                "trials_null": int(args.trials_null),
                "trials_alt": int(args.trials_alt),
                "delta_frac_grid": [float(val) for val in args.delta_frac_grid],
                "stability_grid": [float(val) for val in args.stability_grid],
            }
            handle.write(json.dumps(payload) + "\n")

    print(f"[shard-grid] wrote {shard_count} shards to {out_path}")
    for shard_idx, shard_jobs in enumerate(shards):
        print(f"  shard {shard_idx}: {len(shard_jobs)} base jobs ({len(shard_jobs) * len(args.edge_modes)} cells)")
    return out_path


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
