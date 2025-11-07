#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional dependency
    threadpool_limits = None  # type: ignore[assignment]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Benchmark numpy.linalg.eigh throughput across dimensions.")
    parser.add_argument("--dims", type=int, nargs="+", default=[100, 200, 300], help="Matrix dimensions to benchmark.")
    parser.add_argument("--repeats", type=int, default=100, help="Repetitions per dimension (default: 100).")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="BLAS/OpenMP threads used during the benchmark (default: 1).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label for the benchmark (e.g., instance tag).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory (default: reports/aws/bench/<timestamp>).",
    )
    return parser.parse_args(argv)


def _configure_threads(threads: int) -> None:
    capped = max(1, threads)
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(capped)
    if threadpool_limits is not None:  # pragma: no branch - best effort
        try:
            threadpool_limits(limits=capped)
        except Exception:
            pass


def _detect_instance_type() -> str | None:
    env_type = os.environ.get("INSTANCE_TYPE")
    if env_type:
        return env_type
    try:
        req = urllib.request.Request("http://169.254.169.254/latest/meta-data/instance-type", method="GET")
        with urllib.request.urlopen(req, timeout=0.25) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        return None


def _blas_info() -> dict[str, object]:
    try:
        info = np.__config__.get_info("openblas_info")
        if info:
            return {"library": "openblas", "config": info}
        info = np.__config__.get_info("blas_opt_info")
        if info:
            return {"library": "blas_opt", "config": info}
    except Exception:
        return {}
    return {}


def _bench_dimension(dim: int, repeats: int, rng: np.random.Generator) -> dict[str, float]:
    base = rng.standard_normal(size=(dim, dim))
    matrix = base @ base.T + np.eye(dim) * 1e-6
    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        np.linalg.eigh(matrix)
        durations.append(time.perf_counter() - start)
    return {
        "dimension": dim,
        "repeats": repeats,
        "mean_seconds": statistics.fmean(durations) if durations else 0.0,
        "median_seconds": statistics.median(durations) if durations else 0.0,
        "std_seconds": statistics.pstdev(durations) if len(durations) > 1 else 0.0,
        "min_seconds": min(durations) if durations else 0.0,
        "max_seconds": max(durations) if durations else 0.0,
    }


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    _configure_threads(args.threads)
    rng = np.random.default_rng(1337)
    dims = sorted({int(dim) for dim in args.dims if int(dim) > 0})
    if not dims:
        raise ValueError("Provide at least one positive dimension via --dims.")
    repeats = max(1, int(args.repeats))
    results = [_bench_dimension(dim, repeats, rng) for dim in dims]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or Path("reports/aws/bench") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bench.json"

    instance_type = _detect_instance_type()
    instance_family = instance_type.split(".")[0] if instance_type and "." in instance_type else instance_type

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "dims": dims,
        "repeats": repeats,
        "threads": int(args.threads),
        "runner": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "instance_type": instance_type,
        "instance_family": instance_family,
        "git_sha": _git_sha(),
        "blas": _blas_info(),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[bench-linalg] wrote {out_path} ({len(results)} dimensions, repeats={repeats})")
    for entry in results:
        print(
            f"  p={entry['dimension']:>4d}: mean={entry['mean_seconds']*1e3:7.3f}ms "
            f"median={entry['median_seconds']*1e3:7.3f}ms"
        )
    return out_path


def _git_sha() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - git optional
        return "unknown"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
