from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping

try:  # pragma: no cover - optional dependency
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional dependency
    threadpool_limits = None  # type: ignore[assignment]

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_THREADPOOL_CONTROLLER = None


@dataclass(frozen=True)
class ExecModeSettings:
    """Resolved execution-mode settings shared across runners."""

    mode: str
    blas_threads: int
    worker_scale: float


def _set_threadpool_limits(max_threads: int) -> None:
    global _THREADPOOL_CONTROLLER
    if threadpool_limits is None:
        return
    try:
        if _THREADPOOL_CONTROLLER is not None:
            _THREADPOOL_CONTROLLER.restore()  # type: ignore[call-arg]
    except Exception:
        _THREADPOOL_CONTROLLER = None
    try:
        _THREADPOOL_CONTROLLER = threadpool_limits(limits=max_threads)
    except Exception:
        _THREADPOOL_CONTROLLER = None


def _apply_thread_caps(max_threads: int) -> None:
    capped = max(1, int(max_threads))
    for key in THREAD_ENV_VARS:
        os.environ[key] = str(capped)
    _set_threadpool_limits(capped)


def resolve_exec_mode(mode: str | None, *, throughput_threads: int | None = None) -> ExecModeSettings:
    """Return the execution-mode settings without mutating global state."""

    normalized = (mode or "deterministic").strip().lower()
    if normalized not in {"deterministic", "throughput"}:
        normalized = "deterministic"

    if normalized == "throughput":
        threads = throughput_threads or int(os.environ.get("EXEC_THREADS_THROUGHPUT", "4") or 4)
        threads = max(2, min(threads, 8))
        worker_scale = 1.0 / float(threads)
        return ExecModeSettings(mode="throughput", blas_threads=threads, worker_scale=worker_scale)

    return ExecModeSettings(mode="deterministic", blas_threads=1, worker_scale=1.0)


def configure_exec_mode(mode: str | None, *, throughput_threads: int | None = None) -> ExecModeSettings:
    """Resolve and apply execution-mode thread caps."""

    settings = resolve_exec_mode(mode, throughput_threads=throughput_threads)
    _apply_thread_caps(settings.blas_threads)
    return settings


def effective_worker_count(
    settings: ExecModeSettings,
    requested_workers: int | None,
    cpu_count: int | None = None,
) -> int:
    """Return the worker count respecting the resolved execution mode."""

    if requested_workers is not None and requested_workers > 0:
        return int(requested_workers)
    cpus = cpu_count if cpu_count is not None else os.cpu_count()
    if cpus is None or cpus <= 0:
        cpus = 1
    if settings.mode == "throughput":
        estimate = math.floor(cpus * settings.worker_scale)
        return max(1, estimate)
    return max(1, int(cpus))


def thread_caps_snapshot() -> dict[str, str]:
    """Return the current BLAS/OpenMP thread caps for logging."""

    return {var: os.environ.get(var, "") for var in THREAD_ENV_VARS if os.environ.get(var) is not None}


def exec_mode_metadata(settings: ExecModeSettings) -> Mapping[str, object]:
    """Helper to expose execution-mode metadata for run.json payloads."""

    return {
        "exec_mode": settings.mode,
        "blas_threads": settings.blas_threads,
        "thread_caps": thread_caps_snapshot(),
    }
