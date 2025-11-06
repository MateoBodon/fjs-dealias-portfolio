#!/usr/bin/env python3
"""
Launch a command, mirror its output, and record detailed resource metrics.

Outputs:
  - metrics JSON lines capturing wall clock, host CPU %, aggregate process CPU %,
    RSS/USS memory, thread counts, and disk IO.
  - optional summary JSON with peak/average statistics.
  - optional progress updates derived from JSON messages emitted by the workload.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psutil


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(line: str) -> dict[str, Any] | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


@dataclass
class MetricSample:
    timestamp: float
    cpu_total: float
    cpu_process: float
    mem_rss_mb: float
    mem_uss_mb: float
    threads: int
    io_read_mb: float
    io_write_mb: float
    load_avg: tuple[float, float, float]


def _aggregate_process_metrics(proc: psutil.Process) -> tuple[float, float, int, float, float]:
    try:
        children: Iterable[psutil.Process] = proc.children(recursive=True)
    except (psutil.Error, OSError):
        children = []

    processes = [proc]
    processes.extend(children)

    total_cpu = 0.0
    total_rss = 0.0
    total_uss = 0.0
    total_threads = 0

    for handle in processes:
        try:
            total_cpu += handle.cpu_percent(interval=None)
            mem = handle.memory_full_info()
            total_rss += getattr(mem, "rss", 0.0)
            total_uss += getattr(mem, "uss", getattr(mem, "rss", 0.0))
            total_threads += handle.num_threads()
        except (psutil.Error, OSError):
            continue

    return total_cpu, total_rss, total_uss, total_threads, float(len(processes))


def _io_counters(proc: psutil.Process) -> tuple[float, float]:
    read = 0.0
    write = 0.0
    try:
        counters = proc.io_counters()
        read += counters.read_bytes
        write += counters.write_bytes
    except (psutil.Error, OSError, AttributeError):
        pass

    try:
        for child in proc.children(recursive=True):
            try:
                counters = child.io_counters()
                read += counters.read_bytes
                write += counters.write_bytes
            except (psutil.Error, OSError, AttributeError):
                continue
    except (psutil.Error, OSError):
        pass

    return read / (1024 * 1024), write / (1024 * 1024)


def _monitor_loop(
    proc: psutil.Process,
    interval: float,
    metrics_path: Path,
    progress_queue: queue.Queue[dict[str, Any]],
    stop_event: threading.Event,
    hostname: str,
    samples: list[MetricSample],
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_total_history = []
    io_read_prev, io_write_prev = _io_counters(proc)

    with metrics_path.open("w", encoding="utf-8") as handle:
        psutil.cpu_percent(interval=None)
        try:
            proc.cpu_percent(interval=None)
        except (psutil.Error, OSError):
            pass

        while not stop_event.is_set():
            time.sleep(interval)
            timestamp = time.time()

            cpu_total = psutil.cpu_percent(interval=None)
            cpu_process, rss, uss, threads, _ = _aggregate_process_metrics(proc)
            io_read, io_write = _io_counters(proc)
            load_avg = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)

            sample = MetricSample(
                timestamp=timestamp,
                cpu_total=cpu_total,
                cpu_process=cpu_process,
                mem_rss_mb=rss / (1024 * 1024),
                mem_uss_mb=uss / (1024 * 1024),
                threads=threads,
                io_read_mb=max(0.0, io_read - io_read_prev),
                io_write_mb=max(0.0, io_write - io_write_prev),
                load_avg=tuple(float(v) for v in load_avg),
            )
            samples.append(sample)
            cpu_total_history.append(cpu_process)
            io_read_prev, io_write_prev = io_read, io_write

            record = {
                "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                "hostname": hostname,
                "cpu_total_percent": cpu_total,
                "cpu_process_percent": cpu_process,
                "memory_rss_mb": sample.mem_rss_mb,
                "memory_uss_mb": sample.mem_uss_mb,
                "threads": threads,
                "io_read_mb": sample.io_read_mb,
                "io_write_mb": sample.io_write_mb,
                "load_avg": sample.load_avg,
            }
            if not progress_queue.empty():
                try:
                    progress_record = progress_queue.get_nowait()
                    record["progress"] = progress_record
                except queue.Empty:
                    pass
            handle.write(json.dumps(record) + "\n")
            handle.flush()

            if not proc.is_running():
                break

        try:
            cpu_total = psutil.cpu_percent(interval=None)
            cpu_process, rss, uss, threads, _ = _aggregate_process_metrics(proc)
            io_read, io_write = _io_counters(proc)
            load_avg = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
            sample = MetricSample(
                timestamp=time.time(),
                cpu_total=cpu_total,
                cpu_process=cpu_process,
                mem_rss_mb=rss / (1024 * 1024),
                mem_uss_mb=uss / (1024 * 1024),
                threads=threads,
                io_read_mb=max(0.0, io_read - io_read_prev),
                io_write_mb=max(0.0, io_write - io_write_prev),
                load_avg=tuple(float(v) for v in load_avg),
            )
            samples.append(sample)
            record = {
                "timestamp": datetime.fromtimestamp(sample.timestamp, tz=timezone.utc).isoformat(),
                "hostname": hostname,
                "cpu_total_percent": sample.cpu_total,
                "cpu_process_percent": sample.cpu_process,
                "memory_rss_mb": sample.mem_rss_mb,
                "memory_uss_mb": sample.mem_uss_mb,
                "threads": threads,
                "io_read_mb": sample.io_read_mb,
                "io_write_mb": sample.io_write_mb,
                "load_avg": sample.load_avg,
                "final_sample": True,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        except Exception:
            pass
        else:
            io_read_prev, io_write_prev = io_read, io_write

    return None


def _summarise(samples: list[MetricSample]) -> dict[str, Any]:
    if not samples:
        return {}

    cpu_proc = [sample.cpu_process for sample in samples]
    mem_rss = [sample.mem_rss_mb for sample in samples]
    mem_uss = [sample.mem_uss_mb for sample in samples]

    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "samples": len(samples),
        "cpu_process_avg": _avg(cpu_proc),
        "cpu_process_peak": max(cpu_proc, default=0.0),
        "memory_rss_avg_mb": _avg(mem_rss),
        "memory_rss_peak_mb": max(mem_rss, default=0.0),
        "memory_uss_avg_mb": _avg(mem_uss),
        "memory_uss_peak_mb": max(mem_uss, default=0.0),
        "io_read_total_mb": sum(sample.io_read_mb for sample in samples),
        "io_write_total_mb": sum(sample.io_write_mb for sample in samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser("Run a command while logging resource metrics.")
    parser.add_argument("--command-b64", type=str, required=True, help="Base64-encoded shell command to execute.")
    parser.add_argument("--metrics-out", type=Path, required=True, help="Path to write metrics JSONL.")
    parser.add_argument("--summary-out", type=Path, required=True, help="Path to write summary JSON.")
    parser.add_argument("--log-out", type=Path, required=True, help="Path to tee stdout/stderr.")
    parser.add_argument("--interval", type=float, default=5.0, help="Sampling interval in seconds (default: 5).")
    parser.add_argument("--tag", type=str, default=None, help="Optional workload tag for progress logs.")
    parser.add_argument("--cwd", type=Path, default=None, help="Working directory for the command.")
    parser.add_argument(
        "--progress-out",
        type=Path,
        default=None,
        help="Optional JSONL path capturing parsed progress events.",
    )
    args = parser.parse_args()

    command = base64.b64decode(args.command_b64).decode("utf-8")
    cwd = str(args.cwd) if args.cwd else None

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.log_out.parent.mkdir(parents=True, exist_ok=True)
    if args.progress_out:
        args.progress_out.parent.mkdir(parents=True, exist_ok=True)

    start_ts = time.time()
    start_iso = _now_iso()

    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        text=True,
        bufsize=1,
        executable="/bin/bash",
    )

    ps_proc = psutil.Process(proc.pid)
    progress_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    stop_event = threading.Event()
    hostname = os.uname().nodename if hasattr(os, "uname") else "unknown-host"
    samples: list[MetricSample] = []

    metrics_thread = threading.Thread(
        target=_monitor_loop,
        args=(ps_proc, float(args.interval), args.metrics_out, progress_queue, stop_event, hostname, samples),
        daemon=True,
    )
    metrics_thread.start()

    progress_records: list[dict[str, Any]] = []

    with args.log_out.open("w", encoding="utf-8") as log_handle:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            log_handle.write(line)
            log_handle.flush()
            print(line, end="")

            payload = _safe_json(line)
            if payload and "event" in payload:
                event = str(payload["event"])
                if event.endswith("_progress") and "total" in payload and "current" in payload:
                    total = max(int(payload["total"]), 1)
                    current = int(payload["current"])
                    elapsed = time.time() - start_ts
                    remaining = max(total - current, 0)
                    avg_per_unit = elapsed / max(current, 1)
                    eta = avg_per_unit * remaining
                    progress_snapshot = {
                        "event": event,
                        "current": current,
                        "total": total,
                        "percent": (current / total) * 100.0,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta,
                        "timestamp": _now_iso(),
                    }
                    progress_queue.put(progress_snapshot)
                    progress_records.append(progress_snapshot)
                    eta_minutes = eta / 60.0
                    print(
                        f"[monitor] {event}: {current}/{total} "
                        f"({progress_snapshot['percent']:.1f}%) ETA ≈ {eta_minutes:0.1f} min",
                        file=sys.stderr,
                        flush=True,
                    )

    proc.wait()
    stop_event.set()
    metrics_thread.join(timeout=args.interval + 1.0)

    end_ts = time.time()
    end_iso = _now_iso()
    return_code = proc.returncode

    summary = _summarise(samples)
    summary.update(
        {
            "command": command,
            "start_utc": start_iso,
            "end_utc": end_iso,
            "duration_seconds": end_ts - start_ts,
            "return_code": return_code,
            "tag": args.tag,
        }
    )
    if progress_records:
        summary["progress"] = {
            "last": progress_records[-1],
            "events": len(progress_records),
        }

    if args.progress_out:
        with args.progress_out.open("w", encoding="utf-8") as handle:
            for record in progress_records:
                handle.write(json.dumps(record) + "\n")

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return return_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        sys.exit(130)
