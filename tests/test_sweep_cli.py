from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_sweep_dry_run(tmp_path: Path) -> None:
    output_root = tmp_path / "sweep_runs"
    cmd = [
        sys.executable,
        "experiments/equity_panel/sweep_acceptance.py",
        "--dry-run",
        "--output-root",
        str(output_root),
        "--grid",
        "default",
        "--config",
        "experiments/equity_panel/config.smoke.yaml",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        check=True,
    )
    summary_path = output_root.resolve() / "sweep_summary.csv"
    assert summary_path.exists(), f"Expected {summary_path} to be created.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    content = summary_path.read_text().strip()
    assert "delta_frac" in content
