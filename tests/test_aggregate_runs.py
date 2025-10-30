from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def _write_metrics(path: Path) -> None:
    path.write_text(
        "label,strategy,estimator,mean_mse,mean_qlike\n"
        "full,Equal Weight,De-aliased,0.1,0.2\n"
        "full,Equal Weight,Ledoit-Wolf,0.2,0.3\n",
        encoding="utf-8",
    )


def _write_summary(path: Path) -> None:
    payload = {
        "design": "oneway",
        "crisis_label": "smoke",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_runs_cli(tmp_path: Path) -> None:
    run_one = tmp_path / "run_one"
    run_two = tmp_path / "run_two"
    for run_dir in (run_one, run_two):
        run_dir.mkdir()
        _write_metrics(run_dir / "metrics_summary.csv")
        _write_summary(run_dir / "summary.json")

    out_csv = tmp_path / "aggregate.csv"
    out_tex = tmp_path / "aggregate.tex"

    cmd = [
        sys.executable,
        "tools/aggregate_runs.py",
        "--inputs",
        str(tmp_path / "run_*"),
        "--out",
        str(out_csv),
        "--tex-out",
        str(out_tex),
    ]
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True, check=True)
    assert out_csv.exists(), f"Expected aggregate CSV, stderr={result.stderr}"
    assert out_tex.exists(), "Expected LaTeX output file"

    with out_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows, "Aggregated CSV should contain rows"
    assert {"run", "crisis_label", "design"}.issubset(reader.fieldnames or [])
