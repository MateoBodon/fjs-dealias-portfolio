from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass
class RunMeta:
    """Lightweight metadata summary for a single run.

    Fields are intentionally flat and JSON-serialisable.
    """

    git_sha: str
    n_assets: int | None
    window_weeks: int | None
    horizon_weeks: int | None
    replicates_per_week: int | None

    # Detection / de-aliasing controls
    delta: float | None
    delta_frac: float | None
    a_grid: int | None
    signed_a: bool | None
    sigma2_plugin: str | None

    # Outcomes
    detections_total: int
    L: int | None

    # Hashes of generated figure PDFs for provenance
    figure_sha256: dict[str, str]

    # Persist the resolved configuration as captured by the pipeline (optional)
    config_snapshot: Mapping[str, Any] | None


def _git_sha() -> str:
    """Return the short git SHA for the current repository, or 'unknown'."""

    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_pdf_hashes(directory: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not directory.exists():
        return hashes
    for pdf in directory.glob("*.pdf"):
        try:
            hashes[pdf.name] = _sha256_of_file(pdf)
        except Exception:
            continue
    return hashes


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        return None
    return None


def _count_detections(det_summary_path: Path) -> tuple[int, int]:
    """Return (detections_total, L_max) from detection_summary.csv if present."""

    if not det_summary_path.exists():
        return 0, 0
    try:
        df = pd.read_csv(det_summary_path)
    except Exception:
        return 0, 0
    if "n_detections" not in df.columns:
        return 0, 0
    n_det = int(pd.to_numeric(df["n_detections"], errors="coerce").fillna(0).sum())
    L_max = int(pd.to_numeric(df["n_detections"], errors="coerce").fillna(0).max())
    return n_det, L_max


def write_run_meta(
    output_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    delta: float | None = None,
    delta_frac: float | None = None,
    a_grid: int | None = None,
    signed_a: bool | None = None,
    sigma2_plugin: str | None = None,
) -> Path:
    """Create a run_meta.json artifact in ``output_dir``.

    Parameters
    ----------
    output_dir
        Directory where run artifacts (CSVs, figures) live. The file will be
        written as ``run_meta.json`` in this directory.
    config
        Optional resolved configuration mapping captured by the pipeline.
    delta, delta_frac, a_grid, signed_a, sigma2_plugin
        De-aliasing controls to record for reproducibility.

    Returns
    -------
    pathlib.Path
        Path to the written ``run_meta.json`` file.
    """

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary = _load_optional_json(out_path / "summary.json") or {}
    n_assets = int(summary.get("n_assets")) if "n_assets" in summary else None
    window_weeks = (
        int(summary.get("window_weeks")) if "window_weeks" in summary else None
    )
    horizon_weeks = (
        int(summary.get("horizon_weeks")) if "horizon_weeks" in summary else None
    )
    replicates = (
        int(summary.get("replicates_per_week"))
        if "replicates_per_week" in summary
        else None
    )

    det_total, L_max = _count_detections(out_path / "detection_summary.csv")
    pdf_hashes = _collect_pdf_hashes(out_path)

    meta = RunMeta(
        git_sha=_git_sha(),
        n_assets=n_assets,
        window_weeks=window_weeks,
        horizon_weeks=horizon_weeks,
        replicates_per_week=replicates,
        delta=float(delta) if delta is not None else None,
        delta_frac=float(delta_frac) if delta_frac is not None else None,
        a_grid=int(a_grid) if a_grid is not None else None,
        signed_a=bool(signed_a) if signed_a is not None else None,
        sigma2_plugin=str(sigma2_plugin) if sigma2_plugin is not None else None,
        detections_total=int(det_total),
        L=int(L_max) if L_max is not None else None,
        figure_sha256=pdf_hashes,
        config_snapshot=dict(config) if config is not None else None,
    )

    meta_path = out_path / "run_meta.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(meta), fh, indent=2)
    return meta_path

