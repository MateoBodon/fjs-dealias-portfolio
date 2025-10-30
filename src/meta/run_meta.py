from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

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
    code_signature: str | None
    estimator: str | None
    design: str | None
    nested_replicates: int | None
    solver_used: list[str] | None
    label: str | None
    crisis_label: str | None
    preprocess_flags: dict[str, str] | None

    # Outcomes
    detections_total: int
    L: int | None

    # Hashes of generated figure PDFs for provenance
    figure_sha256: dict[str, str]

    # Persist the resolved configuration as captured by the pipeline (optional)
    config_snapshot: Mapping[str, Any] | None

    # Panel metadata
    panel_universe_hash: str | None
    panel_preprocess_flags: dict[str, str] | None


_DEFAULT_SIGNATURE_TARGETS = [
    "src/fjs/dealias.py",
    "src/fjs/mp.py",
    "src/fjs/balanced.py",
    "src/data/panels.py",
    "src/fjs/theta_solver.py",
    "src/meta/cache.py",
    "src/evaluation/evaluate.py",
]

_DEFAULT_SIGNATURE_GLOBS = [
    "src/fjs/balanced_*.py",
    "src/finance/*.py",
]


def code_signature(targets: Iterable[str | Path] | None = None) -> str:
    """Compute a SHA-256 signature over core de-aliasing code."""

    root = Path(__file__).resolve().parents[2]
    paths: list[Path] = []
    if targets is None:
        for item in _DEFAULT_SIGNATURE_TARGETS:
            paths.append(root / item)
        for pattern in _DEFAULT_SIGNATURE_GLOBS:
            paths.extend(sorted(root.glob(pattern)))
    else:
        for item in targets:
            path = Path(item)
            paths.append(path if path.is_absolute() else (root / path))

    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if resolved.exists():
            seen.add(resolved)
            ordered.append(resolved)

    h = hashlib.sha256()
    for path in ordered:
        try:
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
        except Exception:
            continue
    marker = "::".join(str(p) for p in ordered).encode("utf-8")
    h.update(marker)
    return h.hexdigest()


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
    code_signature_hash: str | None = None,
    estimator: str | None = None,
    design: str | None = None,
    nested_replicates: int | None = None,
    preprocess_flags: Mapping[str, Any] | None = None,
    label: str | None = None,
    crisis_label: str | None = None,
    solver_used: Iterable[str] | None = None,
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
    design, nested_replicates
        Balanced design metadata (oneway/nested) and replicates count.
    preprocess_flags
        Optional preprocessing flags applied before balancing.
    label
        Descriptive label for the run (e.g., "full_oneway_...").
    crisis_label
        Optional short tag when the run corresponds to a crisis slice.
    solver_used
        Optional iterable of solver identifiers observed during the run.
    estimator
        Covariance estimator name applied during evaluation, if any.

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
    manifest = _load_optional_json(out_path / "panel_manifest.json") or {}
    manifest_preprocess = (
        {str(k): str(v) for k, v in manifest.get("preprocess_flags", {}).items()}
        if isinstance(manifest.get("preprocess_flags"), dict)
        else None
    )

    summary_design = str(summary.get("design")) if summary.get("design") else None
    if summary_design is None and design is not None:
        summary_design = str(design)

    summary_nested = summary.get("nested_replicates")
    if summary_nested is None and nested_replicates is not None:
        summary_nested = int(nested_replicates)
    elif summary_nested is not None:
        summary_nested = int(summary_nested)

    summary_label = summary.get("label") if summary.get("label") else label
    summary_crisis = summary.get("crisis_label") if summary.get("crisis_label") else None
    if summary_crisis is None and crisis_label is not None:
        summary_crisis = crisis_label

    summary_preprocess = summary.get("preprocess_flags")
    if isinstance(summary_preprocess, dict):
        run_preprocess_flags = {
            str(k): str(v) for k, v in summary_preprocess.items()
        }
    elif preprocess_flags is not None:
        run_preprocess_flags = {str(k): str(v) for k, v in preprocess_flags.items()}
    else:
        run_preprocess_flags = manifest_preprocess

    solver_candidates: list[str] | None
    if isinstance(summary.get("solver_used"), list):
        solver_candidates = [str(item) for item in summary["solver_used"] if item]
    elif solver_used is not None:
        solver_candidates = sorted({str(item) for item in solver_used if item})
    else:
        solver_candidates = None
    if solver_candidates is not None and not solver_candidates:
        solver_candidates = None

    if run_preprocess_flags is not None and not run_preprocess_flags:
        run_preprocess_flags = None

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
        code_signature=str(code_signature_hash) if code_signature_hash else None,
        estimator=str(estimator) if estimator is not None else None,
        design=summary_design,
        nested_replicates=int(summary_nested) if summary_nested is not None else None,
        solver_used=solver_candidates,
        label=str(summary_label) if summary_label is not None else None,
        crisis_label=str(summary_crisis) if summary_crisis is not None else None,
        preprocess_flags=run_preprocess_flags,
        detections_total=int(det_total),
        L=int(L_max) if L_max is not None else None,
        figure_sha256=pdf_hashes,
        config_snapshot=dict(config) if config is not None else None,
        panel_universe_hash=str(manifest.get("universe_hash", "")) or None,
        panel_preprocess_flags=manifest_preprocess,
    )

    meta_path = out_path / "run_meta.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(meta), fh, indent=2)
    return meta_path
