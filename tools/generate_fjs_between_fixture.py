from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fjs.detector_contract import assess_power_curve
from fjs.overlay import OverlayConfig, detect_spikes

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = (
    REPO_ROOT / "docs/artifacts/detector-contract-reference/between_mechanism_v1"
)
DEFAULT_INPUT_SPEC = DEFAULT_ARTIFACT_DIR / "input_spec.json"
PREDECLARATION_COMMIT = "82d1ffc0b2fc7c4c39e820b7aae3c4ad0bcdb43c"
PRODUCTION_REPAIR_COMMIT = "4437571acf4b42bd1f4c7db8a9616b623c5a3a7b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def load_input_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "fjs-between-mechanism-input/v1":
        raise ValueError("Unsupported between-mechanism input schema.")
    if payload.get("inject_mode") != "between":
        raise ValueError("The frozen mechanism fixture requires inject_mode=between.")
    if payload.get("master_seed") != 20260710:
        raise ValueError("The frozen mechanism fixture master seed changed.")
    if payload.get("trial_count") != 12:
        raise ValueError("The frozen mechanism fixture trial count changed.")
    if payload.get("mu_grid") != [0.0, 6.0]:
        raise ValueError("The frozen mechanism fixture mu grid changed.")
    return payload


def draw_trial(
    seed_sequence: np.random.SeedSequence,
    *,
    groups: int,
    replicates: int,
    features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed_sequence)
    direction = rng.standard_normal(features)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Frozen trial produced an invalid planted direction.")
    direction = np.asarray(direction / norm, dtype=np.float64)
    group_scores = np.asarray(rng.standard_normal(groups), dtype=np.float64)
    residuals = np.asarray(
        rng.standard_normal((groups, replicates, features)),
        dtype=np.float64,
    )
    return direction, group_scores, residuals


def build_panel(
    *,
    direction: np.ndarray,
    group_scores: np.ndarray,
    residuals: np.ndarray,
    mu: float,
    within_noise_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    between = np.sqrt(float(mu)) * np.outer(group_scores, direction)
    observations = between[:, None, :] + float(within_noise_scale) * residuals
    groups, replicates, features = observations.shape
    labels = np.repeat(np.arange(groups, dtype=np.intp), replicates)
    return (
        np.asarray(observations.reshape(groups * replicates, features)),
        labels,
    )


def _overlay_config(payload: Mapping[str, Any]) -> OverlayConfig:
    return OverlayConfig(
        q_max=int(payload["q_max"]),
        delta=float(payload["delta"]),
        eps=float(payload["eps"]),
        stability_eta_deg=float(payload["stability_eta_deg"]),
        a_grid=int(payload["a_grid"]),
        require_isolated=bool(payload["require_isolated"]),
        off_component_cap=float(payload["off_component_cap"]),
        edge_mode=str(payload["edge_mode"]),
        gate_mode=str(payload["gate_mode"]),
        coarse_candidate=bool(payload["coarse_candidate"]),
    )


def run_trials(spec: Mapping[str, Any]) -> list[dict[str, object]]:
    design = spec["design"]
    if not isinstance(design, Mapping):
        raise TypeError("design must be a mapping.")
    detector = spec["detector"]
    if not isinstance(detector, Mapping):
        raise TypeError("detector must be a mapping.")

    groups = int(design["groups"])
    replicates = int(design["replicates"])
    features = int(design["features"])
    within_noise_scale = float(design["within_noise_scale"])
    trial_count = int(spec["trial_count"])
    mu_grid = [float(value) for value in spec["mu_grid"]]
    master = np.random.SeedSequence(int(spec["master_seed"]))
    children = master.spawn(trial_count)
    config = _overlay_config(detector)

    rows: list[dict[str, object]] = []
    for trial_index, child in enumerate(children):
        direction, group_scores, residuals = draw_trial(
            child,
            groups=groups,
            replicates=replicates,
            features=features,
        )
        child_key = ".".join(str(value) for value in child.spawn_key)
        for mu in mu_grid:
            observations, labels = build_panel(
                direction=direction,
                group_scores=group_scores,
                residuals=residuals,
                mu=mu,
                within_noise_scale=within_noise_scale,
            )
            stats: dict[str, Any] = {}
            accepted_candidates = detect_spikes(
                observations,
                labels,
                config=config,
                stats=stats,
            )
            pre_gate = stats.get("pre_gate", {})
            raw_count = int(pre_gate.get("raw_outliers_found", 0))
            accepted_count = len(accepted_candidates)
            if accepted_count > 0 and raw_count <= 0:
                raise RuntimeError("Acceptance cannot occur without detection.")
            if accepted_candidates and {
                str(candidate["candidate_source"]) for candidate in accepted_candidates
            } != {"fjs"}:
                raise RuntimeError("Non-FJS candidates entered the mechanism fixture.")
            rows.append(
                {
                    "trial_index": trial_index,
                    "child_spawn_key": child_key,
                    "mu": mu,
                    "inject_mode": "between",
                    "detected": int(raw_count > 0),
                    "accepted": int(accepted_count > 0),
                    "pre_gate_candidate_count": raw_count,
                    "accepted_candidate_count": accepted_count,
                }
            )
    return rows


def aggregate_curve(
    trial_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    mu_values = sorted({float(row["mu"]) for row in trial_rows})
    curve: list[dict[str, object]] = []
    for mu in mu_values:
        cell = [row for row in trial_rows if float(row["mu"]) == mu]
        detected = sum(int(row["detected"]) for row in cell)
        accepted = sum(int(row["accepted"]) for row in cell)
        count = len(cell)
        curve.append(
            {
                "mu": mu,
                "inject_mode": "between",
                "detection_rate": detected / count,
                "acceptance_rate": accepted / count,
                "n_windows": count,
                "n_detected": detected,
                "n_accepted": accepted,
            }
        )
    return curve


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return buffer.getvalue().encode("utf-8")


def render_outputs(
    trial_rows: Sequence[Mapping[str, object]],
) -> tuple[bytes, bytes, list[dict[str, object]]]:
    curve_rows = aggregate_curve(trial_rows)
    curve_bytes = _csv_bytes(
        curve_rows,
        (
            "mu",
            "inject_mode",
            "detection_rate",
            "acceptance_rate",
            "n_windows",
            "n_detected",
            "n_accepted",
        ),
    )
    trial_bytes = _csv_bytes(
        trial_rows,
        (
            "trial_index",
            "child_spawn_key",
            "mu",
            "inject_mode",
            "detected",
            "accepted",
            "pre_gate_candidate_count",
            "accepted_candidate_count",
        ),
    )
    return curve_bytes, trial_bytes, curve_rows


def generate(input_spec: Path, output_dir: Path) -> dict[str, Any]:
    spec = load_input_spec(input_spec)
    trial_rows = run_trials(spec)
    curve_bytes, trial_bytes, curve_rows = render_outputs(trial_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / "curve.csv"
    trials_path = output_dir / "trials.csv"
    curve_path.write_bytes(curve_bytes)
    trials_path.write_bytes(trial_bytes)

    assessment = assess_power_curve(
        curve_rows,
        expected_inject_mode="between",
    )
    generator_path = Path(__file__).resolve()
    manifest = {
        "schema_version": "fjs-between-mechanism-manifest/v1",
        "claim_boundary": spec["claim_boundary"],
        "inject_mode": "between",
        "predeclaration_commit": PREDECLARATION_COMMIT,
        "production_repair_commit": PRODUCTION_REPAIR_COMMIT,
        "source_commit": _git_value("rev-parse", "HEAD"),
        "source_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "inputs": {
            "input_spec": str(input_spec.relative_to(REPO_ROOT)),
            "input_spec_sha256": _sha256(input_spec),
            "generator": str(generator_path.relative_to(REPO_ROOT)),
            "generator_sha256": _sha256(generator_path),
        },
        "outputs": {
            "curve": "curve.csv",
            "curve_sha256": _sha256(curve_path),
            "trials": "trials.csv",
            "trials_sha256": _sha256(trials_path),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "assessment": {
            "passed": assessment.passed,
            "reasons": list(assessment.reasons),
            "null_detection_rate": assessment.null_detection_rate,
            "strong_detection_rate": assessment.strong_detection_rate,
            "strong_acceptance_rate": assessment.strong_acceptance_rate,
            "detection_gain": assessment.detection_gain,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def check_reproduction(input_spec: Path, artifact_dir: Path) -> None:
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="fjs-between-check-") as raw_temp:
        temp_dir = Path(raw_temp)
        generate(input_spec, temp_dir)
        for filename in ("curve.csv", "trials.csv"):
            observed = _sha256(temp_dir / filename)
            expected = str(manifest["outputs"][f"{filename[:-4]}_sha256"])
            if observed != expected:
                raise RuntimeError(
                    f"Reproduction mismatch for {filename}: {observed} != {expected}."
                )
    if _sha256(input_spec) != manifest["inputs"]["input_spec_sha256"]:
        raise RuntimeError("Input-spec hash no longer matches the manifest.")
    if _sha256(Path(__file__).resolve()) != manifest["inputs"]["generator_sha256"]:
        raise RuntimeError("Generator hash no longer matches the manifest.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or reproduce-check the frozen FJS between fixture."
    )
    parser.add_argument("--input-spec", type=Path, default=DEFAULT_INPUT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check:
        check_reproduction(args.input_spec.resolve(), args.output_dir.resolve())
        print("between_mechanism_fixture: reproducible")
        return 0
    manifest = generate(args.input_spec.resolve(), args.output_dir.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
