from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

CandidateSource = Literal["fjs", "coarse", "oracle", "sham"]

CANDIDATE_SOURCES: frozenset[str] = frozenset({"fjs", "coarse", "oracle", "sham"})


def require_candidate_source(candidate: Mapping[str, Any]) -> CandidateSource:
    """Return the explicit candidate source or fail closed.

    Candidate provenance is part of the scientific contract: an FJS detection,
    a generic coarse spectral candidate, an oracle mechanism control, and a
    magnitude-matched sham are different treatments and cannot be inferred from
    solver labels after the fact.
    """

    raw_source = candidate.get("candidate_source")
    source = str(raw_source).strip().lower() if raw_source is not None else ""
    if source not in CANDIDATE_SOURCES:
        allowed = ", ".join(sorted(CANDIDATE_SOURCES))
        raise ValueError(
            "Detection candidate_source must be explicit and one of "
            f"{{{allowed}}}; received {raw_source!r}."
        )
    return cast(CandidateSource, source)


def candidate_source_counts(
    candidates: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    """Count validated candidate sources in stable lexical order."""

    counts = Counter(require_candidate_source(candidate) for candidate in candidates)
    return {source: int(counts[source]) for source in sorted(counts)}


def format_candidate_source_counts(counts: Mapping[str, int]) -> str:
    """Render source counts for CSV diagnostics without losing attribution."""

    return "|".join(f"{source}:{int(counts[source])}" for source in sorted(counts))


@dataclass(frozen=True)
class PowerCurveAssessment:
    passed: bool
    reasons: tuple[str, ...]
    null_detection_rate: float
    strong_detection_rate: float
    strong_acceptance_rate: float
    detection_gain: float


def assess_power_curve(
    rows: Sequence[Mapping[str, Any]],
    *,
    null_fpr_max: float = 0.075,
    strong_power_min: float = 0.80,
    power_gain_min: float = 0.50,
) -> PowerCurveAssessment:
    """Evaluate the bounded detector stop-line against a persisted power curve.

    This is deliberately a reducer, not a simulation harness. Synthetic data
    remain mechanism calibration only; the reducer prevents a flat or
    underpowered curve from being promoted into a real-data performance run.
    """

    if not rows:
        raise ValueError("Power curve must contain at least one row.")

    parsed: list[tuple[float, float, float, int]] = []
    for row in rows:
        missing = {
            key
            for key in ("mu", "detection_rate", "acceptance_rate", "n_windows")
            if key not in row
        }
        if missing:
            raise ValueError(
                "Power curve row is missing required fields: "
                + ", ".join(sorted(missing))
            )
        mu = float(row["mu"])
        detection = float(row["detection_rate"])
        acceptance = float(row["acceptance_rate"])
        n_windows = int(float(row["n_windows"]))
        if n_windows <= 0:
            raise ValueError("Power curve n_windows values must be positive.")
        if not 0.0 <= detection <= 1.0 or not 0.0 <= acceptance <= 1.0:
            raise ValueError("Power curve rates must lie in [0, 1].")
        parsed.append((mu, detection, acceptance, n_windows))

    parsed.sort(key=lambda item: item[0])
    if parsed[0][0] != 0.0:
        raise ValueError("Power curve must include a mu=0 null row.")
    if parsed[-1][0] <= 0.0:
        raise ValueError("Power curve must include at least one positive spike.")

    null_detection = parsed[0][1]
    strong_detection = parsed[-1][1]
    strong_acceptance = parsed[-1][2]
    detection_gain = strong_detection - null_detection
    reasons: list[str] = []
    if null_detection > float(null_fpr_max):
        reasons.append("null_detection_rate_above_limit")
    if strong_detection < float(strong_power_min):
        reasons.append("strong_signal_detection_below_minimum")
    if strong_acceptance < float(strong_power_min):
        reasons.append("strong_signal_acceptance_below_minimum")
    if detection_gain < float(power_gain_min):
        reasons.append("detection_gain_below_minimum")

    detection_rates = [item[1] for item in parsed]
    acceptance_rates = [item[2] for item in parsed]
    tolerance = 1e-12
    if any(
        right + tolerance < left
        for left, right in zip(detection_rates, detection_rates[1:], strict=False)
    ):
        reasons.append("detection_power_not_monotone")
    if any(
        right + tolerance < left
        for left, right in zip(acceptance_rates, acceptance_rates[1:], strict=False)
    ):
        reasons.append("acceptance_power_not_monotone")

    return PowerCurveAssessment(
        passed=not reasons,
        reasons=tuple(reasons),
        null_detection_rate=float(null_detection),
        strong_detection_rate=float(strong_detection),
        strong_acceptance_rate=float(strong_acceptance),
        detection_gain=float(detection_gain),
    )
