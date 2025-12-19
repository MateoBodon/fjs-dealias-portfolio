from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class SkipReasonPrimary(str, Enum):
    CALIBRATION_MISSING = "calibration_missing_p_T"
    NO_OUTLIERS_ABOVE_EDGE = "no_outliers_above_edge"
    INSTABILITY_IN_NEIGHBORHOOD = "instability_in_a_neighborhood"
    OFF_COMPONENT_LEAK = "off_component_leak"
    ENERGY_BELOW_FLOOR = "energy_below_floor"
    INVALID_MU = "invalid_mu"
    MU_NONFINITE = "mu_nonfinite"
    T_VECTOR_COMPUTE_ERROR = "t_vector_compute_error"
    T_VECTOR_TARGET_ZERO = "t_vector_target_zero"
    T_VECTOR_OFF_COMPONENT = "t_vector_off_component"
    EPS_GUARD = "eps_guard"
    NO_ISOLATED_SPIKE = "no_isolated_spike"
    NESTED_GUARD = "nested_guard"
    DIAGNOSTIC_FAILURE = "diagnostic_failure"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# Guardrail counters emitted by dealias diagnostics. Order matters for tie‑breaks.
DIAGNOSTIC_GUARD_KEYS = [
    "edge_buffer",
    "stability_fail",
    "off_component_ratio",
    "energy_floor",
    "neg_mu",
    "eps",
    "tvec_compute_error",
    "tvec_target_zero",
    "tvec_off_component",
    "mu_nonfinite",
]

# Priority order for attributing a dominant guardrail when multiple fire.
DIAG_KEY_PRIORITY = [
    "edge_buffer",
    "stability_fail",
    "off_component_ratio",
    "energy_floor",
    "neg_mu",
    "tvec_off_component",
    "tvec_target_zero",
    "tvec_compute_error",
    "mu_nonfinite",
    "eps",
]

DIAG_KEY_TO_REASON = {
    "edge_buffer": SkipReasonPrimary.NO_OUTLIERS_ABOVE_EDGE,
    "stability_fail": SkipReasonPrimary.INSTABILITY_IN_NEIGHBORHOOD,
    "off_component_ratio": SkipReasonPrimary.OFF_COMPONENT_LEAK,
    "energy_floor": SkipReasonPrimary.ENERGY_BELOW_FLOOR,
    "neg_mu": SkipReasonPrimary.INVALID_MU,
    "eps": SkipReasonPrimary.NO_OUTLIERS_ABOVE_EDGE,
    "tvec_compute_error": SkipReasonPrimary.T_VECTOR_COMPUTE_ERROR,
    "tvec_target_zero": SkipReasonPrimary.T_VECTOR_TARGET_ZERO,
    "tvec_off_component": SkipReasonPrimary.T_VECTOR_OFF_COMPONENT,
    "mu_nonfinite": SkipReasonPrimary.MU_NONFINITE,
}


@dataclass
class SkipAttribution:
    primary: str
    detail: str = ""
    exception_type: str | None = None


def normalise_diag_counts(diag_local: Mapping[str, Any] | None) -> dict[str, int]:
    """Project raw diagnostics to a stable guardrail count dictionary."""

    counts: dict[str, int] = {}
    if not diag_local:
        return counts
    for key in DIAGNOSTIC_GUARD_KEYS:
        try:
            counts[key] = int(diag_local.get(key, 0))
        except Exception:
            counts[key] = 0
    return counts


def infer_primary_reason(
    diag_local: Mapping[str, Any] | None,
    *,
    calibration_missing: bool,
    isolated_spikes: int | None,
    calibration_detail: Mapping[str, Any] | None = None,
) -> SkipAttribution:
    """Map guardrail diagnostics to a stable primary skip reason."""

    if calibration_missing:
        detail = ""
        if calibration_detail:
            parts = []
            edge = calibration_detail.get("edge_mode")
            if edge:
                parts.append(f"edge_mode={edge}")
            p_val = calibration_detail.get("p")
            t_val = calibration_detail.get("t")
            if p_val is not None:
                parts.append(f"p={p_val}")
            if t_val is not None:
                parts.append(f"t={t_val}")
            if parts:
                detail = "; ".join(parts)
        return SkipAttribution(primary=str(SkipReasonPrimary.CALIBRATION_MISSING), detail=detail)

    counts = normalise_diag_counts(diag_local)
    total = sum(counts.values())
    if total > 0:
        best_key = None
        best_count = -1
        for key in DIAG_KEY_PRIORITY:
            count = counts.get(key, 0)
            if count > best_count:
                best_key = key
                best_count = count
        if best_key:
            reason = DIAG_KEY_TO_REASON.get(best_key, SkipReasonPrimary.NO_OUTLIERS_ABOVE_EDGE)
            summary = ", ".join(
                f"{key}={counts[key]}"
                for key in DIAGNOSTIC_GUARD_KEYS
                if counts.get(key, 0) > 0
            )
            detail = f"dominant={best_key} ({best_count})"
            if summary:
                detail = f"{detail}; counts: {summary}"
            return SkipAttribution(primary=str(reason), detail=detail)

    if isolated_spikes is not None and isolated_spikes == 0:
        return SkipAttribution(
            primary=str(SkipReasonPrimary.NO_ISOLATED_SPIKE),
            detail="isolated_spikes=0",
        )

    return SkipAttribution(
        primary=str(SkipReasonPrimary.NO_OUTLIERS_ABOVE_EDGE),
        detail="no guardrail counts; defaulted to edge buffer gate",
    )
