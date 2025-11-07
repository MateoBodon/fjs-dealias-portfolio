from __future__ import annotations

from enum import Enum


class DiagnosticReason(str, Enum):
    ACCEPTED = "accepted"
    NO_DETECTIONS = "no_detections"
    INSUFFICIENT_GROUPS = "insufficient_groups"
    BALANCE_FAILURE = "balance_failure"
    DETECTION_ERROR = "detection_error"
    HOLDOUT_EMPTY = "holdout_empty"
    ALIGNMENT_REJECTED = "alignment_rejected"
    BASELINE_FAILURE = "baseline_failure"
    CONDITION_CAP = "condition_cap"

    def __str__(self) -> str:  # pragma: no cover - invoked implicitly
        return self.value
