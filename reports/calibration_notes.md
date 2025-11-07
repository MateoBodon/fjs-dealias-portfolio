# Calibration Notes

## 2025-11-07 — Lowered δ for deterministic RC
- **Observation:** All deterministic DoW/vol RC windows (tyler + SCM edges) logged `acceptance_rate = 0` across calm and crisis regimes (e.g., DoW-tyler calm/crisis windows: 69/321, all rejected). Crisis buckets in particular showed no accepted detections despite stable edge margins, triggering the decision-rule nudge (“decrease delta by 0.05 when crisis acceptance ≈ 0%”).
- **Change:** Updated `calibration/defaults.json` (generated at 2025-11-07T01:18Z) to reduce `delta` from 0.35 → 0.30 for both Tyler and SCM cells (`G36` × `r12-16|r17-22`, `p64-96`). `delta_frac`, `stability_eta_deg`, and replicate bins are unchanged.
- **Next steps:** Re-run the affected RC targets (DoW tyler/SCM and vol tyler) under deterministic mode to validate that crisis acceptance recovers toward the 2‑5% band while calm remains gated.
