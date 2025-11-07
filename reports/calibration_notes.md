# Calibration Notes

## 2025-11-07 — Lowered δ for deterministic RC
- **Observation:** All deterministic DoW/vol RC windows (tyler + SCM edges) logged `acceptance_rate = 0` across calm and crisis regimes (e.g., DoW-tyler calm/crisis windows: 69/321, all rejected). Crisis buckets in particular showed no accepted detections despite stable edge margins, triggering the decision-rule nudge (“decrease delta by 0.05 when crisis acceptance ≈ 0%”).
- **Change:** Updated `calibration/defaults.json` (generated at 2025-11-07T01:18Z) to reduce `delta` from 0.35 → 0.30 for both Tyler and SCM cells (`G36` × `r12-16|r17-22`, `p64-96`). `delta_frac`, `stability_eta_deg`, and replicate bins are unchanged.
- **Next steps:** Re-run the affected RC targets (DoW tyler/SCM and vol tyler) under deterministic mode to validate that crisis acceptance recovers toward the 2‑5% band while calm remains gated.

## 2025-11-07T19:10Z — Factor-registry rerun (no additional delta change)
- **Observation:** After enforcing the FF5+MOM registry loader (`data/factors/registry.json`, SHA `469d44ad…08ca`), deterministic DoW (Tyler + SCM) and vol-state (Tyler) RCs still reported `percent_changed = 0` and `reason_code ∈ {detection_error, balance_failure}` for every window (full run counts: DoW detection_error 471 / balance_failure 342 / holdout_empty 25; Vol detection_error 97 / balance_failure 761 / holdout_empty 34). Prewhiten telemetry now shows `mode_effective = ff5mom` with `r2_mean ≈ 0.39`, so the zero-acceptance issue is upstream of the factor ingestion.
- **Decision:** Keep `delta = 0.30` and the existing η / Δ_frac settings; further threshold relaxations would not resolve the `detect_spikes` failures logged per window. Focus next steps on stabilising the overlay detector before re-calibrating.
- **Artifacts:** `reports/rc-20251107/{dow-tyler,dow-scm,vol-tyler}/run.json` capture the registry-backed runs plus acceptance histograms (`acceptance_hist_{dow,vol}.png`, `edge_margin_hist_{dow,vol}.png`).
