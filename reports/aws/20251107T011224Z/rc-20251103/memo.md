# RC 2025-11-03 Memo

## Summary
- Day-of-Week and volatility-state replicates ran on 2023-01-01 to 2023-06-30 daily returns with 40×10 window/horizon; no overlay activations were accepted across full, calm, or crisis regimes.
- Baseline covariance shrinkage used RIE (dow) and EWMA (vol) with post-substitution metrics matching baseline performance (ΔMSE ≈ 0).
- Prewhitening summary for both runs recorded mean R² ≈ 0.00, indicating the factor proxy contributed minimally on this smoke slice.

## Key Artifacts
- `reports/rc-20251103/dow/full/metrics.csv` — regime metrics and ΔMSE vs baseline.
- `reports/rc-20251103/dow/overlay_toggle.md` — detection telemetry per regime.
- `reports/rc-20251103/vol/full/metrics.csv` — EWMA baseline comparison for volatility-state design.
- `reports/rc-20251103/vol/overlay_toggle.md` — overlay acceptance summary (all zeros).

## Limits & Next Steps
- Smoke configuration trims the sample to six months; rerun with extended history before publishing final claims.
- Detection gating remains inactive; evaluate calibration thresholds against higher-volatility slices to validate the 2% FPR guardrail.
- Integrate gallery/memo automation (`make rc` / `make gallery`) once final RC dataset is selected.
